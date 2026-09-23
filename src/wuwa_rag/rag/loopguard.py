"""生成侧复读兜底：检测并截断「整句重复 / 周期块循环」的退化输出。

为什么需要它：aemeath 是 8B 角色扮演模型，在「资料不够/检索不到」时容易陷入
枚举与独白循环。repeat_penalty 等采样参数只能降低概率，压不死（而且加码惩罚会
反过来造成「提前收尾丢行」，见 config.LLM_REPEAT_PENALTY）；这里做最后一道闸——
命中即中断生成并截断复读尾巴。

两类退化，都在本项目实测到过：
  ① **整句重复**：同一条（不短的）句子出现 ≥3 次。
  ② **周期块循环**：一组行（周期 2~LLM_LOOP_PERIOD_MAX 行）整体重复 ≥3 遍。
     实测「清宵配队」输出把 6 行一组的目标配队循环了 5 遍，而且每行尾部还带
     `*1`…`*28` 的计数后缀（`守岸人 + 尤诺*17`）——后缀让**每一行看起来都唯一**，
     纯精确判重完全抓不到（这正是线上漏网的原因）。所以周期检测时先把行尾计数
     标记剥掉再比较。
     ⚠️ 剥标记**只用于周期序列比较**，不参与精确计数：否则「- 贝币×5000」与
     「- 贝币×10000」会被归一成同一行，真实材料表会被误杀。

判重按行/句级：实测退化样本的特征是整行周期性重复，而非单字重复，
token 级 n-gram 检测对这种形态不敏感且容易误杀正常排比。
"""
from __future__ import annotations

import re
from typing import NamedTuple

# 句末终止符（含中文标点与波浪号，角色扮演模型爱用「~」结尾）与换行
_SENT_END = re.compile(r"[。！？!?~～\n]+")
# 归一化时剔除所有空白：退化样本会出现「C O S T」这种被空格撑开的字符
_WS = re.compile(r"\s+")
# 行尾计数标记（`*17` / `×3` / `x5`），仅用于周期比较（见模块 docstring 的 ⚠️）
_COUNTER_TAIL = re.compile(r"[*×x]\s*\d+$")

# 默认阈值（chain 会从 config 覆盖；直接调用的测试用这些）
_DEF_MAX_REPEAT = 2        # 同一行最多允许出现的次数
_DEF_MIN_CHARS = 12        # 「长句」门槛：≥ 此长度才做精确重复计数
_DEF_MIN_ITEM = 4          # 参与判重的单行最短长度（归一化后），滤掉「嗯~」「好的。」
_DEF_PERIOD_MAX = 12       # 周期长度上限（行）
_DEF_MIN_CYCLES = 3        # 同一周期至少重复几遍才判退化


class _Cand(NamedTuple):
    """一行候选（已闭合，可安全判重）。"""

    start: int      # 在原文中的起始偏移
    end: int        # 在原文中的结束偏移（含终止符）
    key: str        # 精确重复用：归一化文本
    pkey: str       # 周期比较用：key 再去掉行尾计数标记


def _norm(sent: str) -> str:
    return _WS.sub("", sent)


def _period_key(key: str) -> str:
    """去掉行尾计数标记；全被剥掉时退回原 key（避免空 key 让所有行相等）。"""
    out = _COUNTER_TAIL.sub("", key)
    return out or key


def _iter_closed_sentences(text: str, start: int = 0):
    """从 start 开始逐句产出 (句首偏移, 句子文本, 下一句起点)。

    句子文本**不含**终止符；只产出已闭合的句子（后面必须紧跟终止符），
    未闭合的尾句会被后续 token 改写，提前判重会误杀。

    feed() 与 trim_loop() 必须共用这个迭代器：两处若各自切句，
    一边含终止符一边不含，归一化后的 key 就对不上（_norm 只去空白、不去标点）。
    """
    pos = start
    n = len(text)
    while pos < n:
        m = _SENT_END.search(text, pos)
        if not m:
            return
        yield pos, text[pos:m.start()], m.end()
        pos = m.end()


def _collect(text: str, min_item: int) -> list[_Cand]:
    """抽出所有可判重的行（跳过过短碎句、Markdown 表格行、分隔线）。

    表格行必须排除——突破材料类答案的表格行（`|` 开头）按换行切分后天然重复，
    会误杀。
    """
    out: list[_Cand] = []
    for start, sent, end in _iter_closed_sentences(text):
        s = sent.strip()
        if len(s) < min_item or s.startswith("|") or s.startswith("---"):
            continue
        key = _norm(s)
        if not key:
            continue
        out.append(_Cand(start, end, key, _period_key(key)))
    return out


def _cycle_at(seg: list[str], p: int, min_cycles: int) -> bool:
    """seg 是否正好是「同一个 p 行周期重复 min_cycles 遍」。"""
    cyc = seg[:p]
    if len(set(cyc)) < 2:
        # 整块都是同一行 → 交给精确重复规则，避免把「嗯~嗯~嗯~」也算成周期
        return False
    return all(seg[i * p:(i + 1) * p] == cyc for i in range(1, min_cycles))


def _extend(keys: list[str], start: int, p: int) -> int:
    """向前扩展：把更早的、同一周期的重复块也一起纳入（截断要连同它们切掉）。"""
    while start - p >= 0 and keys[start - p:start] == keys[start:start + p]:
        start -= p
    return start


def _period_start_tail(keys: list[str], period_max: int, min_cycles: int) -> int | None:
    """**只看尾部**的周期检测——流式 feed() 专用。

    它每收到一批 token 都要跑一次，必须便宜：只在**尾部 span + p 行**这个窗口内试
    所有对齐位置（而不是整篇扫描），复杂度 O(period_max²·min_cycles)，与已生成的行数
    无关。

    为什么要留 p 行的滑动余量：退化块后面经常还跟着几行别的内容（实测「清宵配队」的
    32 行清单之后还接着「具体循环轴嘛…」和几行循环轴），整齐对齐在结尾的窗口就不是
    周期块，只查 n-span 一种切法会漏掉。
    """
    n = len(keys)
    for p in range(2, min(period_max, n // min_cycles) + 1):
        span = p * min_cycles
        lo = max(0, n - span - p + 1)
        for start in range(lo, n - span + 1):
            if _cycle_at(keys[start:start + span], p, min_cycles):
                return _extend(keys, start, p)
    return None


def _period_start(keys: list[str], period_max: int, min_cycles: int) -> int | None:
    """**全文扫描**的周期检测——`find_degenerate_start`（每次生成只跑一次）专用。

    ⚠️ 不能只查尾部：实测「清宵配队」的退化块**后面还接着正常内容**（循环轴那几行
    写在 32 行退化清单之后），只查尾部就完全漏掉——截断点算不出来，trim 什么都不做，
    流式兜底等于白做（这是本轮真跑回归抓到的 bug）。
    """
    n = len(keys)
    best: int | None = None
    for p in range(2, min(period_max, n // min_cycles) + 1):
        span = p * min_cycles
        for end in range(span, n + 1):
            if not _cycle_at(keys[end - span:end], p, min_cycles):
                continue
            # end 递增 → 第一个命中的就是该周期里最早的那块；再向前扩展同周期的块
            start = _extend(keys, end - span, p)
            if best is None or start < best:
                best = start
            break
    return best


def find_degenerate_start(
    text: str,
    max_repeat: int = _DEF_MAX_REPEAT,
    min_chars: int = _DEF_MIN_CHARS,
    min_item: int = _DEF_MIN_ITEM,
    period_max: int = _DEF_PERIOD_MAX,
    min_cycles: int = _DEF_MIN_CYCLES,
) -> int | None:
    """返回「应从哪个字符偏移截断」——取所有命中里**最早**的那个；无命中返回 None。"""
    cands = _collect(text, min_item)
    if not cands:
        return None

    cuts: list[int] = []

    # ① 整句重复：出现次数 > max_repeat 的句子 → 从它**首次**出现处切
    if min_chars <= 0:
        counted = [c for c in cands]
    else:
        counted = [c for c in cands if len(c.key) >= min_chars]
    first: dict[str, int] = {}
    seen: dict[str, int] = {}
    for c in counted:
        first.setdefault(c.key, c.start)
        seen[c.key] = seen.get(c.key, 0) + 1
    for key, n in seen.items():
        if n > max_repeat:
            cuts.append(first[key])

    # ② 周期块循环：从周期块首次出现处切
    pidx = _period_start([c.pkey for c in cands], period_max, min_cycles)
    if pidx is not None:
        cuts.append(cands[pidx].start)

    return min(cuts) if cuts else None


class LoopGuard:
    """累积式复读检测，流式与非流式共用。

    流式：每收到一个 token 调一次 feed(delta)，命中立即中断生成，
          不让模型继续写满 num_predict（这是流式路径独有的止损能力）。
    非流式：整篇文本一次性 feed 即可。
    """

    def __init__(
        self,
        max_repeat: int = _DEF_MAX_REPEAT,
        min_chars: int = _DEF_MIN_CHARS,
        min_item: int = _DEF_MIN_ITEM,
        period_max: int = _DEF_PERIOD_MAX,
        min_cycles: int = _DEF_MIN_CYCLES,
    ) -> None:
        # max_repeat 是「同一句允许出现的次数」，超过即判定复读
        self.max_repeat = max(1, max_repeat)
        self.min_chars = max(1, min_chars)
        self.min_item = max(1, min_item)
        self.period_max = max(2, period_max)
        self.min_cycles = max(2, min_cycles)
        self._text = ""     # 内部累积，调用方无需自己拼接全文
        self._scanned = 0   # 已判重过的文本前缀长度（闭合边界）
        self._cands: list[_Cand] = []
        self._counts: dict[str, int] = {}

    def feed(self, delta: str) -> str | None:
        """追加增量文本；返回触发复读的那一行文本，未触发返回 None。"""
        if not delta:
            return None
        self._text += delta

        for start, sent, end in _iter_closed_sentences(self._text, self._scanned):
            self._scanned = end
            s = sent.strip()
            # 跳过：过短的碎句（「好的。」「嗯~」）、Markdown 表格行、分隔线。
            # 表格行必须排除——突破材料类答案的表格行按换行切分后天然重复，会误杀。
            if len(s) < self.min_item or s.startswith("|") or s.startswith("---"):
                continue
            key = _norm(s)
            if not key:
                continue
            self._cands.append(_Cand(start, end, key, _period_key(key)))
            if len(key) >= self.min_chars:
                self._counts[key] = self._counts.get(key, 0) + 1
                if self._counts[key] > self.max_repeat:
                    return s

        # 尾部周期检测（便宜版）：退化块的尾巴总在最新生成的那几行里
        pidx = _period_start_tail([c.pkey for c in self._cands], self.period_max, self.min_cycles)
        if pidx is not None:
            c = self._cands[pidx]
            return self._text[c.start:c.end].strip()
        return None


def trim_loop(
    text: str,
    sentence: str | None = None,
    max_repeat: int = _DEF_MAX_REPEAT,
    min_chars: int = _DEF_MIN_CHARS,
    min_item: int = _DEF_MIN_ITEM,
    period_max: int = _DEF_PERIOD_MAX,
    min_cycles: int = _DEF_MIN_CYCLES,
) -> str:
    """把复读尾巴切掉：保留退化区之前的内容。

    触发复读说明后面的输出已经废了，与其留几份重复句给用户看，
    不如整段切掉，补省略号让前端呈现上有个自然收尾。

    切点优先用 `find_degenerate_start` 在**传入文本**上重算——流式路径下 guard 内部
    文本含触发那一块、而调用方的 `answer` 不含，两边长度口径不同；重算可以保证偏移
    落在正确的文本上。重算没命中时退回老逻辑（`sentence` 首次出现处），因为退化块
    只差一行就可能凑不满判重条件。
    """
    cut = find_degenerate_start(text, max_repeat, min_chars, min_item, period_max, min_cycles)
    if cut is None and sentence:
        key = _norm(sentence)
        if key:
            for start, sent, _end in _iter_closed_sentences(text):
                if _norm(sent.strip()) == key:
                    cut = start
                    break
    if cut is None:
        return text
    return (text[:cut].rstrip() + "……").strip()
