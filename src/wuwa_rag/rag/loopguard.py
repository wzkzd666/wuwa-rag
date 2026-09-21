"""生成侧复读兜底：检测并截断「整句周期性重复」的退化输出。

为什么需要它：aemeath 是 8B 角色扮演模型，在「检索不到资料」时容易陷入
人设独白循环（同一句话反复说）。repeat_penalty 等采样参数只能降低概率，
压不死；这里做最后一道闸——命中即中断生成并截断复读尾巴。

判重按句子级：实测退化样本的特征是整句/整段周期性重复，而非单字重复，
token 级 n-gram 检测对这种形态不敏感且容易误杀正常排比。
"""
from __future__ import annotations

import re

# 句末终止符（含中文标点与波浪号，角色扮演模型爱用「~」结尾）
_SENT_END = re.compile(r"[。！？!?~～\n]+")
# 归一化时剔除所有空白：退化样本会出现「C O S T」这种被空格撑开的字符
_WS = re.compile(r"\s+")


def _norm(sent: str) -> str:
    return _WS.sub("", sent)


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


class LoopGuard:
    """累积式复读检测，流式与非流式共用。

    流式：每收到一个 token 调一次 feed(delta)，命中立即中断生成，
          不让模型继续写满 num_predict（这是流式路径独有的止损能力）。
    非流式：整篇文本一次性 feed 即可。
    """

    def __init__(self, max_repeat: int = 2, min_chars: int = 12) -> None:
        # max_repeat 是「同一句允许出现的次数」，超过即判定复读
        self.max_repeat = max(1, max_repeat)
        self.min_chars = max(1, min_chars)
        self._counts: dict[str, int] = {}
        self._scanned = 0   # 已判重过的文本前缀长度（闭合边界）
        self._text = ""     # 内部累积，调用方无需自己拼接全文

    def feed(self, delta: str) -> str | None:
        """追加增量文本；返回触发复读的那个句子，未触发返回 None。"""
        if not delta:
            return None
        self._text += delta

        hit: str | None = None
        scanned = self._scanned
        for _start, sent, end in _iter_closed_sentences(self._text, self._scanned):
            scanned = end
            s = sent.strip()
            # 跳过：过短的碎句（「好的。」「嗯~」）、Markdown 表格行、分隔线。
            # 表格行必须排除——突破材料类答案的表格行按换行切分后天然重复，会误杀。
            if len(s) < self.min_chars or s.startswith("|") or s.startswith("---"):
                continue
            key = _norm(s)
            if not key:
                continue
            self._counts[key] = self._counts.get(key, 0) + 1
            if self._counts[key] > self.max_repeat:
                hit = s
                break

        self._scanned = scanned
        return hit


def trim_loop(text: str, sentence: str) -> str:
    """把复读尾巴切掉：保留触发句首次出现之前的内容。

    触发复读说明后面的输出已经废了，与其留几份重复句给用户看，
    不如整段切掉，补省略号让前端呈现上有个自然收尾。
    """
    key = _norm(sentence)
    if not key:
        return text
    for start, sent, _end in _iter_closed_sentences(text):
        if _norm(sent.strip()) == key:
            return (text[:start].rstrip() + "……").strip()
    return text
