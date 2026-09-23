"""文本清洗与拼装：写入端（建索引）和读取端（重排/生成）共用一份，避免两边逻辑漂移。"""
from __future__ import annotations

import re

# wiki 表格没有表头时，导出工具会填「列1 / 列2」占位符，实测 56/102 个 chunk 中招。
# 只删「占位表头 + 紧跟的分隔行」这个组合，正常表格（表头是「身份/名称」）不受影响。
_RE_PH_BLOCK = re.compile(
    r"^[ \t]*\|(?:[ \t]*列\d+[ \t]*\|)+[ \t]*\r?\n"      # | 列1 | 列2 |
    r"[ \t]*\|(?:[ \t]*:?-{3,}:?[ \t]*\|)+[ \t]*\r?$",   # | --- | --- |
    re.M,
)
_RE_PH_ROW = re.compile(r"^[ \t]*\|(?:[ \t]*列\d+[ \t]*\|)+[ \t]*\r?$", re.M)
_RE_BLANKS = re.compile(r"\n{3,}")


def strip_placeholder_table(text: str) -> str:
    """删掉「| 列1 | 列2 |」占位表头及其分隔行。幂等，重复调用无害。"""
    text = _RE_PH_BLOCK.sub("", text)
    text = _RE_PH_ROW.sub("", text)
    return _RE_BLANKS.sub("\n\n", text).strip()


_WS_CHARS = " \t\u00a0"
# 技能说明里常夹着「按键图标」：KuroBBS 富文本把它们写成 <img>（alt 是没用的 blob: URL），
# 转 Markdown 时图标整块消失，只剩下原先写在图标之间的连字符「+」——于是语料里出现
# 「·浮声一刹·凌霄：++；+++」「短按++，获取【静质量能】」这种残渣，模型会照抄进答案
# （实测「清霄的技能是什么」答出 `- 「浮声一刹·凌霄」：++；+++`）。
_RE_PLUS_RUN = re.compile(r"\+{2,}")
# 清掉成串的之后还剩 `：或+` 这种尾巴（`+++或+` 的那个单 +）。只删紧跟 或/：/，/、/·
# 且后面就是空白或句读的孤立 +，绝不碰 `攻击+攻击`。
_RE_PLUS_TAIL = re.compile(r"(?<=[或：，、·])\+(?=[\s；;，,、。]|$)")


def _neighbor(text: str, i: int, step: int) -> str:
    """取 i 位置往 step 方向的第一个非空白字符（越界返回空串）。"""
    while 0 <= i < len(text) and text[i] in _WS_CHARS:
        i += step
    return text[i] if 0 <= i < len(text) else ""


def strip_icon_placeholders(text: str) -> str:
    """删掉「按键图标」消失后残留的 `+` 串（幂等）。

    规则只针对**连续 2 个及以上的 `+`**，且两端不能是数字/百分号（那是伤害式，如
    `26.92%+40.38%+67.29%`）。实测全语料 6572 块（其中含 `+` 的 754 块）：

      - 连续串（≥2 个 `+`）共 8 处：**数学型 0 处、非数学型 8 处，且 100% 是图标残渣**
        ——这正是「只吃 ≥2 连续」这条规则安全的依据：连着的 `+` 在语料里从来不是数学式；
      - 单个 `+` 共 7265 处：数学型 6648 处（`11.33% + 11.33%` 之类）必须保留；
        非数学型 617 处全是有效内容——配队 `布兰特+长离`、连招 `【锯环·疾攻】+【锯环·终结】`、
        声骸词条 `攻击+攻击`——**所以单个 `+` 一律不动**。

    最终全语料只删掉 21 个 `+`（落在 4 个块：弗洛洛/清宵/琳奈/莫宁 的「技能说明」）。
    """
    if "+" not in text:
        return text

    def _sub(m: re.Match[str]) -> str:
        left = _neighbor(text, m.start() - 1, -1)
        right = _neighbor(text, m.end(), 1)
        # ⚠️ 判空必须显式写：`"" in "%."` 恒为 True，会把「行尾的 + 串」当成数学式留下来
        # （实测 `：++；+++` 只会清掉前半段，尾巴原样留下）。
        is_math = (left and (left.isdigit() or left == "%")) or (
            right and (right.isdigit() or right in "%.")
        )
        return m.group(0) if is_math else ""

    return _RE_PLUS_TAIL.sub("", _RE_PLUS_RUN.sub(_sub, text))


# wiki 表格里「多个配队」挤在同一格，用一长串连字符当分隔：
#   | 队伍组成 | 清宵+达妮娅+莫宁---------------------------清宵+琳奈+莫宁 |
# 转成文本后这串 `-` 没有语义，模型会照抄，还会误以为「后面另有一条」。
# ⚠️ 门槛必须设 `{4,}` 而不是 `{3,}`：markdown 表格分隔行 `| --- | --- |` 正好是 3 个连字符，
# 全语料 `-{3,}` 有 19893 处是表格分隔行，而 `-{4,}` 只有 **60 处、100% 是配队分隔**（已实测）。
_RE_DASH_RUN = re.compile(r"-{4,}")


def strip_dash_run(text: str) -> str:
    """把单元格里当分隔符用的长连串连字符换成顿号（幂等）。"""
    return _RE_DASH_RUN.sub("、", text)


# ---------- 读取端：配队「或」组收窄（问谁锁谁） ----------
# 鸣潮配队写法 `A/B/C+D/E+F+G`：`/` 之间是同位置三选一（**或**），`+` 之间是不同位置
# （**和**），一支队伍只有 3 个人。用户要求「同位置识别为或；问的是谁，那位就固定在场，
# 他所在的『或』组只保留他」。
# ⚠️ 这条规则**写进提示词不管用**：写 `_SYSTEM`（远前置）时模型照旧输出
# `守岸人 / 维里奈 / 白芷`（未锁定），和 `[n]` 那次是**同一个教训**——8B 对长 system 里
# 的规则遵守度低。所以做成**确定性字符串变换**：进 prompt 之前先把「或」组收窄成单个名字，
# 模型只需要照抄，不需要它理解规则；天然幂等、可单测、零概率残留。
_OR_TOKEN = r"[^\s/+，。；：、（）()\[\]|*_#`“”\"']{1,12}"
_RE_OR_GROUP = re.compile(rf"{_OR_TOKEN}(?:/{_OR_TOKEN})+")


def lock_focus(text: str, focus: str) -> str:
    """把含 focus 的「或」组收窄成 focus 本身（幂等）。

    只动**斜杠短语**，且必须 focus 恰好是该短语的一个备选::

        lock_focus("守岸人/维里奈/白芷+吟霖/长离/散华+卡卡罗", "守岸人")
        -> "守岸人+吟霖/长离/散华+卡卡罗"

    focus 不在其中的斜杠短语（别人的配队、`攻击/防御` 这类非配队斜杠）原样保留；
    不含 focus 的整段文本直接短路返回，不做任何扫描。
    """
    if not focus or not text or focus not in text:
        return text

    def _sub(m: re.Match[str]) -> str:
        return focus if focus in m.group(0).split("/") else m.group(0)

    return _RE_OR_GROUP.sub(_sub, text)


def join_breadcrumb(breadcrumb: str, text: str) -> str:
    """面包屑 + 正文，只补不重复。

    实测 102 个 chunk 里 74 个的 text 本身就以 breadcrumb 开头
    （MarkdownHeaderTextSplitter 开了 strip_headers=False，标题行留在正文里），
    无条件拼接会让面包屑出现两遍——build_index 和 rerank 都踩过这个坑。
    """
    bc = (breadcrumb or "").strip()
    tx = (text or "").strip()
    if not bc or tx.startswith(bc):
        return tx
    return f"{bc}\n{tx}"


def embed_input(breadcrumb: str, text: str) -> str:
    """写入端：建 BM25 / Chroma 之前的最终文本。"""
    return strip_dash_run(
        strip_icon_placeholders(strip_placeholder_table(join_breadcrumb(breadcrumb, text)))
    )


def chunk_text(d: dict) -> str:
    """读取端：从检索结果 dict 取干净文本。幂等——写入端已洗过也不会二次伤害。"""
    return strip_dash_run(
        strip_icon_placeholders(
            strip_placeholder_table(join_breadcrumb(d.get("breadcrumb", ""), d.get("text", "")))
        )
    )


# ---------- 输出侧：来源标记 `[n]` ----------
# 对照实验（直连 Ollama、固定 seed）坐实的模型微调习惯：prompt 里只要出现「依据资料作答」
# **配上资料块**，aemeath 就会在句末缀 `[1]` `[2]` 这类来源标记；纯问题不给资料时不写，
# 资料里去掉方括号照样写 → 是「有资料可依」这件事诱发的，不是语料残留。
# prompt 侧只能**概率**压住（末尾放泛化禁止语后，3 轮里仍有 2 轮出现，还会自编到 `[10]`，
# 同一个 `[1]` 重复用十几次），所以输出侧再兜一道。
# 语料正文从不含 `[n]`（游戏术语用的是中文方括号 `【】`，不受影响），删除零损失。
_RE_REF_MARK = re.compile(r"\[\d{1,2}\]")
# 流式下 `[1]` 可能被切成 `[` / `1` / `]` 三个 token，先扣住「可能成为标记开头」的尾巴
_RE_REF_PENDING = re.compile(r"\[\d{0,2}$")


def strip_ref_marks(text: str) -> str:
    """删掉答案里的来源标记 `[n]`（幂等）。"""
    return _RE_REF_MARK.sub("", text)


# ---------- 输出侧：重复列表项 ----------
# 实测「清宵配队」会把同一支队伍列两遍（4 支队伍列成 6 项）：**图谱事实与参考文档给了
# 同一批信息、只是组合里角色名的先后不同**（图谱 `清宵+莫宁+达妮娅` vs 文档 `清宵+达妮娅+莫宁`），
# 8B 读成「还有一批」，于是又列一遍。`_SYSTEM` 里「每条只出现一次」对 8B 遵守度不稳
# （实测约半数轮次仍重复），输出侧兜一道。**只丢「整行内容完全相同」的列表行**，
# 不做任何语义合并，真实内容不受影响。
_LIST_MARKS = "-*•"


def dedup_list_items(text: str) -> str:
    """去掉重复的列表项：整行内容（去空白）相同则丢弃后出现的那次（幂等）。"""
    seen: set[str] = set()
    out: list[str] = []
    for line in text.split("\n"):
        body = line.strip()
        if body[:1] in tuple(_LIST_MARKS):
            key = body.lstrip(_LIST_MARKS + " \t").strip()
            if key:
                if key in seen:
                    continue
                seen.add(key)
        out.append(line)
    return "\n".join(out)


class AnswerFilter:
    """流式答案清洗：① 剥来源标记 `[n]`；② 丢掉重复的列表项。

    两件事都要缓冲，但粒度不同：
    - `[n]` 会被切成多个 token，所以扣住「疑似标记开头」的尾巴（最多 3 字符）；
    - 列表项去重要看整行，所以**只在行首是列表标记时**才缓冲到行末；其余内容逐字直通，
      打字机效果不受影响。
    """

    def __init__(self) -> None:
        self._ref_buf = ""
        self._head = True      # 是否处于行首
        self._cand = ""        # 正在缓冲的候选列表行
        self._seen: set[str] = set()
        self.removed_marks = 0
        self.removed_lines = 0

    def _keep_line(self, line: str) -> str:
        key = line.strip().lstrip(_LIST_MARKS + " \t").strip()
        if key:
            if key in self._seen:
                self.removed_lines += 1
                return ""
            self._seen.add(key)
        return line

    def _route(self, text: str) -> str:
        out: list[str] = []
        for ch in text:
            if self._cand:
                self._cand += ch
                if ch == "\n":
                    out.append(self._keep_line(self._cand))
                    self._cand = ""
                    self._head = True
                continue
            if self._head and ch in _LIST_MARKS:
                self._cand = ch          # 疑似列表行：缓冲到行末再决定
                self._head = False
                continue
            out.append(ch)
            self._head = ch == "\n"
        return "".join(out)

    def feed(self, token: str) -> str:
        self._ref_buf += token
        m = _RE_REF_PENDING.search(self._ref_buf)
        if m:
            # 尾巴形如 `[` / `[1` / `[12`：可能是标记前半截，先扣住不下发
            head, self._ref_buf = self._ref_buf[: m.start()], m.group(0)
        else:
            head, self._ref_buf = self._ref_buf, ""
        self.removed_marks += len(_RE_REF_MARK.findall(head))
        return self._route(_RE_REF_MARK.sub("", head))

    def flush(self) -> str:
        tail, self._ref_buf = self._ref_buf, ""
        self.removed_marks += len(_RE_REF_MARK.findall(tail))
        out = self._route(_RE_REF_MARK.sub("", tail))
        if self._cand:                  # 最后一行没有换行符
            out += self._keep_line(self._cand)
            self._cand = ""
        return out
