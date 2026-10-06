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
    `26.92%+40.38%+67.29%`）。实测全语料 6741 块（其中含 `+` 的 783 块）：

      - 连续串（≥2 个 `+`）共 8 处：**数学型 0 处、非数学型 8 处，且 100% 是图标残渣**
        ——这正是「只吃 ≥2 连续」这条规则安全的依据：连着的 `+` 在语料里从来不是数学式；
      - 单个 `+` 共 7895 处：数学型 7171 处（`11.33% + 11.33%` 之类）必须保留；
        非数学型 724 处全是有效内容——配队 `布兰特+长离`、连招 `【锯环·疾攻】+【锯环·终结】`、
        声骸词条 `攻击+攻击`——**所以单个 `+` 一律不动**。

    最终全语料只删掉 21 个 `+`（落在 4 个块：弗洛洛/清宵/琳奈/莫宁 的「基础资料」）。
    规则的核心结论——连续串 8 处、数学型 0、删 21 个/4 块——在 6572 块与 6741 块
    两次语料快照上口径一致，语料增长未动摇它。
    """
    if "+" not in text:
        return text

    def _sub(m: re.Match[str]) -> str:
        left = _neighbor(text, m.start() - 1, -1)
        right = _neighbor(text, m.end(), 1)
        # 注意：判空必须显式判断——"" in "%." 恒为 True，会把行尾的 + 串当成数学式保留下来
        # （实测 `：++；+++` 只会清掉前半段，尾巴原样留下）。
        is_math = (left and (left.isdigit() or left == "%")) or (
            right and (right.isdigit() or right in "%.")
        )
        return m.group(0) if is_math else ""

    return _RE_PLUS_TAIL.sub("", _RE_PLUS_RUN.sub(_sub, text))


# wiki 表格里「多个配队」挤在同一格，用一长串连字符当分隔：
#   | 队伍组成 | 清宵+达妮娅+莫宁---------------------------清宵+琳奈+莫宁 |
# 转成文本后这串 `-` 没有语义，模型会照抄，还会误以为「后面另有一条」。
# 注意：门槛必须是 {4,} 而非 {3,}：markdown 表格分隔行 | --- | --- | 正好是 3 个连字符，
# 全语料 `-{3,}` 有 19893 处是表格分隔行，而 `-{4,}` 只有 **60 处、100% 是配队分隔**（已实测）。
_RE_DASH_RUN = re.compile(r"-{4,}")


def strip_dash_run(text: str) -> str:
    """把单元格里当分隔符用的长连串连字符换成顿号（幂等）。"""
    return _RE_DASH_RUN.sub("、", text)


# ---------- 读取端：配队「或」组收窄（问谁锁谁） ----------
# 鸣潮配队写法 `A/B/C+D/E+F+G`：`/` 之间是同位置三选一（**或**），`+` 之间是不同位置
# （**和**），一支队伍只有 3 个人。用户要求「同位置识别为或；问的是谁，那位就固定在场，
# 他所在的『或』组只保留他」。
# 约束：该规则写进提示词无效：置于 _SYSTEM（前置位置）时模型照旧输出
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
# 流式扣尾：以下尾巴先扣住不下发，等看清全貌再判。
# ① 未闭合的括号前缀 —— `[1]` 会被切成 `[` / `1` / `]`；中文标记 `[图谱]` 会被切成
#    [图 / 谱]，仅扣数字会让它在流式输出中一闪而过。
# ② **已闭合**的连续单位数字括号组 `[4][3][3]` —— 这是 COST 串还没写完的样子。
#    注意：缺少这一步会产生缺陷（收敛过程中复现过三次）：
#      第一次：完全不扣 → 每个 `[4]` 一闭合就被单独释放并按引用标记删掉，
#              `_restore_cost_runs` 永远看不到连续 ≥3 组，`COST 组合成 [4][3][3][1][1]`
#              在流式下被删成「COST 组合成 的」（数值丢失）。
#      第二次：扣 `{2,}`（≥2 组）仍不够 → 单字切片时**第一组到达时只有 1 组**，
#              照样扣不住被删。故必须 `{1,}`：单个 `[4]` 也先扣住，等看清是不是 COST。
#      第三次：扣 `{1,}` 仍不够 → 下一个 `[` 到达时，「未闭合前缀」分支只匹配到末尾那个
#              `[`，于是把它**前面已闭合的组**当 head 释放出去；此时 head 只有 1 组，
#              不足 _restore_cost_runs 要求的 ≥3 组，仍会被当作引用标记删除。
#    ⇒ 正确做法：把「已闭合的组 + 其后未闭合的 `[` 前缀」当**一个整体**扣住（见下方三分支）。
#    代价：普通引用标记 `[1]` 会多扣几个字符再释放（后续字符到达或 flush 时），无感。
#    只扣单位数字（`[\d]`）：COST 档位只有 1/3/4，都是个位数；`[12]` 这种两位必是引用标记，
#    不匹配任一分支 → 立即释放删除，行为正确。
# 三分支（按「最长优先」排列，re.search 取最左匹配）：
#   ① 已闭合组 + 未闭合前缀：`[4][3][` —— 组可能还要长，整体扣住
#   ② 仅已闭合组：`[4][3][3]` —— 下一个字符可能就是 `[`，仍可能长成 COST，扣住
#   ③ 仅未闭合前缀：`[` / `[1` / `[图` / `[图谱` —— 标记或中文标记的前半截
_RE_COST_TAIL = re.compile(r"(?:\[\d\])+\[[^\]\[]{0,4}$|(?:\[\d\])+$|\[[^\]\[]{0,4}$")
# 扣尾长度帽：COST 是 5 件一组（`[4][3][3][1][1]` = 15 字符），32 字符足够宽裕。
# 超过即判定「不可能是 COST」，立刻释放清洗，防止病态输入下无限缓冲。
_REF_BUF_MAX = 32

# ---- 补充两类边界情况（均在实测输出中出现过）----
# ① 中文来源标记：模型自创的出处标注（`彻空冥雷[图谱]`），_RE_REF_MARK 只认数字，完全漏网。
#    只收白名单词，不做「括号里是中文就删」——正文可能合法出现半角括号包裹的短词。
_SOURCE_WORDS = ("图谱", "资料", "文档", "联网", "网络", "来源", "参考", "搜索",
                 "wiki", "WIKI", "web", "WEB", "web_search")
_RE_SRC_MARK = re.compile(r"\[\s*(?:" + "|".join(map(re.escape, _SOURCE_WORDS)) + r")\s*\]")
# ② COST 数值被模型写成 `[4][3][3][1][1]`：这是**数据**不是标记，按 ① 的规则删会变成
#    「COST 组合成 的「彻空冥雷」」——数值被吃掉。语料真实写法是裸数字（`COST 43311`），
#    所以这种情况只摘括号、把数字还原成串。
#    判别依据（游戏数据特征，非猜测）：鸣潮声骸 COST 只有 1/3/4 三档，5 件一组；
#    引用标记则常出现 2/5/6… 且很少连续 ≥3 个。故「连续 ≥3 个单位数字括号且全属 {1,3,4}」
#    判为 COST，其余按引用标记删除。
_COST_DIGITS = frozenset("134")
_RE_COST_RUN = re.compile(r"(?:\[\d\]){3,}")
_RE_COST_ONE = re.compile(r"\[(\d)\]")


def _restore_cost_runs(text: str) -> str:
    """把 `[4][3][3][1][1]` 这类 COST 串还原成 `43311`；不像 COST 的留给引用规则删。"""
    def repl(m: re.Match[str]) -> str:
        digits = _RE_COST_ONE.findall(m.group(0))
        if len(digits) >= 3 and all(d in _COST_DIGITS for d in digits):
            return "".join(digits)
        return m.group(0)          # 含 2/5/6 等 → 是引用标记序列，原样留给下一步删
    return _RE_COST_RUN.sub(repl, text)


def strip_ref_marks(text: str) -> str:
    """删掉答案里的来源标记（数字 `[1]` 与中文 `[图谱]`），并还原被括号包住的 COST 数字串。

    顺序有讲究：先还原 COST（否则连续数字括号会被当引用标记整串删掉，丢数据），
    再删中文标记，最后删剩余数字标记。幂等。
    """
    return _strip_marks(text)[0]


def _strip_marks(text: str) -> tuple[str, int]:
    """清洗实现：返回 (清洗后文本, 删除的标记数)。

    非流式（generate_node 收尾）与流式（AnswerFilter）共用同一套规则——两处曾经各写
    一遍，结果流式侧漏掉中文标记 `[图谱]`（打字机里会闪现）且 COST 串被切碎误删。
    计数口径 = 三类标记的总命中数，供日志报告清洗量。
    """
    text = _restore_cost_runs(text)
    n = len(_RE_SRC_MARK.findall(text)) + len(_RE_REF_MARK.findall(text))
    text = _RE_SRC_MARK.sub("", text)
    text = _RE_REF_MARK.sub("", text)
    return text, n


# ---------- 输出侧：百分比数值的「单位串味」 ----------
# 实测（问「爱弥斯的六链是什么」）：资料原文是
#   「暴击固定为80%，暴击伤害固定为275%」（data/raw/爱弥斯.md 共鸣链六链）
# 模型输出却是「固定八十万暴击伤害」+「两百七十五暴击伤害」——三种走形同时出现：
#   ① 凭空加量纲：80% → 八十**万**；
#   ② 丢单位：275% → 两百七十五（`%` 没了）；
#   ③ 并把化：把「暴击(率)」与「暴击伤害」两个独立分句并成一句。
# 提示词侧只对「满级数值表 / 突破材料表」点名要求原样照抄（见 chain 的输出格式块），
# 共鸣链不在其列，所以这里补一道**确定性**兜底：术语后紧跟的数值若缺单位就补回 `%`。
# 只认「游戏里恒为百分比的术语」白名单，不做通用数值判断，且只在**缺单位**时才补；
# 写作 `暴击伤害提升60%` 这类本就带单位的文本一字不动（幂等）。
# 术语表：**把常见的连接后缀一并写进词条**，而不是在数值体前写 `(?:为|是)?`。
# 原因（两轮实测踩到）：可选组配中文数字分支时，「**定为**275」里的「定为」会被
# 当中文数字吃掉；而 `[\s：:]*` 又不允许跨汉字，于是 `固定为275` 永远匹配不上。
# 拆开写还有第二个好处：「暴击」与「固定为」是两条独立词条，正则按最长优先匹配，
# 不会因为共用前缀而误吞。
# 「固定为 / 固定 / 为 / 是 / 提升 / 提升至」这些连接后缀**逐个枚举**，见下方注释。
_PERMILL_BASE = (
    "暴击伤害", "暴击率", "暴击",
    "伤害加成", "伤害提升", "伤害减免", "治疗效果加成", "治疗加成",
    "防御力", "攻击力", "生命值", "共鸣效率", "属性伤害加成", "全属性伤害加成",
)
# 连接后缀：单独枚举 + 组合。真实语料写法是「暴击固定为80%」——术语与数值之间
# 夹了「固定为」两个汉字，不枚举就连不上（实测「固定八十万暴击伤害」完全漏检）。
_PERMILL_CONNECT = ("", "提升", "提升至", "增加", "减少", "为", "是",
                    "固定", "固定为", "固定是")
_PERMILL_TERMS = tuple(
    sorted({b + c for b in _PERMILL_BASE for c in _PERMILL_CONNECT},
           key=len, reverse=True)
)
# 数值体：**只认阿拉伯数字**（可带小数）。
# 早先版本写成 `[0-9]{1,3}` 并允许 `[：:为是]?` 可选连接词，结果 `%` 反例被拆碎：
# `275%` 里 `{1,3}` 只吃 `27`，正则回溯后把后面那个 `5` 当成新一次匹配的数值，
# 于是补出 `27%5%`。修法（多轮实测收敛，每条都是踩过才写的）：
#
#   ① 阿拉伯数字上限放开（`\d+`），不给 `{n,m}` 回溯切碎的机会。
#   ② 数值体后必须整段结束，排除字符是 **`[0-9.*%％]`**：
#      · `.` 千万不能漏——全语料回归实测 479 行误伤全部来自小数点：`1.20%` 的小数
#        分支回溯失败后只吃 `1`，而 `.` 不在排除集里，于是补成 `1%.20%`（把小数点
#        当成了句子结束符）。排除掉 `.` 之后，`1.20%` 只能整段匹配或完全不匹配。
#      · `*` 也不能漏——语料里有被加粗符号劈开的畸形行 `****暴击提升2**.80%**`
#        （`**2**.80%` 是排版残渣），补 `%` 只会把残渣固化成 `2%**.80%`。
#   ③ 术语与数值之间**只允许空白与中英冒号**（`[\s：:]*`），不允许出现汉字。
#      连接后缀（「固定为」「提升」等）已**逐个枚举进 `_PERMILL_TERMS`**，所以不需要
#      在这里开可选组；若写成可选组，中文数字分支会把「**定为**275」里的「定为」
#      当成数字吃掉，`固定为275` 反而永远匹配不上（实测踩到）。
#   ④ 数值后**不得紧跟另一个百分比术语**（`_RE_TERM_AHEAD`）。这条来自全语料回归：
#      面板表写法是「攻击力：47暴击伤害：10.8%」——`47` 是**固定值**不是百分比，
#      但它后面紧跟着 `暴击伤害`，若只看「后面不是 %/数字」就会误补成 `攻击力：47%`。
#      加后向否定后，`47` 因后面有术语而不匹配，`10.8%` 因已带 `%` 也不匹配。
#   ⑤ 数值后**不得带量纲**（`(?![%％倍万亿])`）。`暴击=80万` 里的「万」说明这不是
#      百分比，必须整条跳过；中文数字同理（见下）。
#
# 为什么**不支持中文数字**：模型把 `275%` 写成「两百七十五」本身就是「改写成中文
# 数字」的违规行为（项目规则明令禁止），补一个 `%` 只会把它固化下来，产出
# 「两百七十五%」这种半中半西的写法，比不补更糟。中文数字一律交给提示词侧约束，
# 规则侧只处理阿拉伯数字这一种规范形态。
_RE_ARABIC = r"\d+(?:\.\d+)?"
# 后向否定用的「另一个术语」前瞻：任意白名单术语出现在数值之后即判为非百分比。
_RE_TERM_AHEAD = r"(?!(?:" + "|".join(map(re.escape, _PERMILL_TERMS)) + r"))"
_RE_PCT_MISSING = re.compile(
    r"(" + "|".join(map(re.escape, _PERMILL_TERMS)) + r")"
    r"[\s：:]*"
    + _RE_ARABIC + r"(?![0-9.*%％])"
    + _RE_TERM_AHEAD
    + r"(?!\s*[%％倍万亿])"
)


def fix_percent_units(text: str) -> str:
    """给缺单位的百分比术语补回 `%`（幂等）。

    仅处理「暴击伤害 275」这类**术语紧跟裸数值**的形态；数值本身（阿拉伯或中文数字）
    不改写、不换算，只在后面补一个 `%`。带单位、带「倍/万/亿」的写法一律跳过。
    """
    if not text:
        return text
    return _RE_PCT_MISSING.sub(lambda m: f"{m.group(0)}%", text)


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

    def _key_of(self, line: str) -> str:
        return line.strip().lstrip(_LIST_MARKS + " \t").strip()

    def _keep_line(self, line: str) -> str:
        key = self._key_of(line)
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
        # 扣尾：把「可能是 COST 串或标记前半截」的尾巴先扣住不下发（见 _RE_COST_TAIL 注释）。
        m = _RE_COST_TAIL.search(self._ref_buf)
        cut = m.start() if m else len(self._ref_buf)
        # 长度帽：缓冲超过上限说明不可能是 COST（5 件一组），强制释放，防无限缓冲。
        if cut == 0 and len(self._ref_buf) > _REF_BUF_MAX:
            cut = len(self._ref_buf) - _REF_BUF_MAX
        head, self._ref_buf = self._ref_buf[:cut], self._ref_buf[cut:]
        clean, removed = _strip_marks(head)
        self.removed_marks += removed
        return self._route(clean)

    def request_more(self) -> str:
        """收尾的「补漏」调用：把扣住的尾巴全部发出，但**继续接受**后续 token。

        唯一用途是「已在流里吐完、还要追加一段修正文本」的场景（chain.ask_stream 的
        单位修正）：先清空缓冲让前端对齐，再 feed 修正文本。若不用这里收尾，扣在
        `_ref_buf` / `_cand` 里的尾巴会排到修正文本后面，出现明显错位。
        `flush` 只清 `_ref_buf`（行候选留给「token 恰好断在行中」时继续累积），
        这里必须把 `_cand` 一并结算，否则修正段会被并进未完成的那一行、去重 key 失真。
        """
        out = self.flush()
        self._cand = ""
        self._head = True
        return out

    def flush(self) -> str:
        """吐出缓冲里扣住的全部内容（流结束时调用一次）。

        注意：只负责 `_ref_buf`，**不代表流结束**——`_cand` 与去重集合 `_seen` 保留，
        因此可以在 flush 之后继续 feed（见 request_more）。
        """
        tail, self._ref_buf = self._ref_buf, ""
        clean, removed = _strip_marks(tail)
        self.removed_marks += removed
        out = self._route(clean)
        if self._cand:                  # 最后一行没有换行符
            out += self._keep_line(self._cand)
            self._cand = ""
        return out
