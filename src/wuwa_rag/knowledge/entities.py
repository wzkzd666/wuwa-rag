"""角色名解析：名册（种子 ∪ 库内实际角色）+ LLM 兜底。

用于问答时识别「知识库里还没有」的角色，以便自动触发爬取+建库。

⚠️ 名册**不是**写死的清单。`SEED_CHARACTER_NAMES` 只是冷启动种子（空库时规则层
也得认得常见角色名），真正的名册是 `get_roster()` = **种子 ∪ PG `documents` 里
已入库的角色**。所以新角色入库后自动进名册，不需要改代码。

规则优先：先在名册里匹配；名册没命中再走一次轻量 LLM 抽取——兜底保留，作用是
识别**连库都还没有**的全新角色（wiki 刚出的那种）。
"""
from __future__ import annotations

import json
import re
import time

import jieba
import psycopg
from langchain_core.messages import HumanMessage, SystemMessage

from wuwa_rag.core.db import get_cursor
from wuwa_rag.core.llm import get_tool_llm
from wuwa_rag.knowledge import domain_terms
from wuwa_rag.ww_logger import get_logger

log = get_logger('rag')

# 冷启动种子名册：只在「库是空的」或「库读不出来」时兜底。
# 它不是权威清单，别往这里追加新角色——入库即可生效，见 get_roster()。
SEED_CHARACTER_NAMES: set[str] = {
    "丹瑾", "丽贝卡", "仇远", "今汐", "凌阳", "千咲", "卜灵", "卡卡罗", "卡提希娅", "吟霖",
    "嘉贝莉娜", "坎特蕾拉", "夏空", "奥古斯塔", "守岸人", "安可", "尤诺", "布兰特", "弗洛洛",
    "忌炎", "折枝", "散华", "桃祈", "椿", "洛可可", "洛瑟菈", "清宵", "渊武",
    "漂泊者-男-导电", "漂泊者-男-气动", "漂泊者-男-湮灭", "漂泊者-男-衍射",
    "灯灯", "炽霞", "爱弥斯", "珂莱塔", "琳奈", "白芷", "相里要", "秋水",
    "秧秧", "秧秧·玄翎", "穗穗", "绯雪", "维里奈", "莫宁", "莫特斐", "菲比",
    "西格莉卡", "赞妮", "达妮娅", "釉瑚", "鉴心", "长离", "陆·赫斯", "露帕", "露西",
}

# 别名对应规则表
CHARACTER_ALIASES: dict[str, str] = {
    "光主": "漂泊者-男-衍射",
    "风主": "漂泊者-男-气动",
    "暗主": "漂泊者-男-湮灭",
    "电主": "漂泊者-男-导电",
    "卡提": "卡提希娅",          # wiki 配队表里的简称，实测散落在多个角色页
}

# ---------- 动态名册：种子 ∪ 库内实际角色 ----------
_ROSTER_TTL = 60.0
# (采集时刻 monotonic, 名册快照)。**空快照 = 还没采过**，不能拿它当「名册为空」用。
_roster_snap: tuple[float, frozenset[str]] = (0.0, frozenset())


async def get_roster() -> set[str]:
    """当前名册 = 种子名册 ∪ 库里已入库的角色；TTL 60s 缓存。

    数据源取 PG `documents`（唯一真源）而不是 Neo4j：Neo4j 是可重建的派生索引，
    而「名册里该不该有这个角色」必须跟真源走 —— 角色被删除时 documents 先没了，
    名册就该立刻不再包含它。

    读库失败不抛（问答链路不能因为一次名册查询挂掉），回落种子名册并告警。
    """
    global _roster_snap
    now = time.monotonic()
    if _roster_snap[1] and now - _roster_snap[0] < _ROSTER_TTL:
        return set(_roster_snap[1])
    in_db: set[str] = set()
    try:
        async with get_cursor() as cur:
            await cur.execute(
                "SELECT DISTINCT character FROM documents WHERE deleted_at IS NULL")
            in_db = {r[0] for r in await cur.fetchall() if r[0]}
    except psycopg.Error as exc:
        # 只吞数据库层故障（连接断、超时、SQL 错）——名册查不到时回落种子名册，
        # 问答链路不该因为一次名册查询挂掉。其余异常属编程错误，应如实暴露。
        log.warning("读取库内角色失败（本轮回落种子名册）: %s", exc)
        return set(SEED_CHARACTER_NAMES)
    full = frozenset(SEED_CHARACTER_NAMES | in_db)
    _roster_snap = (now, full)
    return set(full)


def current_names() -> set[str]:
    """同步读最近一次名册快照（尚未采过则回落种子）。

    给 `normalize_character_name` 这类**同步**调用方用：图谱抽取与检索侧是同步
    批处理，不该为了一次归一去 await。快照由 `get_roster()` 负责刷新。
    """
    return set(_roster_snap[1]) if _roster_snap[1] else set(SEED_CHARACTER_NAMES)


def invalidate_roster() -> None:
    """让下一次 `get_roster()` 立刻重采（入库/删除角色后调用，不必等 TTL）。"""
    global _roster_snap
    _roster_snap = (0.0, frozenset())


# ---------- 角色名提及判定（单字名必须走分词） ----------
#
# 为什么需要它：角色「心」(Hsin) 与「椿」都是**单字名**，而「心」是高频汉字。
# 原先 `nlu.extract_characters` 用 `len(n) >= 2` 一刀切滤掉单字名，后果是这两个角色
# **永远识别不出来**：`characters` 恒空 → `graph.verify_node` 的「按角色清库重爬」分支
# 要求 `chars` 非空、永不触发 → 额度耗尽后落到**联网兜底**（用户报的现象）。
# 而 `entities._rule_candidates` 反向也不安全：它用裸子串匹配（`n in question`），
# 「核心玩法是什么」会命中「心」——若该角色不在库，就会触发一次无谓的爬取并阻塞等待。
#
# 解法：单字名要求「**独立成词**」才算提及，多字名沿用子串匹配（行为已实测稳定，
# 且 `漂泊者-男-衍射` 这类带连字符的名字会被分词器切开，不适合走分词）。
# 实测（jieba 独立实例 + 角色名词典，16/16 全对）：
#   命中：心的声骸怎么配 / 心的突破材料 / 椿怎么玩 / 心和椿谁强
#   不误判：核心玩法 / 中心思想 / 我很关心剧情 / 开心 / 心情不错 / 决心要练她 / 用心练她
#   多角色比较句也对：「卡卡罗和心谁强」-> {卡卡罗, 心}、「鉴心和心谁强」-> {鉴心, 心}
#
# ⚠️ **绝不能用全局 `jieba.add_word`**：BM25 稀疏路的 pickle 内含**自定义词典快照**，
#    建索引与查询必须分词一致（见 knowledge/index/bm25.py）。污染全局词典会让
#    已建好的 BM25 索引与查询侧分词不一致，召回质量静默劣化。故用独立 Tokenizer 实例。
_TOK_CACHE: tuple[frozenset[str], jieba.Tokenizer] | None = None


# 单字名的第二道判据要靠它在**原文**里的相邻字，理由：
#
# `心声骸推荐` 里的「心声」是 jieba 默认词典的真词，最长匹配会把它整体切走，
# 分词结果为 `心声/骸/推荐` —— 单字名「心」这个 token 根本不出现，角色识别不到。
# 后果不是「少认一个名字」：`characters` 为空会让 `graph.verify_node` 的
# 「按角色清库重爬」分支永不触发（它要求 chars 非空），额度耗尽后落到**联网兜底**。
#
# ⚠️ 这里采用「原文裸子串扫描」而不是「往分词器词典里加 `心+后缀` 组合词」：
# 加了 `心玩法` 之后，`核心玩法` 的切分会从 `核心/玩法` 变成 `核/心玩法` ——
# 组合词比原词更长的匹配会**连带改变无关词的切分**，凡「核心+后缀」都有这个风险。
# 原文扫描只多看一眼相邻字，不碰分词器，副作用为零。
#
# 领域词表：只收《鸣潮》资料里跟在角色名后面的**实义名词**。
# 「心 + 声骸/共鸣链/命座/配队/突破/武器/遗器/…」→ 认定在谈论该角色；
# 「心情 / 心愿 / 心思」这类不含表内字，天然被排除。
#
# ⚠️ 这份是**兜底表**，不是主表。主表由 `domain_terms` 从语料自动派生
# （wiki 分类法子词 ∩ 多字角色名锚点共现），新角色入库即自动覆盖、无需维护；
# 只在派生表还没建好时（冷启动 / 首次提问）才用这份手写的。
# 它留着还有两个作用：派生表为空时保证判据不至于完全失效；派生表是语料相关的
# 「数据兜底」，这份是「规则兜底」，两者互补。
_DOMAIN_SUFFIXES: tuple[str, ...] = (
    "声骸", "共鸣链", "共鸣", "命座", "配队", "阵容", "搭配",
    "养成", "培养", "练度", "专精", "精通", "技能", "天赋",
    "属性", "元素", "武器", "遗器", "毕业", "进阶", "攻略", "强度",
)
# 领域词窗口半径（字）。派生词表里词条长短不一（「共鸣链」3 字、「声骸」2 字），
# 只看紧邻的下一个词会漏掉「心 / 心声 / 骸 / 推荐」这种被词典词粘住的写法。
_DOMAIN_WINDOW = 14
# 汉字判定：单字名左边若是汉字，说明它多半是某个词内部（核心 / 关心 / 决心 / 担心），
# 此时即使窗口里有领域词也不认 —— 「核心声骸」说的是声骸本身，不是角色「心」。
_HAN = re.compile(r"[一-鿿]")


def _name_tokenizer(names: frozenset[str]) -> jieba.Tokenizer:
    """按名册构建（并缓存）一个独立分词器。名册不变就复用，变了一次重建。

    ⚠️ 必须缓存：每个 `jieba.Tokenizer` 实例各自持有词典状态，新建实例要重新加载
    默认词典（约 1s 量级的阻塞 CPU 开销）。名册只在角色入库/删除时才变，
    用 `frozenset(names)` 做缓存键即可，正常问答路径命中缓存、零开销。
    """
    global _TOK_CACHE
    if _TOK_CACHE is not None and _TOK_CACHE[0] == names:
        return _TOK_CACHE[1]
    tok = jieba.Tokenizer()
    for n in names:
        if n:
            # 高频权重确保整名成词：否则「卡卡罗和心谁强」会被切成
            # ['卡卡','罗和心','谁','强']（实测），单字名与相邻字粘连。
            tok.add_word(n, freq=100000)
    tok.initialize()                 # 预热：把词典加载的阻塞开销挪到构建这一次
    _TOK_CACHE = (names, tok)
    log.info("角色名分词器已重建（词典 %d 词）", len(names))
    return tok


def find_mentions(text: str, known) -> list[tuple[int, str]]:
    """返回 `(起始位置, 角色名)`，按文本位置升序。同一角色只保留**首次**出现。

    位置信息是给 `graph._inject_far_characters` 用的——「开头聊的那位」要取摘要里
    **最靠前**的名字，只给一个无序集合就实现不了。

    三类判据并存，各取所长：
      - 多字名：正则子串匹配（长度降序拼接，防「秧秧」抢走「秧秧·玄翎」的匹配）；
      - 单字名分词整词：`心` 独立成词时直接认（`心の声骸怎么配` / `帮我配个心`）；
      - 单字名被词典词粘住时：靠**领域词证据**兜底（`心声骸推荐` —— 「心声」是词典
        真词，最长匹配会吃掉「心」，分词通道看不见它）。词表见 `domain_terms`。

    单字名还分「歧义名」与「普通名」：单字名一律走上面两条（多字名不受影响），
    领域词证据只用于**补救分词漏掉**的情况，不会给多字名带来任何变化。
    """
    if not text:
        return []
    names = {n for n in known if n}
    if not names:
        # known 为空时必须挡：空正则在每个位置都能匹配出无意义片段
        return []
    single = {n for n in names if len(n) == 1}
    multi = names - single

    found: dict[str, int] = {}
    if multi:
        pattern = "|".join(re.escape(n) for n in sorted(multi, key=len, reverse=True))
        for m in re.finditer(pattern, text):
            found.setdefault(m.group(0), m.start())
    if single:
        tok = _name_tokenizer(frozenset(names))
        # ⚠️ **必须 HMM=False**。HMM 是 jieba 的新词发现，会把「单字角色名 + 紧邻的
        # 普通字」当成一个未登录词合成出来。实测：「心配队」被切成 ['心配','队']，
        # 「心和椿配队」被切成 ['心和椿','配队'] —— 而 `心配` **根本不在词典里**
        # （FREQ 查无），纯属 HMM 臆造。角色名一旦和相邻字粘连，单字名就整轮识别不到。
        # 关掉后：「心配队」-> ['心','配','队']、「心和椿配队」-> ['心','和','椿','配','队']，
        # 同时「核心/关心/开心/中心思想/决心」这些**词典里的真词**仍然完整不拆
        # （它们靠 FREQ 权重切分，不依赖 HMM），负样本一个都没退化。
        # 代价：未登录的领域词（如「声骸」不在默认词典）会被拆成 ['声','骸']。
        for word, start, _end in tok.tokenize(text, HMM=False):
            if word in single:
                found.setdefault(word, start)

    # 单字名兜底：原文里的「单字名 + 领域后缀」裸子串（分词看不到的那一类，
    # 见 `_DOMAIN_SUFFIXES` 的说明）。判据两条：**左边是文本起点或非汉字**（否则
    # 「核心声骸」这种会把角色「心」当出来）+ **右边紧跟领域后缀**。
    # ⚠️ 已知取舍：名字前面带汉字时不认（`给心声骸推荐` 认不出「心」）。宁可漏、
    # 不能错认 —— 错认会把另一个角色的资料当答案，代价远大于漏。
    if single:
        # 领域词表 = **并集**：派生表（自动覆盖新角色）∪ 手写兜底表（冷启动/数据兜底）。
        # ⚠️ 必须是并集而不是「派生表非空就只用它」：派生表是语料相关的，
        # 某些词（实测「声骸」在被更长的分类法标签吞掉后可能缺席）一缺席，
        # 「心声骸」这类问法就会从「能认」退回「认不出」。手写表只多几十个词，
        # 换取的是派生表任何缺失都不影响判据下限。
        terms = set(domain_terms.current()) | set(_DOMAIN_SUFFIXES)
        for n in single:
            if n in found:
                continue
            start = 0
            while True:
                i = text.find(n, start)
                if i < 0:
                    break
                start = i + 1
                if i and _HAN.match(text[i - 1]):
                    continue
                ctx = text[max(0, i - _DOMAIN_WINDOW): i + len(n) + _DOMAIN_WINDOW]
                if any(t in ctx for t in terms):
                    found.setdefault(n, i)
                    break

    # 子串消歧：仅用于**多字名之间**（「秧秧」是「秧秧·玄翎」的前缀，同句命中时留长的）。
    # ⚠️ 分词命中的单字名**不参与**消歧：`鉴心` 含 `心`，若参与会把
    # 「鉴心和心谁强」里的 `心` 误删（实测：两个角色都在比较句里，应同时命中）。
    # 单字名由「独立成词」这条判据保证精确，无需再靠消歧。
    drop = {
        a for a in found
        if len(a) > 1 and any(b != a and a in b for b in found if len(b) > 1)
    }
    return sorted(
        ((pos, n) for n, pos in found.items() if n not in drop),
        key=lambda t: t[0],
    )


def mentioned_names(text: str, known) -> list[str]:
    """文本里提到的角色名（按出现顺序）。`find_mentions` 的只取名版本。"""
    return [n for _pos, n in find_mentions(text, known)]


# 漂泊者：性别维度归一为[男]
_POVER_ATTRS = ("导电", "气动", "湮灭", "衍射", "热熔", "冷凝")
_POVER_DEFAULT = "漂泊者-男-导电"  # 用户只说「漂泊者」不带属性时的默认分支；改这里换默认


# ---------- 队伍串里的角色名归一（图谱 team 清洗 / 别名统一） ----------
# wiki 的「配队推荐」表把「角色名 + 位置标签/注释」挤在同一格，还有一批别名写法。
# 这些串会被 `_extract_teammates` 原样存进 Neo4j 的 `teams`，再原样进 prompt，
# aemeath 于是念出「渊武其他输出」「维里奈（高熟练度）」这种非角色名。
#
# 语料盘点（245 条 SYNERGIZES_WITH 关系的全部 teams 串）：23 个非名册 token，
# 分五类 ——
#   ① 漂泊者变体    `漂泊者·湮灭` / `漂泊者-湮灭`（都缺 `男-`）      32 处
#   ② 括号注释      `维里奈（高熟练度）` / `夏空（进阶轴，卡提双三剑下落）`
#   ③ 位置标签粘连  `渊武其他输出` / `凌阳等主输出`
#      注意：这两处不是抽取缺陷，wiki 原文即如此（各出现 1 次，
#      `| 吟霖配队 | 守岸人/维里奈/白芷+吟霖+相里要/卡卡罗/今汐/渊武其他输出 |`）。
#      读作「渊武【其他输出】」「凌阳【等】主输出」。
#   ④ 纯位置标签    `主输出` / `副输出`（通用模板行的占位槽位）
#   ⑤ 说明文字      `或者作为奶位配合任意队伍`；以及全角 `＋` 没被拆开的 `釉瑚＋散华＋折枝`
#
# 设计红线：每一步归一都必须落回名册（known）才算有效，不允许按猜测切分。
# wiki 以后出现同形的新词也不会被误切——这是本函数正确性的唯一来源。
_SLOT_TAIL_WORDS: tuple[str, ...] = (
    # 长词必须排在短词前面：`next()` 取第一个 endswith 命中的，
    # 否则 `渊武其他输出` 会先被 `输出` 切成 `渊武其他`（不是名册名，卡死）。
    "其他输出", "主输出", "副输出", "主C", "副C",
    "奶辅", "奶位", "输出", "辅助", "治疗", "奶", "等", "的",
)
_RE_NAME_NOTE = re.compile(r"[（(][^）)]*[）)]")
_RE_POVER_NAME = re.compile(r"^漂泊者[·•・\-－]?(导电|气动|湮灭|衍射|热熔|冷凝)$")


def normalize_character_name(tok: str, known: set[str] | None = None) -> str | None:
    """把一个队伍 token 归一到名册标准角色名；无法归一返回 `None`。

    `None` 的含义是「这一格不是角色名」——位置标签、说明文字、残缺片段，
    由调用方决定是剔除该格还是丢弃整条队伍。

    归一链（每步都要落回名册才认）::

        `维里奈（高熟练度）` -> 剥括号 -> `维里奈`        （规则②）
        `漂泊者·湮灭`        -> 补男属性 -> `漂泊者-男-湮灭`（规则③）
        `渊武其他输出`       -> 剥位置词 -> `渊武`        （规则④）
        `凌阳等主输出`       -> 剥「主输出」再剥「等」-> `凌阳`
        `折枝 或者作为奶位配合任意队伍` -> 前缀最长匹配 -> `折枝`（规则⑤）
        `主输出`             -> 全不中 -> None
    """
    names = current_names() if known is None else known
    t = (tok or "").strip()
    if not t:
        return None

    # ① 原样命中名册 / 别名表
    if t in names:
        return t
    if t in CHARACTER_ALIASES:
        return CHARACTER_ALIASES[t]

    # ② 剥括号注释（`维里奈（高熟练度）`、`莫宁（0链爱）`、`千咲（2链绯雪）`）
    bare = _RE_NAME_NOTE.sub("", t).strip()
    if bare != t:
        if bare in names:
            return bare
        if bare in CHARACTER_ALIASES:
            return CHARACTER_ALIASES[bare]

    # ③ 漂泊者家族补全（`漂泊者·湮灭` / `漂泊者-湮灭` -> `漂泊者-男-湮灭`）
    m = _RE_POVER_NAME.match(t)
    if m:
        cand = f"漂泊者-男-{m.group(1)}"
        if cand in names:
            return cand

    # ④ 剥尾部位置标签与「等/的」，最多 3 层（`凌阳等主输出` = 「等」+「主输出」）
    cur = t
    for _ in range(3):
        nxt = next(
            (cur[: -len(w)] for w in _SLOT_TAIL_WORDS
             if cur.endswith(w) and len(cur) > len(w)),
            "",
        )
        if not nxt:
            break
        if nxt in names:
            return nxt
        cur = nxt

    # ⑤ 最后手段：前缀最长匹配。`折枝 或者作为奶位配合任意队伍` -> `折枝`。
    #    门槛 `len(t) > 4` 是必须的：短 token（`主输出`）不该被前缀匹配救回来。
    if len(t) > 4:
        for n in sorted(names, key=len, reverse=True):
            if n and t.startswith(n):
                return n
    return None


def normalize_team(team: str, known: set[str] | None = None) -> str:
    """把一条队伍串逐格归一到标准角色名，无法归一的格剔除（幂等）。

        `漂泊者·湮灭+维里奈`                          -> `漂泊者-男-湮灭+维里奈`
        `守岸人/维里奈/白芷+秧秧+主输出`               -> `守岸人/维里奈/白芷+秧秧`
        `釉瑚＋散华＋折枝 或者作为奶位配合任意队伍`      -> `釉瑚+散华+折枝`
        `守岸人+主输出+副输出`                         -> `守岸人`

    全角 `＋` / `／` 一并规范成半角；剔除后为空的格丢掉；整串无可用的格时返回 `""`。
    剔除后队伍会变短（如 3 段退化为 2 段），这是刻意设计：保留占位槽位比缺一格更糟
    （模型会把 `主输出` 念成一个角色）。调用方要按**归一后**的段数再做剪枝判断。
    """
    if not team:
        return ""
    s = team.replace("＋", "+").replace("／", "/")
    groups: list[str] = []
    for g in s.split("+"):
        parts = [p for p in (normalize_character_name(x, known) for x in g.split("/")) if p]
        if parts:
            groups.append("/".join(dict.fromkeys(parts)))   # 同格去重且保序
    return "+".join(groups)


def _pover_resolve(question: str) -> str | None:
    """漂泊者特例：性别强制男；属性取用户提到的，没有则用 _POVER_DEFAULT。"""
    if "漂泊者" not in question:
        return None
    attr = next((a for a in _POVER_ATTRS if a in question), None)
    return f"漂泊者-男-{attr}" if attr else _POVER_DEFAULT


async def _llm_candidates(question: str) -> list[str]:
    """名册没命中时的 LLM 兜底，走 tool 模型 qwen3:8b（抽取任务，非 chat）。

    输出不要再用名册过滤：规则层已覆盖名册内角色（兜底触发率约 0%），
    这个兜底的唯一价值就是识别「名册里还没有的新角色」以触发自动爬取；
    拿名册过滤等于把该功能废掉。幻觉名由爬取侧 CharacterNotFound 兜住。
    """
    try:
        resp = await get_tool_llm().ainvoke([
            SystemMessage(content=(
                "你是《鸣潮》wiki 的角色名抽取器。只输出一个 JSON："
                '{"characters": ["角色名"]}，没有具体角色就 {"characters": []}。'
                "不要编造，不要多余文字。")),
            HumanMessage(content="问题：" + question),
        ])
    except Exception as exc:  # noqa: BLE001 —— LLM 不可用（Ollama 没起/网络断）不能挡住问答主链
        log.warning("LLM 角色抽取调用失败: %s", exc)
        return []

    # 解析层单独收窄：模型输出畸形属可预期情形，只认这几类。
    # `.get` 在解析出列表而非字典时会抛 AttributeError，一并挡掉。
    try:
        txt = (resp.content or "").strip()
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            return []
        data = json.loads(m.group(0))
        if not isinstance(data, dict):
            return []
        return [str(c) for c in data.get("characters", []) if c]
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as exc:
        log.warning("LLM 角色抽取输出无法解析: %s", exc)
        return []



async def _rule_candidates(question: str) -> list[str]:
    """名册规则匹配，吃的是**动态名册**（种子 ∪ 库内角色，见 `get_roster`）。

    漂泊者走特例（性别归男）；其余经 `find_mentions` 判定（单字名要求独立成词，
    见该函数注释——裸子串匹配会让「核心玩法」误命中角色「心」，而 `len>=2` 的
    旧过滤又会让「心」「椿」永远匹配不到）。

    命中结果按名排序：`hits` 是 set，不排的话多角色问句的候选顺序每轮都不同，
    日志和后续自动爬取的入队顺序都跟着飘。⚠️ 这里刻意**不用** `find_mentions`
    的位置序：本函数的产物是「待爬取角色集合」，顺序只影响日志与入队次序，
    按名排序才稳定可复现。
    """
    names = await get_roster()
    hits: set[str] = set()
    p = _pover_resolve(question)
    if p:
        hits.add(p)
    hits.update(mentioned_names(question, names))
    # 别名都是 2 字以上（光主/风主/暗主/电主/卡提），子串匹配安全，不必走分词。
    for alias, std in CHARACTER_ALIASES.items():
        if alias in question:
            hits.add(std)
    return sorted(hits)


async def resolve_candidates(question: str) -> list[str]:
    rule = await _rule_candidates(question)
    if rule:
        log.info("角色抽取: 规则命中 %s", rule)
        return rule
    llm = await _llm_candidates(question)
    log.info("角色抽取: LLM兜底 %s", llm)
    return llm
