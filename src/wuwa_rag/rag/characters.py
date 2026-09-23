"""角色名解析：静态角色名册 + LLM 兜底。
用于问答时识别「知识库里还没有」的角色，以便自动触发爬取+建库。
规则优先：先在名册(CHARACTER_NAMES / CHARACTER_ALIASES)里做匹配；名册没命中再走一次轻量 LLM 抽取。
"""
from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from ..ww_logger import get_logger
from .llm import get_tool_llm

log = get_logger('rag')

# 全量角色名
CHARACTER_NAMES: set[str] = {
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

# 漂泊者：性别维度归一为[男]
_POVER_ATTRS = ("导电", "气动", "湮灭", "衍射", "热熔", "冷凝")
_POVER_DEFAULT = "漂泊者-男-导电"  # 用户只说「漂泊者」不带属性时的默认分支；改这里换默认


# ---------- 队伍串里的角色名归一（图谱 team 清洗 / 别名统一） ----------
# wiki 的「配队推荐」表把「角色名 + 位置标签/注释」挤在同一格，还有一批别名写法。
# 这些串会被 `_extract_teammates` 原样存进 Neo4j 的 `teams`，再原样进 prompt，
# aemeath 于是念出「渊武其他输出」「维里奈（高熟练度）」这种非角色名。
#
# 实测盘点（2026-09-22，245 条 SYNERGIZES_WITH 关系的全部 teams 串）：23 个非名册 token，
# 分五类 ——
#   ① 漂泊者变体    `漂泊者·湮灭` / `漂泊者-湮灭`（都缺 `男-`）      32 处
#   ② 括号注释      `维里奈（高熟练度）` / `夏空（进阶轴，卡提双三剑下落）`
#   ③ 位置标签粘连  `渊武其他输出` / `凌阳等主输出`
#      ⚠️ 这两个**不是抽取 bug**，wiki 原文就这么写（各出现 1 次，
#      `| 吟霖配队 | 守岸人/维里奈/白芷+吟霖+相里要/卡卡罗/今汐/渊武其他输出 |`）。
#      读作「渊武【其他输出】」「凌阳【等】主输出」。
#   ④ 纯位置标签    `主输出` / `副输出`（通用模板行的占位槽位）
#   ⑤ 说明文字      `或者作为奶位配合任意队伍`；以及全角 `＋` 没被拆开的 `釉瑚＋散华＋折枝`
#
# ⚠️⚠️ 唯一的设计红线：**每一步归一都必须落回名册（`known`）才算数，绝不「猜着切」**。
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
    names = CHARACTER_NAMES if known is None else known
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
    ⚠️ 剔除后队伍会**变短**（如 3 段退化成 2 段），这是刻意的：留着占位槽位比缺一格更糟
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

    ⚠️ 输出不要用 CHARACTER_NAMES 过滤：规则层已覆盖名册内角色（实测兜底触发率 0%），
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
        txt = resp.content.strip()
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            return []
        return [str(c) for c in json.loads(m.group(0)).get("characters", []) if c]
    except Exception as exc:
        log.warning("LLM 角色抽取失败: %s", exc)
        return []



def _rule_candidates(question: str) -> list[str]:
    """名册规则匹配。漂泊者走特例（性别归男）；其余按 CHARACTER_NAMES / CHARACTER_ALIASES 匹配。"""
    hits: set[str] = set()
    p = _pover_resolve(question)
    if p:
        hits.add(p)
    for n in CHARACTER_NAMES:
        if n and n in question:
            hits.add(n)
    for alias, std in CHARACTER_ALIASES.items():
        if alias in question:
            hits.add(std)
    out = [s for s in hits]
    return out


async def resolve_candidates(question: str) -> list[str]:
    rule = _rule_candidates(question)
    if rule:
        log.info("角色抽取: 规则命中 %s", rule)
        return rule
    llm = await _llm_candidates(question)
    log.info("角色抽取: LLM兜底 %s", llm)
    return llm
