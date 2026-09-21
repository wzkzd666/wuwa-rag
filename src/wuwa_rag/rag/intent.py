"""意图识别 + 实体抽取。

槽位/意图分类规则优先（实测 16/16 覆盖、零漏检，无命中时 classify() 回落 hybrid
图谱+向量双跑，本身即安全兜底）。主题分类（闲聊 vs 游戏）走 qwen3:8b agent——
这类判断规则词典覆盖不全（「家人怎么这么晚才来」无任何游戏信号也无闲聊词典），
但只在规则全无信号时才调用，有信号时零额外延迟。

注意 slots 的 key 必须与 retrievers.py 的 CYPHER / SLOT_LABEL 一一对应，
若将来接入 LLM 抽取槽位，必须用白名单过滤非法槽位名。
"""
from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..ww_logger import get_logger
from .llm import get_tool_llm

log = get_logger("rag")

# 事实型槽位：能直接查图谱
SLOT_PATTERNS: dict[str, str] = {
    "属性":    r"属性|元素|武器类型|什么武器|性别|出生|哪里人|稀有度",
    "属性反查": r"导电|冷凝|热熔|气动|衍射|湮灭",
    "技能":    r"技能|常态攻击|普攻|共鸣技能|共鸣解放|变奏|延奏|共鸣回路|大招",
    "共鸣链":  r"共鸣链|几链|命座|一链|六链|满链",
    "突破材料": r"突破|材料|素材|需要什么|消耗|多少个",
    "声骸":    r"声骸|配装|套装|cost|COST|词条|毕业|主词条|副词条",
    "武器":    r"武器|专武|装备",
    "队友":    r"队友|组队|配队|和谁|一起|阵容|队伍",
}

# 语义型：需要文档描述支撑
SEMANTIC_PATTERNS: tuple[str, ...] = (
    r"怎么|如何|怎样|为什么|原因|理由|依据|优势|好处|队友|组队|配队|和谁|一起|阵容|队伍|思路|玩法|攻略|讲解|介绍|分析|评价|强吗|值得|机制|原理",
)

# 属性值（鸣潮共 6 种）——「导电角色有哪些」里没有「属性」二字，只有属性值
ELEMENTS = ("导电", "冷凝", "热熔", "气动", "衍射", "湮灭")
ELEMENT_RE = "|".join(ELEMENTS)
STAGE_RE = r"([一二三四五六]阶突破|[一二三四五六]阶)"


def detect_stage(question: str) -> str:
    m = re.search(STAGE_RE, question)
    return m.group(1) if m else ""


def detect_element(question: str) -> str:
    m = re.search(ELEMENT_RE, question)
    return m.group(0) if m else ""


def detect_slots(question: str) -> list[str]:
    """根据字典提取图谱关键词"""
    return [slot for slot, pat in SLOT_PATTERNS.items() if re.search(pat, question)]


def is_semantic(question: str) -> bool:
    """提取语义关键词"""
    return any(re.search(p, question) for p in SEMANTIC_PATTERNS)


def classify(question: str, slots: list[str]) -> str:
    """根据关键词提取情况，返回预计处理方式，混合兜底"""
    has_fact, has_sem = bool(slots), is_semantic(question)
    if has_fact and has_sem:
        return "hybrid"
    if has_fact:
        return "fact"
    if has_sem:
        return "semantic"
    return "hybrid"         


def extract_characters(question: str, known: list[str]) -> list[str]:
    """返回所有命中的角色名（支持「卡卡罗和吟霖谁更强」这类多角色提问）。
    包含消歧：仅当 A 是 B 的子串时才剔除 A。
    len>=2 过滤单字，避免「他/她」这类误命中。
    """
    hits = [n for n in known if n and len(n) >= 2 and n in question]
    return [h for h in hits if not any(h != o and h in o for o in hits)]


# ---------- 主题分类：闲聊 vs 游戏（qwen3:8b agent）----------

_TOPIC_SYSTEM = """你是《鸣潮》问答助手的意图分类器。判断用户这句话属于哪一类，只输出一个 JSON：
{"topic": "game"} 或 {"topic": "chitchat"}
- game：询问《鸣潮》游戏内容（角色、声骸、配装、共鸣链、突破材料、技能、属性、配队、版本、玩法攻略等），或明显想查资料的问题。
- chitchat：日常寒暄、问候、情绪表达、与游戏无关的闲聊（如问好、问近况、撒娇、天气、玩笑）。
拿不准时输出 game。不要输出 JSON 以外的任何文字。"""

_TOPIC_EXAMPLES = (
    ("卡卡罗毕业配装用什么声骸", "game"),
    ("秧秧怎么玩", "game"),
    ("家人怎么这么晚才来，今天过得怎么样", "chitchat"),
    ("你好呀，今天心情不错", "chitchat"),
    ("我有点累了，陪我聊会儿", "chitchat"),
)


def rule_has_signal(question: str) -> bool:
    """规则层是否抓到了任何信号（槽位/语义/属性/阶段）。

    已不被主链使用（chain 里的闲聊判定用更严格的「无角色名+无槽位+无属性+无阶段」，
    因为 SEMANTIC_PATTERNS 的「怎么」会误命中闲聊句），保留仅作离线分析用。
    """
    return bool(detect_slots(question)) or is_semantic(question) or bool(detect_element(question)) or bool(detect_stage(question))


async def classify_topic(question: str) -> str:
    """判断主题是闲聊还是游戏提问。返回 "chitchat" 或 "game"。

    调用时机见 chain.intent_node：无角色名且无槽位/属性/阶段信号时才调，
    游戏提问大多被规则直接拦下，不承担额外延迟。解析失败一律回落 game——
    把闲聊误送 RAG 顶多答得生硬，把真问题误判闲聊则直接丢知识，代价不对称。

    少样本用单轮补全（Human 提问 / AI 答 JSON 交替），最后一轮 AI 前缀吃掉
    `{"`，模型只需续写 `"topic": "..."}` —— 比把完整 JSON 塞进 Human 消息
    省输出 token，分类实测更快。
    """
    msgs: list = [SystemMessage(content=_TOPIC_SYSTEM)]
    for q, t in _TOPIC_EXAMPLES:
        msgs.append(HumanMessage(content=f"问题：{q}"))
        msgs.append(AIMessage(content=f'{{"topic": "{t}"}}'))
    msgs.append(HumanMessage(content=f"问题：{question}"))
    msgs.append(AIMessage(content='{"'))
    try:
        resp = await get_tool_llm().ainvoke(msgs)
        txt = '{"' + (resp.content or "")
        m = re.search(r"\{.*?\}", txt, re.S)
        if not m:
            log.warning("主题分类: 输出无 JSON，回落 game：%r", (resp.content or "")[:60])
            return "game"
        topic = json.loads(m.group(0)).get("topic", "")
        return topic if topic in ("game", "chitchat") else "game"
    except Exception as exc:
        log.warning("主题分类失败，回落 game: %s", exc)
        return "game"
