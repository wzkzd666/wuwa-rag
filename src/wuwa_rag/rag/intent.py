"""意图识别 + 实体抽取。

规则优先，不上 LLM 分类：固定领域下规则准确率更高、毫秒级、零 GPU 成本，
且结果可回归测试。
"""
from __future__ import annotations

import re

# 事实型槽位：能直接查图谱
SLOT_PATTERNS: dict[str, str] = {
    "属性":    r"属性|元素|武器类型|什么武器|性别|出生|哪里人|稀有度",
    "属性反查": r"导电|冷凝|热熔|气动|衍射|湮灭",
    "技能":    r"技能|常态攻击|普攻|共鸣技能|共鸣解放|变奏|延奏|共鸣回路|大招",
    "共鸣链":  r"共鸣链|几链|命座|一链|六链|满链",
    "突破材料": r"突破|材料|素材|需要什么|消耗|多少个",
    "配装":    r"配装|声骸|套装|cost|COST|词条|毕业",
    "武器":    r"武器|专武|装备",
    "队友":    r"队友|组队|配队|和谁|一起|阵容|队伍",
}

# 语义型：需要文档描述支撑
SEMANTIC_PATTERNS: tuple[str, ...] = (
    r"怎么|如何|怎样|为什么|原因|理由|依据|优势|好处|思路|玩法|攻略|讲解|介绍|分析|评价|强吗|值得|机制|原理",
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
