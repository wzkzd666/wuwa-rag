"""图状态。total=False：每个节点只回自己负责的字段。"""
from __future__ import annotations

from typing_extensions import TypedDict


class RagState(TypedDict, total=False):
    question: str
    intent: str            # fact / semantic / hybrid
    slots: list[str]       # 命中的事实槽位
    characters: list[str]         # 识别出的角色名列表
    graph_facts: str       # 图谱事实（已格式化）
    docs: list[dict]       # 向量 + 稀疏召回
    context: str
    answer: str
    element: str
    stage: str
    history: list[dict]