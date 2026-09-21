"""图状态。total=False：每个节点只回自己负责的字段。"""
from __future__ import annotations

from typing_extensions import TypedDict


class RagState(TypedDict, total=False):
    question: str
    search_query: str      # 追问改写后的自包含问句（检索/意图用它；history 存原句）
    intent: str            # fact / semantic / hybrid / chitchat
    slots: list[str]       # 命中的事实槽位
    characters: list[str]         # 识别出的角色名列表
    graph_facts: str       # 图谱事实（已格式化）
    docs: list[dict]       # 向量 + 稀疏召回
    context: str
    answer: str
    truncated: bool        # 复读兜底触发、答案被截断过
    element: str
    stage: str
    history: list[dict]
    context_summary: str   # 被滑出窗口的旧轮次的滚动摘要（改写器输入；checkpointer 持久化）