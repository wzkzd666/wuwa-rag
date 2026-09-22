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
    # ---- 验证升级闭环（verify→重检索→刷新→联网）----
    verify_stage: str      # ok / retry / refresh / web / exhausted：验证节点写入，路由据此走
    refined_query: str     # verifier 判不匹配时给的更精确检索式
    retry_count: int       # 已重检索次数（防图内死循环，配 VERIFY_MAX_RETRY）
    refreshed: bool        # 本轮是否已触发过按角色刷新（只刷一次）
    used_web: bool         # 是否走了千帆联网兜底
    web_facts: str         # 千帆联网搜索结果（生成上下文第三级资料）