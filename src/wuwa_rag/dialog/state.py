"""图状态。total=False：每个节点只回自己负责的字段。"""
from __future__ import annotations

from typing_extensions import TypedDict


class RagState(TypedDict, total=False):
    question: str
    user_id: int           # 当前登录用户 id，用于取该用户的云端模型配置。
                           # 约束：state 中只放 id，明文 api_key 绝不写入——checkpointer 会把
                           # state 持久化进 PG，放进去等于把用户密钥落盘到另一张表。
    search_query: str      # 追问改写后的自包含问句（检索/意图用它；history 存原句）
    intent: str            # fact / semantic / hybrid / chitchat / time
    need_time: bool        # 本轮是否要用到服务端真值时间。纯时间问题（intent=time）走
                           # time_node；「顺带问了时间」的混合问句（intent 仍是
                           # fact/semantic/hybrid/chitchat）由 chain 在本节点里额外注入
                           # 真值（见 intent.mentions_time）。
    slots: list[str]       # 命中的事实槽位
    characters: list[str]         # 识别出的角色名列表
    graph_facts: str       # 图谱事实（已格式化）
    docs: list[dict]       # 向量 + 稀疏召回
    context: str
    answer: str
    music_action: str        # 音乐动作 play/pause/next/prev/stop/status；空=本轮不是音乐指令
    music_keyword: str       # 点歌关键词（仅 music_action == "play" 用）
    music_result: str        # **已经执行过**的音乐工具结果。API 层为了让歌早点开始放，
                             # 会在进图之前先跑一次（见 api/app.py 的 _stream_answer_inner），
                             # 结果从这里带进来，图里据此**跳过重复调用**（见 _run_music）。
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
    # ---- 用户画像（user_facts 表）----
    user_context: str      # 该用户的画像事实串（「主玩角色是守岸人；萌新」），注入生成 prompt
    # ---- 情绪标签（供 TTS 使用）----
    emotion: str           # 本轮答案的情绪标签（受限枚举，见 rag/emotion.py）
