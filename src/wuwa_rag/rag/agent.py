"""Step 8 可选：Agent 模式——让 LLM 自己决定调哪个工具。

默认不启用（见 chain.py 的规则路由）：工具循环会把一次问答从 ~3s 拖到 15s+，
而规则路由实测 16/16 全覆盖、零漏检。留着作为规则失效时的兜底路径。

模型分工：工具调用走 qwen3:8b（get_tool_llm），不用 aemeath——
aemeath 是 chat 专用的角色扮演模型，人设会干扰工具选择，且它不带 tool 训练。
"""
from __future__ import annotations

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from .llm import get_tool_llm
from .tools import TOOLS

# 实测：原先这段弱 prompt 会让 qwen3:8b 把参数名写成单数 character，pydantic 忽略未知字段、
# characters 取默认空列表 → 工具返回空 → 模型接着编造共鸣链名称（「追光者」「星海行舟」均非真实数据）。
# 强化后：显式列出参数名与取值域 + 空结果铁律，实测参数名正确、工具返回 409 字、空结果不再编造。
_SYSTEM = """你是《鸣潮》角色养成助手，通过调用工具查资料后作答。
术语：声骸=套装，共鸣链=命座，贝币=货币。

调用 graph_search 时参数名必须严格如下（注意 characters 是复数、且必须是列表）：
  characters: 角色名列表，例如 ["卡卡罗"]
  slots: 槽位列表，取值只能是 属性/属性反查/技能/共鸣链/突破材料/声骸/武器/队友
  element: 属性值，仅属性反查用
  stage: 突破阶段，仅突破材料用

铁律：
- 只依据工具返回的内容回答，禁止编造任何游戏数据。
- 工具返回空字符串即代表查不到，此时只回一句「知识库里没有这项资料」，不得自行补全。"""


def build_agent():
    llm = get_tool_llm().bind_tools(TOOLS)

    async def call_model(state: MessagesState) -> dict:
        resp = await llm.ainvoke([SystemMessage(content=_SYSTEM), *state["messages"]])
        return {"messages": [resp]}

    def should_continue(state: MessagesState) -> str:
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    g = StateGraph(MessagesState)
    g.add_node("agent", call_model)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()
