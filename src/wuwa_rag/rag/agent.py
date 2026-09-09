"""Step 8 可选：Agent 模式——让 LLM 自己决定调哪个工具。

默认不启用：qwen3:8b 带 thinking，工具循环会把一次问答从 ~3s 拖到 15s+，
而规则路由当前 100% 准。留着等换大模型再开，主链不受影响。
"""
from __future__ import annotations

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from .llm import get_chat_llm
from .tools import TOOLS

_SYSTEM = """你是《鸣潮》角色养成助手。先判断该查图谱还是搜文档，再回答。
术语：声骸=套装，共鸣链=命座，贝币=货币。只依据工具返回的内容回答，不要编造。"""


def build_agent():
    llm = get_chat_llm().bind_tools(TOOLS)

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
