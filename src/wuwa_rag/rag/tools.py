"""Step 8：工具层。

把「查图谱 / 搜文档」两个能力包成 LangChain 标准 Tool，两种用法：
  L1（默认）chain.py 直接 await TOOL.ainvoke({...})——确定性调度，零幻觉、零额外延迟。
  L2（可选）agent.py 里 llm.bind_tools(TOOLS) 让模型自己决定调哪个——8B 模型不稳且慢，
           代码留着，等换大模型再开。
"""
from __future__ import annotations

import asyncio

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..config import get_settings
from ..retrieval.rerank import rerank
from ..ww_logger import get_logger
from .retrievers import graph_search, vector_search

s = get_settings()
log = get_logger("rag")


class GraphSearchInput(BaseModel):
    characters: list[str] = Field(default_factory=list, description="角色名列表，如 ['卡卡罗']")
    slots: list[str] = Field(
        default_factory=list,
        description="要查的槽位：属性/属性反查/技能/共鸣链/突破材料/声骸/武器/队友",
    )
    element: str = Field("", description="属性值，仅属性反查用：导电/冷凝/热熔/气动/衍射/湮灭")
    stage: str = Field("", description="突破阶段，如 '六阶突破'，仅突破材料用")


@tool("graph_search", args_schema=GraphSearchInput)
async def graph_search_tool(characters: list[str], slots: list[str],
                            element: str = "", stage: str = "") -> str:
    """查《鸣潮》知识图谱：角色属性、技能、共鸣链、突破材料、声骸、武器、队友。
    问「XX 是什么属性」「XX 六阶突破要什么材料」这类有确定答案的问题时用。
    查不到时返回空字符串。"""
    facts = await graph_search(characters, slots, element, stage)
    log.info("tool graph_search 角色=%s 槽位=%s -> %d 字", characters, slots, len(facts))
    return facts


class VectorSearchInput(BaseModel):
    query: str = Field(..., description="检索问句")
    topk: int = Field(s.TOPK_RERANK, description="返回条数")


@tool("vector_search", args_schema=VectorSearchInput)
async def vector_search_tool(query: str, topk: int = 6) -> list[dict]:
    """在攻略原文里检索（稠密向量 + BM25 融合，再经 bge-reranker 精排）。
    问「怎么玩」「为什么」「思路」「强吗」这类需要原文描述支撑的问题时用。
    每条含 text / breadcrumb / rerank_score。"""
    docs = await asyncio.to_thread(vector_search, query)
    docs = await asyncio.to_thread(rerank, query, docs, topk)
    log.info("tool vector_search '%s' -> %d 条", query, len(docs))
    return docs


TOOLS = [graph_search_tool, vector_search_tool]
TOOL_REGISTRY = {t.name: t for t in TOOLS}
