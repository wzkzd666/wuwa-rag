"""Step 8：工具层。

把「查图谱 / 搜文档 / 取当前时间」三个能力包成 LangChain 标准 Tool，两种用法：
  L1（默认）chain.py 直接 await TOOL.ainvoke({...})——确定性调度，零幻觉、零额外延迟。
  L2（可选）agent.py 里 llm.bind_tools(TOOLS) 让模型自己决定调哪个——8B 模型不稳且慢，
           代码留着，等换大模型再开。
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from wuwa_rag.config import get_settings
from wuwa_rag.knowledge.index.rerank import rerank
from wuwa_rag.knowledge.retrieve import graph_search, vector_search
from wuwa_rag.ww_logger import get_logger

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
async def graph_search_tool(characters: list[str], slots: list[str], element: str = "", stage: str = "") -> str:
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


# 星期与时段的中文名。时段由 24 小时制直接映射，交给工具算好：
# 「现在几点」的答案里几乎必然带「下午/晚上」这类词，让 8B 模型自己换算
# 24 小时制会有概率算错（把 16:42 说成「上午快五点」），而这是纯查表，没有推理价值。
_WEEKDAYS = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")
_PERIODS = (
    (5, "凌晨"), (9, "早上"), (12, "上午"), (13, "中午"),
    (18, "下午"), (23, "晚上"), (24, "深夜"),
)


def _local_now() -> datetime:
    """服务端「现在」，按 TZ_NAME 解释。

    为什么不用 datetime.now()：容器/云主机默认 TZ=UTC，直接读系统本地时区会让
    「现在几点」整整错 8 小时。为什么不用 datetime.utcnow()：它返回 naive 对象，
    再叠加时区换算极易二次出错，Python 3.12 起也已弃用。
    TZ_NAME 不可用时（如 Windows 缺 tzdata）回落系统本地时区并告警——宁可时区可疑，
    也不能让一次问时间把问答整条链路弄挂。
    """
    s = get_settings()
    name = (getattr(s, "TZ_NAME", "") or "").strip()
    if name:
        try:
            from zoneinfo import ZoneInfo
            return datetime.now(ZoneInfo(name))
        except Exception as exc:
            log.warning("时区 %r 不可用，回落系统本地时区：%s", name, exc)
    return datetime.now().astimezone()


def now_text() -> str:
    """当前时间文本，形如：2026-09-30 16:52 星期三（下午4点52分，UTC+08:00）。

    为什么两个时间写法都给：实测（8 次重复抽样）只给 24 小时制时，aemeath 有 3/8 次
    把「16:5x」换算成「三点五x」——24h→12h 的心算它做不稳，而「现在几点」的答案里
    必然要出现口语说法。把口语形态（时段 + 12 小时制点数，含「上午/下午」）预先算好
    摆进上下文，这一步就不再需要模型推理。分钟固定两位（9 点 05 分）避免「9点5分」。
    秒不带：对「现在几点」没有信息量，且模型多抄一个会变的数字没有收益。
    时区偏移显式写出，跨时区部署时答案自证口径。
    """
    dt = _local_now()
    off = dt.utcoffset() or timedelta(0)
    total = int(off.total_seconds())
    sign = "+" if total >= 0 else "-"
    hh, mm = divmod(abs(total) // 60, 60)
    period = next(label for start, label in _PERIODS if dt.hour < start)
    spoken = f"{period}{dt.hour % 12 or 12}点{dt.minute:02d}分"
    return (f"{dt:%Y-%m-%d %H:%M} {_WEEKDAYS[dt.weekday()]}"
            f"（{spoken}，UTC{sign}{hh:02d}:{mm:02d}）")


@tool("current_time")
async def current_time_tool() -> str:
    """查当前真实时间（服务端时钟，含年月日、星期、时/分、上午下午、时区偏移）。
    问「现在几点」「今天几号」「今天星期几」「当前日期」这类问题时用。
    模型自身没有时钟，凭训练数据猜日期必然出错，这类问题一律用本工具取真值。"""
    out = now_text()
    log.info("tool current_time -> %s", out)
    return out


TOOLS = [graph_search_tool, vector_search_tool, current_time_tool]
TOOL_REGISTRY = {t.name: t for t in TOOLS}
