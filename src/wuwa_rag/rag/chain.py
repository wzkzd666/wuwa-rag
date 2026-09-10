"""Step 7：LangGraph 编排。"""
from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from ..config import ensure_dirs, get_settings
from ..graph.neo4j_client import get_session
from ..ww_logger import get_logger
from .intent import classify, detect_slots, extract_characters, detect_element, detect_stage
from .llm import get_chat_llm
from .tools import graph_search_tool, vector_search_tool
from .state import RagState
from .memory import get_checkpointer
from ..text import chunk_text
from .characters import resolve_candidates
from ..worker import build_pipeline, CharacterNotFound

setting = get_settings()
log = get_logger("rag")

_SYSTEM = """你是《鸣潮》角色养成助手。只依据给定资料回答，不要编造。
资料里没有的内容，直接说"资料里没有提到"。
资料里列了几条就答几条，不要自行补充「没有提到其他/更多」这类说明。
术语对照（资料用词和用户提问可能不同，按下表对应理解）：
- 声骸 = 角色装备，资料里也写作「套装」；COST 是声骸的费用点数组合
- 共鸣链 = 相当于命座，序号 1~6 对应一链到六链
- 贝币 = 游戏货币
- 突破阶段：一阶~六阶

回答用中文，简洁，要点化。"""


async def _known_characters() -> list[str]:
    async with get_session() as s:
        rows = await (await s.run("MATCH (c:Character) RETURN c.name AS n")).data()
    return [r["n"] for r in rows]


async def intent_node(state: RagState) -> dict:
    q = state["question"]
    slots = detect_slots(q)
    chars = extract_characters(q, await _known_characters()) or state.get("characters", [])
    element = detect_element(q) or state.get("element", "")
    stage = detect_stage(q)
    intent = classify(q, slots)
    log.info("意图=%s 槽位=%s 角色=%s 属性=%s 阶段=%s", intent, slots, chars or "未识别", element or "-", stage or "-")
    return {"intent": intent, "slots": slots, "characters": chars, "element": element, "stage": stage}


def _route(state: RagState) -> str:
    """返回分支名，落到哪个节点交给下面的 mapping 决定。"""
    return state["intent"]


async def graph_node(state: RagState) -> dict:
    facts = await graph_search_tool.ainvoke({
        "characters" : state.get("characters", []), 
        "slots" : state.get("slots", []),
        "element" : state.get("element",""),
        "stage" : state.get("stage","")
    })
    return {"graph_facts": facts}


def _after_graph(state: RagState) -> str:
    """graph 之后：hybrid 还要补向量，fact 直接生成。"""
    return "need_vector" if state["intent"] == "hybrid" else "done"


async def vector_node(state: RagState) -> dict:
    q = state["question"]
    docs = await vector_search_tool.ainvoke({"query": q, "topk": setting.TOPK_RERANK})

    # 多角色：为每个角色各补一轮召回，避免只召回到其中一个
    for c in (state.get("characters") or [])[1:]:
        docs += await vector_search_tool.ainvoke({
            "query": f"{c} {q}", "topk": setting.TOPK_RERANK/len(state.get("characters",[1]))
        })

    seen: set[str] = set()
    merged: list[dict] = []
    for d in docs:
        if d["chunk_id"] not in seen:
            seen.add(d["chunk_id"])
            merged.append(d)
    return {"docs": merged}


async def generate_node(state: RagState) -> dict:
    parts: list[str] = []
    # 拼接历史对话
    history = (state.get("history", []))[-setting.MAX_HISTORY_TURNS*2:]
    if history:
        parts.append("## 对话历史\n"+"\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}" for m in history
        ))
    
    if state.get("graph_facts"):
        parts.append("## 图谱事实\n" + state["graph_facts"])
    docs = state.get("docs") or []
    if docs:
        parts.append("## 参考文档\n" + "\n\n".join(
            f"[{i + 1}] {chunk_text(d)}"
            for i, d in enumerate(docs)
        ))
    context = "\n\n".join(parts) or "（没有检索到任何资料）"

    prompt = f"{_SYSTEM}\n\n## 资料\n{context}\n\n## 问题\n{state['question']}"
    try:
        resp = await get_chat_llm().ainvoke([HumanMessage(content=prompt)])
        answer = resp.content
    except Exception as exc:                      
        log.error("LLM 调用失败: %s", exc)
        answer = "抱歉，模型服务暂时不可用，请稍后再试。"
    # 回写历史
    new_history = history +[
        {"role": "user", "content": state["question"]},
        {"role": "assistant", "content": answer}
    ]
    return {"context": context, "answer": answer, "history": new_history[-setting.MAX_HISTORY_TURNS*2:]}


def build_graph() -> StateGraph:
    g = StateGraph(RagState)
    g.add_node("intent", intent_node)
    g.add_node("graph", graph_node)
    g.add_node("vector", vector_node)
    g.add_node("generate", generate_node)

    g.add_edge(START, "intent")
    g.add_conditional_edges("intent", _route, {
        "fact": "graph", "semantic": "vector", "hybrid": "graph",
    })
    g.add_conditional_edges("graph", _after_graph, {
        "need_vector": "vector", "done": "generate",
    })
    g.add_edge("vector", "generate")
    g.add_edge("generate", END)
    return g


_compiled = None


async def get_chain():
    global _compiled
    if _compiled is None:
        cp = await get_checkpointer()
        _compiled = build_graph().compile(checkpointer=cp)
    return _compiled


async def _crawl_and_wait(names: list[str], timeout: int = 180) -> bool:
    """对给定角色触发全流水线并阻塞等待；任一角色抓不到(CharacterNotFound)返回 False。
    用 asyncio.to_thread 跑阻塞的 result.get，避免卡住 FastAPI 事件循环。
    """
    from celery.exceptions import TimeoutError as CeleryTimeout
    results = []
    for n in names:
        r = build_pipeline(n).apply_async()
        results.append(r)
        log.info("自动爬取: 已入队 chain_id=%s 角色=%s", r.id, n)
    try:
        for r in results:
            await asyncio.to_thread(r.get, timeout=timeout)
        return True
    except CharacterNotFound:
        return False
    except CeleryTimeout:
        log.warning("自动爬取: 等待超时(%ss)，视为抓取失败", timeout)
        return False


async def ensure_characters(question: str) -> tuple[list[str], bool, list[str]]:
    """返回 (候选角色名, 是否可答, 本次实际新爬的角色名)。"""
    candidates = await resolve_candidates(question)
    if not candidates:
        log.info("自动爬取: 未识别到角色，走普通问答")
        return [], True, []
    known = set(await _known_characters())
    to_crawl = [c for c in candidates if c not in known]
    if not to_crawl:
        log.info("自动爬取: 角色已在知识库 %s，无需爬取", candidates)
        return candidates, True, []
    log.info("自动爬取: 库外角色 %s -> 触发流水线(爬→分块→入库→索引→图谱)", to_crawl)
    ok = await _crawl_and_wait(to_crawl)
    if ok:
        log.info("自动爬取: %s 建库完成，继续回答", to_crawl)
    else:
        log.warning("自动爬取: %s 抓取失败/不存在，将回「不知道」", to_crawl)
    return candidates, ok, to_crawl


async def ask(question: str, thread_id: str = "default") -> dict:
    candidates, ok, crawled = await ensure_characters(question)
    if candidates and not ok:
        log.info("自动爬取: 最终回「不知道」(角色=%s)", candidates)
        return {
            "answer": "不知道（知识库里没有这个角色，尝试联网抓取也没找到）。",
            "characters": candidates, "intent": "", "slots": [], "docs": 0,
        }
    chain = await get_chain()
    injected = crawled if crawled else []
    return await chain.ainvoke(
        {"question": question, "characters": injected},
        config={"configurable": {"thread_id": thread_id}},
    )



async def _main() -> None:
    ensure_dirs()
    for q in ("卡卡罗毕业配装用什么声骸", "卡卡罗六阶突破要多少贝币", "卡卡罗怎么玩"):
        r = await ask(q)
        print("=" * 60)
        print("问题:", q)
        print("意图:", r.get("intent"), "| 槽位:", r.get("slots"),
              "| 角色:", r.get("characters"), "| 属性:", r.get("element") or "-")
        print("\n--- 回答 ---\n", r.get("answer"))


def main() -> None:
    asyncio.run(_main(), loop_factory=asyncio.SelectorEventLoop)


if __name__ == "__main__":
    main()
