"""Step 7：LangGraph 编排。"""
from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from ..config import ensure_dirs, get_settings
from ..graph.neo4j_client import get_session
from ..ww_logger import get_logger
from .intent import classify, detect_slots, extract_characters, detect_element, detect_stage
from .llm import get_chat_llm, no_think_marker
from .loopguard import LoopGuard, trim_loop
from .tools import graph_search_tool, vector_search_tool
from .state import RagState
from .memory import get_checkpointer
from ..text import chunk_text
from .characters import resolve_candidates
from ..worker import build_pipeline, CharacterNotFound

setting = get_settings()
log = get_logger("rag")

# 注意：人设由 aemeath 模型自带（Modelfile 的 SYSTEM），这里不再引导口吻——
# 叠加人设指令会让模型把注意力放在「表演」而非「答题」上，是复读独白的诱因之一。
# 本提示词只负责三件事：术语对照、答题约束、防重复。
# 另：Ollama 的 system 参数会覆盖 Modelfile 内置 SYSTEM，所以这些约束必须拼在
# HumanMessage 里，不能改成 SystemMessage 传，否则人设会丢。
_SYSTEM = """依据下面的资料作答。术语对照（资料用词和提问可能不同，按下表理解）：
- 声骸 = 角色装备，资料里也写作「套装」；COST 是声骸的费用点数组合
- 共鸣链 = 相当于命座，序号 1~6 对应一链到六链
- 贝币 = 游戏货币
- 突破阶段：一阶~六阶

作答要求：
- 用中文，简洁、要点化；资料里列了几条就答几条，不要自行补充「没有提到其他/更多」。
- 资料里没有的内容，用一句话说不知道就停住，不要展开、不要举例、不要反复解释。
- 严禁重复：同一句话、同一段落、同一句口头禅只能说一次。答完即止，不要为凑长度反复输出相同内容。"""


def _new_loop_guard() -> LoopGuard:
    """每次生成都要新建一个 guard：它内部有累积状态，跨请求复用会串味。"""
    return LoopGuard(
        max_repeat=setting.LLM_LOOP_MAX_REPEAT,
        min_chars=setting.LLM_LOOP_MIN_CHARS,
    )



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
    chars = state.get("characters") or []
    n = max(len(chars), 1)

    docs = await vector_search_tool.ainvoke({"query": q, "topk": setting.TOPK_RERANK})

    # 多角色：每个非首角色各补一轮召回，避免只召回到其中一个
    # 用整数除 // 并设下限 2，避免 6/4=1.5 被截断成 1 导致漏召
    per = max(2, setting.TOPK_RERANK // n)
    for c in chars[1:]:
        docs += await vector_search_tool.ainvoke({
            "query": f"{c} {q}", "topk": per
        })

    seen: set[str] = set()
    merged: list[dict] = []
    for d in docs:
        if d["chunk_id"] not in seen:
            seen.add(d["chunk_id"])
            merged.append(d)
    return {"docs": merged}


def _build_context(state: RagState) -> str:
    parts: list[str] = []
    history = (state.get("history", []))[-setting.MAX_HISTORY_TURNS * 2:]
    if history:
        parts.append("## 对话历史\n" + "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}" for m in history
        ))
    if state.get("graph_facts"):
        parts.append("## 图谱事实\n" + state["graph_facts"])
    docs = state.get("docs") or []
    if docs:
        parts.append("## 参考文档\n" + "\n\n".join(
            f"[{i + 1}] {chunk_text(d)}" for i, d in enumerate(docs)
        ))
    # 「有没有资料」必须看图谱事实/文档，不能用 join 结果是否为空来判断：
    # 多轮对话时 history 非空，join 永远有内容，原先的 or 兜底就永远不触发，
    # 零资料信号被吞掉 → 模型收不到约束 → 退化成自由发挥（人设独白 + 复读）。
    if not state.get("graph_facts") and not docs:
        # 这里不要再写「## 资料」标题：_build_prompt 已经加了，重复标题会干扰模型
        parts.append(
            "（本次没有检索到任何资料。请只用一句话说明你不清楚，然后立即停止；"
            "不要解释原因，不要重复这句话，不要补充任何其他内容。）"
        )
    return "\n\n".join(parts)


def _build_prompt(context: str, question: str) -> str:
    # /no_think 放在末尾：Qwen3 的软开关，关掉思考模式（CoT 会降低角色扮演质量，
    # 且 thinking 泄漏会加剧复读）。原 serve_amis.py 在服务端强制关，弃用后由此接手。
    marker = no_think_marker()
    tail = f"\n\n{marker}" if marker else ""
    return f"{_SYSTEM}\n\n## 资料\n{context}\n\n## 问题\n{question}{tail}"


async def generate_node(state: RagState) -> dict:
    context = _build_context(state)
    prompt = _build_prompt(context, state["question"])
    try:
        resp = await get_chat_llm().ainvoke([HumanMessage(content=prompt)])
        answer = resp.content
    except Exception as exc:
        log.error("LLM 调用失败: %s", exc)
        answer = "抱歉，模型服务暂时不可用，请稍后再试。"
        answer_ok = False
    else:
        answer_ok = True

    # 复读兜底放在 try 之外：检测/截断自身若出错，不该被误报成「模型服务不可用」
    truncated = False
    if answer_ok:
        # 采样参数只能降低复读概率，压不死；这里是最后一道闸。
        # 非流式拿到的是完整文本，整篇喂一次即可。
        hit = _new_loop_guard().feed(answer)
        if hit:
            log.warning("复读检测命中，已截断答案（触发句=%.30s… 原长=%d）", hit, len(answer))
            answer = trim_loop(answer, hit)
            truncated = True

    history = (state.get("history", []))[-setting.MAX_HISTORY_TURNS * 2:]
    new_history = history + [
        {"role": "user", "content": state["question"]},
        {"role": "assistant", "content": answer},
    ]
    return {
        "context": context,
        "answer": answer,
        "truncated": truncated,
        "history": new_history[-setting.MAX_HISTORY_TURNS * 2:],
    }


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


async def ask_stream(question: str, thread_id: str = "default"):
    """流式问答：先检索（非流式，首屏延迟），再逐 token 吐答案。
    yield dict：{'token': str}（增量）/ {'status': 'retrieving'}（开始检索）/
                {'done': True, ...}（结束，带元数据）。"""
    candidates, ok, crawled = await ensure_characters(question)
    if candidates and not ok:
        yield {"token": "不知道（知识库里没有这个角色，尝试联网抓取也没找到）。"}
        yield {"done": True}
        return

    yield {"status": "retrieving"}  # 前端可显示「检索中…」

    slots = detect_slots(question)
    chars = extract_characters(question, await _known_characters()) or []
    element = detect_element(question)
    stage = detect_stage(question)
    intent = classify(question, slots)

    state: RagState = {
        "question": question,
        "characters": crawled or chars,
        "slots": slots, "element": element, "stage": stage, "intent": intent,
    }
    if intent in ("fact", "hybrid"):
        state["graph_facts"] = (await graph_node(state))["graph_facts"]
    if intent in ("semantic", "hybrid"):
        state["docs"] = (await vector_node(state))["docs"]

    prompt = _build_prompt(_build_context(state), question)

    guard = _new_loop_guard()
    full: list[str] = []
    hit: str | None = None   # 循环外要用，先初始化避免依赖循环内赋值
    truncated = False
    async for chunk in get_chat_llm().astream([HumanMessage(content=prompt)]):
        if not chunk.content:
            continue
        hit = guard.feed(chunk.content)
        if hit:
            # 提前止损：break 会关闭生成器，Ollama 侧随之中断，不会继续写满 num_predict
            log.warning("流式复读检测命中，中断生成（触发句=%.30s…）", hit)
            truncated = True
            break
        full.append(chunk.content)
        yield {"token": chunk.content}

    answer = "".join(full)
    if truncated:
        # token 已经吐给前端、收不回来，所以把截断后的权威全文放进 done 事件，
        # 由前端覆盖已渲染内容。
        answer = trim_loop(answer, hit)

    yield {
        "done": True,
        "answer": answer,
        "intent": intent,
        "slots": slots,
        "characters": chars,
        "docs": len(state.get("docs") or []),
        "truncated": truncated,
    }


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
