"""Step 7：LangGraph 编排。"""
from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from ..config import ensure_dirs, get_settings
from ..graph.neo4j_client import get_session
from ..text import chunk_text
from ..worker import CharacterNotFound, build_pipeline
from ..ww_logger import get_logger
from .characters import resolve_candidates
from .intent import (
    classify,
    classify_topic,
    detect_element,
    detect_slots,
    detect_stage,
    extract_characters,
)
from .llm import get_chat_llm
from .loopguard import LoopGuard, trim_loop
from .memory import get_checkpointer
from .state import RagState
from .tools import graph_search_tool, vector_search_tool

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

# 流式阶段文案：只收真正干活的节点。LangGraph / _route / _after_graph 是图容器与
# 路由函数（实测 astream_events 也会为它们发 on_chain_start），对用户无意义，排除。
# 检索+重排实测约 19s，而生成仅 1~2s —— 这段静默期正是用户焦虑的来源。
_STAGE_LABELS = {
    "intent": "分析问题类型与角色",
    "chitchat": "陪家人聊两句",
    "graph": "查询角色关系图谱",
    "vector": "检索并重排相关资料",
    "generate": "整理答案中",
}


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

    # 闲聊分流（qwen3:8b 主题 agent）：仅在「无角色名 且 无槽位 且 无属性/阶段」时才调用。
    # 槽位非空几乎必然是游戏提问（实测「秧秧怎么玩」这类靠语义命中；真闲聊句槽位为空，
    # 误触发 rule_has_signal 的是 SEMANTIC_PATTERNS 的「怎么」——所以判据用 slots 而非它）。
    # 有角色名绝不当闲聊（「你好呀卡卡罗」是提问）；LLM 解析失败回落 game。
    if not chars and not slots and not stage and not element:
        topic = await classify_topic(q)
        if topic == "chitchat":
            intent = "chitchat"
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
    # 不在这里注入 /no_think：实测它对 aemeath 无效（仍 38s + 'v' 泄漏前缀 + 触发
    # Ollama 500）。思考模式由 llm.py 的 .bind(think=False) 统一关闭。
    return f"{_SYSTEM}\n\n## 资料\n{context}\n\n## 问题\n{question}"


async def generate_node(state: RagState):
    """生成节点：支持流式与非流式两种调用方式。

    - 非流式（ask → chain.ainvoke）：LangGraph 自动收集所有 yield 的最终状态，
      行为等价于返回 dict。
    - 流式（ask_stream → chain.astream_events）：每个 yield 都会产出
      on_chain_stream 事件，前端可从 on_chat_model_stream 抽 token。

    内部用 llm.astream() 逐 token 生成，同时做复读检测。
    """
    context = _build_context(state)
    prompt = _build_prompt(context, state["question"])

    guard = _new_loop_guard()
    full: list[str] = []
    hit: str | None = None
    truncated = False
    answer_ok = True

    try:
        async for chunk in get_chat_llm().astream([HumanMessage(content=prompt)]):
            if not chunk.content:
                continue
            h = guard.feed(chunk.content)
            if h:
                log.warning("流式复读检测命中，中断生成（触发句=%.30s…）", h)
                hit = h
                truncated = True
                break
            full.append(chunk.content)
            # 增量 yield：让 astream_events 产出 on_chain_stream 事件
            # 注意：answer_chunk 是临时字段，不在 RagState 里声明，
            # LangGraph 会忽略未声明字段的持久化，但事件里仍然能看到。
            yield {"answer_chunk": chunk.content}
    except Exception as exc:
        log.error("LLM 调用失败: %s", exc)
        answer_ok = False
        if not full:
            full.append("抱歉，模型服务暂时不可用，请稍后再试。")

    answer = "".join(full)
    if answer_ok and truncated and hit:
        answer = trim_loop(answer, hit)

    history = (state.get("history", []))[-setting.MAX_HISTORY_TURNS * 2:]
    new_history = history + [
        {"role": "user", "content": state["question"]},
        {"role": "assistant", "content": answer},
    ]
    # 最终状态：answer/truncated/history 是 RagState 声明的字段，会被 checkpointer 持久化
    yield {
        "context": context,
        "answer": answer,
        "truncated": truncated,
        "history": new_history[-setting.MAX_HISTORY_TURNS * 2:],
    }


# 闲聊人设提示：模型自带人设，这里只给最小引导；不带术语表与 RAG 答题约束，
# 否则会诱发「资料里没有…」式拒答独白（正是复读的温床）。
_CHITCHAT_HINT = "家人在跟你闲聊，没有要查资料。自然、简短地回应，不要提资料、检索或知识库。"


async def chitchat_node(state: RagState):
    """闲聊分支：不挂检索，带对话历史，人设自然回应。streaming node。"""
    msgs: list = []
    for m in (state.get("history", []))[-setting.MAX_HISTORY_TURNS * 2:]:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        else:
            msgs.append(AIMessage(content=m["content"]))
    msgs.append(HumanMessage(content=f"{_CHITCHAT_HINT}\n\n家人说：{state['question']}"))

    guard = _new_loop_guard()
    full: list[str] = []
    hit: str | None = None
    truncated = False
    answer_ok = True
    try:
        async for chunk in get_chat_llm().astream(msgs):
            if not chunk.content:
                continue
            h = guard.feed(chunk.content)
            if h:
                log.warning("闲聊复读检测命中，中断生成（触发句=%.30s…）", h)
                hit = h
                truncated = True
                break
            full.append(chunk.content)
            yield {"answer_chunk": chunk.content}
    except Exception as exc:
        log.error("闲聊生成失败: %s", exc)
        answer_ok = False
        if not full:
            full.append("诶？我刚才走神了……你再说一遍嘛。")

    answer = "".join(full)
    if answer_ok and truncated and hit:
        answer = trim_loop(answer, hit)

    history = (state.get("history", []))[-setting.MAX_HISTORY_TURNS * 2:]
    new_history = history + [
        {"role": "user", "content": state["question"]},
        {"role": "assistant", "content": answer},
    ]
    yield {
        "answer": answer,
        "truncated": truncated,
        "history": new_history[-setting.MAX_HISTORY_TURNS * 2:],
    }


def build_graph() -> StateGraph:
    g = StateGraph(RagState)
    g.add_node("intent", intent_node)
    g.add_node("chitchat", chitchat_node)
    g.add_node("graph", graph_node)
    g.add_node("vector", vector_node)
    g.add_node("generate", generate_node)

    g.add_edge(START, "intent")
    g.add_conditional_edges("intent", _route, {
        "fact": "graph", "semantic": "vector", "hybrid": "graph", "chitchat": "chitchat",
    })
    g.add_conditional_edges("graph", _after_graph, {
        "need_vector": "vector", "done": "generate",
    })
    g.add_edge("vector", "generate")
    g.add_edge("generate", END)
    g.add_edge("chitchat", END)
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
    """流式问答：走 LangGraph astream_events，checkpointer 自动管理多轮记忆。

    yield dict：
      {'token': str}                        答案增量
      {'status': 'retrieving'}              兼容旧前端的粗粒度状态
      {'stage': str, 'label': str}          细粒度阶段（前端显示「正在…」用）
      {'done': True, ...}                   结束，带元数据

    与 ask() 共享同一张图和 checkpointer，thread_id 真正生效。
    """
    # 阶段文案：检索+重排实测约 19s、生成仅 1~2s，所以必须让用户看到在干什么
    yield {"stage": "crawl", "label": "检查角色是否在知识库中"}

    candidates, ok, crawled = await ensure_characters(question)
    if candidates and not ok:
        yield {"token": "不知道（知识库里没有这个角色，尝试联网抓取也没找到）。"}
        yield {"done": True}
        return

    yield {"status": "retrieving"}  # 前端可显示「检索中…」

    chain = await get_chain()
    injected = crawled if crawled else []
    input_state = {"question": question, "characters": injected}
    config = {"configurable": {"thread_id": thread_id}}

    # 从事件流抽 token + 阶段 + 最终元数据
    full: list[str] = []
    final_state: dict = {}
    emitted_stages: set[str] = set()   # streaming node 会触发两次 on_chain_start，去重
    async for event in chain.astream_events(input_state, version="v2", config=config):
        kind = event.get("event", "")
        name = event.get("name", "")

        # 节点开始事件：转成用户可读的阶段提示（同一节点只发一次）
        if kind == "on_chain_start" and name in _STAGE_LABELS and name not in emitted_stages:
            emitted_stages.add(name)
            yield {"stage": name, "label": _STAGE_LABELS[name]}

        # token 级事件：**必须按节点过滤**。intent_node 里的主题分类器也调 LLM，
        # 它的流式事件同样挂在 on_chat_model_stream 上（metadata.langgraph_node='intent'），
        # 不过滤会把 {"topic":"chitchat"} 这类分类输出当答案吐给前端（实测发生过）。
        elif kind == "on_chat_model_stream":
            node = event.get("metadata", {}).get("langgraph_node", "")
            if node not in ("generate", "chitchat"):
                continue
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                full.append(chunk.content)
                yield {"token": chunk.content}

        # 图执行结束事件：拿最终完整状态（含 intent/slots/characters/docs/truncated/answer）
        elif kind == "on_chain_end" and name == "LangGraph":
            output = event.get("data", {}).get("output", {})
            if isinstance(output, dict):
                final_state = output

    answer = "".join(full)
    # 如果 generate_node 内部已截断，final_state["answer"] 是截断后的权威全文
    if final_state.get("truncated") and final_state.get("answer"):
        answer = final_state["answer"]

    yield {
        "done": True,
        "answer": answer,
        "intent": final_state.get("intent", ""),
        "slots": final_state.get("slots") or [],
        "characters": final_state.get("characters") or [],
        "docs": len(final_state.get("docs") or []),
        "truncated": bool(final_state.get("truncated")),
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
