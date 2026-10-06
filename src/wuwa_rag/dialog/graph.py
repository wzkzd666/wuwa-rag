"""Step 7：LangGraph 编排。"""
from __future__ import annotations

import asyncio
import re
import time
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph

from wuwa_rag.config import ensure_dirs, get_settings
from wuwa_rag.core import llmstore
from wuwa_rag.core.llm import get_chat_llm
from wuwa_rag.dialog.guard import LoopGuard, trim_loop
from wuwa_rag.dialog.memory import get_checkpointer
from wuwa_rag.dialog.nlu import (
    classify,
    classify_topic,
    detect_element,
    detect_slots,
    detect_stage,
    extract_characters,
    is_identity,
    is_self_intro,
    is_time_question,
    mentions_time,
    rewrite_query,
    summarize_turns,
    veto_ambiguous_names,
)
from wuwa_rag.dialog.prompt import (
    build_context,
    build_prompt,
    doc_sources,
)
from wuwa_rag.dialog.state import RagState
from wuwa_rag.dialog.tools import current_time_tool, graph_search_tool, vector_search_tool
from wuwa_rag.knowledge import domain_terms
from wuwa_rag.knowledge.entities import find_mentions, resolve_candidates
from wuwa_rag.knowledge.graph.neo4j_client import get_session
from wuwa_rag.knowledge.retrieve import fetch_chunks
from wuwa_rag.services import persona, tts
from wuwa_rag.services.emotion import detect_emotion
from wuwa_rag.services.verify import verify_knowledge
from wuwa_rag.services.websearch import web_search
from wuwa_rag.tasks.worker import (
    CharacterNotFound,
    build_pipeline,
    build_refresh_pipeline,
    reset_progress,
)
from wuwa_rag.text import (
    AnswerFilter,
    chunk_text,
    dedup_list_items,
    fix_percent_units,
    strip_ref_marks,
)
from wuwa_rag.ww_logger import get_logger

setting = get_settings()
log = get_logger("rag")

# 有界线程池：供 fetch_chunks 等同步阻塞调用使用。
# asyncio.to_thread 用默认 ThreadPoolExecutor（max_workers=40），并发高时占满线程资源。
# 限 8 个线程足够（检索不是高并发场景），避免与 Celery worker / Ollama 抢线程。
_IO_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="rag-io")


async def _run_io(func, *args, **kwargs):
    """把同步阻塞函数放到有界线程池跑，替代 asyncio.to_thread。"""
    from functools import partial
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_IO_EXECUTOR, partial(func, *args, **kwargs))

# 流式阶段文案：只收真正干活的节点。LangGraph / _route / _after_graph 是图容器与
# 路由函数（实测 astream_events 也会为它们发 on_chain_start），对用户无意义，排除。
# 检索+重排实测约 19s，而生成仅 1~2s —— 这段静默期正是用户焦虑的来源。
_STAGE_LABELS = {
    "intent": "分析问题类型与角色",
    "chitchat": "陪家人聊两句",
    "time": "查看当前时间",
    "graph": "查询角色关系图谱",
    "vector": "检索并重排相关资料",
    "verify": "核对资料是否对题",
    "web": "联网搜索补充知识",
    "generate": "整理答案中",
}

# 内部 LLM 调用的标签：这些调用的输出是**结构化中间结果**（摘要/审查/情绪），
# 绝不能出现在用户答案里。它们都发生在 generate/chitchat 节点**内部**，其流式回调
# metadata.langgraph_node 就是 'generate'，**节点名过滤挡不住**，只能靠标签丢弃。
# 实测教训（两轮，同一类泄漏）：
#   ① 未过滤 wwa:summary → 摘要整句被拼进答案尾巴（六轮深会话测试抓到）；
#   ② 未过滤 wwa:emotion 时情绪 JSON 同样会拼进答案（接入情绪标签时已复现）。
# 约束：新增任何「在生成节点内部调用 LLM」的功能，都必须打标签并加入该集合。
# wwa:verify 不在此列：verify_node 是**独立图节点**，节点名不在放行白名单
# （generate/chitchat），已被节点过滤挡住；但打标签无害且更稳，故一并列入。
_INTERNAL_TAGS = frozenset({"wwa:summary", "wwa:emotion", "wwa:verify"})


def _new_loop_guard() -> LoopGuard:
    """每次生成都要新建一个 guard：它内部有累积状态，跨请求复用会串味。"""
    return LoopGuard(
        max_repeat=setting.LLM_LOOP_MAX_REPEAT,
        min_chars=setting.LLM_LOOP_MIN_CHARS,
        min_item=setting.LLM_LOOP_MIN_ITEM,
        period_max=setting.LLM_LOOP_PERIOD_MAX,
        min_cycles=setting.LLM_LOOP_MIN_CYCLES,
    )


def _trim_loop(text: str, sentence: str | None) -> str:
    """按 config 阈值截断复读尾巴（guard 与 trim 必须用同一套阈值）。"""
    return trim_loop(
        text, sentence,
        max_repeat=setting.LLM_LOOP_MAX_REPEAT,
        min_chars=setting.LLM_LOOP_MIN_CHARS,
        min_item=setting.LLM_LOOP_MIN_ITEM,
        period_max=setting.LLM_LOOP_PERIOD_MAX,
        min_cycles=setting.LLM_LOOP_MIN_CYCLES,
    )


async def _chat_client(state: RagState, strict: bool = False) -> tuple[Runnable, bool, bool]:
    """取本轮 chat 客户端，返回 `(client, is_cloud, emotion_via_cloud)`。

    需求：① 默认用项目自带 agent（本地 Ollama aemeath）；② 用户配了自己的云端
    OpenAI 兼容 API 就走云端。判定顺序：state 里有 `user_id` → `llmstore.get_runtime`
    取该用户配置（含解密后的 key）→ 取到走云端，取不到（未配 / 已停用 / 会话未
    输入加密口令 / 解密失败）一律**回落本地默认**，问答不因配置问题中断。

    返回值 is_cloud 决定人设注入方式，两条规则相反（判断错误会丢失作答效果）：
      - 本地 aemeath：人设烧在 Modelfile SYSTEM，**不得**发 SystemMessage（会覆盖人设），
        `prompt.SYSTEM_PROMPT` 走 HumanMessage 前缀（现状，逐字节不变）；
      - 云端通用模型：不认识爱弥斯，**必须**发 `SystemMessage(persona.cloud_system())`，
        且 prompt 不再前缀 `SYSTEM_PROMPT`（避免术语表/作答要求重复两遍）。

    明文 key 只存在于返回的 client 实例中，不得写入 state —— checkpointer 会把
    state 持久化进 PG，写进去等于把用户密钥落盘到另一张表。

    第三个返回值 `emotion_via_cloud`：**该用户是否勾选了让自己的模型兼任情绪判定**
    （`llmstore.emotion_enabled`，默认否）。为真时把本 client 交给 rag/emotion.py，
    否则情绪判定走本地 tool 模型（分工说明见该模块 docstring）。
    """
    uid = state.get("user_id")
    if uid:
        try:
            cfg = await llmstore.get_runtime(int(uid))
        except Exception as exc:  # noqa: BLE001 —— 配置读不出来就回落本地默认，不因配置问题中断问答
            log.warning("读取云端模型配置失败，回落本地默认：%s", exc)
            cfg = None
        if cfg:
            return (get_chat_llm(strict=strict, cloud_cfg=cfg), True,
                    bool(cfg.get("emotion_enabled")))
    return get_chat_llm(strict=strict), False, False


async def _commit_history(state: RagState, answer: str) -> dict:
    """本轮问答写回记忆，并在轮次被挤出窗口时滚动压缩（B）。

    evicted 必须在截断**之前**取：checkpointer 里只存截断后的窗口，
    事后再也拿不到被挤出的轮次。只有真有 eviction 才调摘要模型——
    窗口没满时无需压缩，省一次调用。摘要失败保持原摘要（回落不压缩）。
    生成与闲聊两个节点共用。
    """
    prev = state.get("history", [])
    window = setting.MAX_HISTORY_TURNS * 2
    grown = prev + [
        {"role": "user", "content": state["question"]},
        {"role": "assistant", "content": answer},
    ]
    new_history = grown[-window:]
    out: dict = {"history": new_history}
    evicted = grown[:-window]
    if evicted:
        out["context_summary"] = await summarize_turns(
            evicted, state.get("context_summary", ""))
    return out



# 名册 TTL 缓存：每轮问答 intent_node 与 ensure_characters 至少各查一次 Neo4j，
# 而名册只在入库时变化——60s 缓存直接省掉一次往返。入库成功后主动失效。
_KNOWN_TTL = 60.0
_known_cache: tuple[float, list[str]] = (0.0, [])
_known_lock = asyncio.Lock()   # 防并发竞态：多个请求同时判空→同时查 Neo4j→同时写缓存


async def _known_characters() -> list[str]:
    global _known_cache
    now = time.monotonic()
    if _known_cache[1] and now - _known_cache[0] < _KNOWN_TTL:
        return _known_cache[1]
    async with _known_lock:
        # double-check：等锁期间可能已有另一个协程刷新了缓存
        now = time.monotonic()
        if _known_cache[1] and now - _known_cache[0] < _KNOWN_TTL:
            return _known_cache[1]
        async with get_session() as s:
            rows = await (await s.run("MATCH (c:Character) RETURN c.name AS n")).data()
        names = [r["n"] for r in rows]
        if names:                  # 空结果（Neo4j 抖动）不缓存，避免脏 60s
            _known_cache = (now, names)
        return names


def _invalidate_known_cache() -> None:
    """新角色建库完成后调用，下一轮立刻能看到。"""
    global _known_cache
    _known_cache = (0.0, [])


# 远指代信号：问句（含改写句）里出现这些词，说明指代对象在窗口外的摘要里。
# 「开头聊的那位武器推荐什么」即便改写器没解析出名字，图谱检索也需要角色。
_FAR_REF_RE = re.compile(r"开头|先前|之前|前面|上次|刚才|最初|那位")


def _inject_far_characters(
    sq: str, chars: list[str], summary: str, known: set[str],
) -> list[str]:
    """前角色注入（尾巴修复）：远指代问句从滚动摘要补「前面的角色」。

    - 触发只看**远指代词**（开头/之前/那位…），不看 chars 是否为空——比较句
      （「开头那位和长离谁强」）原有角色照常保留，前角色**追加**进去，
      extract_characters 的多角色抽取与每角色补召回原功能不受影响。
    - 「开头聊的那位」= 摘要里**最先**出现的名字（摘要按谈话顺序保留角色名），
      取法与 focus_anchors 的最近优先相反，按摘要文本位置排序。
    - 改写成功时远指代词已被替换掉，sq 不再命中 → 天然 no-op；已在 chars 中也不重复。

    ⚠️ 名字匹配一律走 `entities.find_mentions`，**不要**在这里另写一套正则：
    本函数原先自建「长度降序 + re.finditer」，与 `nlu.extract_characters` 是同一语义
    的两处判据（本项目对这种重复已有大量踩坑记录）。更要紧的是它有实际缺陷——
    裸子串匹配会让摘要里的「核心词条」「关心剧情」误命中单字角色「心」，
    注入一个用户从没聊过的角色。find_mentions 对单字名要求独立成词，
    并且**返回值本身就是按文本位置升序**的，「最靠前」直接取第一个即可。
    """
    if not summary or not known or not _FAR_REF_RE.search(sq):
        # known 为空时 find_mentions 会直接返回 []（内部有挡），此处一并短路省一次调用
        return chars
    hits = find_mentions(summary, known)
    if not hits:
        return chars
    name = hits[0][1]          # (位置, 名字)；按摘要文本位置取最靠前那个
    if name in chars:
        return chars
    log.info("前角色注入: %s（远指代=%r 摘要=%r）", name, sq[:24], summary[:40])
    return [*chars, name]


async def intent_node(state: RagState) -> dict:
    q = state["question"]
    # 领域词表（单字角色名消歧用）过期就丢一个后台重建，本轮仍用旧词表、不阻塞问答。
    # 冷启动时是空表，find_mentions 会回落到 entities 的手写兜底表。
    domain_terms.schedule_warmup()
    known = await _known_characters()
    # 追问改写（A+B 输入）：滚动摘要（窗口外压缩记忆）+ 焦点锚点（全量文本提角色，
    # 不怕 120 字截断丢名）+ 最近 2 轮短原文。检索信号全部吃改写句——
    # 「那她配什么声骸」单拿原句必落空，补出角色名才能命中。
    # 生成侧仍用原句+history（prompt.build_context 里有对话历史），展示不受影响。
    history = (state.get("history", []))[-setting.MAX_HISTORY_TURNS * 2:]
    summary = state.get("context_summary", "")
    sq = await rewrite_query(q, history, known=known, summary=summary)

    slots = detect_slots(sq)
    chars = extract_characters(sq, known) or state.get("characters", [])
    # 远指代兜底：改写器偶发解析失败/半解析时，摘要里的「前角色」兜住
    chars = _inject_far_characters(sq, chars, summary, set(known))
    # 歧义角色名的 LLM 裁决（默认关闭，见 config.NAME_LLM_ADJUDICATION —— 实测
    # qwen3:8b 两个问法都不可用：一个只会答「是」、另一个只会答「不是」，都会误伤）。
    # 放在 `_inject_far_characters` 之后：远指代补进来的名字也该受同一道裁决。
    if chars and setting.NAME_LLM_ADJUDICATION:
        chars = await veto_ambiguous_names(sq, chars)
    element = detect_element(sq) or state.get("element", "")
    stage = detect_stage(sq)
    intent = classify(sq, slots)

    # 约束：不走向量时必须标为 fact，intent 字段不得名不副实。
    # intent 是对外字段（`AskOut.intent` / SSE done.intent），「报 hybrid 却 docs=0」
    # 会让前端与排查都读到假信息。
    # 概括性配队（slots 恰为 ['队友'] 且未指名，见 _is_team_overview）只在图谱就能答全
    # （含「或」的模板已完整），**语义上就是 fact**。在此降级后：
    #   · `_route` 自然走 fact 分支（graph → verify），
    #   · 那条更宽的「hybrid → 补向量」规则也就不再命中 ——
    #   **不需要、也不该在 _after_graph 里再写一条例外**（双判据必然迟早不同步）。
    # 为什么必须在这里而不是 _after_graph：_after_graph 是**路由函数**，只能返回分支名，
    # 改不了 state；而 intent 要如实落到 state 里对外。
    if _is_team_overview(slots, chars):
        intent = "fact"

    # 闲聊分流，两层：
    # ⓪ 时间类（「现在几点」）——最优先判且**不调 LLM**：答案只可能来自服务端时钟
    #    （见 tools.current_time），走检索是空手而归、走闲聊是让模型猜。判据用原句 q。
    #    条件带 `not slots`：像「秧秧共鸣链几号节点要多少材料」这种真游戏问题若被抢走
    #    就丢了检索（时间问法在无槽位时才成立）。
    #    另外算一个 need_time（弱信号）：句子里**顺带**问了时间（「现在几点，顺便说说
    #    今汐的共鸣链」）时不动路由，只让检索/闲聊分支把服务端真值一并带上。
    #    两条判据都来自 intent.py，规则硬编码、零额外延迟（判据本身只是几个正则）。
    # ① 人格/身份类硬信号（问「你」的台词/名字/身份）——即使改写出了角色名，本质也
    #    是问人格不是查资料，直接判 chitchat。用**原句 q** 判：改写 sq 已把「你」补成
    #    角色名（「你的台词」→「爱弥斯的台词」），第二人称信号会丢。aemeath 人设由模型
    #    自带，走 RAG 反而召回大段角色剧情文案整段倾倒（实测「你的台词是什么」→ 809 字）。
    # ①′ 自我介绍硬信号（「我是颗粒」「叫我小星就行」）——同判 chitchat，与①方向相反
    #    （①问助手自己，①′陈述用户自己）。为什么必须走规则、不能只靠②的主题分类器：
    #    纯自我介绍句 `classify()` 兜底给 `hybrid`，会触发全量检索 → 查无资料 →
    #    `characters` 空、`verify_node` 的按角色重爬不触发 → 一路升级到**联网兜底**
    #    （用户实测「我是颗粒」走了联网查知识）。这类句子的正确出口只有「闲聊 + 写画像」，
    #    检索必然空手。判据同样用**原句 q**（改写器可能把它改得面目全非）。
    #    ⚠️ 带 `not slots`：复合句「我是萌新，守岸人怎么玩」虽被 is_self_intro 挡在
    #    句尾逗号处（不会命中），这里再加 not slots 是双保险——真游戏提问绝不被抢走。
    # ② qwen3:8b 主题 agent：仅在「无角色名 且 无槽位 且 无属性/阶段」时才调用。
    #    槽位非空几乎必然是游戏提问（实测「秧秧怎么玩」这类靠语义命中；真闲聊句槽位为空）。
    #    不能用 SEMANTIC_PATTERNS 当判据——「怎么」会误命中闲聊句（「怎么这么晚才来」）。
    #    有角色名绝不当闲聊（「你好呀卡卡罗」是提问）；LLM 解析失败回落 game。
    #    判据用改写句 sq：追问「那她配什么声骸」原句无角色，改写后有——不会误入闲聊。
    pure_time = is_time_question(q) and not slots
    # mentions_time 是 is_time_question 的超集，所以不必再 or 一次 pure_time
    need_time = mentions_time(q)

    if pure_time:
        intent = "time"
    elif (is_identity(q) or is_self_intro(q)) and not slots:
        intent = "chitchat"
    elif not chars and not slots and not stage and not element:
        topic = await classify_topic(sq)
        if topic == "chitchat":
            intent = "chitchat"
    log.info("意图=%s 槽位=%s 角色=%s 属性=%s 阶段=%s%s%s", intent, slots, chars or "未识别",
             element or "-", stage or "-", " | 顺带问时间" if need_time and intent != "time" else "",
             f" | 改写={sq!r}" if sq != q else "")
    return {"search_query": sq, "intent": intent, "slots": slots,
            "characters": chars, "element": element, "stage": stage,
            "need_time": need_time}


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


# 鸣潮一队只有 3 个人 —— 满 3 个角色名才算「点名一支具体队伍」。
_TEAM_SLOTS = 3


def _is_named_team(slots, characters) -> bool:
    """点名一支具体队伍：≥3 个角色名 + 问的是配队（队友槽位）。

    阈值取 3 而非 2：只有 2 个名字时用户尚未确定队伍（图谱会列出所有
    「含这两人」的队），补的向量是「这帮人相关的正文」，答非所问、还白等十几秒。

    为什么指名时要补向量：**队名在图谱、打法循环在正文描述里**（wiki 配队页正文
    有出招顺序，如「守岸人：AAAA-Z-E-Q-AAAA-Z-R」）。
    """
    return len(characters or []) >= _TEAM_SLOTS and "队友" in (slots or [])


def _is_team_overview(slots, characters) -> bool:
    """**概括性**问配队：只问了配队（slots 恰为 `['队友']`），且**没有**指名一支具体队伍。

    概括性询问配队不再走向量检索，作答结束后可追加一句
    『你对哪个队伍感兴趣，需要我给你详细介绍吗』。

    为什么能不走向量：图谱侧泛问给的就是含「或」的模板
    （`守岸人+吟霖/长离/散华+卡卡罗`），信息已完整；走向量只会把正文里**别的**队伍
    的打法描述召回来（答非所问），还要多等一次检索 + 重排（实测约 15~19s）。

    判据限定 `slots == ['队友']`：同时命中其它槽位时（「守岸人配队和声骸」
    → `['队友','声骸']`）时那些槽位仍需要向量，别一刀切。

    本函数是唯一的判据来源（state 版包装为 _named_team / _team_overview）。两处依赖它，
    必须口径一致：
      · `intent_node` —— 命中则把 intent 从 `hybrid` 降级为 **`fact`**（不走向量就该是 fact）；
      · `_should_ask_team` —— 命中则在答案末尾追问一句。
    """
    return list(slots or []) == ["队友"] and not _is_named_team(slots, characters)


def _named_team(state: RagState) -> bool:
    """state 版包装（见 _is_named_team）。"""
    return _is_named_team(state.get("slots"), state.get("characters"))


def _team_overview(state: RagState) -> bool:
    """state 版包装（见 _is_team_overview）。"""
    return _is_team_overview(state.get("slots"), state.get("characters"))


def _should_ask_team(state: RagState) -> bool:
    """要不要在答案末尾追问「对哪支队伍感兴趣」。

    两个条件缺一不可：① 本轮是概括性配队；② **图谱侧确实给出了队伍**
    （`graph_facts` 里出现了队友块标题）——否则问了也是无源之水
    （角色不在库 / 该角色 wiki 没有配队段时，图谱给不出队伍）。
    """
    return _team_overview(state) and "【可组队伍" in (state.get("graph_facts") or "")


# 概括性配队答完后的一句追问。**固定文案走确定性拼接**，
# 不求模型生成：这类"元话语"8B 会写得千奇百怪、时有时无，而且写进提示词就有
# negative-example 污染风险（见文件头铁律）。措辞带一点点 aemeath 语气。
_TEAM_FOLLOWUP = "你对哪个队伍感兴趣？需要我给你详细介绍一下吗~"


def _after_graph(state: RagState) -> str:
    """graph 之后：hybrid 还要补向量，其余直接进 verify。

    「概括性配队不走向量」由 intent 字段承载，本函数不再重复判断 ——
    `intent_node` 已把这类问题的 intent 从 `hybrid` **降级为 `fact`**（见 _is_team_overview），
    于是 `_route` 走 fact 分支、这里 `state["intent"] == "hybrid"` 也不再成立，自然只走图谱。
    **不要再在这里加一条 `if _team_overview(state): return "done"`**：同一个语义挂两条判据
    迟早不同步；更要紧的是 intent 字段是对外字段，必须与实际走的路径一致
    （intent 不允许例外：未走向量检索时一律标记为 fact）。

    例外一：技能类问题（slots 含「技能」）强制补一轮向量。原因：图谱的 HAS_SKILL
    每个 kind 只存了**技能名**（见 graph/extract.py::_extract_skills 取首个加粗串），
    没有效果描述与数值——问「爱弥斯共鸣解放」只喂得到「共鸣解放=飞至启明之时」
    这 7 行名字，模型无从作答（实测答「至于具体效果嘛……我记不清了啦」）。
    技能描述在原文里（爱弥斯 61 个技能 chunk 含「造成热熔伤害」「消耗全部【同步率】」
    等），必须走向量才能拿到。代价：技能类多一次检索 + 重排（约 15~19s）。

    例外二：指名具体队伍（见 _named_team，≥3 个角色名）——同理，队名在图谱、打法在正文。
    """
    if (state["intent"] == "hybrid" or "技能" in (state.get("slots") or [])
            or _named_team(state)):
        return "need_vector"
    return "done"


async def vector_node(state: RagState) -> dict:
    # 检索吃改写句（intent_node 产出）：追问「那她配什么声骸」原句召不到秧秧的文档
    q = state.get("search_query") or state["question"]
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


# ───────────── 验证升级闭环：verify → 刷新/重检索 → 联网 → 生成 ─────────────

async def _refresh_and_wait(chars: list[str], timeout: int) -> bool:
    """按角色清库重爬（build_refresh_pipeline），阻塞等待。失败返回 False。"""
    from celery.exceptions import TimeoutError as CeleryTimeout
    results = []
    for n in chars:
        reset_progress(n)          # 前端进度重走五步，不然一直显示上一轮全绿
        r = build_refresh_pipeline(n).apply_async()
        results.append(r)
        log.info("资料刷新: 已入队 chain_id=%s 角色=%s", r.id, n)
    t0 = time.perf_counter()
    try:
        for r in results:
            await _run_io(r.get, timeout=timeout)
    except CharacterNotFound:
        log.warning("资料刷新: 角色 %s wiki 上不存在，刷新终止", chars)
        return False
    except CeleryTimeout:
        log.warning("资料刷新: 等待超时(%ss)，按失败处理", timeout)
        return False
    except Exception as exc:  # noqa: BLE001 —— r.get() 会透传流水线任务内的**任意**异常
        # 上面已单独接住 CharacterNotFound（wiki 上没这个角色）与 CeleryTimeout（等待超时）；
        # 这里兜的是五步链里真实抛出的各种错误（抓取失败、入库约束冲突、索引写坏……）。
        # 刷新的定位是「尽力重爬一次」，失败就降级走重检索/联网，不该让整轮问答崩掉。
        log.warning("资料刷新失败: %s", exc)
        return False
    log.info("资料刷新: %s 重建完成，耗时 %.1fs", chars, time.perf_counter() - t0)
    _invalidate_known_cache()
    return True


async def verify_node(state: RagState) -> dict:
    """生成前把关：资料真能回答问题吗？不匹配按代价从低到高逐级升级。

    路径：ok→generate；
    不匹配且已识别角色且没刷过 → 清库重爬（刷新即一次重检索机会）；
    不匹配但还有重试额度（无角色可刷 / 刷过仍歪）→ 用 refined 重检索；
    额度用尽 → 千帆联网（key 空/失败则跳过）；都没有 → exhausted 直接生成
    （空/不匹配材料由 prompt.build_context 的「一句话不知道」约束兜底）。
    防死循环：retry_count 上限 VERIFY_MAX_RETRY + refreshed 只一次 + used_web 只一次，
    verify 最多进两次。
    """
    if not setting.VERIFY_ENABLED:
        return {"verify_stage": "ok"}

    sq = state.get("search_query") or state["question"]
    match, refined = await verify_knowledge(sq, state.get("graph_facts", ""), state.get("docs") or [])
    if match:
        return {"verify_stage": "ok"}

    retry = state.get("retry_count", 0)
    chars = state.get("characters") or []
    updates: dict = {"retry_count": retry + 1, "refined_query": refined}

    if chars and not state.get("refreshed"):
        # 缓存/脏数据不匹配：按角色清库重爬，回来后重检索（refined 若更优则用）
        updates["refreshed"] = True
        if refined and refined != sq:
            updates["search_query"] = refined
            updates |= await _rescan(refined, state.get("characters"))
        ok = await _refresh_and_wait(chars, setting.REFRESH_WAIT_TIMEOUT)
        updates["verify_stage"] = "refreshed" if ok else _retry_or_web(state, refined)
        if updates["verify_stage"] != "refreshed":
            updates["retry_count"] = setting.VERIFY_MAX_RETRY  # 刷新失败不再耗额度
        return updates

    if retry < setting.VERIFY_MAX_RETRY:
        # 召回跑题：verifier 的精确检索式重检索一轮。
        # 注意别用 _retry_or_web 给这里定 stage——它会直接跳到 web/exhausted，
        # "retrieval" 永远出不来（实测漏网：无角色题因此丢失精化重检索机会）。
        if refined:
            updates["search_query"] = refined
            updates |= await _rescan(refined, state.get("characters"))
        updates["verify_stage"] = "retrieval"
        return updates

    updates["verify_stage"] = "web" if _web_available() and not state.get("used_web") else "exhausted"
    return updates


def _web_available() -> bool:
    return bool(get_settings().QIANFAN_API_KEY.strip())


def _retry_or_web(state: RagState, refined: str) -> str:
    """重检索后仍不匹配（或刷新失败）的下一级：还有 web 额度走 web，否则收口。"""
    if state.get("retry_count", 0) + 1 < setting.VERIFY_MAX_RETRY:
        return "retrieval"
    if _web_available() and not state.get("used_web"):
        return "web"
    return "exhausted"


async def _rescan(sq: str, fallback_chars: list[str] | None = None) -> dict:
    """refined 检索式落地前重算规则信号，让重检索走对通道（槽位/角色/属性/阶段）。

    fallback_chars：verifier 给的 refined 会**凭空塞角色**（实测：问「我的技能有
    哪些」角色未识别，refined='今汐 技能' → characters 被改成今汐，答非所问；
    另一轮 refined='卡卡罗 技能' 同理）。本轮已识别的角色优先，refined 抽不出
    名字时不顶替它。
    """
    known = await _known_characters()          # async：直接读缓存名册（TTL 内零成本）
    return {
        "slots": detect_slots(sq),
        "characters": extract_characters(sq, known) or (fallback_chars or []),
        "element": detect_element(sq),
        "stage": detect_stage(sq),
    }


async def web_node(state: RagState) -> dict:
    """联网兜底：千帆实时搜索。失败不报错，降级走生成侧「一句话不知道」。"""
    q = state.get("refined_query") or state.get("search_query") or state["question"]
    ok, text = await web_search(q)
    log.info("联网兜底: %s (%d 字)", "命中" if ok else "无结果", len(text))
    return {"used_web": True, "web_facts": (text if ok else "")[:1500]}


def _after_verify(state: RagState) -> str:
    stage = state.get("verify_stage", "ok")
    if stage == "ok" or stage == "exhausted":
        return "generate"
    if stage == "refreshed":
        # 刷新完成 → 按意图重走检索（fact 查图谱，semantic 走向量，hybrid 图谱起步）
        return "graph" if state.get("intent") in ("fact", "hybrid") else "vector"
    if stage == "retrieval":
        return "graph" if state.get("intent") in ("fact", "hybrid") else "vector"
    if stage == "web":
        return "web"
    return "generate"


# ───────────── 满级数值表 / 突破材料表：确定性补料 ─────────────
# 为什么不靠向量召回拿这两类材料：
#   技能数值表是 11 列宽表（等级 | Lv 1 … Lv 10），突破材料每块只有 70~90 字，
#   在 bge-reranker 眼里都赢不过大段机制描述——实测问「共鸣解放伤害倍率」时
#   top6 里数值表 0 条，top6 分数还挤在 0.9983~0.9994 毫无区分度。
#   但这两类材料的元数据（character + component）足以精确定位，直接按元数据取必中。
#   取到之后**在 Python 里替模型读表**：把「满级列」和材料格解析成成对条目再喂进去。
#   8B 模型读 11 列宽表会串列（实测把 Lv7 读数读成中文数字），替它读比让它读可靠。
_SKILL_TABS = ("常态攻击", "共鸣技能", "共鸣回路", "共鸣解放", "变奏技能", "延奏技能", "谐度破坏")
_MATERIAL_SLOT_RE = re.compile(r"突破|材料|素材|培养|养成|练度")
# 声骸/武器走确定性补料时，认领这两个槽位（组件名本身不写死，由 `chunks.component` 决定）
_ECHO_SLOTS = ("声骸", "武器")
_MATERIAL_ITEM_RE = re.compile(r"([\u4e00-\u9fa5A-Za-z·]+?)\s*[x×](\d+)")
_SEP_CELL_RE = re.compile(r"^[-:\s]*$")
_LEVEL_CELL_RE = re.compile(r"^L?V?\.?\d+$")          # 表头的 LV.2 / Lv 3 之类
_STAGE_ORDER = ("一阶突破", "二阶突破", "三阶突破", "四阶突破", "五阶突破", "六阶突破")


def _split_cells(line: str) -> list[str] | None:
    """markdown 表格行 -> 单元格；分隔行/占位表头/非表格行返回 None。"""
    s = line.strip()
    if not s.startswith("|"):
        return None
    cells = [c.strip() for c in s.strip("|").split("|")]
    if all(_SEP_CELL_RE.match(c) for c in cells):       # | --- | --- |
        return None
    if all(re.fullmatch(r"列\d+", c) for c in cells):   # | 列1 | 列2 |
        return None
    return cells


def _max_level_rows(text: str) -> list[tuple[str, str]]:
    """从技能数值表里抽 (行名, 满级值)，原样照抄不做任何换算。

    只认表头里独立成格的「Lv 10」：技能突破材料表的表头是「LV.2…LV.10」（带点、
    且没有 Lv 1），不会被误判成倍率表。非表格行会让表头失效，避免跨表串列。
    """
    rows: list[tuple[str, str]] = []
    col = -1
    for line in text.splitlines():
        cells = _split_cells(line)
        if cells is None:
            if not line.strip().startswith("|"):
                col = -1
            continue
        if "Lv 10" in cells:                 # 表头（重复出现的表头也在此重定位）
            col = cells.index("Lv 10")
            continue
        if col < 0 or len(cells) <= col:
            continue
        name, val = cells[0], cells[col]
        if name and val:
            rows.append((name, val))
    return rows


def _material_items(text: str) -> list[str]:
    """从材料表里抽「名称×数量」条目。

    两类表都要吃：角色突破材料是规整两列（低频隧花声核x4 | 贝币x5000），
    技能突破材料把一整阶的材料挤在一格里（低频隧花声核x2残翼偏振体x2贝币x1500），
    所以按「x数量」逐个切分，而不是按列取。
    """
    items: list[str] = []
    for line in text.splitlines():
        cells = _split_cells(line)
        if cells is None:
            continue
        for c in cells:
            if not c or c == "[图]" or _LEVEL_CELL_RE.match(c):
                continue
            found = _MATERIAL_ITEM_RE.findall(c)
            items += [f"{n}×{q}" for n, q in found] if found else [c]
    return items


def _max_level_material(text: str) -> list[str]:
    """技能突破材料表 -> 满级(Lv10) 那一档的材料清单。

    这张表把 9 个等级并排，模型直接照抄会把 9 档材料拼成一张混在一起的清单
    （实测「中频×3、低频×3、全频×4…贝币×100000」全糊成一团，语义失真），
    所以只取最后一格。两种块形都要吃：
      · 带「LV.10」表头的 9 列表 —— 直接按表头定位满级列；
      · 占位表头（| 列1 | 列2 |）下直接铺 9 格的那种（实测常态攻击就是这形状）——
        没有等级表头，只能按「最长的一行 = 分等级展开行」兜底，
        2 格的强化小表（高频隧花声核x3 | 贝币x50000）因此被排除。
    """
    col = -1
    tail = ""
    longest: list[str] = []
    for line in text.splitlines():
        cells = _split_cells(line)
        if cells is None:
            if not line.strip().startswith("|"):
                col = -1
            continue
        if "LV.10" in cells:
            col = cells.index("LV.10")
            continue
        if col >= 0 and len(cells) > col and cells[col]:
            tail = cells[col]
        elif len(cells) > len(longest):
            longest = cells
    cell = tail or (longest[-1] if len(longest) >= 6 else "")
    if not cell:
        return []
    found = _MATERIAL_ITEM_RE.findall(cell)
    return [f"{n}×{q}" for n, q in found] if found else [cell]


async def _skill_value_block(char: str, sq: str, docs: list[dict]) -> str:
    """技能数值表 -> 满级(Lv 10)数值块；取不到返回空串。"""
    chunks = await _run_io(fetch_chunks, char, "技能介绍")
    if not chunks:
        return ""
    tabs = sorted(
        {(d.get("tab") or "").strip() for d in chunks if (d.get("tab") or "").strip()},
        key=lambda t: _SKILL_TABS.index(t) if t in _SKILL_TABS else len(_SKILL_TABS),
    )
    # 优先按问句点名的技能页签；没点名（只问「技能」）就用本轮检索命中的页签，
    # 命中顺序即重排得分顺序，天然是「最相关的那两个技能」；检索也没命中
    # （问句只写了「技能」二字、召回的块不含技能介绍页）时兜底到两个主力页签——
    # 既然问题已明确在问技能（slots 含「技能」），就给倍率，不放空。
    picked = [t for t in tabs if t in sq]
    if not picked:
        seen: set[str] = set()
        for d in docs:
            t = d.get("tab") or ""
            if d.get("component") == "技能介绍" and t in tabs and t not in seen:
                seen.add(t)
                picked.append(t)
    if not picked:
        picked = [t for t in ("共鸣解放", "共鸣技能") if t in tabs]
    picked = picked[:2]
    if not picked:
        return ""

    parts: list[str] = []
    for tab in picked:
        rows: dict[str, str] = {}
        for d in chunks:
            if (d.get("tab") or "").strip() != tab:
                continue
            for name, val in _max_level_rows(chunk_text(d)):
                rows.setdefault(name, val)      # 同名同行去重：分块切开的表拼回来
        if rows:
            parts.append(f"### {tab}\n" + "\n".join(f"- {n}：{v}" for n, v in rows.items()))
    if not parts:
        return ""
    return (
        "## 满级数值表（Lv 10，原文照抄）\n"
        "（本问必须把下表逐行完整列出：数值与「+」「*」「%」原样保留，不得换算、"
        "不得合并、不得改成中文数字、不得说「记不清」。）\n" + "\n".join(parts)
    )


async def _material_block(char: str) -> str:
    """角色突破材料（一阶~六阶）+ 技能突破材料（满级一档）材料表；取不到返回空串。"""
    chunks = await _run_io(fetch_chunks, char, "角色突破材料")
    by_stage: dict[str, list[str]] = {}
    for d in chunks:
        stage = (d.get("breadcrumb") or "").split("›")[-1].strip()
        items = _material_items(chunk_text(d))
        if items:
            by_stage[stage] = items

    skill_chunks = await _run_io(fetch_chunks, char, "技能突破材料")
    by_tab: dict[str, list[str]] = {}
    for d in skill_chunks:
        tab = (d.get("tab") or "").strip()
        if not tab or tab in by_tab:
            continue                       # 同名页签多块，取到第一份完整表就够
        items = _max_level_material(chunk_text(d))
        if items:
            by_tab[tab] = items

    if not by_stage and not by_tab:
        return ""
    # 中文按 Unicode 排是「一三二五六四」，必须按游戏阶段名显式排序
    order = [s for s in _STAGE_ORDER if s in by_stage]
    order += [s for s in by_stage if s not in _STAGE_ORDER]
    tabs = [t for t in _SKILL_TABS if t in by_tab] + [t for t in by_tab if t not in _SKILL_TABS]
    lines = [f"### {s}\n" + "、".join(by_stage[s]) for s in order]
    lines += [f"- {t}（满级 Lv10 一档）：" + "、".join(by_tab[t]) for t in tabs]
    return (
        "## 突破材料表（原文照抄）\n"
        "（回答培养/突破问题时必须把下表逐条完整列出，材料名与数量原样保留。）\n"
        + "\n".join(lines)
    )


async def _echo_block(char: str, slots: list[str]) -> str:
    """声骸套装 / 武器推荐表（原文照抄）；取不到返回空串。

    ⚠️ 组件名**不写死**：wiki 的板块叫什么由数据决定（`chunks.component`），
    这里一次 `fetch_chunks(角色)` 取出该角色全部块，再按**槽位关键词**去挑组件
    （问「声骸」挑名字里带「声骸」的板块，问「武器」挑带「武器」的）。
    板块改名、加板块都不用改代码 —— 写死组件名的话，wiki 一改名就静默失效。

    只接「声骸 / 武器」两个槽位：技能、突破材料各有专用块（`_skill_value_block` /
    `_material_block`），这里再取一遍会重复。

    ⚠️ 为什么必须补这一块（用户实测踩到）：问「心声骸推荐」，模型把「心声骸」当成
    一个完整的词，答成「我先替你看看那位朋友的心声骸」—— 检索其实是对的（`COST 43311`
    原样答出），是**生成侧的分词**出的问题。把「本问解析（角色=心 / 内容=声骸）」与
    本块原文一起喂进去，模型就没有再猜分词的余地。
    """
    wanted = [s for s in slots if s in _ECHO_SLOTS]
    if not wanted:
        return ""
    chunks = await _run_io(fetch_chunks, char)
    if not chunks:
        return ""
    picked = {d.get("component") for d in chunks
              if d.get("component") and any(s in d["component"] for s in wanted)}
    if not picked:
        return ""
    body = "\n".join(chunk_text(d) for d in chunks if d.get("component") in picked).strip()
    if not body:
        return ""
    return (
        "## 声骸 / 武器推荐表（原文照抄）\n"
        f"（本问只答「{char}」本人的推荐。下表逐行完整列出，套装名与武器名是专有名词、"
        "逐字照抄（不改字、不换词、不用近义写法），COST 与词条里的数字和符号原样保留。）\n"
        + body
    )


async def _value_blocks(state: RagState) -> list[str]:
    """按问题类型补「满级数值表 / 突破材料表 / 声骸武器推荐表」。取不到返回空，绝不影响主链路。"""
    chars = state.get("characters") or []
    if not chars:
        return []
    sq = state.get("search_query") or state["question"]
    slots = state.get("slots") or []
    blocks: list[str] = []
    try:
        if "技能" in slots or any(t in sq for t in _SKILL_TABS):
            b = await _skill_value_block(chars[0], sq, state.get("docs") or [])
            if b:
                blocks.append(b)
        if _MATERIAL_SLOT_RE.search(sq):
            b = await _material_block(chars[0])
            if b:
                blocks.append(b)
        # 声骸/武器放最后：prompt 里越靠后的块离问句越近（近因位），而问「X 声骸」
        # 时它才是主料；上面两块是次要补充
        if "声骸" in slots or "武器" in slots:
            b = await _echo_block(chars[0], slots)
            if b:
                blocks.append(b)
    except Exception as exc:  # noqa: BLE001 —— 补料是增益不是依赖，失败只记日志
        # ⚠️ 这里刻意保持宽捕获：补料块（满级数值表 / 突破材料表）依赖 Chroma 元数据查询
        # + 正则读表，任一环节形态变化都可能抛新异常。它只是让答案更完整，
        # **绝不能**因为它失败就让整轮问答崩掉（用户至少还能拿到检索结果）。
        log.warning("补数值/材料表失败，跳过: %s", exc)
    return blocks


async def generate_node(state: RagState):
    """生成节点：支持流式与非流式两种调用方式。

    - 非流式（ask → chain.ainvoke）：LangGraph 自动收集所有 yield 的最终状态，
      行为等价于返回 dict。
    - 流式（ask_stream → chain.astream_events）：token 从 on_chat_model_stream 抽取
      （llm.astream 自带回调），节点内部不再逐 token yield 中间态。

    内部用 llm.astream() 逐 token 生成，同时做复读检测。
    """
    # 确定性补料：技能问题补满级数值表、培养问题补突破材料表（见 _value_blocks）
    blocks = await _value_blocks(state)
    context = build_context(state, blocks)
    if blocks:
        log.info("补料 %d 块（%s）", len(blocks),
                 " + ".join(b.splitlines()[0].lstrip("# ") for b in blocks))
    # provider：本地 aemeath（默认）或用户自配云端。is_cloud 决定人设注入方式（见 _chat_client）
    client, is_cloud, emotion_via_cloud = await _chat_client(state, strict=bool(blocks))
    # 混合时间问句（既查资料又问时间，见 intent.mentions_time）：把服务端真值补进 prompt
    now = await _now_if_needed(state)
    prompt = build_prompt(context, state["question"], blocks=blocks,
                           characters=state.get("characters"),
                           team_focus=_named_team(state),
                           user_context=state.get("user_context") or "",
                           now=now,
                           cloud=is_cloud)
    # 云端：人设走 SystemMessage（模型不认识爱弥斯，必须注入）；
    # 本地：只发 HumanMessage（人设在 Modelfile，发 system 会覆盖）。
    msgs = ([SystemMessage(content=persona.cloud_system()), HumanMessage(content=prompt)]
            if is_cloud else [HumanMessage(content=prompt)])

    guard = _new_loop_guard()
    full: list[str] = []
    hit: str | None = None
    truncated = False
    answer_ok = True

    try:
        async for chunk in client.astream(msgs):
            if not chunk.content:
                continue
            h = guard.feed(chunk.content)
            if h:
                log.warning("流式复读检测命中，中断生成（触发句=%.30s…）", h)
                hit = h
                truncated = True
                break
            full.append(chunk.content)
            # 不再逐 token yield 中间态：前端 token 取自 on_chat_model_stream
            # （llm.astream 自带回调），逐 token yield 只产生无人消费的
            # on_chain_stream 事件；str 字段本就覆盖非拼接，中间值无意义。
    except Exception as exc:  # noqa: BLE001 —— 流式生成必须宽捕获，见下
        # ⚠️ 绝不能收窄：astream 期间可能出任何错（Ollama 500 / 连接中断 / 解码失败 /
        # 模型返回畸形 chunk）。这里是**最后一道用户可见兜底**——漏掉一种异常类型，
        # 用户就会收到一个裸崩而不是「模型服务暂时不可用」。宁可宽捕获保住兜底话术。
        log.error("LLM 调用失败: %s", exc)
        answer_ok = False
        if not full:
            full.append("抱歉，模型服务暂时不可用，请稍后再试。")

    answer = "".join(full)
    if answer_ok and truncated and hit:
        answer = _trim_loop(answer, hit)
    # 输出侧兜底清洗：剥来源标记 `[n]`（提示词只能概率压住）+ 去重重复的列表项（图谱与
    # 参考文档给的是同一批队伍、只是角色名先后不同，8B 会读成「还有一批」再列一遍）
    # + 补回缺失的百分号（术语紧跟裸数值，见 text.fix_percent_units）。
    clean = dedup_list_items(fix_percent_units(strip_ref_marks(answer)))
    if clean != answer:
        log.info("输出清洗：去掉 %d 个字符（来源标记 / 重复列表项 / 缺失单位）",
                 len(answer) - len(clean))
        answer = clean

    # 概括性配队：答完追问一句，引导用户点名具体队伍（点名后才会补向量检索打法）。
    # 用**确定性拼接**而非提示词（见 _TEAM_FOLLOWUP 注释）；答案被复读闸截断时不加
    # （半截答案后面跟一句追问很突兀）。流式路径由 ask_stream 补吐同一段差量。
    if answer_ok and not truncated and _should_ask_team(state):
        answer = answer.rstrip() + "\n\n" + _TEAM_FOLLOWUP
        log.info("概括性配队：答案末尾追加追问句（图谱侧已给出队伍）")

    # 情绪标签（供 TTS 使用）：仅在 TTS 可用时才判定——判定需要额外一次模型调用，
    # TTS 默认关闭时该字段无人消费，没有理由付出这一次延迟。
    # 可用性按**当前用户**判（用户自持凭据优先，其次全局兜底，见 rag/tts.resolve）。
    # 判定模型：默认本地 8b；配了自定义云端模型且该用户开启兼任时才复用 chat_client。
    emotion = ""
    if setting.EMOTION_ENABLED and await tts.available(state.get("user_id")):
        emotion = await detect_emotion(
            answer, chat_client=client if emotion_via_cloud else None)

    # 最终状态：answer/truncated/history/context_summary 是 RagState 声明字段，
    # 由 checkpointer 持久化；_commit_history 顺带压缩被挤出窗口的旧轮次（B）。
    committed = await _commit_history(state, answer)
    yield {
        "context": context,
        "answer": answer,
        "emotion": emotion,
        "truncated": truncated,
        **committed,
    }


# 「不检索」类分支的人设提示：模型自带人设，这里只给最小引导；不带术语表与 RAG
# 答题约束，否则会诱发「资料里没有…」式拒答独白（正是复读的温床）。
_CHITCHAT_HINT = "家人在跟你闲聊，没有要查资料。自然、简短地回应，不要提资料、检索或知识库。"

# 时间分支的场景提示：把工具取到的**真实时间**原样摆到模型面前，只要求它照说。
# 提示词铁律（见本模块顶部说明）：只写正面要求，不写「不要编时间」这类否定句——
# aemeath 会把点名的反例当样本照抄。
# 「照抄括号里的口语说法」这句是实测加上的：只给 24 小时制时模型会把 16:5x 心算成
# 「三点五x」（8 次抽样错 3 次），明确指向预先生成的口语写法后不再出错；
# 末尾的「一两句说完」是长度约束——短答案既贴爱弥斯「话多」的人设又不至于写飞。
_TIME_HINT = (
    "家人问的是现在的时间。服务端时钟刚查到：{now}。\n"
    "用你自己的口吻把这个时间讲给家人听，一两句说完就好："
    "把上面的年月日、星期三和括号里的口语说法（上午/下午几点几分）讲出来即可。"
)

# 「顺带问时间」的场景提示（state.need_time，见 nlu.mentions_time）。
# 与 _TIME_HINT 的区别：那边整句都在问时间，这边另有一个主诉求（查资料或闲聊），
# 时间只是附带一句——所以措辞必须点明「别丢掉原来的话题」，否则模型容易只答时间。
_TIME_ASIDE_HINT = (
    "家人这句话里也问了现在的时间。服务端时钟刚查到：{now}。\n"
    "顺带用你自己的口吻把这个时间讲一句就好，别丢掉上面那个话题本身。"
)

# 用户画像注入（「不检索」分支专用）。
#
# ⚠️ 为什么必须有这一段：`user_context` 原先只在 `generate_node` → `prompt.build_prompt`
# 里注入，而闲聊走的是 `_chat_turn`（chitchat_node / time_node 共用），**完全拿不到画像**。
# 后果正是用户报的现象：说「我是颗粒」后昵称确实进了 `user_facts`（实测已入库），
# 但下一句「你好呀」走闲聊时模型对此一无所知，于是「记住了却用不上」。
# 自我介绍被 `nlu.is_self_intro` 判进 chitchat 之后，这条注入就成了「记住 → 用得上」的
# 关键一环，缺它则前一半修复没有意义。
#
# ⚠️ 措辞与位置是实测选型定的，改动时别违背这三条：
#   ① **档案要放在用户原话之后**（近因位）。放之前命中率极低（0~4/8），放最后 7/8 ——
#      与本模块 `_build_prompt` 把「小档案」「输出格式」压在 prompt 末尾是同一条经验。
#   ② **措辞越强硬越差**。明确要求「用在开头、直接喊出」反而掉到 1/3，还诱发人称混淆
#      （模型以为用户叫它「颗粒」）；强调「是这位家人自己希望的叫法」直接 0/8。
#      当前措辞既点明了称呼归属（是用户的、不是助手的，故零混淆），又没下硬指令。
#   ③ 仍遵守本模块铁律：只写正面要求，不写反例。
_PROFILE_HINT = (
    "## 这位用户的小档案\n"
    "{profile}\n"
    "上面这位家人希望你用档案里的称呼喊TA。"
)


async def _now_if_needed(state: RagState) -> str:
    """本轮**顺带**问了时间就取服务端真值，否则返回空串。

    L1 直调（与 time_node 同一把工具、同一口径）：确定性调度，零幻觉。
    纯时间问题（intent=time）不走这里——它由 time_node 独占处理，本函数只服务
    「既问资料又问时间」的混合问句（intent 仍是 fact/semantic/hybrid/chitchat）。
    工具异常返回空串：少说一句时间而已，绝不能让检索那条主链路跟着挂掉。
    """
    if not state.get("need_time"):
        return ""
    try:
        return await current_time_tool.ainvoke({})
    except Exception as exc:  # noqa: BLE001 —— 少说一句时间而已，绝不能让检索主链路跟着挂
        log.error("current_time 工具调用失败（混合时间问句）: %s", exc)
        return ""


async def _chat_turn(state: RagState, hint: str):
    """「不检索」类分支的公共实现：闲聊 chitchat_node 与时间 time_node 共用。

    两处的人设口径、云端/本地 system 注入规则、流式累积、复读兜底、情绪判定、
    记忆写回**完全一致**，唯一差别就是那句场景提示 hint —— 所以不复制两份代码：
    双份实现的必然结局是修了一处忘另一处（本项目对「同一语义挂两条判据」已有
    多次踩坑记录）。streaming node：内部累积全文，只在收尾 yield 一次最终状态。
    """
    client, is_cloud, emotion_via_cloud = await _chat_client(state)
    msgs: list = []
    # 云端：人设走 SystemMessage（见 _chat_client 的两条相反规则）。
    # 本地：不发 system —— aemeath 人设在 Modelfile，发了会覆盖。
    if is_cloud:
        msgs.append(SystemMessage(content=persona.cloud_system()))
    for m in (state.get("history", []))[-setting.MAX_HISTORY_TURNS * 2:]:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        else:
            msgs.append(AIMessage(content=m["content"]))
    # 画像注入（见 _PROFILE_HINT）：顺序是「场景 hint → 用户原话 → 画像」。
    # ⚠️ 画像必须排在**用户原话之后**（近因位）—— 这是实测选型的结果，不是风格偏好：
    #    同一画像同一问句各抽 8 次，档案放原话之前命中率 4/8，放之后 **7/8**
    #    （完整对比表见 _PROFILE_HINT 注释）。与本模块 `_build_prompt` 把「输出格式」
    #    硬要求压在 prompt 末尾是同一条经验。
    user_ctx = (state.get("user_context") or "").strip()
    profile_block = _PROFILE_HINT.format(profile=user_ctx) if user_ctx else ""
    parts = [p for p in (hint, f"家人说：{state['question']}", profile_block) if p]
    msgs.append(HumanMessage(content="\n\n".join(parts)))

    guard = _new_loop_guard()
    full: list[str] = []
    hit: str | None = None
    truncated = False
    answer_ok = True
    try:
        async for chunk in client.astream(msgs):
            if not chunk.content:
                continue
            h = guard.feed(chunk.content)
            if h:
                log.warning("闲聊复读检测命中，中断生成（触发句=%.30s…）", h)
                hit = h
                truncated = True
                break
            full.append(chunk.content)
            # 同 generate_node：不再逐 token yield 中间态（前端 token 走
            # on_chat_model_stream，此处中间 yield 无人消费）
    except Exception as exc:  # noqa: BLE001 —— 同 generate_node：流式生成必须宽捕获保住兜底话术
        log.error("闲聊生成失败: %s", exc)
        answer_ok = False
        if not full:
            full.append("诶？我刚才走神了……你再说一遍嘛。")

    answer = "".join(full)
    if answer_ok and truncated and hit:
        answer = _trim_loop(answer, hit)

    # 情绪标签：与 generate_node 使用同一开关（仅在 TTS 可用时判定，按当前用户判）。
    # 闲聊是 playful/cheerful 的高发场景，情绪对语音表现力价值最大。
    emotion = ""
    if setting.EMOTION_ENABLED and await tts.available(state.get("user_id")):
        emotion = await detect_emotion(
            answer, chat_client=client if emotion_via_cloud else None)

    committed = await _commit_history(state, answer)
    yield {
        "answer": answer,
        "emotion": emotion,
        "truncated": truncated,
        **committed,
    }


async def chitchat_node(state: RagState):
    """闲聊分支：不挂检索，带对话历史，人设自然回应。streaming node。

    如果这句闲聊里**顺带**问了时间（「今天几号呀，天气真好」这类），把服务端真值
    一并交给它 —— 不为此另开分支：答案的形态仍然是闲聊，只是多了一条可信依据。
    （纯时间问题走 time_node；两者的差别只是提示词，见 _chat_turn。）
    """
    hint = _CHITCHAT_HINT
    now = await _now_if_needed(state)
    if now:
        hint = f"{_CHITCHAT_HINT}\n\n{_TIME_ASIDE_HINT.format(now=now)}"
    async for out in _chat_turn(state, hint):
        yield out


async def time_node(state: RagState):
    """时间分支：**确定性**调 current_time 工具取服务端真值，再交模型用爱弥斯口吻说出。

    为什么单开一条分支：模型没有时钟，训练数据里的「今天」永远停在训练期附近；
    而「现在几点」这类问句经 classify_topic 会被判成闲聊（它确实与游戏无关），落到
    chitchat_node 后模型对真实时间一无所知，只能含糊其辞（「应该是下午吧」）或自信地
    报一个错日期。主题分类器那层 LLM 判断在这里没有价值——时间的事实来源只有服务端时钟。

    走 L1 直调（而非 agent.py 的 bind_tools 自主选工具）与本项目其余工具口径一致：
    确定性调度，零幻觉、零额外延迟。工具异常不静默——hint 里明确告知没读到，
    让模型按人设说不知道，而不是顺手编一个时间出来。
    """
    now = ""
    try:
        now = await current_time_tool.ainvoke({})
    except Exception as exc:  # noqa: BLE001 —— 读不到时钟就让模型如实说没拿到，不中断问答
        log.error("current_time 工具调用失败: %s", exc)
    hint = _TIME_HINT.format(now=now) if now else (
        "家人问的是现在的时间，但服务端时钟这次没读到。"
        "用你自己的口吻老实说没拿到当前时间，让家人稍后再问一次。"
    )
    async for out in _chat_turn(state, hint):
        yield out


def build_graph() -> StateGraph:
    g = StateGraph(RagState)
    g.add_node("intent", intent_node)
    g.add_node("chitchat", chitchat_node)
    g.add_node("time", time_node)
    g.add_node("graph", graph_node)
    g.add_node("vector", vector_node)
    g.add_node("verify", verify_node)
    g.add_node("web", web_node)
    g.add_node("generate", generate_node)

    g.add_edge(START, "intent")
    g.add_conditional_edges("intent", _route, {
        "fact": "graph", "semantic": "vector", "hybrid": "graph",
        "chitchat": "chitchat", "time": "time",
    })
    # 检索完先进 verify 把关（材料空/跑题在此发现），不再直达 generate
    g.add_conditional_edges("graph", _after_graph, {
        "need_vector": "vector", "done": "verify",
    })
    g.add_edge("vector", "verify")
    # 映射键必须等于 _after_verify 的返回值（图节点名），不是 verify_stage 的值
    g.add_conditional_edges("verify", _after_verify, {
        "generate": "generate", "graph": "graph", "vector": "vector", "web": "web",
    })
    g.add_edge("web", "generate")
    g.add_edge("generate", END)
    g.add_edge("chitchat", END)
    g.add_edge("time", END)
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
    用 _run_io 跑阻塞的 result.get，避免卡住 FastAPI 事件循环。
    """
    from celery.exceptions import TimeoutError as CeleryTimeout
    results = []
    for n in names:
        r = build_pipeline(n).apply_async()
        results.append(r)
        log.info("自动爬取: 已入队 chain_id=%s 角色=%s", r.id, n)
    t0 = time.perf_counter()
    try:
        for r in results:
            await _run_io(r.get, timeout=timeout)
    except CharacterNotFound:
        return False
    except CeleryTimeout:
        log.warning("自动爬取: 等待超时(%ss) 角色=%s，视为抓取失败", timeout, names)
        return False
    # 新角色已进图谱：失效名册缓存，下一轮 _known_characters 立刻看得到
    _invalidate_known_cache()
    log.info("自动爬取: %s 建库完成，耗时 %.1fs", names, time.perf_counter() - t0)
    return True


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


def _fresh_state(question: str, characters: list[str], user_context: str = "",
                 user_id: int | None = None, history: list[dict] | None = None) -> dict:
    """每轮问答的初始状态：检索侧字段全部清零。

    docs/graph_facts/web_facts 等是普通字段（无 reducer），checkpointer 会把上一轮
    的值原样带进本轮。实测后果：fact/chitchat 分支压根不跑 vector_node，却把上轮
    的 9 条旧文档当成本轮「## 参考文档」拼进 prompt —— 三个不同问题（我的技能有
    哪些 / 在游戏里有没有技能 / 爱弥斯共鸣解放）产出同样 78 字的同一套话；同时
    prompt.build_context 的零资料判定、verify_knowledge 的审查、前端「N 条引用」全部被
    污染。所以在入口显式清零，用 input_state 覆盖 checkpointer 的旧值。

    `history` 只在「重新生成」时传：那一步已经把转录表里要重答的那一轮删掉了，
    必须把模型侧的记忆（checkpointer 里的 history 窗口）也**替换**成删完之后的版本，
    否则模型仍记得自己刚被删掉的那个回答，重新生成大概率吐出同一段话。
    传 None（默认）表示不动 —— checkpointer 的 history 照常累积，这是正常问答路径。
    """
    state: dict = {
        "question": question,
        "characters": characters,
        "docs": [],
        "graph_facts": "",
        "web_facts": "",
        "verify_stage": "",
        "refined_query": "",
        "retry_count": 0,
        "refreshed": False,
        "used_web": False,
        # 同 verify_stage 一类的「每轮必须清零」字段：checkpointer 会把上一轮的值原样带进
        # 本轮，上一轮顺带问了时间、这一轮没问，若不清零会白给一句时间。
        "need_time": False,
        "user_context": user_context,   # 用户画像事实串；checkpointer 会跨轮带旧值，每轮显式覆盖
        # 约束：user_id 必须每轮显式覆盖（与 user_context 同理，但后果更严重）：
        # checkpointer 会跨轮带旧值，若不清零，同一 thread_id 上换人提问时，
        # _chat_client 会读到**上一个用户**的云端配置 —— 等于用别人的 API-KEY 跑自己的问题。
        # 无登录/未配置时为 None，_chat_client 直接回落本地默认。
        "user_id": user_id,
    }
    if history is not None:
        state["history"] = history
    return state


def _unknown_text(names: list[str]) -> str:
    """角色不在知识库、联网抓取也没找到时的兜底话术。

    早期版本是「不知道（知识库里没有这个角色，尝试联网抓取也没找到）。」，
     ——事实没错但整体缺乏角色口吻。改成爱弥斯的表达，同时保留三件事实：翻了本地、联网找过、确实没有。
    """
    who = "、".join(names) if names else "这个角色"
    return (
        f"诶…「{who}」我在小本本里翻遍了都没找着，刚也联网去问了一圈，还是没着落。"
        "这个我确实不清楚呀，家人帮我确认一下名字嘛~"
    )


async def ask(question: str, thread_id: str = "default", user_context: str = "",
              user_id: int | None = None, history: list[dict] | None = None) -> dict:
    candidates, ok, crawled = await ensure_characters(question)
    if candidates and not ok:
        log.info("自动爬取: 最终回「不知道」(角色=%s)", candidates)
        return {
            "answer": _unknown_text(candidates),
            "characters": candidates, "intent": "", "slots": [], "docs": 0,
        }
    chain = await get_chain()
    injected = crawled if crawled else []
    return await chain.ainvoke(
        _fresh_state(question, injected, user_context, user_id, history),
        config={"configurable": {"thread_id": thread_id}},
    )


async def ask_stream(question: str, thread_id: str = "default", user_context: str = "",
                     user_id: int | None = None, history: list[dict] | None = None):
    """流式问答：走 LangGraph astream_events，checkpointer 自动管理多轮记忆。

    yield dict：
      {'token': str}                        答案增量
      {'status': 'retrieving'}              兼容旧前端的粗粒度状态
      {'stage': str, 'label': str}          细粒度阶段（前端显示「正在…」用）
      {'done': True, ...}                   结束，带元数据

    与 ask() 共享同一张图和 checkpointer，thread_id 真正生效。
    """
    # 阶段文案：检索+重排实测约 19s、生成仅 1~2s，所以必须让用户看到在干什么
    t0 = time.perf_counter()
    yield {"stage": "crawl", "label": "检查角色是否在知识库中"}

    candidates, ok, crawled = await ensure_characters(question)
    if candidates and not ok:
        text = _unknown_text(candidates)
        yield {"token": text}
        # done 事件补齐字段：原来只 `{"done": True}`，前端 meta 全是 undefined、
        # 引用面板也拿不到 sources（与正常路径的 done 结构对齐）。
        yield {
            "done": True, "answer": text, "intent": "", "slots": [],
            "characters": candidates, "docs": 0, "sources": [], "truncated": False,
            # thread_id 必须回传：会话现在是服务端建的（前端不传 thread_id 时由后端生成），
            # 前端要靠这个事件认领新建的会话 id，否则「新建对话」后第一句问答
            # 前端永远不知道自己的会话叫什么。
            "thread_id": thread_id,
        }
        return

    yield {"status": "retrieving"}  # 前端可显示「检索中…」

    chain = await get_chain()
    injected = crawled if crawled else []
    input_state = _fresh_state(question, injected, user_context, user_id, history)
    config = {"configurable": {"thread_id": thread_id}}

    # 从事件流抽 token + 阶段 + 最终元数据
    full: list[str] = []               # 模型原稿（旁路抽取，用于算 done.answer）
    sent: list[str] = []               # 实际下发给前端的正文（收尾只补差量时当前缀基准）
    final_state: dict = {}
    emitted_stages: set[str] = set()   # streaming node 会触发两次 on_chain_start，去重
    answer_filter = AnswerFilter()     # 流式剥 `[n]` + 去重重复列表项（见 text.AnswerFilter）
    async for event in chain.astream_events(input_state, version="v2", config=config):
        kind = event.get("event", "")
        name = event.get("name", "")

        # 节点开始事件：转成用户可读的阶段提示（同一节点只发一次）
        if kind == "on_chain_start" and name in _STAGE_LABELS and name not in emitted_stages:
            emitted_stages.add(name)
            yield {"stage": name, "label": _STAGE_LABELS[name]}

        # token 级事件：两道过滤，缺一不可。
        # ① 标签过滤（_INTERNAL_TAGS）：摘要/审查/情绪这些**结构化中间结果**都产生于
        #    generate/chitchat 节点**内部**，其 ainvoke 流式回调 node 就是 'generate'，
        #    下面的节点白名单挡不住 —— 只能靠标签丢弃。实测两轮同类泄漏：
        #    未挡 wwa:summary 时摘要整句被拼进答案尾巴；接情绪标签时预判到 wwa:emotion 同样。
        # ② 节点过滤：intent_node 里的主题分类器也调 LLM，事件同样挂 on_chat_model_stream
        #    （node='intent'），不过滤会把 {"topic":"chitchat"} 当答案吐给前端（实测发生过）。
        #    白名单 = 三个会产出「给用户看的正文」的节点（generate/chitchat/time），
        #    新增这类节点必须同步加进来，否则答案会被静默丢弃、前端一个字都收不到。
        elif kind == "on_chat_model_stream":
            if _INTERNAL_TAGS & set(event.get("tags") or []):
                continue
            node = event.get("metadata", {}).get("langgraph_node", "")
            if node not in ("generate", "chitchat", "time"):
                continue
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                full.append(chunk.content)
                # 输出侧清洗：剥 `[n]` 来源标记 + 去重重复列表项（见 text.AnswerFilter）
                safe = answer_filter.feed(chunk.content)
                if safe:
                    sent.append(safe)
                    yield {"token": safe}

        # 图执行结束事件：拿最终完整状态（含 intent/slots/characters/docs/truncated/answer）
        elif kind == "on_chain_end" and name == "LangGraph":
            output = event.get("data", {}).get("output", {})
            if isinstance(output, dict):
                final_state = output

    # 收尾：先吐出过滤器扣住的尾巴（被切成多 token 的 `[n]` 开头、最后一行列表项），
    # 再做单位修正——顺序反了会让修正文本排在被扣住的尾巴前面（见 AnswerFilter.request_more）。
    tail_out = answer_filter.request_more()
    if tail_out:
        sent.append(tail_out)
        yield {"token": tail_out}
    if answer_filter.removed_marks or answer_filter.removed_lines:
        log.info("流式清洗：来源标记 %d 处 / 重复列表项 %d 行",
                 answer_filter.removed_marks, answer_filter.removed_lines)

    answer = dedup_list_items(fix_percent_units(strip_ref_marks("".join(full))))
    # 如果 generate_node 内部已截断，final_state["answer"] 是截断后的权威全文
    if final_state.get("truncated") and final_state.get("answer"):
        answer = final_state["answer"]

    # 单位修正的**流式补吐**：token 是旁路抽取的，屏幕上还是模型原稿；把修正后的
    # 完整文本在滤器里过一遍（去重状态与刚才同源），只把**新增的后缀**吐出去。
    # 截断/已收口时跳过（与下面追问句同一口径：半截答案后面补东西反而突兀）。
    #
    # ⚠️ 判据必须用「已下发的正文」`sent` 做前缀比较，**不能**用 AnswerFilter.would_append：
    # 它只看列表行的去重集合 `_seen`，而 `_route` 对非列表行是无条件直通的——答案里
    # 没有列表行时 `_seen` 恒为空、would_append 恒为 True，`feed(answer)` 会把**整篇正文
    # 原样重吐一遍**。实测（单元级复现 + 端到端）：闲聊/时间这类无列表行的短答案整段重复
    # 一遍（48 字答案流出 96 字）；含列表行的事实答案则是开头的非列表行被重吐一遍
    # （`你好呀…` 之后又跟一句开头）。前端最终会用 `done.answer` 覆盖，所以屏幕上的终稿
    # 没错，但流式过程中的重复/错位是实打实的，且与「只补后缀」的原意相反。
    already = "".join(sent)
    if not final_state.get("truncated") and answer.startswith(already) and len(answer) > len(already):
        for chunk in answer_filter.feed(answer[len(already):]):
            if chunk:
                sent.append(chunk)
                yield {"token": chunk}

    # 概括性配队：generate_node 已把追问句拼进 final_state["answer"]，但流式的 token
    # 是**旁路抽取**的（on_chat_model_stream），只有模型生成的正文，必须把这段差量补吐
    # 出去——否则前端少显示一句、与 done.answer 不一致。endswith 防重复。
    if not final_state.get("truncated") and _should_ask_team(final_state):
        suffix = "\n\n" + _TEAM_FOLLOWUP
        if not answer.endswith(_TEAM_FOLLOWUP):
            yield {"token": suffix}
            answer = answer.rstrip() + suffix

    # 全链路耗时：排查「慢在检索还是生成」刚需（阶段事件只给顺序不给时长）
    log.info(
        "流式完成 thread=%s 耗时=%.1fs intent=%s 角色=%s docs=%d 答案=%d字%s",
        thread_id, time.perf_counter() - t0, final_state.get("intent", "-"),
        final_state.get("characters") or "-", len(final_state.get("docs") or []),
        len(answer), " [截断]" if final_state.get("truncated") else "",
    )

    yield {
        "done": True,
        "answer": answer,
        "intent": final_state.get("intent", ""),
        "slots": final_state.get("slots") or [],
        "characters": final_state.get("characters") or [],
        "docs": len(final_state.get("docs") or []),
        "sources": doc_sources(final_state.get("docs") or []),
        "truncated": bool(final_state.get("truncated")),
        "emotion": final_state.get("emotion", ""),   # TTS 情绪（未开启时为空串）
        "thread_id": thread_id,   # 见上：供前端认领服务端新建的会话
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
