"""Step 7：LangGraph 编排。"""
from __future__ import annotations

import asyncio
import re
import time

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from ..config import ensure_dirs, get_settings
from ..graph.neo4j_client import get_session
from ..text import chunk_text
from ..worker import (
    CharacterNotFound,
    build_pipeline,
    build_refresh_pipeline,
    reset_progress,
)
from ..ww_logger import get_logger
from .characters import resolve_candidates
from .intent import (
    classify,
    classify_topic,
    detect_element,
    detect_slots,
    detect_stage,
    extract_characters,
    is_identity,
    rewrite_query,
    summarize_turns,
)
from .llm import get_chat_llm
from .loopguard import LoopGuard, trim_loop
from .memory import get_checkpointer
from .retrievers import fetch_chunks
from .state import RagState
from .tools import graph_search_tool, vector_search_tool
from .verify import verify_knowledge
from .websearch import web_search

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
- 资料里出现「满级数值表」或「突破材料表」时，必须把该表**逐行完整列出**：数值、材料名与
  「+」「*」「×」「%」原样照抄，不得换算、不得合并同类项、不得改写成中文数字、不得略过。
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
    "verify": "核对资料是否对题",
    "web": "联网搜索补充知识",
    "generate": "整理答案中",
}


def _new_loop_guard() -> LoopGuard:
    """每次生成都要新建一个 guard：它内部有累积状态，跨请求复用会串味。"""
    return LoopGuard(
        max_repeat=setting.LLM_LOOP_MAX_REPEAT,
        min_chars=setting.LLM_LOOP_MIN_CHARS,
    )


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


async def _known_characters() -> list[str]:
    global _known_cache
    now = time.monotonic()
    if _known_cache[1] and now - _known_cache[0] < _KNOWN_TTL:
        return _known_cache[1]
    async with get_session() as s:
        rows = await (await s.run("MATCH (c:Character) RETURN c.name AS n")).data()
    names = [r["n"] for r in rows]
    if names:                      # 空结果（Neo4j 抖动）不缓存，避免脏 60s
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
    """
    if not summary or not known or not _FAR_REF_RE.search(sq):
        # known 为空时空正则会在每个位置匹配出垃圾，必须挡
        return chars
    # 名字长度降序：「秧秧」是「秧秧·玄翎」的前缀，短的在前会误抢匹配
    pattern = "|".join(re.escape(n) for n in sorted(known, key=len, reverse=True) if n)
    hits = [m.group(0) for m in re.finditer(pattern, summary)]
    if not hits:
        return chars
    name = hits[0]
    if name in chars:
        return chars
    log.info("前角色注入: %s（远指代=%r 摘要=%r）", name, sq[:24], summary[:40])
    return [*chars, name]


async def intent_node(state: RagState) -> dict:
    q = state["question"]
    known = await _known_characters()
    # 追问改写（A+B 输入）：滚动摘要（窗口外压缩记忆）+ 焦点锚点（全量文本提角色，
    # 不怕 120 字截断丢名）+ 最近 2 轮短原文。检索信号全部吃改写句——
    # 「那她配什么声骸」单拿原句必落空，补出角色名才能命中。
    # 生成侧仍用原句+history（_build_context 里有对话历史），展示不受影响。
    history = (state.get("history", []))[-setting.MAX_HISTORY_TURNS * 2:]
    summary = state.get("context_summary", "")
    sq = await rewrite_query(q, history, known=known, summary=summary)

    slots = detect_slots(sq)
    chars = extract_characters(sq, known) or state.get("characters", [])
    # 远指代兜底：改写器偶发解析失败/半解析时，摘要里的「前角色」兜住
    chars = _inject_far_characters(sq, chars, summary, set(known))
    element = detect_element(sq) or state.get("element", "")
    stage = detect_stage(sq)
    intent = classify(sq, slots)

    # 闲聊分流，两层：
    # ① 人格/身份类硬信号（问「你」的台词/名字/身份）——即使改写出了角色名，本质也
    #    是问人格不是查资料，直接判 chitchat。用**原句 q** 判：改写 sq 已把「你」补成
    #    角色名（「你的台词」→「爱弥斯的台词」），第二人称信号会丢。aemeath 人设由模型
    #    自带，走 RAG 反而召回大段角色剧情文案整段倾倒（实测「你的台词是什么」→ 809 字）。
    # ② qwen3:8b 主题 agent：仅在「无角色名 且 无槽位 且 无属性/阶段」时才调用。
    #    槽位非空几乎必然是游戏提问（实测「秧秧怎么玩」这类靠语义命中；真闲聊句槽位为空）。
    #    不能用 SEMANTIC_PATTERNS 当判据——「怎么」会误命中闲聊句（「怎么这么晚才来」）。
    #    有角色名绝不当闲聊（「你好呀卡卡罗」是提问）；LLM 解析失败回落 game。
    #    判据用改写句 sq：追问「那她配什么声骸」原句无角色，改写后有——不会误入闲聊。
    if is_identity(q) and not slots:
        intent = "chitchat"
    elif not chars and not slots and not stage and not element:
        topic = await classify_topic(sq)
        if topic == "chitchat":
            intent = "chitchat"
    log.info("意图=%s 槽位=%s 角色=%s 属性=%s 阶段=%s%s", intent, slots, chars or "未识别",
             element or "-", stage or "-", f" | 改写={sq!r}" if sq != q else "")
    return {"search_query": sq, "intent": intent, "slots": slots,
            "characters": chars, "element": element, "stage": stage}


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
    """graph 之后：hybrid 还要补向量，其余直接进 verify。

    例外：技能类问题（slots 含「技能」）强制补一轮向量。原因：图谱的 HAS_SKILL
    每个 kind 只存了**技能名**（见 graph/extract.py::_extract_skills 取首个加粗串），
    没有效果描述与数值——问「爱弥斯共鸣解放」只喂得到「共鸣解放=飞至启明之时」
    这 7 行名字，模型无从作答（实测答「至于具体效果嘛……我记不清了啦」）。
    技能描述在原文里（爱弥斯 61 个技能 chunk 含「造成热熔伤害」「消耗全部【同步率】」
    等），必须走向量才能拿到。代价：技能类多一次检索 + 重排（约 15~19s）。
    """
    if state["intent"] == "hybrid" or "技能" in (state.get("slots") or []):
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
            await asyncio.to_thread(r.get, timeout=timeout)
    except CharacterNotFound:
        log.warning("资料刷新: 角色 %s wiki 上不存在，刷新终止", chars)
        return False
    except CeleryTimeout:
        log.warning("资料刷新: 等待超时(%ss)，按失败处理", timeout)
        return False
    except Exception as exc:
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
    （空/不匹配材料由 _build_context 的「一句话不知道」约束兜底）。
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
    chunks = await asyncio.to_thread(fetch_chunks, char, "技能介绍")
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
    chunks = await asyncio.to_thread(fetch_chunks, char, "角色突破材料")
    by_stage: dict[str, list[str]] = {}
    for d in chunks:
        stage = (d.get("breadcrumb") or "").split("›")[-1].strip()
        items = _material_items(chunk_text(d))
        if items:
            by_stage[stage] = items

    skill_chunks = await asyncio.to_thread(fetch_chunks, char, "技能突破材料")
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


async def _value_blocks(state: RagState) -> list[str]:
    """按问题类型补「满级数值表 / 突破材料表」。取不到返回空，绝不影响主链路。"""
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
    except Exception as exc:               # 补料是增益不是依赖，失败只记日志
        log.warning("补数值/材料表失败，跳过: %s", exc)
    return blocks


def _build_context(state: RagState, extra: list[str] | None = None) -> str:
    parts: list[str] = []
    # 注意：不要把 context_summary 塞进生成上下文——实测 aemeath 会把摘要句
    # 原样复述进答案（「…讨论声骸选择及毕业配装…」这种第三人称腔调穿帮）。
    # 摘要只喂给 rewrite_query 消解指代；生成侧靠窗口内 history + 改写后的检索结果。
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
    web_facts = state.get("web_facts") or ""
    if web_facts:
        parts.append("## 联网搜索资料\n（实时搜索结果，本地资料不足时以本段为准）\n" + web_facts)
    # 确定性补料排在最后（紧贴 ## 问题）。真凶其实不在位置：实测把这张满级数值表
    # 放在 ## 资料 第一节时，aemeath 会「只抄前几行就收尾」并补一句「其他参数未在该
    # 列表中」，换表格/纯文本/编号/中文数值都无效——根因是 repeat_penalty=1.3 把彼此
    # 高度相似的表行罚到写不下去（降到 1.15 即 7 行全出，见 config.LLM_REPEAT_PENALTY）。
    # 位置只是第二道保险：同样的坏参数下，块放末尾确实能抄全，放开头会截断。
    for b in extra or ():
        parts.append(b)
    # 「有没有资料」必须看图谱事实/文档/联网结果/确定性补料，不能用 join 结果是否为空
    # 来判断：多轮对话时 history 非空，join 永远有内容，原先的 or 兜底就永远不触发，
    # 零资料信号被吞掉 → 模型收不到约束 → 退化成自由发挥（人设独白 + 复读）。
    if not (state.get("graph_facts") or docs or web_facts or extra):
        # 这里不要再写「## 资料」标题：_build_prompt 已经加了，重复标题会干扰模型
        parts.append(
            "（本次没有检索到任何资料。请只用一句话说明你不清楚，然后立即停止；"
            "不要解释原因，不要重复这句话，不要补充任何其他内容。）"
        )
    return "\n\n".join(parts)


def _build_prompt(context: str, question: str, blocks: list[str] | None = None) -> str:
    # 不在这里注入 /no_think：实测它对 aemeath 无效（仍 38s + 'v' 泄漏前缀 + 触发
    # Ollama 500）。思考模式由 llm.py 的 .bind(think=False) 统一关闭。
    prompt = f"{_SYSTEM}\n\n## 资料\n{context}\n\n## 问题\n{question}"
    blocks = blocks or []
    if not blocks:
        return prompt
    # 指令必须压在 prompt **末尾**：_SYSTEM 里那条规则实测只能让模型「带上几个数」，
    # 面对长表仍会概括成「各需不同数量」而不逐行列（实测）。近因位置 + 点名禁止的
    # 偷懒写法，才能把它按回照抄状态。
    #
    # ⚠ 2026-09-22 踩坑：不要写「只照抄那两张表」这种话。它有两个反作用——
    #  ①「只」字会让模型把技能介绍/人设口吻整个砍掉，退化成纯数据倾倒（用户原话
    #    「怎么变成这种垃圾回复了…不能丢人设」）；
    #  ② 点名「突破材料表」会让模型以为该有材料表，于是跑去「## 参考文档」里翻材料
    #    表一起列出来——问技能却蹦出材料就是这么来的（实测）。
    # 正确写法：先保住「说人话的介绍」，再只点名**本轮真正补了的那几张表**，
    # 并显式禁止主动扩列没被问到的内容。
    has_value = any("满级数值表" in b for b in blocks)
    has_mat = any("突破材料表" in b for b in blocks)
    lines = ["\n\n## 输出格式（硬要求）"]
    lines.append("1) 先用你自己的口吻把内容讲清楚（这是什么、怎么打、什么手感），正常说话，"
                 "不要只丢数字，也不要写成机械报表；")
    step = 2
    if has_value:
        lines.append(f"{step}) 然后把「满级数值表」里每一行都列出来，写成「- 名称：数值」，"
                     "一行都不许省；")
        step += 1
    if has_mat:
        lines.append(f"{step}) 再把「突破材料表」里每一条都列出来，写成「- 材料名×数量」，"
                     "一条都不许省；")
    lines.append("禁止写成「各需不同数量」「材料如上」「数值都在资料里」这类概述，"
                 "禁止换算、合并、四舍五入或改成中文数字。")
    if has_mat:
        lines.append("「## 参考文档」里分等级展开的长表不要照抄，更不要把不同等级、不同技能的"
                     "材料拼成一张混在一起的清单——数值与材料一律以上面两张表为准，"
                     "且不要主动列出没被问到的内容。")
    else:
        lines.append("「## 参考文档」里分等级展开的长表（尤其是各种材料表）一律不要照抄，"
                     "更不要主动列出没被问到的内容——数值只以「满级数值表」为准。")
    return prompt + "\n".join(lines)


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
    context = _build_context(state, blocks)
    if blocks:
        log.info("补料 %d 块（%s）", len(blocks),
                 " + ".join(b.splitlines()[0].lstrip("# ") for b in blocks))
    prompt = _build_prompt(context, state["question"], blocks=blocks)

    guard = _new_loop_guard()
    full: list[str] = []
    hit: str | None = None
    truncated = False
    answer_ok = True

    try:
        async for chunk in get_chat_llm(strict=bool(blocks)).astream([HumanMessage(content=prompt)]):
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
    except Exception as exc:
        log.error("LLM 调用失败: %s", exc)
        answer_ok = False
        if not full:
            full.append("抱歉，模型服务暂时不可用，请稍后再试。")

    answer = "".join(full)
    if answer_ok and truncated and hit:
        answer = trim_loop(answer, hit)

    # 最终状态：answer/truncated/history/context_summary 是 RagState 声明字段，
    # 由 checkpointer 持久化；_commit_history 顺带压缩被挤出窗口的旧轮次（B）。
    committed = await _commit_history(state, answer)
    yield {
        "context": context,
        "answer": answer,
        "truncated": truncated,
        **committed,
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
            # 同 generate_node：不再逐 token yield 中间态（前端 token 走
            # on_chat_model_stream，此处中间 yield 无人消费）
    except Exception as exc:
        log.error("闲聊生成失败: %s", exc)
        answer_ok = False
        if not full:
            full.append("诶？我刚才走神了……你再说一遍嘛。")

    answer = "".join(full)
    if answer_ok and truncated and hit:
        answer = trim_loop(answer, hit)

    committed = await _commit_history(state, answer)
    yield {
        "answer": answer,
        "truncated": truncated,
        **committed,
    }


def build_graph() -> StateGraph:
    g = StateGraph(RagState)
    g.add_node("intent", intent_node)
    g.add_node("chitchat", chitchat_node)
    g.add_node("graph", graph_node)
    g.add_node("vector", vector_node)
    g.add_node("verify", verify_node)
    g.add_node("web", web_node)
    g.add_node("generate", generate_node)

    g.add_edge(START, "intent")
    g.add_conditional_edges("intent", _route, {
        "fact": "graph", "semantic": "vector", "hybrid": "graph", "chitchat": "chitchat",
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
    t0 = time.perf_counter()
    try:
        for r in results:
            await asyncio.to_thread(r.get, timeout=timeout)
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


def _fresh_state(question: str, characters: list[str]) -> dict:
    """每轮问答的初始状态：检索侧字段全部清零。

    docs/graph_facts/web_facts 等是普通字段（无 reducer），checkpointer 会把上一轮
    的值原样带进本轮。实测后果：fact/chitchat 分支压根不跑 vector_node，却把上轮
    的 9 条旧文档当成本轮「## 参考文档」拼进 prompt —— 三个不同问题（我的技能有
    哪些 / 在游戏里有没有技能 / 爱弥斯共鸣解放）产出同样 78 字的同一套话；同时
    _build_context 的零资料判定、verify_knowledge 的审查、前端「N 条引用」全部被
    污染。所以在入口显式清零，用 input_state 覆盖 checkpointer 的旧值。
    """
    return {
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
    }


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
        _fresh_state(question, injected),
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
    t0 = time.perf_counter()
    yield {"stage": "crawl", "label": "检查角色是否在知识库中"}

    candidates, ok, crawled = await ensure_characters(question)
    if candidates and not ok:
        yield {"token": "不知道（知识库里没有这个角色，尝试联网抓取也没找到）。"}
        yield {"done": True}
        return

    yield {"status": "retrieving"}  # 前端可显示「检索中…」

    chain = await get_chain()
    injected = crawled if crawled else []
    input_state = _fresh_state(question, injected)
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
        # 另挡 wwa:summary 标签一路：_commit_history 在 generate/chitchat 节点**内部**
        # 调 summarize_turns，其 ainvoke 的流式回调 node='generate'，节点过滤挡不住，
        # 实测摘要整句被拼进答案尾巴（六轮深会话测试抓到，intent 侧已打标签）。
        elif kind == "on_chat_model_stream":
            if "wwa:summary" in (event.get("tags") or []):
                continue
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
