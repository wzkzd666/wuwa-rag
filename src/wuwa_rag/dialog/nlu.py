"""意图识别 + 实体抽取。

槽位/意图分类规则优先（实测 16/16 覆盖、零漏检，无命中时 classify() 回落 hybrid
图谱+向量双跑，本身即安全兜底）。主题分类（闲聊 vs 游戏）走 qwen3:8b agent——
这类判断规则词典覆盖不全（「家人怎么这么晚才来」无任何游戏信号也无闲聊词典），
但只在规则全无信号时才调用，有信号时零额外延迟。
另有两条**规则硬信号**不经过主题分类器、直接定性：
  · `is_identity`（问「你」的台词/名字/身份）→ chitchat；
  · `is_time_question`（「现在几点」）→ time（由 chain.time_node 调 current_time 取真值）。
两者都用**原句**判（改写器会把「你」补成角色名，第二人称信号会丢）。
另有一条**弱信号** `mentions_time`（只是顺带提了时间，例如「现在几点，顺便说说今汐的
共鸣链」）：不改变路由，只让 chain 在照常检索/闲聊之外**额外**把真值时间交给模型
（state.need_time，由 chain 写入）。

注意 slots 的 key 必须与 retrievers.py 的 CYPHER / SLOT_LABEL 一一对应，
若将来接入 LLM 抽取槽位，必须用白名单过滤非法槽位名。
"""
from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from wuwa_rag.config import get_settings
from wuwa_rag.core.llm import get_tool_llm
from wuwa_rag.ww_logger import get_logger

log = get_logger("rag")

# 事实型槽位：能直接查图谱
SLOT_PATTERNS: dict[str, str] = {
    "属性":    r"属性|元素|武器类型|什么武器|性别|出生|哪里人|稀有度",
    "属性反查": r"导电|冷凝|热熔|气动|衍射|湮灭",
    "技能":    r"技能|常态攻击|普攻|共鸣技能|共鸣解放|变奏|延奏|共鸣回路|大招",
    "共鸣链":  r"共鸣链|几链|命座|一链|六链|满链",
    "突破材料": r"突破|材料|素材|需要什么|消耗|多少个",
    "声骸":    r"声骸|配装|套装|cost|COST|词条|毕业|主词条|副词条",
    "武器":    r"武器|专武|装备",
    "队友":    r"队友|组队|配队|和谁|一起|阵容|队伍",
}

# 语义型：需要文档描述支撑
SEMANTIC_PATTERNS: tuple[str, ...] = (
    r"怎么|如何|怎样|为什么|原因|理由|依据|优势|好处|队友|组队|配队|和谁|一起|阵容|队伍|思路|玩法|攻略|讲解|介绍|分析|评价|强吗|值得|机制|原理",
)

# 人格/身份类硬信号：问「你」的台词/口头禅/名字/身份，是角色人格不是游戏数值资料。
# aemeath 人设由模型自带（Modelfile SYSTEM），这类应走 chitchat 让人设自由发挥；
# 走 RAG 反而召回大段「角色故事/珍贵之物」剧情文案被整段倾倒（实测「你的台词是什
# 么」→ hybrid → 809 字剧情故事，答案里混入无关内容）。注意必须用**原句**匹配：
# 改写器会把「你」补成角色名（「你的台词」→「爱弥斯的台词」），第二人称信号丢失。
_IDENTITY_PATTERNS: tuple[str, ...] = (
    r"你的台词", r"你的语音", r"你的语录", r"你的口头禅", r"你的口头语",
    r"你是谁", r"你叫什么", r"你叫啥", r"你的名字", r"你的身份",
)

# 时间类硬信号：问「现在几点 / 今天几号 / 星期几 / 当前日期」。
# 这类问题规则能全覆盖（问法就那么几种），答案来自服务端真实时钟（tools.current_time
# 工具），既不检索也不该让模型猜——模型没有时钟，凭空作答必然给出训练期附近的日期。
#
# 判据是**白名单 + 句末锚定**，不维护动词黑名单（黑名单永远漏，且与项目一贯口径相反）：
#   ① 「星期几 / 周几 / 礼拜几 / 几月几号」自身无歧义，任意位置命中；
#   ② 整句就是时间问法（可选礼貌前缀/锚点，时间词收尾）：「几点」「几点了」「请问现在几点」；
#   ③ 「几点 / 几号」太常见，**只有落在句末**才算问时间，且需要有「现在/今天/当前」这类
#      锚点佐证——锚点让「爱弥斯现在几点」「帮我看看现在几点」照样命中。
#      游戏提问里时间词后面总跟着动作词或名词，实测反例：「鸣潮几点刷新」「今天几点开服」
#      「卡卡罗几点上线」「日常委托几点」（无锚点）「秧秧共鸣链几号节点要多少材料」。
#      句末 + 锚点两条一起用就切掉了整类，无需枚举动词。
#      ⚠️ 锚点不能省：省了「今天有时间吗」「我今天有点累」会被误吞。
# ⚠️ 不要往里加「什么时候」：它太宽，「秧秧什么时候上线」「这个声骸什么时候出的」都是
# 游戏提问，误判会把真问题踢出检索链路。与 classify_topic 的回落口径同理——代价不对称。
# 判据用**原句**（与 is_identity 一致），改写器可能把「现在几点」改得面目全非。
_TAIL = r"[了呀呢啊吧]*[？?。!！~～·.…]*$"          # 收尾语气词 + 标点
_TIME_ANCHOR = r"(现在|此刻|这会儿|眼下|当前|今天|今日)(是|的)?(什么|多少|几)?"
_TIME_TAIL_WORD = r"(时间|日期|时刻|星期|周几|几点(几分)?|几分)"
_TIME_PATTERNS: tuple[str, ...] = (
    # ① 星期/日期问法自身无歧义，但**仍要句末锚定**：句中出现「星期几」就抢走的写法
    #    会把「今天星期几呀，长离怎么玩」整句踢进时间分支，长离那半句直接没人答。
    #    收尾锚定后，只有「…星期几(了呀)？」这种整句问法才算纯时间问题。
    r"(星期几|周几|礼拜几|几月几[日号])" + _TAIL,
    r"^(请问|家人|你)?(现在|今天|今日)?(几点|几号|几分)" + _TAIL,
    _TIME_ANCHOR + _TIME_TAIL_WORD + r"(是多少|是几|是什么|多少)?" + _TAIL,
    r"(日期|时间)[是到](多少|几|什么)" + _TAIL,
)

# 时间**提及**（比 _TIME_PATTERNS 宽一档）：句子里除了别的诉求还顺带问了时间。
# 与上面的区别只在**不要求句末收尾**——「现在几点，顺便说说今汐的共鸣链」整句不是
# 时间问法，但里面那半句是真的在问时间。
# ⚠️ 依然只收带锚点的写法，裸「几点 / 几号」一概不收（理由见 mentions_time docstring）。
_TIME_MENTION_PATTERNS: tuple[str, ...] = (
    r"(现在|此刻|这会儿|眼下|当前|如今)(是|的)?(什么|多少|几)?"
    r"(时间|日期|时刻|星期|周几|几点|几号|几分)",
    r"(今天|今日)(是|的)?(几号|几月几|星期几|周几|礼拜几|什么日子)",
    r"星期几", r"周几", r"礼拜几", r"几月几[日号]",
)

# 属性值（鸣潮共 6 种）——「导电角色有哪些」里没有「属性」二字，只有属性值
ELEMENTS = ("导电", "冷凝", "热熔", "气动", "衍射", "湮灭")
ELEMENT_RE = "|".join(ELEMENTS)
STAGE_RE = r"([一二三四五六]阶突破|[一二三四五六]阶)"


def detect_stage(question: str) -> str:
    m = re.search(STAGE_RE, question)
    return m.group(1) if m else ""


def detect_element(question: str) -> str:
    m = re.search(ELEMENT_RE, question)
    return m.group(0) if m else ""


def detect_slots(question: str) -> list[str]:
    """根据字典提取图谱关键词"""
    return [slot for slot, pat in SLOT_PATTERNS.items() if re.search(pat, question)]


def is_semantic(question: str) -> bool:
    """提取语义关键词"""
    return any(re.search(p, question) for p in SEMANTIC_PATTERNS)


def is_identity(question: str) -> bool:
    """是否在问角色人格（台词/口头禅/名字/身份），而非游戏数值资料。"""
    return any(re.search(p, question) for p in _IDENTITY_PATTERNS)


def is_time_question(question: str) -> bool:
    """是否在问当前时间/日期。

    命中即走 chain.time_node：**确定性调 tools.current_time** 取服务端真值，
    再让爱弥斯用自己的口吻说出这个时间。放在 classify_topic 之前判——主题分类器
    会把「现在几点」判成 chitchat（它确实与游戏无关），落到 chitchat_node 后模型
    对真实时间一无所知，只能含糊其辞或编一个出来。
    """
    return any(re.search(p, question) for p in _TIME_PATTERNS)


def mentions_time(question: str) -> bool:
    """句子里是否**顺带**问了时间（不要求整句都是时间问法）。

    与 `is_time_question` 的分工（两者是包含关系，纯时间问题必然也命中这里）：
      · `is_time_question` = 整句就是一个时间问法 → 走 time 分支：不检索，只调工具；
      · `mentions_time`    = 除了别的诉求还提到时间（「现在几点，顺便说说今汐的共鸣链」）
                          → 照常检索/闲聊，**另外**把服务端真值一并交给模型。

    ⚠️ 判据同样只用**带锚点**的写法，绝不放进裸的「几点 / 几号」：
    裸词会把「鸣潮几点刷新」「秧秧共鸣链几号节点要多少材料」这类真游戏问题也算成
    「问了时间」，于是往资料答案里平白塞一句「现在是…」，把刷新时刻/节点号的答案带偏。
    宁可漏（少说一句时间），不可滥（污染资料答案）——与 classify_topic 的回落口径同理。

    实现上是 `is_time_question` 的**真超集**（纯时间问题必然也命中这里），调用方据
    `is_time_question` 决定走不走 time 分支、据本函数决定要不要带真值时间。
    """
    return is_time_question(question) or any(
        re.search(p, question) for p in _TIME_MENTION_PATTERNS)


def classify(question: str, slots: list[str]) -> str:
    """根据关键词提取情况，返回预计处理方式，混合兜底"""
    has_fact, has_sem = bool(slots), is_semantic(question)
    if has_fact and has_sem:
        return "hybrid"
    if has_fact:
        return "fact"
    if has_sem:
        return "semantic"
    return "hybrid"         


def extract_characters(question: str, known: list[str]) -> list[str]:
    """返回所有命中的角色名（支持「卡卡罗和吟霖谁更强」这类多角色提问）。
    包含消歧：仅当 A 是 B 的子串时才剔除 A。
    len>=2 过滤单字，避免「他/她」这类误命中。
    """
    hits = [n for n in known if n and len(n) >= 2 and n in question]
    return [h for h in hits if not any(h != o and h in o for o in hits)]


# ---------- 主题分类：闲聊 vs 游戏（qwen3:8b agent）----------

_TOPIC_SYSTEM = """你是《鸣潮》问答助手的意图分类器。判断用户这句话属于哪一类，只输出一个 JSON：
{"topic": "game"} 或 {"topic": "chitchat"}
- game：询问《鸣潮》游戏内容（角色、声骸、配装、共鸣链、突破材料、技能、属性、配队、版本、玩法攻略等），或明显想查资料的问题。
- chitchat：日常寒暄、问候、情绪表达、与游戏无关的闲聊（如问好、问近况、撒娇、天气、玩笑）。
拿不准时输出 game。不要输出 JSON 以外的任何文字。"""

_TOPIC_EXAMPLES = (
    ("卡卡罗毕业配装用什么声骸", "game"),
    ("秧秧怎么玩", "game"),
    ("家人怎么这么晚才来，今天过得怎么样", "chitchat"),
    ("你好呀，今天心情不错", "chitchat"),
    ("我有点累了，陪我聊会儿", "chitchat"),
)


async def classify_topic(question: str) -> str:
    """判断主题是闲聊还是游戏提问。返回 "chitchat" 或 "game"。

    调用时机见 chain.intent_node：无角色名且无槽位/属性/阶段信号时才调，
    游戏提问大多被规则直接拦下，不承担额外延迟。解析失败一律回落 game——
    把闲聊误送 RAG 顶多答得生硬，把真问题误判闲聊则直接丢知识，代价不对称。

    少样本用单轮补全（Human 提问 / AI 答 JSON 交替），最后一轮 AI 前缀吃掉
    `{"`，模型只需续写 `"topic": "..."}` —— 比把完整 JSON 塞进 Human 消息
    省输出 token，分类实测更快。
    """
    msgs: list = [SystemMessage(content=_TOPIC_SYSTEM)]
    for q, t in _TOPIC_EXAMPLES:
        msgs.append(HumanMessage(content=f"问题：{q}"))
        msgs.append(AIMessage(content=f'{{"topic": "{t}"}}'))
    msgs.append(HumanMessage(content=f"问题：{question}"))
    msgs.append(AIMessage(content='{"'))
    try:
        resp = await get_tool_llm().ainvoke(msgs)
        txt = '{"' + (resp.content or "")
        m = re.search(r"\{.*?\}", txt, re.S)
        if not m:
            log.warning("主题分类: 输出无 JSON，回落 game：%r", (resp.content or "")[:60])
            return "game"
        topic = json.loads(m.group(0)).get("topic", "")
        return topic if topic in ("game", "chitchat") else "game"
    except Exception as exc:
        log.warning("主题分类失败，回落 game: %s", exc)
        return "game"


# ---------- 追问改写：把指代残缺的追问补成自包含问句（qwen3:8b agent）----------

_REWRITE_SYSTEM = """你是《鸣潮》问答助手的查询改写器。用户的问题可能带指代（她/它/他/那个/再/那/换成/开头那位…），需要结合对话上下文把它改写成一个不需要上下文就能看懂、适合拿去检索的自包含问句。
规则：
- 只输出改写后的一句话，不要引号、不要 JSON、不要解释。
- 把「她/它/他/那位」替换成具体的角色名；省略主语的要补上。
- 上下文里有角色锚点（话题角色）时，指代优先解析为锚点角色。
- 「话题角色」按最近提及排序：「她/他/它/那位」默认指**列表第一个**（最近讨论的角色）。
- 「开头/之前/前面聊的那位」这类**远指代**，解析为**摘要**里提到的角色（摘要按谈话顺序保留角色名，「开头聊的」= 摘要里最先出现的名字），不是最近话题。
- 如果问题本身已自包含（无指代、无省略），原样输出。
- 不改写主题，不加信息，不回答问题。

示例：
上下文：
更早对话摘要：卡卡罗毕业配装推荐彻空冥雷。
话题角色: 长离、今汐
最近对话：
用户: 长离的共鸣链效果
助手: 第一链提高抗打断
问题：开头聊的那位武器推荐什么
改写为：卡卡罗的武器推荐是什么"""


def focus_anchors(history: list[dict], known: list[str]) -> str:
    """A·结构化焦点压缩：从历史里提炼「话题角色 + 聊过的槽位」微型锚点。

    零 LLM、零延迟（全现成正则）。存在的意义：rewrite_query 里每轮原文截 120
    字符，角色名出现在长回答的深处就会被截丢；锚点用全量文本提名字，不受截断影响。
    角色按**最近提及优先**排序（倒序遍历历史）——实测正序时「她」在两个角色间
    歧义，8b 会放弃改写；配合 _REWRITE_SYSTEM 的「默认第一个」规则消歧。
    """
    chars: list[str] = []
    slots: list[str] = []
    for m in reversed(history):
        for c in extract_characters(m.get("content", ""), known):
            if c not in chars:
                chars.append(c)
        for s in detect_slots(m.get("content", "")):
            if s not in slots:
                slots.append(s)
    if not chars and not slots:
        return ""
    parts = []
    if chars:
        parts.append(f"话题角色: {'、'.join(chars[:3])}")
    if slots:
        parts.append(f"聊过的方面: {'、'.join(slots[:4])}")
    return "；".join(parts)


_SUMMARY_SYSTEM = """你是对话压缩器。把「已有摘要」和「即将被遗忘的旧对话」合并压缩成一句不超过80字的会话摘要，只保留：聊过哪些《鸣潮》角色、涉及哪些方面（声骸/配队/突破/共鸣链等）、用户的偏好倾向。
- **角色名是最高优先级信息，必须逐字保留**，其次才是细节。宁可丢细节也不能丢名字。
- 只输出摘要正文，不要前缀、不要引号、不要解释。
- 旧摘要里的信息如果新对话没再提及，仍要保留（除非与新增内容冲突）。"""

_DEGRADED_MARK = "（摘要失败，话题未知）"


async def summarize_turns(evicted: list[dict], prev_summary: str) -> str:
    """B·滚动摘要：压缩将被滑出记忆窗口的轮次。

    模型选型实测：0.6b 合并多轮时会**丢角色名**（输出「讨论了声骸组合及相关话题」
    这类空话），而摘要的价值恰恰在保住名字，所以压缩器用 qwen3:8b（get_tool_llm）。
    实测首次压缩 12.9s、滚动合并 0.2s（输入短时有 KV 前缀缓存）。

    evicted 是被挤出窗口的旧轮次（调用方必须在截断 history 前取好——checkpointer
    里只有截断后的窗口，事后拿不到）。失败/超长时把 prev_summary 追加降级标记
    返回——标记的意义是让**下一轮重新压缩完整窗口**来修复（此时被压缩的原文还全在
    窗口内）；若原样吞掉，坏摘要会被一路继承、再也修不好。
    """
    s = get_settings()
    text = "\n".join(
        f"{'用户' if m.get('role') == 'user' else '助手'}: {(m.get('content') or '')[:200]}"
        for m in evicted
    )
    if not text:
        return prev_summary
    msgs = [
        SystemMessage(content=_SUMMARY_SYSTEM),
        HumanMessage(content=f"已有摘要：{prev_summary or '（无）'}\n\n旧对话：\n{text}\n\n合并摘要："),
    ]
    try:
        # tags 打标：ainvoke 内部同样产生 on_chat_model_stream 事件，且与 generate_node
        # 同属一个节点（langgraph_node='generate'），节点过滤挡不住它——实测摘要文本
        # 曾作为尾巴拼进流式答案。下游按 "wwa:summary" 标签丢弃（见 chain.ask_stream）。
        resp = await get_tool_llm().ainvoke(msgs, config={"tags": ["wwa:summary"]})
        out = (resp.content or "").strip().strip('"「」\'')
        out = out.splitlines()[0].strip() if out else ""
        if not out or len(out) > s.SUMMARY_MAX_CHARS:
            log.warning("摘要异常(%r)，标记降级待下轮修复", out[:60])
            return f"{prev_summary} {_DEGRADED_MARK}".strip()
        log.info("滚动摘要: %r（窗口外 %d 条）", out, len(evicted))
        return out
    except Exception as exc:
        log.warning("摘要失败，标记降级待下轮修复: %s", exc)
        return f"{prev_summary} {_DEGRADED_MARK}".strip()


async def rewrite_query(
    question: str, history: list[dict], *, known: list[str] | None = None, summary: str = "",
) -> str:
    """追问改写：有上下文（历史或摘要）才调 LLM，无历史直接原句返回省一次调用。

    输入三路合并（A+B）：滚动摘要（窗口外的压缩记忆）+ 焦点锚点（结构化角色信号）
    + 最近 2 轮短原文。检索（意图/槽位/图谱/向量）都吃改写句——「那她配什么声骸」
    单拿去召回必落空，补出角色名后才能命中。任何失败（异常/空/过长）一律回落
    原句，改写是增益不是依赖，绝不能因它把问答弄挂。history 与展示仍用原句。
    """
    if not history and not summary:
        return question
    bits = []
    if summary:
        bits.append(f"更早对话摘要：{summary}")
    if history:
        if known:
            anchors = focus_anchors(history, known)
            if anchors:
                bits.append(anchors)
        turns = history[-4:]   # 最近两轮足够定位指代对象，也更省 token
        ctx = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content'][:120]}" for m in turns
        )
        bits.append(f"最近对话：\n{ctx}")
    msgs = [
        SystemMessage(content=_REWRITE_SYSTEM),
        HumanMessage(content="\n".join(bits) + f"\n\n原始问题：{question}\n改写为："),
    ]
    try:
        resp = await get_tool_llm().ainvoke(msgs)
        out = (resp.content or "").strip().strip('"「」\'').splitlines()[0].strip() if resp.content else ""
        if not out or len(out) > max(len(question) * 4, 60):
            log.warning("查询改写: 输出异常(%r)，用原句", out[:60])
            return question
        if out != question:
            log.info("查询改写: %r -> %r", question, out)
        return out
    except Exception as exc:
        log.warning("查询改写失败，用原句: %s", exc)
        return question
