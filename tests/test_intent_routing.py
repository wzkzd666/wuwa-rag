"""意图路由：intent_node 的分流判据。

这里守两条独立的修复，都是「用户说一句话，系统走错分支」级别的问题。

① 单字角色名参与路由
-------------------
`intent_node` 的 `chars` 来自 `extract_characters`。单字名被滤掉时，
「心的声骸怎么配」会 chars=[] → 图谱空 → 资料空 → verify 判不匹配 →
chars 仍空所以**跳过按角色重爬** → 额度耗尽 → 落到联网兜底。
用户视角就是「角色明明爬好了，问了还是去联网搜」。

② 自我介绍必须走闲聊，不能走检索
--------------------------------
「我是颗粒」这类句子，`classify()` 兜底会给 `hybrid`（无槽位无语义词时的默认值），
于是一路走到全量检索 → 查无资料 → chars 空不重爬 → **联网兜底**。
用户实测到的现象正是「我说我是颗粒，它去联网查『颗粒』是什么」。
这类句子的正确出口只有「闲聊 + 写画像」，检索必然空手。
修法是在 `intent_node` 加 `is_self_intro(q)` 硬信号（①′ 分支），与问助手身份的
`is_identity`（① 分支）方向相反但同层。

⚠️ 主题分类器被 stub 成一律返回 'game'（见 conftest.offline_intent）
---------------------------------------------------------------
这是**刻意的最坏情形**。闲聊分流若只在 LLM 判 chitchat 时才成立，那就不叫规则兜底。
stub 成 game 之后 `我是颗粒` 仍判 chitchat，才证明硬信号真的独立生效。
同理，负样本 `核心玩法是什么` 在 stub 成 game 后应落到 semantic/game 侧，
绝不能被 is_self_intro 抢走。

⚠️ 断言必须同时验 intent 与 characters
------------------------------------
见 conftest 模块 docstring 的血泪教训：只断言 intent 会「假绿」。
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from wuwa_rag.dialog import nlu as nlu_mod
from wuwa_rag.dialog.nlu import (
    asks_about_own_nickname,
    is_identity,
    is_material_question,
    is_self_intro,
    veto_ambiguous_names,
)

# (问句, 期望 intent, 期望 characters)。期望值取自 2026-10-06 实测（主题分类器 stub 成 game）。
ROUTE_CASES = [
    # —— ① 单字角色名：chars 必须非空，否则下游一路升级到联网
    ("心的声骸怎么配", "hybrid", ["心"]),
    ("心和椿配队", "fact", ["心", "椿"]),
    ("心配队", "fact", ["心"]),
    ("卡卡罗和心谁强", "hybrid", ["卡卡罗", "心"]),
    ("守岸人配队", "fact", ["守岸人"]),
    # —— ② 自我介绍 / 问助手身份 → chitchat（规则硬信号，不靠 LLM）
    ("我是颗粒", "chitchat", []),
    ("你是谁", "chitchat", []),
    # —— 时间类：最优先判，且不调 LLM
    ("现在几点", "time", []),
    # —— 负样本：真游戏提问绝不能被自我介绍规则抢走
    ("我是萌新，守岸人怎么玩", "semantic", ["守岸人"]),
    ("核心玩法是什么", "semantic", []),
]

# is_self_intro 的完整样本集（实测 30 条全对）。
# 正样本 = 陈述用户自己；负样本含两类陷阱：
#   · 疑问句（我是谁 / 我叫什么 / 我是玩家吗）——问的是助手或反问，不是自报家门；
#   · 复合句（我是萌新，守岸人怎么玩）——真游戏提问，被抢走就丢检索。
SELF_INTRO_POSITIVE = [
    "我是颗粒", "我叫颗粒", "我是萌新", "我是新来的", "我是新人",
    "你可以叫我颗粒", "叫我颗粒就行", "大家好我是颗粒", "嗯我是颗粒", "记住我叫颗粒",
    "我的游戏ID是颗粒", "我的名字是小星", "喊我阿明", "人家是新手",
    "我是老玩家", "我是回归玩家", "我叫椿", "我是心", "我是Hsin", "叫我小星吧",
]
SELF_INTRO_NEGATIVE = [
    "我是萌新，守岸人怎么玩", "守岸人怎么玩", "我是谁", "你是谁", "心的声骸怎么配",
    "我是来问问题的，卡卡罗配队", "你觉得我是谁", "我是导电属性的吗",
    "我叫什么", "我是玩家吗",
]


@pytest.mark.parametrize("question,expected_intent,expected_chars", ROUTE_CASES)
async def test_intent_node_路由(offline_intent, question, expected_intent, expected_chars):
    out = await offline_intent(question)
    # 双断言：intent 与 characters 都要对（只验 intent 会假绿，见模块 docstring）
    assert out["intent"] == expected_intent, f"{question!r} intent={out['intent']}"
    assert out["characters"] == expected_chars, f"{question!r} chars={out['characters']}"


# 「培养要花什么」判据（决定是否附突破材料表）。前 5 条是 2026-10-23 实测到的
# **两边口径不一致**的问法：旧的 graph 正则认不出「需要什么/消耗」，
# 而 nlu 的槽位词表又认不出「练度/培养」。
MATERIAL_YES = [
    "卡卡罗突破材料", "突破到六阶要什么", "练度怎么提升", "卡卡罗培养路线",
    "60级需要什么", "升满要消耗多少", "声骸突破材料",
]
# 关键是「声骸需要什么」这一类：槽位词表会命中「需要什么」，但用户问的是声骸搭配，
# 附上「必须逐条照抄」的突破材料表反而会把答案带偏 —— 补料判据必须比槽位窄。
MATERIAL_NO = [
    "声骸需要什么", "武器选什么", "共鸣解放倍率", "心怎么配队", "", "主词条选什么",
    # 「消耗/需要多少」在鸣潮还指**技能的能量开销**，不是培养材料（2026-10-23 审查抓到的
    # 假阳性：弱词单独命中时把「必须逐条照抄」的突破材料表挂到了技能能量问题上）。
    "共鸣回路消耗多少能量", "延奏技能消耗多少协奏", "技能需要消耗多少能量",
]


@pytest.mark.parametrize("question", MATERIAL_YES)
def test_材料补料判据_该附表的问法都认(question: str) -> None:
    assert is_material_question(question), f"{question!r} 应判为培养材料问题"


@pytest.mark.parametrize("question", MATERIAL_NO)
def test_材料补料判据_拿不准就不附表(question: str) -> None:
    assert not is_material_question(question), f"{question!r} 不该附突破材料表"


async def test_裸指代转澄清_且各硬信号出口不被抢走(offline_intent, monkeypatch) -> None:
    """守 `clarify` 分支的**可达性**。

    它曾经是恒假条件：判据写成 `intent == "chitchat"`，却挂在 `classify()` 之后——
    而 `classify()` 只可能返回 hybrid/fact/semantic，闲聊判定根本还没发生
    （它在下面的 ⓪①①′② 分流里才赋值）。于是反问分支从上线起一次都没执行过。
    2026-10-23 实测：说「那个」落到闲聊，模型自己圆出一句「怎么突然这么小声呀」——
    正是 `clarify` 存在的理由（不许猜着答）所禁止的行为。

    这里把主题分类器 stub 成 'chitchat'（clarify 唯一可能的上游），断言三件事：
      ① 纯指代/重试/语气词 → clarify（真的会反问）；
      ② 语义完整的招呼 → 仍是 chitchat（不许把正常闲聊一律打回反问）；
      ③ 有自己出口的硬信号（问助手身份、自我介绍、时间）不许被改道。
    音乐回指（「继续」带历史）由 if/elif 顺序天然优先，见同一函数里的 ① 之前的 music 分支。
    """
    from wuwa_rag.dialog import graph as graph_mod

    async def _topic_chitchat(_question: str) -> str:
        return "chitchat"

    monkeypatch.setattr(graph_mod, "classify_topic", _topic_chitchat)

    for q in ["那个", "这个", "上面那个", "继续", "嗯", "啊"]:
        out = await offline_intent(q)
        assert out["intent"] == "clarify", f"{q!r} 应转澄清，实得 {out['intent']}"

    for q in ["你好呀", "今天天气不错"]:
        out = await offline_intent(q)
        assert out["intent"] == "chitchat", f"{q!r} 语义完整，不该被打回反问：{out['intent']}"

    # ③ 硬信号各自守出口：这三条都在 ② 之前的分支里落定，不会被澄清改道
    assert (await offline_intent("你是谁"))["intent"] == "chitchat"
    assert (await offline_intent("我是颗粒"))["intent"] == "chitchat"
    assert (await offline_intent("现在几点"))["intent"] == "time"


async def test_自我介绍走闲聊_即使主题分类器判游戏(offline_intent):
    """最坏情形下的兜底证明：classify_topic 被 stub 成一律 'game'，仍须判 chitchat。"""
    out = await offline_intent("我是颗粒")
    assert out["intent"] == "chitchat"


async def test_复合句不被自我介绍抢走(offline_intent):
    """「我是萌新，守岸人怎么玩」带槽位/角色名，必须留给检索侧。

    两道保险：is_self_intro 本身在句尾逗号处停下不命中；分流条件另带 `not slots`。
    """
    out = await offline_intent("我是萌新，守岸人怎么玩")
    assert out["intent"] != "chitchat"
    assert out["characters"] == ["守岸人"]


@pytest.mark.parametrize("question", SELF_INTRO_POSITIVE)
def test_is_self_intro_正样本(question: str) -> None:
    assert is_self_intro(question), f"{question!r} 应判为自我介绍"


@pytest.mark.parametrize("question", SELF_INTRO_NEGATIVE)
def test_is_self_intro_负样本(question: str) -> None:
    assert not is_self_intro(question), f"{question!r} 不应判为自我介绍"


def test_is_identity_问助手身份() -> None:
    """① 分支的硬信号：问的是助手自己，与 is_self_intro 方向相反。"""
    for q in ("你是谁", "你叫什么名字", "你的名字", "你的台词是什么", "你的口头禅", "你的身份是什么"):
        assert is_identity(q), f"{q!r} 应判为身份类"


def test_is_identity_不覆盖问底层模型() -> None:
    """⚠️ 边界钉桩：`_IDENTITY_PATTERNS` 判的是**角色人格**（台词/口头禅/名字/身份），
    不是「底层用的是哪个 LLM」。实测「你是什么模型」→ False，这是**设计如此**。

    写这条是为了防止后人误判成「覆盖缺口」而去加正则——问模型属于产品身份问题，
    走 chitchat 让 aemeath 用人设回答即可，不需要 is_identity 这道硬信号。
    真要改，先想清楚：加进去会让「你是什么模型」跳过主题分类器，
    而它其实和「你是谁」需要的处理并不相同。
    """
    for q in ("你是什么模型", "你用的什么模型", "你是GPT吗"):
        assert not is_identity(q), f"{q!r} 不应被 is_identity 命中（问的是底层 LLM，非角色人格）"


def test_两条硬信号不互相越界() -> None:
    """is_identity 与 is_self_intro 必须各管一头，不能把对方的样本也吃进来。

    「我是谁」是疑问句（问助手/反问），两条都不该判正——它既不是自报家门，
    也不是在问助手的身份设定。
    """
    assert not is_identity("我是颗粒")
    assert not is_self_intro("你是谁")
    assert not is_identity("我是谁")
    assert not is_self_intro("我是谁")


# ---------- 歧义角色名的 LLM 裁决（config.NAME_LLM_ADJUDICATION，默认关闭）----------

class _FakeLLM:
    """按脚本返回预设文本的 tool LLM 替身。记录收到的 user 消息以便断言调用范围。"""

    def __init__(self, script, boom: bool = False) -> None:
        self._script = script
        self._boom = boom
        self.seen: list[str] = []

    async def ainvoke(self, messages):
        self.seen.append(messages[-1][1])
        if self._boom:
            raise RuntimeError("ollama 没起")
        text = self._script.pop(0) if self._script else ""
        return AIMessage(content=text)


async def test_裁决_说不是角色就否决(monkeypatch) -> None:
    llm = _FakeLLM(['{"refers_to_character": false}'])
    monkeypatch.setattr(nlu_mod, "get_tool_llm", lambda: llm)
    assert await veto_ambiguous_names("心算不算强", ["心"]) == []


async def test_裁决_说是角色就保留(monkeypatch) -> None:
    llm = _FakeLLM(['{"refers_to_character": true}'])
    monkeypatch.setattr(nlu_mod, "get_tool_llm", lambda: llm)
    assert await veto_ambiguous_names("心声骸推荐", ["心"]) == ["心"]


async def test_裁决_输出畸形或模型挂掉都保留规则结论(monkeypatch) -> None:
    """裁决是增益：任何异常都**不得**改写规则层已经给出的名字。"""
    for llm in (_FakeLLM(["不是 JSON"]), _FakeLLM([], boom=True), _FakeLLM(["{oops"])):
        monkeypatch.setattr(nlu_mod, "get_tool_llm", lambda fake=llm: fake)
        assert await veto_ambiguous_names("心值得练吗", ["心"]) == ["心"]


async def test_裁决_多字名不送去模型(monkeypatch) -> None:
    """多字名规则层几乎不会错，交给模型只会引入风险 —— 必须一个都不送。"""
    llm = _FakeLLM(['{"refers_to_character": false}'])
    monkeypatch.setattr(nlu_mod, "get_tool_llm", lambda: llm)
    assert await veto_ambiguous_names("守岸人配队", ["守岸人"]) == ["守岸人"]
    assert llm.seen == []


# ---------- 「问用户自己是谁」（is_self_intro 的疑问形态）----------

def test_问自己是谁_只认画像里的昵称且只认疑问句() -> None:
    """「颗粒是谁」曾走向量检索 → 答成爱弥斯自述（用户实测）。

    修法是把它判进 chitchat：正确答案只可能来自画像（user_facts 里有
    「用户自称是颗粒」），检索必然空手。昵称从画像文本里取，不硬编码任何名字。
    """
    ctx = "用户自称是颗粒"
    # 疑问形态：命中
    for q in ("颗粒是谁", "颗粒叫什么", "颗粒是谁呀", "颗粒是哪位", "颗粒是什么人"):
        assert asks_about_own_nickname(q, ctx), q
    # 游戏角色问句不受影响（「谁」不等于在问用户）
    assert not asks_about_own_nickname("守岸人是谁", ctx)
    assert not asks_about_own_nickname("心是谁", ctx)
    # 昵称不在问句里 → 不命中（不能把「守岸人是颗粒吗」这类也拉进闲聊）
    assert not asks_about_own_nickname("守岸人是颗粒吗", ctx)
    # 是非问不算「问是谁」
    assert not asks_about_own_nickname("颗粒是不是萌新", ctx)
    # 画像为空 → 判定不了就不猜
    assert not asks_about_own_nickname("颗粒是谁", "")
    # 换个昵称自动生效（证明没硬编码）
    assert asks_about_own_nickname("小星是谁", "用户自称是小星")


async def test_问自己是谁_已在意图路由里判进闲聊(offline_intent) -> None:
    """端到端过一遍 intent_node：必须落在 chitchat，且不指向任何游戏角色。

    走 offline_intent 夹具（stub 名册/改写/主题分类器），所以这条断言证明的是
    **规则本身**独立生效 —— 即使主题分类器被 stub 成一律 'game'（最坏情形），
    仍然判 chitchat。
    """
    st = await offline_intent("颗粒是谁", user_context="用户自称是颗粒")
    assert st["intent"] == "chitchat", st
    assert not st.get("characters"), st
    # 反例对照：同样带「谁」，但没指名用户昵称 → 仍走检索（游戏问题不受影响）
    st2 = await offline_intent("守岸人是谁", user_context="用户自称是颗粒")
    assert st2["intent"] != "chitchat", st2
    assert st2.get("characters") == ["守岸人"], st2


def test_对话图能编译_且每条边的端点都已注册为节点() -> None:
    """守「边引用了未注册节点」这类构图错误 —— 它是**编译期**才暴露的。

    背景：`clarify` 分支曾经只加了路由表 `"clarify": "clarify"` 和
    `add_edge("clarify", END)`，却漏掉 `add_node("clarify", ...)`，
    于是 LangGraph 编译时抛 `Found edge starting at unknown node 'clarify'`。
    因为 `_compiled` 是懒加载（首次问答才编译），**进程启动一切正常、
    第一条提问才「生成中断」**，只看启动日志根本发现不了。

    这条测试不需要 LLM / 数据库 / 事件循环，纯构图，毫秒级，
    但能拦住「加节点时漏注册」这一类所有变体。
    """
    from wuwa_rag.dialog.graph import build_graph

    compiled = build_graph().compile()   # 构图有问题会直接在这里抛
    nodes = set(compiled.get_graph().nodes)
    assert "clarify" in nodes
    for edge in compiled.get_graph().edges:
        assert edge.source in nodes, f"边的起点不是已注册节点: {edge.source}"
        assert edge.target in nodes, f"边的终点不是已注册节点: {edge.target}"


def test_产正文的节点必须全在流式白名单里() -> None:
    """守「分支能生成答案,却被流式 token 白名单挡成空气泡」。

    上一条测试断言了 `"clarify" in nodes`,而 2026-10-23 抓到的 bug 恰恰是**它全绿**
    的情况下发生的:节点注册了、路由表指了、`_chat_turn` 也正常生成了,
    只有 `ask_stream` 的白名单漏了 `clarify`。token 是从 `on_chat_model_stream`
    **旁路抽取**的,名单漏一个名字不抛异常、不报错,只让 `done.answer` 变成空串 ——
    前端留下一条永远空白的气泡,而服务端日志里答案写得明明白白。
    （「没听懂→反问」是任何含糊输入都会走的活路径,不是死代码。）

    判据**不抄节点名清单**,而是从源码现取:调 `_chat_turn(` 或 `.astream(` 的节点
    就是「会产出给用户看的正文」的节点。将来新加一条分支忘了进 `_TEXT_NODES`,
    这条当场红,而不是等用户看到空气泡。
    """
    import inspect

    from wuwa_rag.dialog import graph as graph_mod

    producers: set[str] = set()
    for obj in vars(graph_mod).values():
        name = getattr(obj, "__name__", "")
        if not (callable(obj) and name.endswith("_node")):
            continue
        src = inspect.getsource(obj)
        if "_chat_turn(" in src or ".astream(" in src:
            producers.add(name[: -len("_node")])

    assert producers, "一个产正文的节点都没解析到，多半是 _chat_turn/.astream 的写法变了"
    assert producers == set(graph_mod._TEXT_NODES), (
        f"产正文节点与流式白名单不一致：只在节点={sorted(producers - set(graph_mod._TEXT_NODES))}"
        f"、只在白名单={sorted(set(graph_mod._TEXT_NODES) - producers)}"
        "——漏白名单的那条分支前端一个字都收不到"
    )
    # 回归钉桩：2026-10-23 实测漏的就是它
    assert "clarify" in graph_mod._TEXT_NODES
    # 反问这条分支也得有阶段提示，否则用户只看到三个点干转（上限 7 字，见 _STAGE_LABELS）
    assert graph_mod._STAGE_LABELS["clarify"]
