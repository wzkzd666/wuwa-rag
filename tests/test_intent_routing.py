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

from wuwa_rag.dialog.nlu import is_identity, is_self_intro

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
