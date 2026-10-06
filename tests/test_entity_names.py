"""角色名提及判定（entities.find_mentions / nlu.extract_characters）。

守的是 2026-10-06 修的那个 bug：`extract_characters` 有一道 `len(n) >= 2` 过滤，
把**单字角色名**（心 / 椿）一刀切滤掉。后果不是「少识别一个名字」这么轻——
`characters` 恒空会让 `verify_node` 的「按角色清库重爬」分支永不触发（它要求 chars
非空），额度耗尽后 verify_stage 落到 `web`，**新角色爬好了问答照样去联网**。

所以这里的断言分两组，缺一不可：
  · 正样本：单字名在各种搭配下都要被认出（尤其 `心配队`——见下方 HMM 说明）；
  · 负样本：含同字的常用词绝不能误命中（核心 / 中心 / 关心 / 开心 / 心情 / 决心 / 用心）。
负样本是「不敢简单把过滤放开成 len >= 1」的原因：放开会把半个中文世界都当成角色名。

⚠️ `心配队` 这条是**回归哨兵**，删了会丢掉一次真实踩坑的记录
------------------------------------------------------------
判据用的是 jieba 分词「单字名必须独立成词」。默认开着 HMM 时，
`心配队` 会被切成 `['心配', '队']`——「心配」不在词典里，是 HMM 新词发现**臆造**
出来的词，于是「心」又被吞掉。修法是在 `find_mentions` 里显式 `HMM=False`。
这条用例就是钉住它：哪天有人为了别的分词效果把 HMM 打开，这里会立刻红。

⚠️ `心声骸推荐` 这条是**第二道回归哨兵**（`HMM=False` 挡不住的一类）
------------------------------------------------------------
`HMM=False` 只能挡住「HMM 臆造新词」造成的粘连，**挡不住词典里已有的真词**。
`心声` 正是 jieba 默认词典里的词，最长匹配优先 → `心声骸推荐` 被切成
`心声/骸/推荐`，`心` 这个 token 根本不出现 → 角色识别不到 → `characters` 为空
→ `verify_node` 的「按角色清库重爬」分支永不触发 → 落到联网兜底（用户实测报的现象）。
注意这类问法与已覆盖的「心的声骸怎么配」只差一个「的」，极易漏测。
兜底判据是**原文裸子串**扫描（不改分词器，理由见 `entities._DOMAIN_SUFFIXES`）。
"""
from __future__ import annotations

import pytest

from wuwa_rag.dialog.nlu import extract_characters
from wuwa_rag.knowledge import domain_terms
from wuwa_rag.knowledge.entities import find_mentions, mentioned_names

# (问句, 期望识别出的角色集合)。期望值全部来自实测输出，非推测。
POSITIVE = [
    ("心配队", {"心"}),                       # ← HMM=False 回归哨兵，见模块 docstring
    ("心和椿配队", {"心", "椿"}),
    ("心的声骸", {"心"}),
    ("心的队友", {"心"}),
    ("心+椿+守岸人配队", {"心", "椿", "守岸人"}),
    ("椿配队", {"椿"}),
    ("心怎么配队", {"心"}),
    ("心的突破材料", {"心"}),
    ("卡卡罗和心谁强", {"卡卡罗", "心"}),
    ("鉴心和心谁强", {"鉴心", "心"}),          # 多字名与单字名同现，二者都要留住
    ("秧秧·玄翎的武器", {"秧秧·玄翎"}),        # 带间隔号的形态变体
    ("守岸人配队", {"守岸人"}),
    # ↓ 「心声」是词典真词，分词看不到「心」，靠原文兜底判据救回
    ("心声骸推荐", {"心"}),                    # ← 第二道回归哨兵
    ("心声骸", {"心"}),
    ("心声骸怎么配", {"心"}),
    ("心声骸配队", {"心"}),
    ("心连招", {"心"}),
]

# 含单字角色名同字的常用词：一个都不能误命中。
NEGATIVE = [
    "核心玩法是什么",
    "这个中心思想",
    "我很关心剧情",
    "开心",
    "心情不错",
    "决心要练她",
    "用心练她",
    "声骸核心词条",
    "这个中心很关键",
    "担心自己练不好",
    # ↓ 兜底判据的边界：名字左边是汉字时不认，否则会把这些当成在谈角色「心」
    "核心声骸",          # 「声骸」在右边，但「心」是「核心」的内部字
    "核心属性",
    "声骸的核心机制",
    "心态调整",
    "心愿",
    "连招怎么打",        # 「连招」在表内，但没出现角色名
]


@pytest.mark.parametrize("question,expected", POSITIVE)
def test_extract_characters_正样本(question: str, expected: set[str], roster) -> None:
    assert set(extract_characters(question, roster)) == expected


@pytest.mark.parametrize("question", NEGATIVE)
def test_extract_characters_负样本不误命中(question: str, roster) -> None:
    got = extract_characters(question, roster)
    assert got == [], f"{question!r} 误识别为角色名：{got}"


def test_find_mentions_返回位置与名字(roster) -> None:
    """find_mentions 是底层判据，除名字外还返回**位置**（远指代注入要靠它取「最靠前的」）。

    顺序必须是文中出现顺序，而不是名册顺序——`_inject_far_characters` 依赖这点。
    """
    got = find_mentions("卡卡罗和心谁强", roster)
    assert [name for _, name in got] == ["卡卡罗", "心"]
    # 位置单调递增，且确实落在原文对应处
    positions = [pos for pos, _ in got]
    assert positions == sorted(positions)
    for pos, name in got:
        assert "卡卡罗和心谁强".startswith(name, pos)


def test_mentioned_names_与_extract_characters_口径一致(roster) -> None:
    """两个入口必须同口径——它们共用 `find_mentions`，这条钉住「别哪天又各写一套」。"""
    for question, expected in POSITIVE:
        assert set(mentioned_names(question, roster)) == expected


def test_单字名兜底判据_位置取首字(roster) -> None:
    """兜底判据命中的「心」必须报**首字位置**。

    `graph._inject_far_characters` 要靠位置取摘要里「最靠前」的角色名，位置错了
    远指代（「开头聊的那位」）就会选错人。
    """
    assert find_mentions("心声骸推荐", roster) == [(0, "心")]
    assert find_mentions("心声骸", roster) == [(0, "心")]


# ---------- 领域词表驱动（词表是数据，不是硬编码）----------

def test_领域词表驱动_派生表独有的词能认出(roster, monkeypatch) -> None:
    """判据由**词表**驱动：词表里有的词就认。

    用一个虚构词做探针 —— 它既不在 jieba 词典、也不在手写兜底表里，
    因此能认出它只可能是派生领域词表起了作用。
    （只断言正方向：负方向无法用同一条句子构造 —— 分词通道可能独立命中，
    「词表为空时不该认」这句话得另找 glued 样例来钉，见下面那条并集测试。）
    """
    monkeypatch.setattr(domain_terms, "current", lambda: frozenset({"虚构领域词"}))
    assert extract_characters("心虚构领域词推荐", roster) == ["心"]


def test_领域词表_派生表非空时手写兜底表仍然生效(roster, monkeypatch) -> None:
    """回归：`领域词表 = 派生表 ∪ 手写表`。

    这条钉住一个真实写错过的实现 —— 当时写成「派生表非空就只用它」，于是
    「声骸」这类**派生表里没有**（wiki 分类法标签是「声骸套装推荐」，
    而 jieba 不认「声骸」这个未登录词）的词整批失效，
    `心声骸推荐`/`心声骸`/`心声骸怎么配` 三条从能认变成认不出。
    """
    monkeypatch.setattr(domain_terms, "current", lambda: frozenset({"共鸣链", "武器"}))
    for q in ("心声骸推荐", "心声骸", "心声骸怎么配", "心声骸配队"):
        assert set(extract_characters(q, roster)) == {"心"}, q


# ⚠️ **已知残留误报**（不是本层引入的，别在这里加断言钉死）
# 「心算不算强 / 心累了吗 / 心什么 / 心太软」会被判成提及角色「心」——
# 来源是**分词通道**：`心算`/`心累`/`太软` 都在 jieba 词典里，但 jieba 仍把
# 这些句切成 `心/算不算`、`心/累了吗`、`心/太软`，`心` 成了独立 token。
# 领域词证据只兜「分词漏掉」的情况，管不了「分词判错」。
# 彻底解决要靠第 ④ 层（tool LLM 二次裁决），当前未启用。

