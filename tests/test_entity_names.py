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
"""
from __future__ import annotations

import pytest

from wuwa_rag.dialog.nlu import extract_characters
from wuwa_rag.knowledge.entities import find_mentions, mentioned_names

# (问句, 期望识别出的角色集合)。期望值全部来自 2026-10-06 的实测输出，非推测。
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
