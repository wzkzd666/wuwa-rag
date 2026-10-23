"""答案级质量闸门 `dialog.quality.check_answer` 的用例（全离线，不碰模型与服务）。

这个模块的存在意义是「把已经踩过的答案故障形态钉成可复跑的检查」，
所以每条用例都拿**当年那个坏形态**当正样本，另配一条**干净答案**当负样本 ——
只有前者能红，才证明检查不是恒过的摆设（本仓反复吃过「条件恒假 = 静默失效」的亏）。
"""
from __future__ import annotations

from wuwa_rag.dialog.graph import _TEAM_FOLLOWUP
from wuwa_rag.dialog.quality import check_answer

# 干净答案：四步清洗后都不该再变、没有标签、没有角标
CLEAN = ("我的配队呀~家人可以这样组：\n\n"
         "- 守岸人+吟霖/长离/散华+卡卡罗\n"
         "- 清宵+守岸人+尤诺\n\n"
         "暴击固定为80%，暴击伤害固定为275%。")


def test_干净答案一条不报() -> None:
    assert check_answer(CLEAN) == []
    # 带画像昵称的干净答案也不许误报（unquote 只剥**恰好包裹昵称**的那层引号）
    assert check_answer("颗粒刚刚让我放歌，现在去听吗？", nicknames=["颗粒"]) == []


def test_空答案必须报并且说出真正的原因() -> None:
    got = check_answer("   ")
    assert len(got) == 1 and "_TEXT_NODES" in got[0]


def test_引用角标没剥净() -> None:
    assert "引用角标 [n] 没剥净" in check_answer("守岸人很强 [1] 可以这样打。")


def test_百分比丢单位没修() -> None:
    # 当年那次的形态：术语后面跟着裸数值（`2.80` 后面没有 %）
    assert "百分比单位串味没修" in check_answer("六链之后暴击提升2.80，很稳的一档。")


def test_重复列表行没去() -> None:
    dup = "- 守岸人+吟霖+卡卡罗\n- 守岸人+吟霖+卡卡罗\n- 清宵+守岸人+尤诺"
    assert "重复列表行没去" in check_answer(dup)


def test_昵称外层引号没剥只在给出昵称时查() -> None:
    bad = "「颗粒」刚刚让我放一首歌来着。"
    assert "昵称外层引号没剥" in check_answer(bad, nicknames=["颗粒"])
    # 没给昵称就**不许猜**：别的「强调」引号一概不动（歌名《…》、术语【…】同理）
    assert check_answer(bad) == []


def test_情绪标签泄进正文() -> None:
    # EMOTION_TAGS 的**值**才是标签（[excited]），键是情绪名 —— 查错方向会恒不报
    assert "情绪标签泄进正文：['[excited]']" in check_answer("[excited]我来啦~家人好。")


def test_概括性配队的追问两态() -> None:
    with_tail = CLEAN + "\n\n" + _TEAM_FOLLOWUP
    assert check_answer(with_tail, team_followup=True) == []
    assert "概括性配队缺那句追问（流式侧要单独补吐）" in check_answer(CLEAN, team_followup=True)
    # 不是泛问时不许报（否则每一条普通答案都被判成缺追问）
    assert check_answer(CLEAN, team_followup=False) == []
