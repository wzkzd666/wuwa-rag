"""角色名归一（entities.normalize_*）与配队剪枝判据（retrieve._covers 等）回归。

AGENTS.md 的实测盘点在此钉死：23 个非名册 token 五类形态、归一链每一步
「必须落回名册才算数」的设计红线、以及「覆盖剪枝按位置组、不摊平名字」
与「等价对不得双双消失」两条判据。

全量离线：`known` 一律显式传入，不碰 PG / Neo4j。
"""
from __future__ import annotations

import pytest

from wuwa_rag.knowledge.entities import normalize_character_name, normalize_team
from wuwa_rag.knowledge.retrieve import (
    _covers,
    _is_placeholder_team,
    _split_teams,
    _team_groups,
    _team_key,
)

# 覆盖五类形态所需的最小名册（与真实库形态一致：漂泊者带 `男-`、含多字/单字名）
KNOWN = {
    "维里奈", "白芷", "守岸人", "秧秧", "渊武", "凌阳", "折枝", "散华", "釉瑚",
    "卡卡罗", "吟霖", "长离", "今汐", "相里要", "莫特斐", "漂泊者-男-湮灭",
    "漂泊者-男-气动", "漂泊者-男-衍射", "漂泊者-男-导电", "卡提希娅", "夏空",
    "千咲", "莫宁", "灯灯", "仇远", "嘉贝莉娜", "洛可可", "椿", "心", "鉴心",
}


# ---------- normalize_character_name：五类实测形态 ----------

NORMALIZE_CASES = [
    # ① 漂泊者变体补 `男-`
    ("漂泊者·湮灭", "漂泊者-男-湮灭"),
    ("漂泊者-湮灭", "漂泊者-男-湮灭"),
    ("漂泊者·气动", "漂泊者-男-气动"),
    # ② 括号注释
    ("维里奈（高熟练度）", "维里奈"),
    ("莫宁（0链爱）", "莫宁"),
    ("千咲（2链绯雪）", "千咲"),
    ("夏空（进阶轴，卡提双三剑下落）", "夏空"),
    # ③ 位置标签粘连
    ("渊武其他输出", "渊武"),
    ("凌阳等主输出", "凌阳"),
    # ④ 纯位置标签 → 剔除
    ("主输出", None),
    ("副输出", None),
    # ⑤ 说明文字：前缀最长匹配（门槛 len>4）
    ("折枝 或者作为奶位配合任意队伍", "折枝"),
    # 别名表
    ("卡提", "卡提希娅"),
    ("暗主", "漂泊者-男-湮灭"),
    ("电主", "漂泊者-男-导电"),
    # 原样命中
    ("守岸人", "守岸人"),
    ("漂泊者-男-衍射", "漂泊者-男-衍射"),
]


@pytest.mark.parametrize(("tok", "want"), NORMALIZE_CASES)
def test_归一链五类形态(tok: str, want: str | None) -> None:
    assert normalize_character_name(tok, KNOWN) == want


def test_绝不猜着切_同形新词不被误切() -> None:
    # 设计红线：每一步都必须落回名册。名册没有「主输出+新词」这种形态时宁可返回 None
    assert normalize_character_name("某某某新角色输出", KNOWN) is None
    assert normalize_character_name("", KNOWN) is None
    assert normalize_character_name("   ", KNOWN) is None


def test_单字名与多字名共存时前缀匹配不误删() -> None:
    # 「鉴心」含「心」——归一必须优先整词命中，而不是被前缀规则切坏
    assert normalize_character_name("鉴心", KNOWN) == "鉴心"
    assert normalize_character_name("心", KNOWN) == "心"


# ---------- normalize_team ----------

def test_队伍串逐格归一() -> None:
    assert (
        normalize_team("漂泊者·湮灭+维里奈", KNOWN)
        == "漂泊者-男-湮灭+维里奈"
    )


def test_认不出的格整格剔除() -> None:
    assert (
        normalize_team("守岸人/维里奈/白芷+秧秧+主输出", KNOWN)
        == "守岸人/维里奈/白芷+秧秧"
    )
    assert normalize_team("守岸人+主输出+副输出", KNOWN) == "守岸人"


def test_全角分隔与说明文字() -> None:
    assert (
        normalize_team("釉瑚＋散华＋折枝 或者作为奶位配合任意队伍", KNOWN)
        == "釉瑚+散华+折枝"
    )


def test_normalize_team_幂等且空串安全() -> None:
    once = normalize_team("守岸人/维里奈/白芷+秧秧+主输出", KNOWN)
    assert normalize_team(once, KNOWN) == once
    assert normalize_team("", KNOWN) == ""


# ---------- 占位串 / 拆分 / 指纹 ----------

def test_占位串判定_去槽位后真人名不足两个() -> None:
    assert _is_placeholder_team("守岸人+主输出+副输出")
    assert _is_placeholder_team("主输出+副输出")
    assert not _is_placeholder_team("守岸人+吟霖+卡卡罗")
    assert not _is_placeholder_team("守岸人/维里奈/白芷+秧秧+卡卡罗")


def test_一条值拆多支队伍并剥前缀() -> None:
    raw = "作为副输出：折枝+今汐/卡卡罗+守岸人；作为主输出：折枝+散华/釉瑚+莫宁"
    out = _split_teams(raw)
    assert out == ["折枝+今汐/卡卡罗+守岸人", "折枝+散华/釉瑚+莫宁"]


def test_镜像队伍指纹相同() -> None:
    assert _team_key("吟霖+灯灯+守岸人") == _team_key("灯灯+吟霖+守岸人")
    assert _team_key("守岸人+吟霖/长离+卡卡罗") == _team_key("卡卡罗+吟霖/长离+守岸人")
    assert _team_key("守岸人+吟霖") != _team_key("守岸人+长离")


# ---------- _covers：逐位覆盖，不摊平 ----------

def g(team: str) -> list[frozenset[str]]:
    return _team_groups(team)


def test_模板逐位覆盖具体队() -> None:
    assert _covers(g("守岸人+吟霖/长离/散华+卡卡罗"), g("守岸人+吟霖+卡卡罗"))


def test_散华不在模板位置则不覆盖() -> None:
    # 摊平名字集合会误判成覆盖——这条就是「必须按位置组」的红线
    assert not _covers(g("守岸人+吟霖/长离+卡卡罗"), g("吟霖+守岸人+散华"))


def test_位置数不同不比较() -> None:
    assert not _covers(g("守岸人+维里奈/白芷"), g("守岸人+吟霖+卡卡罗"))


def test_等价模板互相覆盖_严格性由调用方收窄() -> None:
    a, b = g("守岸人+洛可可+椿/漂泊者-男-湮灭"), g("椿/漂泊者-男-湮灭+洛可可+守岸人")
    assert _covers(a, b) and _covers(b, a)   # 等价对两边互 True：剪枝必须用「严格」判据


def test_模板剪模板_多备选者覆盖少备选者() -> None:
    big = g("守岸人/维里奈/白芷+洛可可+椿/漂泊者-男-湮灭")
    small = g("守岸人/维里奈/白芷+洛可可+漂泊者-男-湮灭")
    assert _covers(big, small) and not _covers(small, big)
