"""`text.py` 回归基线：把 AGENTS.md 里「实测校准」的清洗规则钉成可执行断言。

这些规则每一条都对应一次线上踩坑（残渣照抄、`275%`→`27%5%`、流式整段重复……），
改 `text.py` 前先看这里——动一条规则，对应正/负样本必须同时给出解释。

全量离线：纯字符串进、字符串出，不碰任何外部服务。
"""
from __future__ import annotations

import pytest

from wuwa_rag.text import (
    AnswerFilter,
    chunk_text,
    dedup_list_items,
    embed_input,
    fix_percent_units,
    join_breadcrumb,
    lock_focus,
    strip_dash_run,
    strip_icon_placeholders,
    strip_placeholder_table,
    strip_ref_marks,
)

# ---------- strip_placeholder_table ----------

PH_BLOCK = "| 列1 | 列2 |\n| --- | --- |\n| 忌炎 | 导电 |"


def test_占位表头整块删除而数据行保留() -> None:
    out = strip_placeholder_table(PH_BLOCK)
    assert "列1" not in out
    assert "| 忌炎 | 导电 |" in out


def test_正常表头不受影响() -> None:
    src = "| 身份 | 名称 |\n| --- | --- |\n| 奶辅 | 守岸人 |"
    assert strip_placeholder_table(src) == src


def test_占位表头幂等() -> None:
    once = strip_placeholder_table(PH_BLOCK)
    assert strip_placeholder_table(once) == once


# ---------- strip_icon_placeholders（按键图标残渣） ----------
# AGENTS.md 回归基线：5 正样本必须被清；8 负样本必须一字不动。

ICON_POSITIVE = [
    "·浮声一刹·凌霄：++；+++",
    "短按++，获取【静质量能】",
    "长按++或+，进入「质量效应」",
    "浮声一刹·凌霄：++；+++",
    "释放+++后+，衔接重击",
]


@pytest.mark.parametrize("src", ICON_POSITIVE)
def test_图标残渣正样本必须被清(src: str) -> None:
    out = strip_icon_placeholders(src)
    assert out != src          # 残渣必须被动过
    assert "++" not in out     # 连续串必须清光（单个 + 一律不动，是设计而非漏网）


ICON_NEGATIVE = [
    "26.92%+40.38%+67.29%",
    "200.80%+267.74%*3",
    "布兰特+长离",
    "长离+维里奈",
    "1cost：攻击+攻击",
    "【锯环·疾攻】+【锯环·终结】",
    "热熔伤害加成+攻击",
    "14.28%+16.66%*2",
]


@pytest.mark.parametrize("src", ICON_NEGATIVE)
def test_图标残渣负样本必须一字不动(src: str) -> None:
    assert strip_icon_placeholders(src) == src


def test_行尾加号串不得因判空短路漏清() -> None:
    # "" in "%." 恒为 True 的历史坑：`：++；+++` 必须整串清光，不留尾巴
    # （句读 ：； 是正文，保留）
    assert strip_icon_placeholders("凌霄：++；+++") == "凌霄：；"


# ---------- strip_dash_run ----------

def test_长连字符分隔换成顿号() -> None:
    src = "清宵+达妮娅+莫宁---------------------------清宵+琳奈+莫宁"
    out = strip_dash_run(src)
    assert "、" in out and "-" not in out


def test_三连字符表格分隔行不受影响() -> None:
    src = "| --- | --- |"
    assert strip_dash_run(src) == src


# ---------- lock_focus（问谁锁谁） ----------

def test_锁定或组收窄为本人() -> None:
    src = "守岸人/维里奈/白芷+吟霖/长离/散华+卡卡罗"
    assert lock_focus(src, "守岸人") == "守岸人+吟霖/长离/散华+卡卡罗"


def test_不含焦点的或组原样保留() -> None:
    src = "守岸人+维里奈/白芷+卡卡罗"
    assert lock_focus(src, "守岸人") == src


def test_文本不含焦点直接短路() -> None:
    src = "攻击/防御+生命"
    assert lock_focus(src, "守岸人") == src


def test_lock_focus_幂等() -> None:
    once = lock_focus("守岸人/维里奈/白芷+吟霖/长离/散华+卡卡罗", "守岸人")
    assert lock_focus(once, "守岸人") == once


# ---------- join_breadcrumb / embed_input / chunk_text ----------

def test_面包屑不重复拼接() -> None:
    bc = "忌炎 › 技能介绍"
    assert join_breadcrumb(bc, f"{bc}\n正文") == f"{bc}\n正文"
    assert join_breadcrumb(bc, "正文").startswith(bc)


def test_写入端与读取端共用同一份清洗() -> None:
    bc, body = "清宵 › 技能介绍", "浮声一刹：++；+++"
    assert embed_input(bc, body) == chunk_text({"breadcrumb": bc, "text": body})


# ---------- strip_ref_marks / AnswerFilter ----------

def test_数字引用标记被剥离() -> None:
    assert strip_ref_marks("暴击[1]提升，再看共鸣链[2]") == "暴击提升，再看共鸣链"


def test_中文来源标记被剥离() -> None:
    assert strip_ref_marks("彻空冥雷[图谱]是最优解[资料]") == "彻空冥雷是最优解"


def test_COST_数字串还原而非删除() -> None:
    assert strip_ref_marks("COST 组合成 [4][3][3][1][1] 的套装") == "COST 组合成 43311 的套装"


def test_含非COST档位数字的括号串按标记删除() -> None:
    # 2/5/6 不是 COST 档位 → 判为引用标记序列，整串删掉
    assert strip_ref_marks("见[2][5][6]条目") == "见条目"


def test_ref_marks_幂等() -> None:
    once = strip_ref_marks("暴击[1]，COST [4][3][3][1][1]")
    assert strip_ref_marks(once) == once


def feed_all(chunks: list[str]) -> str:
    f = AnswerFilter()
    return "".join(f.feed(c) for c in chunks) + f.flush()


def test_流式逐字符吐引用标记不泄漏() -> None:
    out = feed_all(list("暴击[1]提升"))
    assert out == "暴击提升"


def test_流式COST串跨token拼回不丢数值() -> None:
    # 三次复现过的坑：单个 [4] 到达时若不扣住，会被当引用标记删掉、restore 永远看不到串
    out = feed_all(["COST ", "[4]", "[3]", "[3]", "[1]", "[1]", " 组合"])
    assert "43311" in out


def test_流式列表行去重只丢整行相同() -> None:
    out = feed_all(["- 清宵+守岸人+尤诺\n", "- 守岸人+清宵+尤诺\n", "- 清宵+守岸人+尤诺\n"])
    assert out.count("清宵+守岸人+尤诺") == 1
    assert "守岸人+清宵+尤诺" in out  # 顺序不同 = 不同行，不许语义合并


def test_短答案不得整段重复() -> None:
    # 2026-09-30 修正的 would_append 判据回归：feed 全文再 flush 不得把正文吐两遍
    f = AnswerFilter()
    out = f.feed("我呀~我来啦~") + f.flush()
    assert out == "我呀~我来啦~"


def test_request_more_结算未完成行后可继续喂() -> None:
    f = AnswerFilter()
    got = f.feed("- 队伍A\n")
    assert got == "- 队伍A\n"
    got += f.request_more() + f.feed("后缀修正[1]") + f.flush()
    assert got == "- 队伍A\n后缀修正"


# ---------- fix_percent_units ----------

PCT_POSITIVE = [
    ("暴击伤害 275", "暴击伤害 275%"),
    ("暴击固定为80，暴击伤害固定为275", "暴击固定为80%，暴击伤害固定为275%"),
    ("暴击提升2.80", "暴击提升2.80%"),
]


@pytest.mark.parametrize(("src", "want"), PCT_POSITIVE)
def test_裸百分比数值补回单位(src: str, want: str) -> None:
    assert fix_percent_units(src) == want


PCT_NEGATIVE = [
    "暴击固定为80%，暴击伤害固定为275%",   # 已带单位（幂等）
    "1.20%",                              # 小数点不能被当成句读切开（479 行误伤教训）
    "275%",                               # 上限放开：不得回溯切成 27%5%
    "攻击力：47暴击伤害：10.8%",            # 47 后面紧跟术语 → 是固定值，不许补
    "暴击=80万",                           # 带量纲 → 不是百分比
    "暴击伤害提升两百七十五",               # 中文数字刻意不支持
    "****暴击提升2**.80%**",               # markdown 残渣：`*` 在排除集里
    "各需不同数量：攻击+攻击",              # 与百分比无关的文本
]


@pytest.mark.parametrize("src", PCT_NEGATIVE)
def test_百分比规则负样本一字不动(src: str) -> None:
    assert fix_percent_units(src) == src


# ---------- dedup_list_items ----------

def test_整行相同才丢弃() -> None:
    src = "- 清宵+莫宁+达妮娅\n- 达妮娅+莫宁+清宵\n- 清宵+莫宁+达妮娅"
    out = dedup_list_items(src)
    assert out.count("清宵+莫宁+达妮娅") == 1
    assert "达妮娅+莫宁+清宵" in out


def test_满级表五行同材料不得误杀() -> None:
    # 语义合并必误杀的反例：行首技能名不同、材料完全相同
    src = "\n".join(
        f"- {skill}满级 Lv10：贝币×10000，声骸经验×8"
        for skill in ("普攻", "重击", "共鸣技能", "共鸣解放", "变奏")
    )
    assert dedup_list_items(src) == src


def test_非列表行不参与判重() -> None:
    src = "这一句出现了两次。\n这一句出现了两次。"
    assert dedup_list_items(src) == src
