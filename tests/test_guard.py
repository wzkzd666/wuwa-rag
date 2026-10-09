"""`dialog/guard.py` 防复读闸回归：退化样本必须命中，真实表格/材料块必须不命中。

AGENTS.md 的基线口径：「退化样本必须命中且截断、干净答案/真实补料块/技能 13 行/
材料 11 行负样本必须**不**命中」。这里用合成样本复刻两类退化形态
（整句重复 / 周期块循环），并钉死三条设计红线：

- 计数标记（`*N`/`×N`）只用于周期比较、不参与精确计数——否则「贝币×5000/×10000」误杀；
- 表格行（`|` 开头）与短碎句天然重复，必须豁免；
- 全文扫描（find_degenerate_start）与尾部窗口（LoopGuard.feed）双档都要能抓到
  「退化块后面还跟着正常内容」的形态——只查尾部曾让 trim 变成空转。
"""
from __future__ import annotations

from wuwa_rag.dialog.guard import LoopGuard, find_degenerate_start, trim_loop

# ---------- 整句重复 ----------

LOOP_SENT = "我曾在雪原上见过一只会飞的雪绒，它替我挡了一顿风雪。"


def _repeat(sentence: str, n: int, tail: str = "") -> str:
    return (sentence + "\n") * n + tail


def test_同一长句出现三次必须命中() -> None:
    text = _repeat(LOOP_SENT, 3)
    cut = find_degenerate_start(text)
    assert cut == 0                       # 从首次出现处截断


def test_同一长句出现两次属允许() -> None:
    assert find_degenerate_start(_repeat(LOOP_SENT, 2)) is None


def test_trim_保留前文并补省略号() -> None:
    text = _repeat(LOOP_SENT, 3, tail="")
    text = "开场白。" + text
    out = trim_loop(text)
    assert out.startswith("开场白。")
    assert out.endswith("……")
    assert LOOP_SENT not in out


# ---------- 周期块循环（清宵配队形态） ----------

def _cyclic_teams(n_cycles: int = 5) -> str:
    """6 行一组的目标配队循环 n 遍，每行带递增计数后缀（*1…×N）。

    后缀让每行**看起来**都唯一——精确判重抓不到，这正是线上漏网形态。
    """
    base = [
        "守岸人 + 尤诺",
        "卡卡罗 + 长离",
        "忌炎 + 莫特斐",
        "今汐 + 仇远",
        "椿 + 露西",
        "折枝 + 散华",
    ]
    lines: list[str] = []
    k = 1
    for _ in range(n_cycles):
        for b in base:
            lines.append(f"- {b}*{k}")
            k += 1
    return "\n".join(lines) + "\n具体循环轴嘛，我再说两句收尾。"


def test_周期块循环五遍必须命中且从首块截断() -> None:
    text = _cyclic_teams(5)
    cut = find_degenerate_start(text)
    assert cut is not None
    assert text[cut:].startswith("- 守岸人 + 尤诺*1")   # 切在**最早**一块


def test_退化块后跟正常内容也能截断() -> None:
    # 只查尾部窗口的旧实现在这里会算不出截断点（本轮真跑回归抓到的 bug）
    text = _cyclic_teams(3) + "循环轴就这些啦~"
    out = trim_loop(text)
    assert "循环轴就这些啦~" not in out
    assert out == "……" or not out.startswith("- 守岸人")


def test_三遍是周期判定的最低门槛() -> None:
    # 两遍（12 行）不构成退化——真实答案里「同一组队伍说两遍」的合法场景存在
    two = _cyclic_teams(2).split("具体循环轴")[0]
    assert find_degenerate_start(two) is None


# ---------- 负样本：真实内容不得误杀 ----------

def test_材料计数标记差异不得被归一误杀() -> None:
    # 三行只差计数后缀：归一后同一行 → 周期块 set 只剩 1 种 → 放行给精确计数；
    # 而精确计数用**未归一**的原 key 且这些短行 <min_chars，两条规则都不该命中。
    text = "\n".join(["- 贝币×5000", "- 贝币×10000", "- 贝币×15000"] * 3)
    assert find_degenerate_start(text) is None


def test_满级表五行同材料仅行首不同不得误杀() -> None:
    # 「满级 Lv10 一档」真实形态：材料完全相同、只有技能名不同——语义合并必误杀
    text = "\n".join(
        f"- {s}满级：贝币×10000，残响声骸×8"
        for s in ("普攻", "重击", "共鸣技能", "共鸣解放", "变奏")
    )
    assert find_degenerate_start(text) is None


def test_技能13行表格天然重复行必须豁免() -> None:
    row = "| Lv10 | 26.92%+40.38%+67.29% | 128 | 10 |"
    text = "技能倍率如下：\n" + "\n".join([row] * 13)
    assert find_degenerate_start(text) is None


def test_短碎句不参与判重() -> None:
    text = "嗯~\n" * 6 + "好的。\n" * 6
    assert find_degenerate_start(text) is None


def test_干净长答案不命中() -> None:
    text = (
        "我呀~是爱弥斯。\n我的角色故事要从雪原说起。\n"
        "共鸣解放是我的招牌，倍率逐行都在资料里。\n"
        "配队方面守岸人+吟霖+卡卡罗是一队。\n"
        "声骸我推荐彻空冥雷，COST 43311。\n"
    )
    assert find_degenerate_start(text) is None


# ---------- 流式 LoopGuard ----------

def test_流式feed在第三次出现时触发() -> None:
    g = LoopGuard()
    assert g.feed(LOOP_SENT + "\n") is None
    assert g.feed(LOOP_SENT + "\n") is None
    hit = g.feed(LOOP_SENT + "\n")
    assert hit is not None and LOOP_SENT[:8] in hit


def test_流式feed逐token喂也能抓到周期循环() -> None:
    g = LoopGuard()
    triggered = False
    for ch in _cyclic_teams(3):
        if g.feed(ch) is not None:
            triggered = True
            break
    assert triggered


def test_流式表格行不触发() -> None:
    g = LoopGuard()
    row = "| 一阶 | 贝币×5000 | 声骸经验×8 |\n"
    for _ in range(10):
        assert g.feed(row) is None
