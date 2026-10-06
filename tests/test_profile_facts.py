"""用户画像的纯规则层：事实分类与注入串。

守的是 2026-10-06 修的「画像不可更新」bug：`user_facts` 原先只增不改，
用户连说三个昵称（「我是颗粒」→「我叫小明」→「我是阿星」）会堆出三条互相矛盾的
事实并**全部**注入 prompt，模型看到「自称颗粒；姓名小明；自称阿星」只能瞎选。

修法是给 `user_facts` 加 `category` 列，同类软删旧值再插新值。这里测的是那个决策点
——`_fact_category` 判得对不对，直接决定「该覆盖的有没有覆盖、不该覆盖的有没有被误删」。

⚠️ 为什么不测 SQL 层的覆盖行为
----------------------------
`save_facts` 的覆盖是「UPDATE 旧行 valid_to=now + INSERT 新行」，依赖 PG。
离线测它只能断言 mock 被怎么调，那测的是 mock 不是 SQL——假绿比没测更糟。
SQL 层的真实行为由**真实库端到端验证**承担（连说三句应收敛为一条）。
本文件只固化不依赖 DB 的规则判定。

⚠️ 可多值事实必须返回 None（不覆盖）
----------------------------------
`_fact_category` 返回 None = 不参与覆盖，照常叠加。这是**刻意**的：
一个人可以有多个本命角色、多套配队，强行覆盖等于静默删掉用户的真实信息。
只有天然单值的类别（昵称、玩家水平）才返回具体 category。
下面 `test_可多值事实不覆盖` 就是钉这个边界——它比正样本更容易被后人「顺手修好」。
"""
from __future__ import annotations

import pytest

from wuwa_rag.services.profile import MAX_FACTS_IN_PROMPT, _fact_category, facts_to_context

# (事实文本, 期望 category)。期望值取自 2026-10-06 实测输出，非推测。
NICKNAME_FACTS = [
    "用户自称是颗粒", "用户名叫小明", "称呼为小星", "游戏ID是颗粒",
    "用户昵称阿明", "姓名：小红", "用户叫做阿星", "叫我小星", "称呼我阿明",
]
LEVEL_FACTS = ["用户自称是萌新", "用户是新手", "用户是老玩家", "用户是回归玩家", "刚入坑不久", "玩了2年"]

# 返回 None = 不覆盖。前四条是可多值事实，后两条是「看起来像但不该算」的陷阱。
NO_CATEGORY_FACTS = [
    "用户主玩守岸人",        # 可以有多个本命
    "常用配队是忌炎+今汐",    # 可以有多套配队
    "偏爱湮灭队",           # 同上
    "喜欢音感仪角色",        # 偏好可多条
    "用户自称是玩家",        # 「玩家」不是昵称（_INTRO_BAD_NICK 挡掉的泛称）
    "我是人",             # 同上，不是有效画像事实
]


@pytest.mark.parametrize("fact", NICKNAME_FACTS)
def test_昵称类事实归为_nickname(fact: str) -> None:
    assert _fact_category(fact) == "nickname", f"{fact!r} 应归 nickname"


@pytest.mark.parametrize("fact", LEVEL_FACTS)
def test_水平类事实归为_level(fact: str) -> None:
    assert _fact_category(fact) == "level", f"{fact!r} 应归 level"


@pytest.mark.parametrize("fact", NO_CATEGORY_FACTS)
def test_可多值事实不覆盖(fact: str) -> None:
    """返回 None 才不会被覆盖逻辑软删——这条比正样本更重要，见模块 docstring。"""
    assert _fact_category(fact) is None, f"{fact!r} 不应参与覆盖（会被误删真实信息）"


def test_昵称与水平分属不同类别() -> None:
    """两类都能覆盖，但必须**互不干扰**：改昵称不能把玩家水平也软删掉。"""
    assert _fact_category("用户自称是颗粒") != _fact_category("用户是老玩家")


def test_空串与异常输入不抛错() -> None:
    """画像是增益不是依赖，判分类绝不能抛异常把问答弄挂。"""
    for fact in ("", " ", "无", None):
        try:
            _fact_category(fact)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001 —— 这里就是要断言「不抛」
            pytest.fail(f"_fact_category({fact!r}) 抛了 {type(exc).__name__}: {exc}")


# ---------- 注入串 ----------

def test_facts_to_context_拼接与空值() -> None:
    assert facts_to_context([{"fact": "用户自称是颗粒"}, {"fact": "用户是老玩家"}]) == "用户自称是颗粒；用户是老玩家"
    assert facts_to_context([]) == ""


def test_facts_to_context_只读_fact_键() -> None:
    """category 是给覆盖逻辑用的，**不参与**注入串（模型不需要知道分类）。"""
    with_cat = facts_to_context([{"fact": "用户自称是颗粒", "category": "nickname"}])
    without = facts_to_context([{"fact": "用户自称是颗粒"}])
    assert with_cat == without == "用户自称是颗粒"


def test_facts_to_context_按上限截断() -> None:
    """超过 MAX_FACTS_IN_PROMPT 只取前 N 条，防止画像把 prompt 撑爆。"""
    many = [{"fact": f"事实{i}"} for i in range(MAX_FACTS_IN_PROMPT + 5)]
    out = facts_to_context(many)
    parts = out.split("；")
    assert len(parts) == MAX_FACTS_IN_PROMPT
    assert parts[0] == "事实0"
    assert parts[-1] == f"事实{MAX_FACTS_IN_PROMPT - 1}"
