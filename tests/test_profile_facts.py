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

from wuwa_rag.services.profile import (
    MAX_FACTS_IN_PROMPT,
    _fact_category,
    _prefix_collides,
    _retract_matching,
    facts_to_context,
    negated_object,
    negated_objects,
)

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


def test_从注入串取昵称() -> None:
    from wuwa_rag.services.profile import nicknames_from_context
    assert nicknames_from_context("用户自称是颗粒") == ["颗粒"]
    assert nicknames_from_context("用户是老玩家；用户自称是阿星；主玩今汐") == ["阿星"]
    assert nicknames_from_context("用户名叫小星") == ["小星"]
    assert nicknames_from_context("喜欢周杰伦") == []
    assert nicknames_from_context("") == []
    # 水平自述不是昵称（噪声表兜住）
    assert nicknames_from_context("用户自称是新手") == []


# 端到端实测（2026-10-10）抓到的两类：判据必须**锚在子句开头**。
# 漏了锚定时，「主玩角色名为今汐」里的「名为」会被当成昵称标记 ——
# 于是模型正当引用的「今汐」也会被 unquote_mentions 剥掉引号（误剥比不剥更糟）。
@pytest.mark.parametrize("ctx,nick", [
    ("用户名为小探", ["小探"]),            # 真实抽取产物（探针账号实测入库形态）
    ("用户叫我小星", ["小星"]),
    ("用户希望被称为阿明", []),            # 没有标记词，不猜
    ("主玩角色名为今汐", []),             # 「名为」在子句中间 -> 不是昵称
    ("常用配队名为三幻神", []),            # 同上
    ("用户喜欢周杰伦", []),
    ("用户是老玩家；主玩角色名为今汐；用户自称是阿星", ["阿星"]),
])
def test_取昵称只认子句开头的标记(ctx: str, nick: list[str]) -> None:
    from wuwa_rag.services.profile import nicknames_from_context
    assert nicknames_from_context(ctx) == nick, ctx


# ---------- 否定撤销（「我不主玩今汐了」要把「主玩今汐」撤掉） ----------


@pytest.mark.parametrize("fact", [
    "用户说不喜欢今汐了",
    "用户说不主玩维里奈了",
    "用户说没在练散华",
    "用户不再用守岸人了",
])
def test_否定式能取出对象(fact: str) -> None:
    assert negated_object(fact)


@pytest.mark.parametrize("fact,expect", [
    ("用户说不喜欢今汐了", "今汐"),
    ("用户说不主玩维里奈了", "维里奈"),
    ("用户说没在练散华，还是长离顺手", "散华"),
])
def test_否定式取到的对象就是那一个名字(fact: str, expect: str) -> None:
    """对象词取错 = 撤错事实，比不撤更糟，所以按名字逐条钉。"""
    assert negated_object(fact) == expect


@pytest.mark.parametrize("fact", [
    "用户主玩今汐",                    # 肯定式：什么都不许撤
    "用户喜欢今汐",
    "用户说今汐很可爱",
    "用户主玩角色换成长离了",           # 「换成X」里的 X 是**要留下**的那个
    "用户和长离一起玩了三年",           # 「不」字不在动词前，不该命中
    "用户不谋而合地选了同一支队伍",      # 成语里的「不」不是否定
    "用户说不太喜欢椿",                 # 单字名刻意不撤：LIKE %椿% 会连累别的句子
    "",
    "用户",
])
def test_这些句子不许触发撤销(fact: str) -> None:
    assert negated_object(fact) == ""


class _Cur:
    def __init__(self, rows):
        self._rows = rows
        self.rowcount = len(rows)

    async def fetchall(self):
        return self._rows


class _Conn:
    """假 conn：记录 execute 调用。authdb 的池是 dict_row，所以这里也按 dict 行返回。"""

    def __init__(self, rows):
        self._rows = rows
        self.calls = []

    async def execute(self, sql, args=None):
        self.calls.append((" ".join(sql.split()), args))
        return _Cur(self._rows if "SELECT" in sql.upper() else [])


@pytest.mark.asyncio
async def test_撤销只删肯定式且不动新事实自己() -> None:
    conn = _Conn([
        {"id": 1, "fact": "用户主玩今汐"},
        {"id": 2, "fact": "用户不喜欢今汐"},          # 否定式旧记录：不许再删
        {"id": 3, "fact": "用户不喜欢今汐了"},          # 与本轮新事实**同文**：不许删
    ])
    new = "用户不喜欢今汐了"
    n = await _retract_matching(conn, "7", "今汐", new)
    assert n == 1, f"只应撤掉那条肯定式，实得 {n}"
    upd = [c for c in conn.calls if c[0].upper().startswith("UPDATE")]
    assert upd and upd[0][1][0] == [1], upd   # execute 的第二参是**参数元组**
    sel = conn.calls[0][1]
    assert sel[0] == "7" and sel[1].endswith("%"), f"LIKE 模式要带通配且转义：{sel}"


@pytest.mark.asyncio
async def test_LIKE_元字符不变成通配() -> None:
    """对象词取自用户原话，`%`/`_` 不转义就会「撤掉所有事实」（和会话搜索同一道坑）。"""
    conn = _Conn([])
    await _retract_matching(conn, "7", "100%_x", "用户不玩100%_x了")
    pattern = conn.calls[0][1][1]
    assert chr(92) + "%" in pattern and chr(92) + "_" in pattern, pattern   # 反斜杠显式拼，别写转义


def test_save_facts_里真的接了否定撤销() -> None:
    """`negated_object` / `_retract_matching` 单测全绿 ≠ 入库那条链接上了。

    本仓反复栽在「函数写好了、没人调用」——所以直接在源码上钉一次：
    插入之前必须先跑否定撤销，且顺序在同类覆盖之后（覆盖是「同类换值」，
    撤销是「按对象删别的类别」，两步都要在 INSERT 前完成）。
    """
    import inspect

    from wuwa_rag.services.profile import save_facts

    body = inspect.getsource(save_facts)
    assert "negated_object(fact)" in body and "_retract_matching(" in body
    assert body.index("_retract_matching(") < body.index("INSERT INTO user_facts")


def test_一句里可以否定多个对象() -> None:
    assert negated_objects("我不喜欢今汐了，也不想练散华") == ["今汐", "散华"]
    assert negated_objects("我主玩今汐") == []


def test_撤销以用户原话为准并拦下正面事实() -> None:
    """抽取器实测会把「我不喜欢今汐了，之前说主玩今汐」整句抽成正面的「主玩今汐」。

    只按 facts 判否定就完全撤不掉，所以 `save_facts` 必须**另外拿到用户原话**
    （参数 `utterance`）：① 按原话里的对象撤旧；② 把本轮抽回来的正面事实拦下不入库。
    """
    import inspect

    from wuwa_rag.services.profile import save_facts

    sig = inspect.signature(save_facts)
    assert "utterance" in sig.parameters, "撤销只能靠用户原话，这个参数不能省"
    body = inspect.getsource(save_facts)
    assert "negated_objects(utterance)" in body, "没按原话取否定对象"
    assert "_RE_NEGATION.search(fact)" in body, "没拦下被否定对象对应的正面事实"


def test_没有新事实时也要能撤销() -> None:
    """抽取器把整句否定丢成空数组是实测形态之一；`if not facts: return 0` 会把撤销一起挡掉。"""
    import inspect

    from wuwa_rag.services.profile import save_facts

    body = inspect.getsource(save_facts)
    assert "if not facts and not negated_objects(utterance)" in body, "早退没放行否定撤销"


def test_宾语提前式的否定撤不动但也不许乱撤() -> None:
    """已知边界（2026-10-24 端到端实测）：「之前不是说主玩今汐吗，现在不喜欢了」
    —— 否定词后面**没有宾语**（中文把对象放在前半句、甚至上一轮），取不到对象。

    取不到就必须什么都不撤。这不是偷懒：猜一个对象去删，错一次就是静默删掉
    用户没说过要删的事实（本模块第一条老规矩：宁可不撤，不可误撤）。
    真要覆盖这类句式得拿名册/上下文做实体绑定，代价与收益不划算，暂不做。
    """
    assert negated_objects("之前不是说主玩今汐吗，现在不喜欢了") == []
    assert negated_object("现在不喜欢了") == ""


# ---------- 前缀歧义：一个对象词不许连累另一个角色 ----------
# 名册里 `秧秧` 与 `秧秧·玄翎` 是两个角色（`漂泊者` 与四个 `漂泊者-男-*` 同理）。
# 裸 `fact LIKE '%秧秧%'` 会把「用户主玩秧秧·玄翎」一起软删 —— 撤错的是**别人**。
# 2026-10-24 由评审子智能体报出、我执行真实函数复现（三条事实里两条被判定要撤）。
@pytest.mark.parametrize("fact,obj,collides", [
    ("用户主玩秧秧·玄翎", "秧秧", True),          # 秧秧 只作为长名的前缀出现 → 判不出是谁
    ("用户主玩秧秧", "秧秧", False),               # 独立出现，照撤
    ("用户常用配队是秧秧·玄翎+守岸人", "秧秧", True),
    ("用户主玩漂泊者-男-衍射", "漂泊者", True),
    ("用户主玩漂泊者", "漂泊者", False),
    ("用户主玩今汐", "今汐", False),               # 名册里没有以「今汐」开头的更长名
    ("用户主玩卡提希娅", "卡提希娅", False),        # 归一之后：自己不算自己的前缀
    ("用户主玩今汐", "长离", False),               # obj 压根不在文本里
])
def test_前缀歧义判定(fact: str, obj: str, collides: bool) -> None:
    assert _prefix_collides(fact, obj) is collides


@pytest.mark.asyncio
async def test_撤销遇前缀歧义时不许动另一个角色() -> None:
    conn = _Conn([
        {"id": 11, "fact": "用户主玩秧秧"},
        {"id": 12, "fact": "用户主玩秧秧·玄翎"},      # **另一个角色**：LIKE 会连它一起捞进来
    ])
    n = await _retract_matching(conn, "7", "秧秧", "")
    assert n == 1, f"只应撤掉秧秧自己那条，实撤 {n} 条"
    upd = [c for c in conn.calls if c[0].upper().startswith("UPDATE")]
    assert upd and upd[0][1][0] == [11], upd


def test_简称要先归一再撤() -> None:
    """「不喜欢卡提」里的 卡提 是 卡提希娅 的别名（名册里两个都在）。

    不归一的话：拿 卡提 去匹配「用户主玩卡提希娅」，前缀判据会把它当成
    「更长的另一个名字」而整批不撤 —— 为了保护数据反而把功能做废了。
    """
    assert negated_objects("我不喜欢卡提") == ["卡提希娅"]
    assert _prefix_collides("用户主玩卡提希娅", "卡提希娅") is False


def test_拦截正面事实与撤销共用同一份前缀判据() -> None:
    """「我不喜欢秧秧，主玩秧秧·玄翎」后半句是**另一个角色**的新信息，不许被拦下不入库。

    同一语义只准一处判据（本仓反复栽在两份实现漂移）：撤销与拦截都必须走 `_prefix_collides`。
    """
    import inspect

    from wuwa_rag.services.profile import save_facts

    body = inspect.getsource(save_facts)
    assert "_prefix_collides(fact, o)" in body, "拦截正面事实没走前缀判据"
    retract = inspect.getsource(_retract_matching)
    assert "_prefix_collides(" in retract, "撤销没走前缀判据"
