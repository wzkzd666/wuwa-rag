"""上下文体积预算 + 确定性补料的覆盖面。

两组判据，各自对应一个真实缺口：

① 上下文体积预算（`prompt._fit`）
   此前只按轮数截历史（MAX_HISTORY_TURNS），没有体积上限。历史里存的是答案全文，
   叠上 6 条文档与补料表后，长轮次会贴到 `LLM_NUM_CTX - MAX_TOKENS` 的输入预算，
   而 Ollama 超出 num_ctx 是从**头部静默截断**（调用侧没有信号）。
   这里钉住：按「历史 → 文档尾部 → 图谱 → 联网」的顺序裁剪；补料与「本问解析」
   不可裁；裁了什么、裁不下去都要留日志。

② 补料覆盖多角色 + 口语页签 + 技能等级（`graph._value_blocks` 一族）
   对比类问句此前只补 `characters[0]`，第二个角色的数值表只能靠向量召回，
   而这类表恰恰是召不回来的。另外问句写「大招 / 普攻」时此前选不中页签，
   点名「7级」时也只会取满级列。

全部离线：`fetch_chunks` 被 stub 成内存数据，不碰 Chroma。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from wuwa_rag.config import Settings
from wuwa_rag.dialog import graph as graph_mod
from wuwa_rag.dialog import prompt as prompt_mod


# ---------- ① 上下文体积预算 ----------

def _history(n: int, size: int = 40) -> list[dict]:
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": chr(65 + i) * size}
        for i in range(n)
    ]


def _docs(n: int, size: int = 40) -> list[dict]:
    return [{"text": f"DOC{i}" + "x" * size, "breadcrumb": f"来源{i}"} for i in range(n)]


def _base_state(**extra) -> dict:
    state = {"question": "问一句", "characters": [], "slots": []}
    state.update(extra)
    return state


def test_预算内不裁剪():
    state = _base_state(history=_history(2), docs=_docs(2))
    out = prompt_mod.build_context(state, ["## 技能数值表（Lv 10，原文照抄）\n- 伤害：100%"],
                                   budget=10_000)
    assert "## 对话历史" in out and "A" * 40 in out and "B" * 40 in out
    assert "## 参考文档" in out and "DOC0" in out and "DOC1" in out
    assert "## 技能数值表（Lv 10，原文照抄）" in out


def test_历史先丢最老的(monkeypatch):
    msgs: list[str] = []
    monkeypatch.setattr(prompt_mod.log, "info", lambda fmt, *a, **k: msgs.append(fmt % a))
    state = _base_state(history=_history(4))
    full = prompt_mod.build_context(state, budget=10_000)
    out = prompt_mod.build_context(state, budget=len(full) - 1)
    assert "A" * 40 not in out          # 最老的一条先走
    assert "B" * 40 in out and "D" * 40 in out
    assert any("上下文裁剪" in m and "对话历史×1" in m for m in msgs)


def test_文档从尾部丢():
    state = _base_state(docs=_docs(3))
    full = prompt_mod.build_context(state, budget=10_000)
    out = prompt_mod.build_context(state, budget=len(full) - 1)
    assert "DOC0" in out and "DOC1" in out
    assert "DOC2" not in out            # 重排名次最低的那份先走


def test_历史裁完才动文档():
    hist, docs = _history(2, 30), _docs(2, 30)
    full = prompt_mod.build_context(_base_state(history=hist, docs=docs), budget=10_000)
    no_hist = prompt_mod.build_context(_base_state(docs=docs), budget=10_000)
    # 预算比「只有文档」还小 1 字符：历史必须全裁掉，随后才轮到文档尾部
    out = prompt_mod.build_context(_base_state(history=hist, docs=docs),
                                   budget=len(no_hist) - 1)
    assert "## 对话历史" not in out
    assert "DOC0" in out and "DOC1" not in out
    assert len(full) > len(no_hist)


def test_补料与本问解析不可裁且超限留痕(monkeypatch):
    warns: list[str] = []
    monkeypatch.setattr(prompt_mod.log, "warning",
                        lambda fmt, *a, **k: warns.append(fmt % a))
    extra = ["## 技能数值表（Lv 10，原文照抄）\n"
             + "\n".join(f"- 数值{i}：{i}%" for i in range(40))]
    state = _base_state(characters=["卡卡罗"], slots=["技能"])
    out = prompt_mod.build_context(state, extra, budget=50)
    assert extra[0] in out
    assert "## 本问解析" in out
    assert any("仍超预算" in w for w in warns)


def test_prompt_尺寸超上限写warning(monkeypatch):
    warns: list[str] = []
    monkeypatch.setattr(prompt_mod.log, "warning",
                        lambda fmt, *a, **k: warns.append(fmt % a))
    monkeypatch.setattr(prompt_mod, "get_settings",
                        lambda: SimpleNamespace(PROMPT_MAX_CHARS=100))
    out = prompt_mod.build_prompt("资料" * 100, "问一句")
    assert len(out) > 100
    assert any("prompt 超预算" in w for w in warns)


def test_prompt_预算内写info(monkeypatch):
    infos: list[str] = []
    monkeypatch.setattr(prompt_mod.log, "info", lambda fmt, *a, **k: infos.append(fmt % a))
    monkeypatch.setattr(prompt_mod, "get_settings",
                        lambda: SimpleNamespace(PROMPT_MAX_CHARS=100_000))
    prompt_mod.build_prompt("资料", "问一句")
    assert any("prompt 尺寸" in m for m in infos)


def test_预算默认值():
    fields = Settings.model_fields
    assert fields["PROMPT_CONTEXT_CHARS"].default == 4200
    assert fields["PROMPT_MAX_CHARS"].default == 6000
    assert fields["SUPPLEMENT_MAX_CHARACTERS"].default == 2
    assert fields["SUPPLEMENT_MAX_CHARS"].default == 4000


# ---------- ② 技能页签 / 等级 / 多角色补料 ----------

def _skill_chunks(char: str, tabs: tuple[str, ...]) -> list[dict]:
    return [
        {
            "chunk_id": f"{char}-{tab}",
            "character": char,
            "component": "技能介绍",
            "tab": tab,
            "text": ("| 等级 | Lv 1 | Lv 7 | Lv 10 |\n"
                     "| --- | --- | --- | --- |\n"
                     f"| {tab}伤害 | 10% | 70% | 100% |"),
        }
        for tab in tabs
    ]


def _material_chunks(char: str) -> list[dict]:
    return [{
        "chunk_id": f"{char}-突破",
        "character": char,
        "component": "角色突破材料",
        "breadcrumb": f"{char} › 角色突破材料 › 一阶突破",
        "text": "| 一阶突破 | 低频隧花声核x4 贝币x5000 |",
    }]


def _echo_chunks(char: str) -> list[dict]:
    return [{
        "chunk_id": f"{char}-声骸",
        "character": char,
        "component": "声骸推荐",
        "text": "套装：彻空冥雷\n武器：时和岁稔",
    }]


FAKE_CHUNKS: dict[tuple[str, str | None], list[dict]] = {}
for _char in ("卡卡罗", "长离", "守岸人"):
    FAKE_CHUNKS[(_char, "技能介绍")] = _skill_chunks(_char, ("共鸣解放", "共鸣技能"))
    FAKE_CHUNKS[(_char, "角色突破材料")] = _material_chunks(_char)
    FAKE_CHUNKS[(_char, None)] = _echo_chunks(_char)


def _fake_fetch(char: str, component: str | None = None) -> list[dict]:
    return [dict(c) for c in FAKE_CHUNKS.get((char, component), [])]


@pytest.fixture
def fake_chunks(monkeypatch):
    monkeypatch.setattr(graph_mod, "fetch_chunks", _fake_fetch)
    return FAKE_CHUNKS


def test_满级列按等级选择():
    text = ("| 名称 | Lv 1 | Lv 7 | Lv 10 |\n"
            "| --- | --- | --- | --- |\n"
            "| 伤害 | 10% | 70% | 100% |")
    assert graph_mod._max_level_rows(text, 7) == [("伤害", "70%")]
    assert graph_mod._max_level_rows(text) == [("伤害", "100%")]


def test_材料表的等级标签不被当成材料():
    text = ("| LV.2 | LV.10 |\n"
            "| --- | --- |\n"
            "| 低频隧花声核x2 | 高频隧花声核x3 |")
    assert graph_mod._max_level_rows(text, 10) == []
    assert graph_mod._max_level_material(text) == ["高频隧花声核×3"]

    spaced = ("| Lv 2 | Lv 10 |\n"
              "| --- | --- |\n"
              "| 低频隧花声核x2 | 高频隧花声核x3 |")
    assert graph_mod._max_level_material(spaced) == ["高频隧花声核×3"]
    assert graph_mod._material_items("| Lv 3 | 低频隧花声核x2 |") == ["低频隧花声核×2"]


def test_前置说明与表头挤在同一行也能识别():
    text = ("**技能数据**| 等级 | Lv 1 | Lv 7 | Lv 10 |\n"
            "| --- | --- | --- | --- |\n"
            "| 伤害 | 10% | 70% | 100% |")
    assert graph_mod._max_level_rows(text, 7) == [("伤害", "70%")]


def test_带点的技能等级表头也能定位():
    text = ("| 技能等级 | Lv.1 | Lv.7 | Lv.10 |\n"
            "| --- | --- | --- | --- |\n"
            "| 伤害 | 10% | 70% | 100% |")
    assert graph_mod._max_level_rows(text, 7) == [("伤害", "70%")]
    assert graph_mod._max_level_rows(text, 10) == [("伤害", "100%")]


def test_占位表头不会被当成数值行():
    text = ("**技能数据**| 列1 | 列2 | 列3 | 列4 |\n"
            "| --- | --- | --- | --- |\n"
            "| 伤害 | 10% | 70% | 100% |")
    assert graph_mod._max_level_rows(text) == []
    assert graph_mod._fragment_level_rows([text]) == []


def test_图标占位不进入数值表():
    text = ("| 等级 | Lv 1 | Lv 10 |\n"
            "| --- | --- | --- |\n"
            "| 伤害 | 10% | 100% |\n"
            "| 特效 | [图] | [图] |")
    assert graph_mod._max_level_rows(text) == [("伤害", "100%")]


def test_分块切开的技能表按表宽拼回整表():
    header = ("卡卡罗 › 技能介绍 › 共鸣技能\n"
              "**技能数据**| 等级 | Lv 1 | Lv 7 | Lv 10 |\n"
              "| --- | --- | --- | --- |\n"
              "| 第一段伤害 | 10% | 70% | 100% |")
    tail = ("卡卡罗 › 技能介绍 › 共鸣技能\n"
            "| 列1 | 列2 | 列3 | 列4 |\n"
            "| --- | --- | --- | --- |\n"
            "| 第二段伤害 | 20% | 80% | 110% |")
    assert graph_mod._fragment_level_rows([header, tail], 7) == [
        ("第一段伤害", "70%"), ("第二段伤害", "80%")]


def test_续块里的窄表不会混进等级表():
    header = ("| 等级 | Lv 1 | Lv 7 | Lv 10 |\n"
              "| --- | --- | --- | --- |\n"
              "| 伤害 | 10% | 70% | 100% |")
    narrow = ("| --- | --- |\n"
              "| 暴击 | 5% |")
    assert graph_mod._fragment_level_rows([header, narrow], 7) == [("伤害", "70%")]


def test_技能等级解析():
    assert graph_mod._requested_skill_level("共鸣解放7级倍率") == 7
    assert graph_mod._requested_skill_level("Lv 7 的共鸣解放") == 7
    assert graph_mod._requested_skill_level("共鸣解放伤害倍率") == 10
    # 角色等级不是技能等级，不能被夹成 10 以外的值，也不改变默认满级
    assert graph_mod._requested_skill_level("80级突破要什么") == 10


def test_口语页签别名():
    assert graph_mod._named_tabs("大招伤害") == ["共鸣解放"]
    assert graph_mod._named_tabs("普攻和E技能") == ["常态攻击", "共鸣技能"]
    assert graph_mod._named_tabs("技能倍率") == []


async def test_口语页签与点名等级真的取到那一列(fake_chunks):
    sq = "大招7级倍率"
    block = await graph_mod._skill_value_block(
        "卡卡罗", sq, [], level=graph_mod._requested_skill_level(sq))
    assert "## 技能数值表（Lv 7，原文照抄）" in block
    assert "### 共鸣解放" in block
    assert "- 共鸣解放伤害：70%" in block
    assert "常态攻击" not in block and "共鸣技能" not in block


async def test_技能表被分块切开时整表都补进块(monkeypatch):
    chunks = [
        {
            "character": "卡卡罗",
            "component": "技能介绍",
            "tab": "共鸣技能",
            "text": ("卡卡罗 › 技能介绍 › 共鸣技能\n"
                     "**技能数据**| 等级 | Lv 1 | Lv 7 | Lv 10 |\n"
                     "| --- | --- | --- | --- |\n"
                     "| 第一段伤害 | 10% | 70% | 100% |"),
        },
        {
            "character": "卡卡罗",
            "component": "技能介绍",
            "tab": "共鸣技能",
            "text": ("卡卡罗 › 技能介绍 › 共鸣技能\n"
                     "| 列1 | 列2 | 列3 | 列4 |\n"
                     "| --- | --- | --- | --- |\n"
                     "| 第二段伤害 | 20% | 80% | 110% |"),
        },
    ]
    monkeypatch.setattr(
        graph_mod, "fetch_chunks",
        lambda char, component=None: [dict(c) for c in chunks]
        if component == "技能介绍" else [])
    block = await graph_mod._skill_value_block("卡卡罗", "共鸣技能7级", [], level=7)
    assert "### 共鸣技能" in block
    assert "- 第一段伤害：70%" in block
    assert "- 第二段伤害：80%" in block


async def test_对比问句两个角色都补技能表(fake_chunks):
    state = {
        "question": "卡卡罗和长离的技能倍率",
        "search_query": "卡卡罗和长离的技能倍率",
        "characters": ["卡卡罗", "长离"],
        "slots": ["技能"],
        "docs": [],
    }
    blocks = await graph_mod._value_blocks(state)
    assert len(blocks) == 2
    assert "技能数值表·卡卡罗" in blocks[0]
    assert "技能数值表·长离" in blocks[1]


async def test_补料角色数受配置限制(fake_chunks, monkeypatch):
    monkeypatch.setattr(graph_mod.setting, "SUPPLEMENT_MAX_CHARACTERS", 2)
    state = {
        "question": "卡卡罗、长离和守岸人的技能倍率",
        "search_query": "卡卡罗、长离和守岸人的技能倍率",
        "characters": ["卡卡罗", "长离", "守岸人"],
        "slots": ["技能"],
        "docs": [],
    }
    blocks = await graph_mod._value_blocks(state)
    assert len(blocks) == 2
    assert all("守岸人" not in b for b in blocks)


async def test_补料超上限时整块跳过并留日志(fake_chunks, monkeypatch):
    state = {
        "question": "卡卡罗和长离的技能倍率与突破材料",
        "search_query": "卡卡罗和长离的技能倍率与突破材料",
        "characters": ["卡卡罗", "长离"],
        "slots": ["技能", "突破材料"],
        "docs": [],
    }
    monkeypatch.setattr(graph_mod.setting, "SUPPLEMENT_MAX_CHARS", 10_000)
    full = await graph_mod._value_blocks(state)
    assert len(full) == 4                     # 两角色 × (技能表 + 材料表)

    msgs: list[str] = []
    monkeypatch.setattr(graph_mod.log, "warning", lambda fmt, *a, **k: msgs.append(fmt % a))
    monkeypatch.setattr(graph_mod.setting, "SUPPLEMENT_MAX_CHARS", len(full[0]) + 1)
    blocks = await graph_mod._value_blocks(state)
    # 半张表比没有表更糟：只放得下第一块时，其余整块跳过，绝不截断后塞进去
    assert blocks == [full[0]]
    assert any("整块跳过" in m and "长离" in m for m in msgs)


async def test_对比问句材料表按角色分块(fake_chunks):
    state = {
        "question": "卡卡罗和长离的突破材料",
        "search_query": "卡卡罗和长离的突破材料",
        "characters": ["卡卡罗", "长离"],
        "slots": ["突破材料"],
        "docs": [],
    }
    blocks = await graph_mod._value_blocks(state)
    assert len(blocks) == 2
    assert "突破材料表·卡卡罗" in blocks[0]
    assert "突破材料表·长离" in blocks[1]


async def test_声骸块多角色时改措辞(fake_chunks):
    single = await graph_mod._echo_block("卡卡罗", ["声骸"], multi=False)
    multi = await graph_mod._echo_block("卡卡罗", ["声骸"], multi=True)
    assert "本问只答" in single
    assert "本问只答" not in multi
    assert "声骸 / 武器推荐表·卡卡罗" in multi
