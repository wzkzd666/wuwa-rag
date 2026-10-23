"""语料对账的判据用例（全离线：只喂合成状态，不碰 PG / Chroma / Neo4j）。

`wuwa-corpcheck` 存在的理由是三次真实事故，所以每条用例都照着**当时那个形状**写：
① 124 块被 `ON CONFLICT (hash)` 静默丢弃 —— 三方索引彼此一致、只有产物行数对不上；
② chunks.jsonl 被单角色增量链写成「只剩一个角色」（或 0 字节）而五步链仍报 success；
③ 36 块在 Chroma 里存的还是改版前的旧口径文本 —— 点数与行数都对得上，文本不对。
"""
from __future__ import annotations

from wuwa_rag.knowledge.corpcheck import (
    FAIL,
    WARN,
    Finding,
    _report,
    compare_by_character,
    compare_doc_texts,
    compare_graph_coverage,
    compare_id_sets,
    find_raw_drift,
)
from wuwa_rag.text import embed_input


def _levels(findings: list[Finding]) -> list[str]:
    return [f.level for f in findings]


def test_总数相同但逐角色不同也必须报() -> None:
    """事故 ① 的形状：全局计数对齐而分布错了，正是「看起来没事」的那一类。"""
    got = compare_by_character({"守岸人": 10, "卡卡罗": 2}, {"守岸人": 2, "卡卡罗": 10})
    assert len(got) == 2
    assert all(f.level == FAIL for f in got)
    assert all("≠" in f.detail for f in got)


def test_产物整角色缺席要给出可抄的补救命令() -> None:
    """事故 ② 的形状：jsonl 里整个角色没了（被单角色链覆盖）。"""
    got = compare_by_character({"卡卡罗": 10}, {"守岸人": 8, "卡卡罗": 10})
    assert len(got) == 1
    assert got[0].level == FAIL and "守岸人" in got[0].detail
    assert got[0].fix, "fail 级不一致必须直接给补救命令，不能只说『有问题』"


def test_逐角色一致时空() -> None:
    assert compare_by_character({"甲": 3, "乙": 4}, {"甲": 3, "乙": 4}) == []


def test_集合对账两个方向分开报() -> None:
    """缺 = 库里有而索引没有（召不回）；多 = 索引里有而库已删（召回旧知识）。
    两者症状相反，绝不能合成一条「数量不一致」。"""
    got = compare_id_sets({"a", "b", "c"}, {"b", "c", "d"}, "稠密索引 chunk_id", "uv run wuwa-index")
    assert len(got) == 2
    joined = " ".join(f.detail for f in got)
    assert "缺 1 个" in joined and "多 1 个" in joined
    assert all(f.fix == "uv run wuwa-index" for f in got)


def test_稠密索引文本口径() -> None:
    """事故 ③：点数、行数、id 集合全都对得上，只有存的文本是改版前那一份。"""
    bc, text = "守岸人 › 编队", "| 队伍组成 | 守岸人+吟霖-----守岸人+长离 "
    fresh = embed_input(bc, text)
    assert fresh != text                                  # 前置：这条清洗真的会改文本
    assert compare_doc_texts({"x": (bc, text)}, {"x": fresh}) == []
    got = compare_doc_texts({"x": (bc, text)}, {"x": bc + "\n" + text})   # 旧口径（没洗过）
    assert len(got) == 1 and got[0].fix == "uv run wuwa-index"
    # 索引里压根没有这个 id 时也算口径不一致（而不是静默通过）
    assert len(compare_doc_texts({"x": (bc, text)}, {})) == 1


def test_raw哈希漂移与缺席分级() -> None:
    sha = "a" * 64
    got = find_raw_drift({"守岸人": sha, "卡卡罗": sha}, {"守岸人": "b" * 64, "卡卡罗": sha})
    assert _levels(got) == [FAIL]
    assert "1 个角色" in got[0].detail and got[0].fix
    # 缺席聚合成**一条** warn：data/raw 整目录不在时逐角色刷屏，会把真正要看的 fail 埋掉
    got2 = find_raw_drift({"甲": sha, "乙": sha, "丙": sha}, {})
    assert len(got2) == 1 and got2[0].level == WARN
    assert "3 个角色" in got2[0].detail


def test_图谱名册两个方向分开分级() -> None:
    """缺角色 = 该角色只剩向量路（fail）；多出来的名字 = 抽取把位置标签当角色
    （2026-09-22 那 4 个垃圾节点），不影响当前问答，warn。"""
    got = compare_graph_coverage({"守岸人", "心"}, {"守岸人", "主输出"})
    assert len(got) == 2
    assert [f.level for f in got if "少" in f.detail] == [FAIL]
    assert [f.level for f in got if "主输出" in f.detail] == [WARN]
    assert compare_graph_coverage({"甲"}, {"甲"}) == []


def test_退出码只看fail不看warn(capsys) -> None:
    """warn 不该把 CI/脚本挡下来；fail 必须非 0 —— 否则这条命令等于没写。"""
    assert _report([Finding(WARN, "图谱角色覆盖", "多一个名字")]) == 0
    assert _report([Finding(FAIL, "逐角色块数", "产物 0 ≠ 库里 8", "uv run wuwa-chunk")]) == 1
    assert _report([]) == 0
    out = capsys.readouterr().out
    assert "uv run wuwa-chunk" in out          # 补救命令要打在结论里，不是只给人一句「有问题」
