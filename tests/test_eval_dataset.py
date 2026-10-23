"""评测集**标注形态**的守卫（离线：只读 chunks.jsonl，不碰任何服务）。

为什么要有：`eval_retrieval.py` 的三种标注里，显式 `chunk_id` 含正文 hash ——
重分块或清洗规则一改就全体腐烂，而指标只会**虚低**，看不出是标注烂了。
片段锚点（`relevant_passages`）是为了解决这件事，但它自己也需要一条门：
「这条逐字片段在今天的语料里到底还找不找得到、是不是唯一」。

`data/` 是 gitignore 的：全新 clone 没有 chunks.jsonl → 相关用例**跳过**而不是失败
（本仓的门禁必须离线可跑，见 AGENTS「全量离线是硬约定」）。
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
CHUNKS = _ROOT / "data" / "chunks" / "chunks.jsonl"
DATASET = _ROOT / "tests" / "retrieval_eval_dataset.json"


def _harness():
    sys.path.insert(0, str(_ROOT / "tests"))
    import eval_retrieval as h  # noqa: PLC0415 —— 只有这几条用例需要它
    return h


def _rows() -> list[dict]:
    if not CHUNKS.exists():
        pytest.skip("data/chunks/chunks.jsonl 不在（全新 clone）：跳过语料相关断言")
    return [json.loads(line) for line in CHUNKS.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_片段锚点解析与选择器互不影响() -> None:
    h = _harness()
    chunks = [{"chunk_id": "甲::a", "character": "甲", "component": "声骸推荐", "tab": "",
               "breadcrumb": "甲 › a", "text": "这是一段可以当锚点的足够长的中文正文内容。"},
              {"chunk_id": "乙::b", "character": "乙", "component": "声骸推荐", "tab": "",
               "breadcrumb": "乙 › b", "text": "另一段正文。"}]
    # 命中 1 块
    assert h.resolve_passages(["足够长的中文正文内容"], chunks) == {"足够长的中文正文内容": ["甲::a"]}
    # 零命中要能**报出来**（体检靠它），不是静默给空集
    assert h.resolve_passages(["语料里没有的字串"], chunks) == {"语料里没有的字串": []}
    # 跨角色同文本 → 多条命中，交由体检提示「按多目标计」
    dup = [{"chunk_id": "甲::c", "character": "甲", "breadcrumb": "", "text": "突破材料一致的情况"},
           {"chunk_id": "乙::c", "character": "乙", "breadcrumb": "", "text": "突破材料一致的情况"}]
    assert len(h.resolve_passages(["突破材料一致的情况"], dup)["突破材料一致的情况"]) == 2
    # 显式 id 优先，其次片段，最后选择器 —— 三者互不串台
    ids, src = h.resolve_relevant({"relevant_chunk_ids": ["甲::a"],
                                   "relevant_passages": ["足够长的中文正文内容"]}, chunks)
    assert (sorted(ids), src) == (["甲::a"], "manual")
    ids, src = h.resolve_relevant({"relevant_passages": ["另一段正文"]}, chunks)
    assert (sorted(ids), src) == (["乙::b"], "passage")
    ids, src = h.resolve_relevant({"relevant_selector": {"characters": ["甲"]}}, chunks)
    assert (sorted(ids), src) == (["甲::a"], "selector")


def test_随评测集的片段锚点在语料里唯一() -> None:
    """把「锚点还有效」变成门禁：改了分块/清洗导致某条锚点找不到或命中多块 → 当场红。"""
    h = _harness()
    rows = _rows()
    ds = json.loads(DATASET.read_text(encoding="utf-8"))
    checked = 0
    for item in ds:
        for snippet in item.get("relevant_passages") or []:
            hits = h.resolve_passages([snippet], rows)[snippet]
            assert len(hits) == 1, f"{item['id']} 的锚点命中 {len(hits)} 块：{snippet[:40]!r}"
            checked += 1
    if not checked:
        pytest.skip("评测集里暂时没有片段锚点")


def test_评测集每条都有可解析的标注() -> None:
    """负样本允许两者皆空，但**不许**出现「写了标注却一条都解析不出来」的静默空标注。"""
    h = _harness()
    rows = _rows()
    for item in json.loads(DATASET.read_text(encoding="utf-8")):
        ids, src = h.resolve_relevant(item, rows)
        has_annotation = bool(item.get("relevant_chunk_ids") or item.get("relevant_selector")
                              or item.get("relevant_passages"))
        assert not has_annotation or ids, f"{item['id']}：写了标注却解析为空（source={src}）"
