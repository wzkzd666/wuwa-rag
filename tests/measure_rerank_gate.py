"""量一件事：`RERANK_MIN_SCORE` 该不该按槽位分档（评审报告建议，先测再决定）。

用法：`uv run python tests/measure_rerank_gate.py`（要 `data/chroma/` 完好 + reranker 权重在位，
**不离线**；文件名不以 `test_` 开头，pytest 不会收集）。

2026-10-24 实测结论：**不分档**。门槛一次都没挡掉过相关块（17 条 0 条），
而**无关的表/材料块分数(0.876)比相关的正文块最低分(0.250)还高** —— 给短块降门槛
只会放进更多无关块。完整数字与推理记在 AGENTS「审过但决定不改的四条」。
以后想动这道门，先重跑本脚本再决定。

判据不是我的感觉：把每条 relevant 块与同问的 irrelevant 块的精排分数摊开，
看「相关」与「无关」的分布在**分组之间**是否真的错开。
如果材料/数值表这类短块的相关分数仍然高于正文类无关分数，那分档门槛救不了什么，
只是多一套要维护的常数（而 AGENTS 已经记过两次「参数一改没人知道」的坑）。
"""
import json
import pathlib
import sys

sys.path.insert(0, "src")

from wuwa_rag.config import get_settings  # noqa: E402
from wuwa_rag.knowledge.retrieve import debug_recall  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
DS = json.loads((ROOT / "tests/retrieval_eval_dataset.json").read_text(encoding="utf-8"))
CHUNKS = [json.loads(line) for line in
          (ROOT / "data/chunks/chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

sys.path.insert(0, str(ROOT / "tests"))
import eval_retrieval as ev  # noqa: E402

TABLEISH = ("技能介绍", "角色突破材料", "突破材料", "共鸣链")


def group_of(cid: str) -> str:
    row = next((r for r in CHUNKS if r["chunk_id"] == cid), None)
    if row is None:
        return "缺失"
    comp = (row.get("component") or "") + "|" + (row.get("tab") or "")
    return "表/材料" if any(k in comp for k in TABLEISH) else "正文"


def main() -> None:
    s = get_settings()
    print(f"RERANK_MIN_SCORE={s.RERANK_MIN_SCORE}  TOPK_RERANK={s.TOPK_RERANK}  "
          f"TOPK_RERANK_IN={s.TOPK_RERANK_IN}")
    rel_by_group: dict[str, list[float]] = {"表/材料": [], "正文": [], "缺失": []}
    irr_by_group: dict[str, list[float]] = {"表/材料": [], "正文": [], "缺失": []}
    dropped_relevant = []
    gate_failures = []
    for item in DS:
        q = item["query"]
        relevant, src = ev.resolve_relevant(item, CHUNKS)
        d = debug_recall(q)
        top = d["reranked"]
        for i, row in enumerate(top):
            g = group_of(row["chunk_id"])
            score = row["score"]
            if score is None:
                continue
            if row["chunk_id"] in relevant:
                rel_by_group[g].append(score)
            elif i < s.TOPK_RERANK:
                irr_by_group[g].append(score)
        if relevant:
            hits = [r for r in top if r["chunk_id"] in relevant]
            if not d["gate"]["passed"] and hits:
                gate_failures.append((q, d["gate"]["top1"], len(hits)))
            elif hits and hits[0]["score"] is not None and hits[0]["score"] < s.RERANK_MIN_SCORE:
                dropped_relevant.append((q, hits[0]["score"]))

    def rng(xs):
        return f"n={len(xs)} min={min(xs):.3f} 中位={sorted(xs)[len(xs)//2]:.3f} max={max(xs):.3f}" if xs else "n=0"

    print("\n【相关块精排分数，按分组】")
    for g in rel_by_group:
        print(f"  {g:8} {rng(rel_by_group[g])}")
    print("【无关但进了 top6 的块，按分组】")
    for g in irr_by_group:
        print(f"  {g:8} {rng(irr_by_group[g])}")
    print(f"\n相关块被门槛挡掉的问句：{len(dropped_relevant)} {dropped_relevant[:5]}")
    print(f"闸门判空但库里其实有相关的问句（=真误拒）：{len(gate_failures)} {gate_failures[:5]}")


if __name__ == "__main__":
    main()
