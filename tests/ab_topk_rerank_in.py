"""TOPK_RERANK_IN 参数 A/B 实测（一次性诊断脚本，非测试）。

背景：评测发现 4 条真缺口里 3 条的相关块落在 RRF 第 22 位，被 TOPK_RERANK_IN=20
切掉（只差 2 位）。本脚本对比多个取值对整个评测集的影响，用数据决定该不该改。

用法：uv run python tests/ab_topk_rerank_in.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "tests"))

from eval_retrieval import (  # noqa: E402
    DATASET_DEFAULT,
    evaluate,
    load_chunks,
)

from wuwa_rag.config import get_settings  # noqa: E402

VARIANTS = [20, 25, 30, 40]   # 20 = 当前生产值


def main() -> int:
    with open(DATASET_DEFAULT, encoding="utf-8") as f:
        dataset = json.load(f)
    chunks = load_chunks()
    print(f"评测集 {len(dataset)} 条 | 语料 {len(chunks)} 块")
    print(f"对比 TOPK_RERANK_IN: {VARIANTS}（20 为当前生产值）\n")

    s = get_settings()
    # ⚠️ 两个参数极易混淆，这里必须分清：
    #   · TOPK_RERANK_IN —— 送进 reranker 的**候选池**大小，是本次 A/B 要变的量；
    #   · TOPK_RERANK    —— 精排后**真正送进 LLM** 的条数（生产 = 6），是必须固定的对照量。
    # 首版把 original 误取成 TOPK_RERANK_IN(20) 再当 topk 传进 evaluate，等于用
    # 「返回 20 条」评测，于是 9 个相关块的 query 竟能算出 recall=1.00（topk=6 时
    # 上限只有 6/9=0.667，数学上不可能）—— 那份对比表整体作废，已重测。
    topk_llm = s.TOPK_RERANK
    original_in = s.TOPK_RERANK_IN
    print(f"固定 topk(送入 LLM)={topk_llm}，变动 TOPK_RERANK_IN，当前生产值={original_in}")

    rows = []
    for v in VARIANTS:
        s.TOPK_RERANK_IN = v
        t0 = time.perf_counter()
        rep = evaluate(dataset, chunks, topk=topk_llm, stage="rerank")
        elapsed = time.perf_counter() - t0
        sm = rep["summary"]
        rows.append((v, sm, elapsed, rep["results"]))
        print(f"  RERANK_IN={v:<3} 完成 ({elapsed:.1f}s)")

    s.TOPK_RERANK_IN = original_in

    print()
    print("=" * 96)
    print(f"{'RERANK_IN':>10} {'人工recall':>11} {'选择器recall':>13} {'合计recall':>11} "
          f"{'真缺口':>7} {'闸门判空':>9} {'MRR':>7} {'总耗时':>8}")
    print("-" * 96)
    for v, sm, el, _ in rows:
        print(f"{v:>10} {sm['manual_only']['recall']:>11.4f} "
              f"{sm['selector_only']['recall']:>13.4f} {sm['positive']['recall']:>11.4f} "
              f"{sm['n_real_gap']:>5}/{sm['positive']['n']:<1} "
              f"{sm['negative_gate_empty_rate']:>9.4f} {sm['positive']['mrr']:>7.4f} "
              f"{el:>7.1f}s")

    # 逐条对比：哪些 query 因改动而变化
    # ⚠️ 基线是 VARIANTS[0]（=20，改动前的旧值），**不是** original_in（当前生产值）。
    #    首版打印成 `RERANK_IN {original_in} → {v}`，在 original_in 已改成 25 时会输出
    #    「25 → 25」这种自相矛盾的标签，让整张对比表没法读。
    print()
    print("【逐条 recall 变化】（只列有变化的）")
    base_v = VARIANTS[0]
    base = {r["id"]: r for r in rows[0][3]}
    for v, sm, el, res in rows[1:]:
        changed = []
        for r in res:
            b = base.get(r["id"])
            if b and abs(b["recall"] - r["recall"]) > 0.001:
                changed.append((r["query"], b["recall"], r["recall"]))
        if changed:
            print(f"  RERANK_IN {base_v} → {v}:")
            for q, a, b in changed:
                arrow = "↑" if b > a else "↓"
                print(f"    {arrow} {q:<28} {a:.2f} → {b:.2f}")
        else:
            print(f"  RERANK_IN {base_v} → {v}: 无变化")
    return 0


if __name__ == "__main__":
    sys.exit(main())
