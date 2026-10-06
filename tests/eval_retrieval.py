"""检索质量离线评测：量化 RRF / rerank / topk 参数改动的效果。

用法：
    uv run python tests/eval_retrieval.py                    # 默认（等同生产 topk=6）
    uv run python tests/eval_retrieval.py --topk 10          # 改送入 LLM 的条数
    uv run python tests/eval_retrieval.py --stage recall     # 只看精排前的召回
    uv run python tests/eval_retrieval.py --json             # 机器可读输出

⚠️ 这是**评测工具**，不是单元测试
--------------------------------
- 依赖真实向量索引（Chroma + BM25）与本地 reranker 权重，所以**不离线**，
  需要 `data/chroma/` 完好。
- 文件名不以 `test_` 开头，`uv run pytest` 不会收集它——缺索引时不会假失败。
- 每条 query 要跑一次 CrossEncoder 精排（约 170ms/条 × 20 候选），15 条约 1 分钟。

⚠️⚠️ 必须走生产链路，不能只调 `vector_search`（2026-10-06 踩过的坑）
------------------------------------------------------------------
生产链路是 `vector_search_tool`：**`vector_search(query)` → `rerank(query, docs, topk)`**。
两个易错点：
  ① `vector_search` **不做精排**，返回的是 RRF 融合后的 20 条中间产物；
     只调它等于在评测一个生产根本不用的结果集，`RERANK_MIN_SCORE` 那道
     「不知道」的门也完全测不到。
  ② `vector_search(question, topk)` 的 `topk` 是**每路召回深度**
     （传给 Chroma 的 `n_results`），**不是**返回条数。把它当返回条数传 6，
     会把 dense 召回从 30 砍到 6，目标块被挤出融合结果——
     实测「长离常态攻击伤害倍率」的目标块本在 RRF 第 7 位，传 topk=6 后
     recall 直接变 0，纯属自造的测量假象。
所以本脚本复刻 `vector_search_tool` 的两步，且召回阶段不传 topk。

两种标注方式（可混用）
--------------------
- `relevant_chunk_ids`：显式 chunk_id 列表。精度最高，适合单目标 query。
- `relevant_selector`：`{"characters": [...], "components": [...]}` 声明式选择器，
  评测时按语料实时解析成 chunk_id 集合。适合「该角色这一类内容都算相关」的语义型 query，
  比手写 id 列表可维护，且语料增长后自动跟上。
- 两者都空 = **负样本**：期望精排的 `RERANK_MIN_SCORE` 闸门把候选全部丢弃
  （即「库内无答案」），这是检索层该有的行为，不是 recall=0 的失败。

⚠️⚠️ 负样本必须是**真域外**，不能是「看着无关但语义沾边」（2026-10-06 踩坑）
--------------------------------------------------------------------------
最初放了「核心玩法是什么」「我很关心剧情」，期望闸门判空，实测判空率只有 0.33，
看着像检索缺陷——**其实是标注设计错了**：
  · 「核心玩法」→ 召回各角色「角色机制 › 核心机制」块，top1 rerank=0.5345；
  · 「我很关心剧情」→ 召回今汐「角色故事 › 在意之人」，top1=0.2185。
语义上确实相关，reranker 给分合理，闸门**不该**判空。这两条真正要验的是
「核心/关心 不误命中角色『心』」——那是**意图路由层**的职责，已由
`tests/test_entity_names.py` 的 NEGATIVE 覆盖，不该塞进检索层评测。
真域外样本实测 6/6 正确判空（天气 / 原神钟离圣遗物 / 快速排序 / 股票基金 /
高铁票 / Python GIL）。其中「原神里钟离的圣遗物怎么配」最有价值：术语与鸣潮
高度相似（圣遗物 ≈ 声骸）但语料里没有，专考闸门能否挡住「像但不是」。

指标
----
- `recall@k` / `precision@k`：经典两件套。
- `MRR`：第一条相关结果的排名倒数均值。对「单目标事实型 query」比 recall 更有信息量
  （目标排第 1 和排第 6 都算 recall=1，但体验天差地别）。
- `gate_empty_rate`：负样本里精排正确判空的比例。
⚠️ 绝对值受评测集规模与标注松紧影响，**只用于同一评测集下不同参数的相对比较**，
不要跨版本比绝对值。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from wuwa_rag.config import get_settings  # noqa: E402
from wuwa_rag.knowledge.index.rerank import rerank  # noqa: E402
from wuwa_rag.knowledge.retrieve import vector_search  # noqa: E402

DATASET_DEFAULT = Path(__file__).parent / "retrieval_eval_dataset.json"
CHUNKS_JSONL = _ROOT / "data" / "chunks" / "chunks.jsonl"


def load_chunks() -> list[dict]:
    rows = []
    with open(CHUNKS_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_selector(sel: dict, chunks: list[dict]) -> set[str]:
    """把声明式选择器解析成 chunk_id 集合。

    `characters` 与 `components` 是**与**关系（都要满足）；各自内部是**或**关系。
    `components` 支持子串匹配（写「声骸」即可同时命中「声骸推荐」「声骸套装推荐」）。
    """
    chars = set(sel.get("characters") or [])
    comps = sel.get("components") or []
    tabs = set(sel.get("tabs") or [])
    out: set[str] = set()
    for r in chunks:
        if chars and r.get("character") not in chars:
            continue
        if comps and not any(c in (r.get("component") or "") for c in comps):
            continue
        if tabs and (r.get("tab") or "") not in tabs:
            continue
        out.add(r["chunk_id"])
    return out


def resolve_relevant(item: dict, chunks: list[dict]) -> tuple[set[str], str]:
    """返回 (相关 chunk_id 集合, 标注来源)。来源用于分组报告。"""
    explicit = set(item.get("relevant_chunk_ids") or [])
    if explicit:
        return explicit, "manual"
    sel = item.get("relevant_selector")
    if sel:
        return resolve_selector(sel, chunks), "selector"
    return set(), "negative"


def run_query(query: str, topk: int, stage: str) -> tuple[list[str], float, int]:
    """复刻生产链路 `vector_search_tool`：召回（不传 topk）→ 精排（传 topk）。

    stage='recall' 时只看精排前，用于判断问题出在召回还是精排。
    """
    t0 = time.perf_counter()
    docs = vector_search(query)              # ⚠️ 刻意不传 topk：它是召回深度，不是返回条数
    n_recall = len(docs)
    if stage == "rerank":
        docs = rerank(query, docs, topk)
    ids = [d["chunk_id"] for d in docs]
    return ids, time.perf_counter() - t0, n_recall


def evaluate(dataset: list[dict], chunks: list[dict], topk: int, stage: str) -> dict:
    results = []
    for item in dataset:
        relevant, source = resolve_relevant(item, chunks)
        ids, elapsed, n_recall = run_query(item["query"], topk, stage)
        retrieved = list(ids)
        retrieved_set = set(retrieved)

        if relevant:
            hits = retrieved_set & relevant
            recall = len(hits) / len(relevant)
            precision = len(hits) / len(retrieved_set) if retrieved_set else 0.0
            # MRR：第一条相关结果的排名倒数
            rr = 0.0
            for rank, cid in enumerate(retrieved, 1):
                if cid in relevant:
                    rr = 1.0 / rank
                    break
            gate_ok = None
        else:
            # 负样本：期望精排闸门判空（库内无答案）
            recall = precision = rr = 0.0
            gate_ok = (len(retrieved) == 0)

        # recall 天花板 = min(topk, 相关数)/相关数：相关块比 topk 多时，
        # 召回不全**不是检索的错**（一次最多送 topk 条），是评测口径的上限。
        # 只有 recall < ceiling 才是真缺口，值得去调 RRF/rerank 参数。
        # ⚠️ 用 topk 而非 len(retrieved)：闸门过滤会让返回数 < topk，
        #    那时天花板仍是「理论最多能返回 topk 条」，不是被过滤后的实际条数。
        if relevant:
            ceiling = min(topk, len(relevant)) / len(relevant)
            gap = round(max(0.0, ceiling - recall), 4)
        else:
            ceiling = gap = 0.0

        results.append({
            "id": item["id"],
            "scene": item.get("scene", ""),
            "source": source,
            "query": item["query"],
            "n_relevant": len(relevant),
            "n_recall_pool": n_recall,
            "n_retrieved": len(retrieved),
            "recall": round(recall, 4),
            "recall_ceiling": round(ceiling, 4),
            "recall_gap": gap,
            "precision": round(precision, 4),
            "mrr": round(rr, 4),
            "first_hit_rank": next((i for i, c in enumerate(retrieved, 1) if c in relevant), None),
            "gate_ok": gate_ok,
            "elapsed_ms": round(elapsed * 1000, 1),
        })

    def _avg(rows, key):
        vals = [r[key] for r in rows]
        return round(sum(vals) / len(vals), 4) if vals else None

    positive = [r for r in results if r["n_relevant"] > 0]
    negative = [r for r in results if r["n_relevant"] == 0]
    manual = [r for r in positive if r["source"] == "manual"]
    selector = [r for r in positive if r["source"] == "selector"]

    return {
        "stage": stage,
        "topk": topk,
        "n_queries": len(results),
        "summary": {
            "positive": {
                "n": len(positive),
                "recall": _avg(positive, "recall"),
                "recall_ceiling": _avg(positive, "recall_ceiling"),
                "recall_gap": _avg(positive, "recall_gap"),
                "precision": _avg(positive, "precision"),
                "mrr": _avg(positive, "mrr"),
            },
            # 人工标注与选择器标注分开报：前者可信，后者是宽松上界，混在一起会互相污染
            "manual_only": {"n": len(manual), "recall": _avg(manual, "recall"),
                            "recall_ceiling": _avg(manual, "recall_ceiling"),
                            "recall_gap": _avg(manual, "recall_gap"),
                            "precision": _avg(manual, "precision"), "mrr": _avg(manual, "mrr")},
            "selector_only": {"n": len(selector), "recall": _avg(selector, "recall"),
                              "recall_ceiling": _avg(selector, "recall_ceiling"),
                              "recall_gap": _avg(selector, "recall_gap"),
                              "precision": _avg(selector, "precision"), "mrr": _avg(selector, "mrr")},
            "negative_gate_empty_rate": (
                round(sum(1 for r in negative if r["gate_ok"]) / len(negative), 4)
                if negative else None
            ),
            # 真缺口条数：recall < ceiling 才算，已达天花板的不计
            "n_real_gap": sum(1 for r in positive if r["recall_gap"] > 0.001),
        },
        "results": results,
    }


def print_report(rep: dict) -> None:
    s = rep["summary"]
    print()
    print("=" * 78)
    print(f"检索质量评测 | stage={rep['stage']} | topk={rep['topk']} | {rep['n_queries']} 条 query")
    print("=" * 78)

    def _row(label, d):
        if not d["n"]:
            print(f"  {label:<26} n=0（无此类用例）")
            return
        print(f"  {label:<26} n={d['n']:<3} recall={d['recall']:.4f} "
              f"(天花板 {d['recall_ceiling']:.4f}, 缺口 {d['recall_gap']:.4f}) "
              f"precision={d['precision']:.4f} MRR={d['mrr']:.4f}")

    print("【按标注可信度分组】—— 看 manual 那行做参数决策，selector 是宽松上界仅供趋势参考")
    _row("人工标注 (manual)", s["manual_only"])
    _row("选择器标注 (selector)", s["selector_only"])
    _row("正样本合计", s["positive"])
    neg = s["negative_gate_empty_rate"]
    neg_txt = f"{neg:.4f}" if neg is not None else "n/a"
    print(f"  {'负样本闸门判空率':<26} {neg_txt}"
          f"   （精排 RERANK_MIN_SCORE 把无关 query 正确判空的比例）")
    print()
    print(f"  🔍 真缺口条数: {s['n_real_gap']} / {s['positive']['n']}"
          f"   （recall < 天花板的才算；已达天花板的是 topk 口径上限，调参数也提不上去）")

    print()
    print("【按场景分组】")
    by_scene = defaultdict(list)
    for r in rep["results"]:
        by_scene[r["scene"]].append(r)
    print(f"  {'场景':<24} {'N':>3} {'recall':>8} {'MRR':>8} {'ms':>8}")
    print("  " + "-" * 58)
    for scene in sorted(by_scene):
        items = by_scene[scene]
        avg_r = sum(i["recall"] for i in items) / len(items)
        avg_m = sum(i["mrr"] for i in items) / len(items)
        avg_t = sum(i["elapsed_ms"] for i in items) / len(items)
        print(f"  {scene:<24} {len(items):>3} {avg_r:>8.4f} {avg_m:>8.4f} {avg_t:>8.1f}")

    print()
    print("【逐条明细】✗ = 真缺口（recall < 天花板，值得调参数）；"
          "△ = 已达天花板（相关块多于 topk，调参无益）")
    for r in rep["results"]:
        if r["n_relevant"] == 0:
            flag = "✓" if r["gate_ok"] else "✗"
            detail = f"闸门判空={r['gate_ok']} 实际返回={r['n_retrieved']}"
        elif r["recall_gap"] > 0.001:
            flag = "✗"
            detail = (f"recall={r['recall']:.2f} 天花板={r['recall_ceiling']:.2f} "
                      f"MRR={r['mrr']:.2f} 首位命中={r['first_hit_rank']} "
                      f"相关={r['n_relevant']} 召回池={r['n_recall_pool']}")
        elif r["recall"] < 0.999:
            flag = "△"
            detail = (f"recall={r['recall']:.2f} 已达天花板={r['recall_ceiling']:.2f} "
                      f"（相关 {r['n_relevant']} > topk {rep['topk']}，非缺陷）")
        else:
            flag = "✓"
            detail = (f"recall=1.00 MRR={r['mrr']:.2f} 首位命中={r['first_hit_rank']} "
                      f"相关={r['n_relevant']}")
        print(f"  {flag} [{r['source']:<8}] {r['query']:<26} {detail}")


def main() -> int:
    ap = argparse.ArgumentParser(description="检索质量离线评测")
    ap.add_argument("--dataset", default=str(DATASET_DEFAULT))
    ap.add_argument("--topk", type=int, default=None,
                    help="精排后送入 LLM 的条数（默认取 config.TOPK_RERANK，与生产一致）")
    ap.add_argument("--stage", choices=["recall", "rerank"], default="rerank",
                    help="recall=只测精排前的召回池；rerank=测完整生产链路（默认）")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    topk = args.topk or get_settings().TOPK_RERANK

    print(f"评测集: {args.dataset}")
    with open(args.dataset, encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"  {len(dataset)} 条 query")

    chunks = load_chunks()
    print(f"语料: {len(chunks)} 块")

    # 标注体检：显式 id 必须真实存在，否则指标恒为 0 且看不出来
    all_ids = {r["chunk_id"] for r in chunks}
    stale = [(i["id"], c) for i in dataset for c in (i.get("relevant_chunk_ids") or [])
             if c not in all_ids]
    if stale:
        print(f"\n⚠️ 发现 {len(stale)} 个标注指向不存在的 chunk_id（指标会虚低）：")
        for iid, c in stale:
            print(f"    {iid}: {c}")
        return 2

    print(f"开始评测 (stage={args.stage}, topk={topk})…")
    rep = evaluate(dataset, chunks, topk, args.stage)

    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    else:
        print_report(rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
