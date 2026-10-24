"""问题索引（给每块生成若干问句再嵌入）的可行性小样本。

2026-10-24 的实测动机：先分清评测集里的真缺口是**哪一种**，再谈要不要加索引 ——

  A 真没召回：dense 与 sparse 都没进展开后的 RRF 前 `TOPK_RERANK_IN` 名；
  B 被截断：进了 RRF，但名次在 `TOPK_RERANK_IN` 之后，被切掉。

问题索引只对**问法与陈述句对不上**有效；若是 B，那是对排序/截断的问题，
多一路索引治不了。判据用**不截断**的 `_rrf([dense_ids, sparse_ids])` 全量名次，
不能拿 `recall_stages()["fused"]`（它已经截到 20 名，`rank > 20` 永远看不到）。

对每个缺口块再量一件事：让 8b 生成 3 个玩家可能问的问题，把它们嵌入后，
按「假设这块以该相似度落进 dense」重算一遍完整 RRF，看能否进前 `TOPK_RERANK_IN`：

  best_q_cos = max cos(用户问句, 生成问句向量)
  假设 dense 距离 = 1 − best_q_cos，假设名次 = 1 + 现有里比它近的个数
  若假设名次 ≤ TOPK_DENSE：把它插进 dense 列表（截到 30）→ 重算 RRF → hyp_rank
  hyp_rank ≤ TOPK_RERANK_IN ⇒ 问题索引能把它推进候选池

但进池不等于进答案：再把它**单独加进该查询的 20 条候选**跑一次真精排，与基线 top6
比对**相关块净增数**。这一步是决定性的 —— 块挤进第 6 名会顶掉另一条相关块，
只看「进没进前 6」会把「一换一」读成「救回来了」（实测「心·四阶突破」就是这种）。

额外的 sanity（不决定结论）：生成问句自己能否召回它的块 —— 若连这都召不回，
说明 8b 生成的问题与块不对应，那是生成质量问题，不能算到索引头上。

用法（需要 Chroma/BM25 完好 + Ollama 在跑；不以 test_ 开头，pytest 不收）：
    uv run python tests/measure_question_index.py
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import eval_retrieval as h  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from wuwa_rag.config import get_settings  # noqa: E402
from wuwa_rag.core.llm import get_tool_llm  # noqa: E402
from wuwa_rag.knowledge.index.rerank import rerank  # noqa: E402
from wuwa_rag.knowledge.retrieve import (  # noqa: E402
    _collection,
    _embedder,
    _rrf,
    recall_stages,
)

MAX_CHUNKS = 6

_ASK = (
    "下面是一段游戏资料。请写出 3 个玩家最可能用这段资料来回答的中文问题。\n"
    "每行一个，不要编号，不要解释，不要输出其它文字。\n\n资料：\n{text}"
)


def _cos(a, b) -> float:
    # 建索引与查询侧都 normalize_embeddings=True（见 embeddings.py），内积即余弦
    return sum(x * y for x, y in zip(a, b))


def _diagnose() -> tuple[list[dict], int, int]:
    """按「不截断的 RRF 名次」把相关块分成真没召回 / 被截断两类。"""
    cfg = get_settings()
    chunks = h.load_chunks()
    dataset = json.loads((ROOT / "tests/retrieval_eval_dataset.json").read_text(encoding="utf-8"))
    gaps: list[dict] = []
    seen: set[str] = set()
    missed = cut = 0
    for item in dataset:
        relevant, _src = h.resolve_relevant(item, chunks)
        if not relevant:
            continue
        st = recall_stages(item["query"])
        dense = [(d["chunk_id"], d["distance"]) for d in st["dense"]]
        sparse_ids = [x["chunk_id"] for x in st["sparse"]]
        dense_ids = [cid for cid, _ in dense]
        fused_full = _rrf([dense_ids, sparse_ids])
        rank_of = {cid: i + 1 for i, cid in enumerate(fused_full)}
        dense_rank = {cid: i + 1 for i, (cid, _) in enumerate(dense)}
        sparse_rank = {cid: i + 1 for i, cid in enumerate(sparse_ids)}
        cutoff = 1.0 - (dense[-1][1] if dense else 1.0)
        for cid in sorted(relevant):
            r = rank_of.get(cid)
            if r is not None and r <= cfg.TOPK_RERANK_IN:
                continue
            if cid in seen:
                continue
            seen.add(cid)
            if r is None:
                missed += 1
            else:
                cut += 1
            gaps.append({
                "query": item["query"],
                "chunk_id": cid,
                "relevant": relevant,
                "dense": dense,
                "sparse_ids": sparse_ids,
                "dense_rank": dense_rank.get(cid),
                "sparse_rank": sparse_rank.get(cid),
                "rrf_rank": r,
                "cutoff": cutoff,
            })
    return gaps, missed, cut


async def _questions(text: str) -> list[str]:
    llm = get_tool_llm()
    msg = await llm.ainvoke([
        SystemMessage(content="你是检索评测助手。"),
        HumanMessage(content=_ASK.format(text=text[:600])),
    ])
    out: list[str] = []
    for line in str(msg.content).splitlines():
        s = line.strip().lstrip("-·*0123456789.、）) ").strip()
        if s:
            out.append(s)
    return out[:3]


async def main() -> int:
    cfg = get_settings()
    gaps, missed, cut = _diagnose()
    print(f"缺口块 {len(gaps)} 个（去重后）：真没召回 {missed} / 被 RRF 前 "
          f"{cfg.TOPK_RERANK_IN} 截掉 {cut}")
    if not gaps:
        print("没有缺口，无需评估问题索引。")
        return 0
    for g in gaps:
        print(f"  [{g['dense_rank'] or '-'}/{g['sparse_rank'] or '-'}/"
              f"{g['rrf_rank'] or '-'}] {g['query']} → {g['chunk_id']}")

    probe = gaps[:MAX_CHUNKS]
    col = _collection()
    got = col.get(ids=[g["chunk_id"] for g in probe],
                  include=["documents", "embeddings", "metadatas"])
    docs = {i: d for i, d in zip(got["ids"], got["documents"])}
    embs = {i: list(e) for i, e in zip(got["ids"], got["embeddings"])}
    metas = {i: m for i, m in zip(got["ids"], got["metadatas"])}
    emb = _embedder()

    in_pool_n = helped = 0
    for g in probe:
        cid = g["chunk_id"]
        qv = emb.embed_query(g["query"])
        text_cos = _cos(qv, embs[cid])
        qs = await _questions(docs[cid] or "")
        best_q, best_qv = -1.0, None
        for s in qs:
            sv = emb.embed_query(s)
            c = _cos(qv, sv)
            if c > best_q:
                best_q, best_qv = c, sv

        hyp_rank = None
        if best_q > 0:
            hyp_dense_rank = 1 + sum(1 for _, d in g["dense"] if d < 1.0 - best_q)
            if hyp_dense_rank <= cfg.TOPK_DENSE:
                ids = [x for x, _ in g["dense"]]
                ids.insert(hyp_dense_rank - 1, cid)
                fused = _rrf([ids[: cfg.TOPK_DENSE], g["sparse_ids"]])
                hyp_rank = fused.index(cid) + 1 if cid in fused else None
        in_pool = hyp_rank is not None and hyp_rank <= cfg.TOPK_RERANK_IN
        in_pool_n += int(in_pool)
        rr_rank = rr_score = sixth = None
        net = 0
        displaced: list[str] = []
        if in_pool:
            pool = recall_stages(g["query"])["docs"]
            base = rerank(g["query"], list(pool), cfg.TOPK_RERANK)
            base_hits = {x["chunk_id"] for x in base} & g["relevant"]
            cand = list(pool) + [
                {"chunk_id": cid, "text": docs[cid] or "", **metas[cid]}
            ]
            ranked = rerank(g["query"], cand, len(cand))
            ids = [x["chunk_id"] for x in ranked]
            rr_rank = ids.index(cid) + 1
            rr_score = ranked[rr_rank - 1]["rerank_score"]
            sixth = ranked[cfg.TOPK_RERANK - 1]["rerank_score"]
            plus_hits = {x["chunk_id"] for x in ranked[: cfg.TOPK_RERANK]} & g["relevant"]
            net = len(plus_hits) - len(base_hits)
            displaced = sorted(c.split("::")[-2] for c in base_hits - plus_hits)
        helped += int(net > 0)

        own_hit = False
        if best_qv is not None:
            res = col.query(query_embeddings=[best_qv], n_results=cfg.TOPK_DENSE)
            own_hit = cid in res["ids"][0]

        print(f"\n[{metas[cid].get('character', '?')}] {metas[cid].get('breadcrumb', cid)}")
        print(f"  问：{g['query']}")
        print(f"  现有：dense={g['dense_rank'] or '未进'} sparse={g['sparse_rank'] or '未进'} "
              f"RRF={g['rrf_rank'] or '未进'}")
        print(f"  cutoff(第{cfg.TOPK_DENSE}名 cos)={g['cutoff']:.4f}  "
              f"text_cos={text_cos:.4f}  best_q_cos={best_q:.4f}")
        verdict = f"能进前 {cfg.TOPK_RERANK_IN}" if in_pool else "仍进不了"
        print(f"  假设走问题索引后 RRF 名次={hyp_rank or '进不了'}  {verdict}")
        if in_pool:
            tail = f"顶掉：{'、'.join(displaced)}" if displaced else "没顶掉别的相关块"
            print(f"  单独加进候选池后真精排：第 {rr_rank} 名 分数 {rr_score:.4f}"
                  f"（第 {cfg.TOPK_RERANK} 名 {sixth:.4f}）→ 相关块净增 {net:+d}（{tail}）")
        for s in qs:
            print(f"    - {s}")
        print(f"  生成问句自召回：{'是' if own_hit else '否'}")

    print(f"\n结论：{in_pool_n}/{len(probe)} 个缺口块在「问题索引」假设下能进候选池，"
          f"其中 {helped}/{len(probe)} 个带来**净**增益（相关块净增 > 0，不是一换一）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(), loop_factory=asyncio.SelectorEventLoop))
