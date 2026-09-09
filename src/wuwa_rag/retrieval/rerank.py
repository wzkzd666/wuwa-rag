"""Step 8：bge-reranker-v2-m3 精排（CPU）。

为什么必须有这一步：
  bge-m3 是双塔——query 和 doc 各自编码后算余弦，两者全程没有交互；
  CrossEncoder 把 (query, doc) 拼在一起过一遍 Transformer，能建模交互，精度高一档。
  代价是不能预计算 doc 向量，只能放在召回之后，且 CPU 上约 170ms/条。

所以：召回放宽（30 条 RRF 融合），精排只做前 20 条，最终只送 6 条给 LLM。
"""
from __future__ import annotations

import os
from functools import lru_cache

import torch
from sentence_transformers import CrossEncoder

from ..config import get_settings
from ..ww_logger import get_logger
from ..text import chunk_text

s = get_settings()
os.environ.setdefault("HF_HOME", s.HF_HOME)
log = get_logger("rag")


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    log.info("加载重排模型 %s（%s，max_length=%d）",
             s.RERANK_MODEL, s.RERANK_DEVICE, s.RERANK_MAX_LENGTH)
    return CrossEncoder(
        s.RERANK_MODEL,
        device=s.RERANK_DEVICE,
        local_files_only=True,
        max_length=s.RERANK_MAX_LENGTH,
    )


def rerank(query: str, docs: list[dict], topk: int | None = None) -> list[dict]:
    """原地给 docs 打 rerank_score，按分降序返回前 topk 条。"""
    if not docs:
        return []

    k = topk or s.TOPK_RERANK
    # 线程数临时设置 + 恢复：全局设会污染同进程的 bge-m3 embedding
    old_threads = torch.get_num_threads()
    torch.set_num_threads(s.RERANK_THREADS)    
    try:
        scores = _model().predict(
            [(query, chunk_text(d)) for d in docs],
            batch_size=s.RERANK_BATCH_SIZE,
            show_progress_bar=False,
        )
    finally:
        torch.set_num_threads(old_threads)

    for d, sc in zip(docs, scores):
        d["rerank_score"] = round(float(sc), 4)
    docs.sort(key=lambda d: d["rerank_score"], reverse=True)

    top = docs[:k]
    if top and top[0]["rerank_score"] < s.RERANK_MIN_SCORE:
        log.info("重排 top1=%.3f < %.2f，判定库内无答案，丢弃全部候选",
                 top[0]["rerank_score"], s.RERANK_MIN_SCORE)
        return []
    log.info("重排 %d 条 -> %d 条，top1=%.3f 末位=%.3f",
             len(docs), len(top), top[0]["rerank_score"], top[-1]["rerank_score"])
    kept = [d for d in top if d["rerank_score"] >= s.RERANK_MIN_SCORE]
    return kept 
