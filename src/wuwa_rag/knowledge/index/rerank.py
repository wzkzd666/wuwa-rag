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

from wuwa_rag.config import get_settings
from wuwa_rag.text import chunk_text
from wuwa_rag.ww_logger import get_logger

s = get_settings()
log = get_logger("rag")
# HF_HOME 必须赶在 torch/sentence_transformers 之前设好：它经 huggingface_hub 在
# **import 时**读取并缓存该路径（实测：import 后再改 os.environ 不生效）。
# 所以下面两行 import 只能留在 setdefault 之后，加 noqa 声明这是有意为之。
os.environ.setdefault("HF_HOME", s.HF_HOME)

import torch  # noqa: E402
from sentence_transformers import CrossEncoder  # noqa: E402


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


def status() -> tuple[bool, str]:
    """精排模型**现在**能不能用；不能用时给原因，不只给布尔。

    `rerank()` 自己失败会判空 + 写 error 日志，但调试出口必须把两件不同的事分开：
    **「精排不可用」**（修环境：权重/HF_HOME/显存）与**「精排可用、只是 top1 没过门槛」**
    （那是真的库里没答案）。判据仍然只有 `_model()` 这一处，这里只把异常翻成人话。
    """
    try:
        _model()
        return True, ""
    except Exception as exc:  # noqa: BLE001 —— 可用性探测绝不能反过来把调用方打挂
        return False, f"{type(exc).__name__}: {exc}"


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
    except Exception as exc:  # noqa: BLE001 —— 精排挂了必须是**可见的失败**
        # 精排就是「库里有没有答案」的那道门（RERANK_MIN_SCORE）。它失效时如果
        # 退回「按 RRF 顺序硬给 top6」，等于把门拆了还装作一切正常：答案照出、
        # 检索照绿，只有用户在追问「为什么答非所问」时才发现 —— 而日志里什么都没有。
        # 所以这里判定为「本轮无可用资料」（走既有的不知道/联网兜底），并把真实原因
        # 连同该查的两件事写进 error 日志。
        log.error("重排失败，本轮判定为库内无答案：%s: %s | 先查 HF_HOME=%s 与 %s 的权重是否就位"
                  "（local_files_only=True，缺文件不会自动下载）",
                  type(exc).__name__, exc, s.HF_HOME, s.RERANK_MODEL)
        return []
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
