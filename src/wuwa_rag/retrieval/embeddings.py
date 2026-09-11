"""bge-m3 向量化：sentence-transformers 本地推理（CPU）。

为什么不用 Ollama/OpenAI 的 embedding 接口：
  GPU 8GB 已被 qwen3-vl:8b 占满，本地 CPU 显式可控；
  且与 reranker(bge-reranker-v2-m3) 同一套 sentence-transformers，代码一致。
"""
from __future__ import annotations
from ..config import get_settings
s = get_settings()
import os
os.environ.setdefault("HF_HOME", s.HF_HOME)

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer






@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    m = SentenceTransformer(s.EMBED_MODEL, device=s.EMBED_DEVICE, local_files_only=True)
    # bge-m3 默认 8192，CPU 上慢一个量级，必须压到 512
    m.max_seq_length = s.EMBED_MAX_SEQ_LENGTH
    return m


class BgeM3Embeddings(Embeddings):
    """langchain 标准接口，可直接喂给 LangChain 的 VectorStore。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return _model().encode(
            texts,
            batch_size=s.EMBED_BATCH_SIZE,
            normalize_embeddings=True,   # 归一化后 cosine == 内积
            show_progress_bar=True,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return _model().encode(text, normalize_embeddings=True).tolist()

