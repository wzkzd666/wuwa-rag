"""应用层 BM25 稀疏检索（jieba 分词 + rank_bm25）。

为什么不用 PG 全文检索：
  ① 规模小（几千块），内存常驻 + pickle 落盘最省事，全量重建只要几秒
  ② 中文要靠自定义词典（鸣潮术语），PG 的 tsvector 对中文基本无能为力
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from ..config import get_settings

_NOISE_RE = re.compile(r"[|\-\s]+")   # 表格竖线/分隔横线/空白：纯噪声
s=get_settings()

@dataclass
class Hit:
    chunk_id: str
    score: float


def add_terms(words: list[str]) -> None:
    """把领域术语灌进 jieba，防止被切碎。"""
    for w in words or []:
        w = (w or "").strip()
        if len(w) >= 2:
            jieba.add_word(w)


def tokenize(text: str) -> list[str]:
    """去噪、分词、列表化"""
    clean = _NOISE_RE.sub(" ", text or "")
    return [t for t in jieba.cut(clean) if t.strip()]


class BM25Index:
    """稀疏路索引。

    terms 跟着一起落盘 —— 保证建索引和查询用的是同一套词典，
    否则两边分词不一致，分数会莫名其妙。
    """

    def __init__(
        self,
        chunk_ids: list[str],
        corpus: list[list[str]],
        terms: list[str] | None = None,
    ):
        """
        chunk_ids:唯一id列表;
        corpus:已经分词好的文档集合;
        terms:建索引时用的自定义词典快照
        """

        self.chunk_ids = chunk_ids
        self.terms = terms or []
        #接收corpus，计算每个词的 IDF（逆文档频率），构建倒排索引
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, topk: int = 30) -> list[Hit]:
        """输入query，得到Hit列表"""
        q = tokenize(query)
        if not q:
            return []
        scores = self.bm25.get_scores(q)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:topk]
        return [Hit(self.chunk_ids[i], float(s)) for i, s in ranked if s > 0]

    def save(self, path: Path | None = None) -> Path:
        """保存，对应的专业术语一并存进去，确保使用时环境稳定"""
        path = path or (s.VECTOR_DIR / "bm25.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {"chunk_ids": self.chunk_ids, "bm25": self.bm25, "terms": self.terms},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "BM25Index":
        path = path or (s.VECTOR_DIR / "bm25.pkl")
        with path.open("rb") as f:
            d = pickle.load(f)
        add_terms(d.get("terms", []))     # 关键：恢复词典，否则查询分词与建索引不一致
        obj = cls.__new__(cls)
        obj.chunk_ids = d["chunk_ids"]
        obj.bm25 = d["bm25"]
        obj.terms = d.get("terms", [])
        return obj
