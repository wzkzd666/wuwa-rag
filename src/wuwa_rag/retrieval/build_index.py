"""Step 5：从 PG 读 chunks，建 Chroma 稠密索引 + BM25 稀疏索引。

两路索引都可随时重建 —— PG 才是真源。
"""
from __future__ import annotations

import asyncio

import chromadb
from chromadb.config import Settings

from ..config import ensure_dirs, get_settings
from ..db import close_pool, get_cursor
from .bm25 import BM25Index, add_terms, tokenize
from .embeddings import BgeM3Embeddings
from ..ww_logger import get_logger

bm25_logger=get_logger('bm25')
vec_logger=get_logger('vec')

async def _load_chunks() -> list[dict]:
    async with get_cursor(commit=False) as cur:
        """只读方式打开pg"""
        await cur.execute(
            """
            SELECT chunk_id, character, module, component, tab,
                   breadcrumb, text, has_table
            FROM chunks ORDER BY id
            """
        )                                           # 按id排序，方便后续BM25使用
        cols = [d.name for d in cur.description]    # 动态获取列名
        return [dict(zip(cols, r)) for r in await cur.fetchall()]


def _terms_of(rows: list[dict]) -> list[str]:
    """把角色名/模块/组件/tab 全部灌进 jieba 词典。"""
    s: set[str] = set()
    for r in rows:
        for k in ("character", "module", "component", "tab"):
            v = (r.get(k) or "").strip()
            if v:
                s.add(v)
    return sorted(s)


def _embed_input(r: dict) -> str:
    """breadcrumb 拼进正文：补正文语义。"""
    return f"{r['breadcrumb']}\n{r['text']}"


def build_sparse(rows: list[dict]) -> None:
    terms = _terms_of(rows)
    add_terms(terms)
    idx = BM25Index(
        [r["chunk_id"] for r in rows],
        [tokenize(_embed_input(r)) for r in rows],
        terms=terms,
    )
    bm25_logger.info(f"BM25  : {len(rows)} 条 / 术语 {len(terms)} 个 -> {idx.save()}")


def build_dense(rows: list[dict]) -> None:
    s = get_settings()
    client = chromadb.PersistentClient(
        path=str(s.VECTOR_DIR / "chroma"),
        settings=Settings(anonymized_telemetry=False),
    )
    col = client.get_or_create_collection(
        name=s.CHUNK_COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    emb = BgeM3Embeddings()
    for i in range(0, len(rows), s.EMBED_BATCH_SIZE):
        batch = rows[i : i + s.EMBED_BATCH_SIZE]
        col.upsert(
            ids=[r["chunk_id"] for r in batch],
            documents=[_embed_input(r) for r in batch],
            embeddings=emb.embed_documents([_embed_input(r) for r in batch]),
            metadatas=[
                {
                    "character": r["character"] or "",
                    "module": r["module"] or "",
                    "component": r["component"] or "",
                    "tab": r["tab"] or "",
                    "breadcrumb": r["breadcrumb"] or "",
                    "has_table": bool(r["has_table"]),
                }
                for r in batch
            ],
        )
        vec_logger.info(f"  dense {min(i + s.EMBED_BATCH_SIZE, len(rows))}/{len(rows)}")
    vec_logger.info(f"Chroma: 共 {col.count()} 条")


async def _main() -> None:
    ensure_dirs()
    rows = await _load_chunks()
    vec_logger.info(f"从 PG 读到 {len(rows)} 块")
    build_sparse(rows)     # 先稀疏：秒级，先验证通不通
    build_dense(rows)      # 后稠密：慢，放最后
    await close_pool()


def main() -> None:
    asyncio.run(_main(), loop_factory=asyncio.SelectorEventLoop)


if __name__ == "__main__":
    main()
