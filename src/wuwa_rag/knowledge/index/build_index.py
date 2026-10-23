"""Step 5：从 PG 读 chunks，建 Chroma 稠密索引 + BM25 稀疏索引。

两路索引都可随时重建 —— PG 才是真源。
"""
from __future__ import annotations

import asyncio
import sys

import chromadb
from chromadb.config import Settings

from wuwa_rag.config import ensure_dirs, get_settings
from wuwa_rag.core.db import close_pool, get_cursor
from wuwa_rag.knowledge.index.bm25 import BM25Index, add_terms, tokenize
from wuwa_rag.knowledge.index.embeddings import BgeM3Embeddings
from wuwa_rag.text import embed_input
from wuwa_rag.ww_logger import get_logger

bm25_logger=get_logger('bm25')
vec_logger=get_logger('vec')


def _embed_input(r: dict) -> str:
    """breadcrumb 补语义——(必须去重)"""
    return embed_input(r["breadcrumb"], r["text"])


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


def plan_reconcile(stored: dict[str, str | None], rows: list[dict], *, force: bool) -> tuple[list[dict], list[str]]:
    """比「索引里存的文本」与「当前口径算出来的文本」，返回 (要重嵌的块, 要删的孤儿 id)。

    为什么按文本比而不是按时间：documents 里存的就是 embed_input(breadcrumb, text)，
    而 text.py 的清洗规则（长连字符、按键图标残渣、占位表头）一改，旧点存的还是旧口径
    文本，查询侧却已经是新口径 —— 两边不一致是实打实的召回损耗（2026-10-24 实测：
    36 块配队文本在索引里仍是 `-----` 串，PG 里早已是顿号）。
    文本没变就不必再跑一遍 embedding（6741 块 CPU 全量重嵌要几十分钟，跳过才是常态）。

    孤儿 id 也要删：chunk_id 含正文 hash，wiki 改版后旧 id 不会自然消失，
    upsert 只增不删，于是「全量重建」其实并不会把旧块清掉 —— 召回到已废的旧知识就是这么来的。
    按角色刷新（delete_dense_by_character）是另一条路，两者都要在：这条管离线重建，那条管在线刷新。
    """
    ids = {r["chunk_id"] for r in rows}
    todo = rows if force else [r for r in rows if stored.get(r["chunk_id"]) != _embed_input(r)]
    orphans = [cid for cid in stored if cid not in ids]
    return todo, orphans


def build_sparse(rows: list[dict]) -> None:
    terms = _terms_of(rows)
    add_terms(terms)
    idx = BM25Index(
        [r["chunk_id"] for r in rows],
        [tokenize(_embed_input(r)) for r in rows],
        terms=terms,
    )
    bm25_logger.info(f"BM25  : {len(rows)} 条 / 术语 {len(terms)} 个 -> {idx.save()}")


def build_dense(rows: list[dict], *, force: bool = False) -> None:
    """稠密路：只重嵌「索引里存的文本 ≠ 当前口径文本」的块，并删掉 PG 已不存在的孤儿块。

    force=True（命令行 `--full`）回到无脑全量重嵌。
    """
    s = get_settings()
    client = chromadb.PersistentClient(
        path=str(s.VECTOR_DIR / "chroma"),
        settings=Settings(anonymized_telemetry=False),
    )
    col = client.get_or_create_collection(
        name=s.CHUNK_COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    got = col.get(include=["documents"])
    stored = dict(zip(got["ids"], got["documents"]))
    todo, orphans = plan_reconcile(stored, rows, force=force)

    if orphans:
        col.delete(ids=orphans)
        vec_logger.info("  删孤儿块 %d 个（PG 里已没有）", len(orphans))
    if not todo:
        vec_logger.info("  %d 块文本口径全一致，跳过重嵌（要强制全量用 --full）", len(rows))
    else:
        vec_logger.info("  待重嵌 %d/%d 块（%s）", len(todo), len(rows),
                        "强制全量" if force else "口径不一致或索引里没有")
        emb = BgeM3Embeddings()     # 没活儿时不加载模型：CPU 上 bge-m3 起步就是十几秒
        for i in range(0, len(todo), s.EMBED_BATCH_SIZE):
            batch = todo[i : i + s.EMBED_BATCH_SIZE]
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
            vec_logger.info(f"  dense {min(i + s.EMBED_BATCH_SIZE, len(todo))}/{len(todo)}")
    vec_logger.info(f"Chroma: 共 {col.count()} 条")


def delete_dense_by_character(character: str) -> int:
    """按角色删 Chroma 旧向量（刷新重爬前清脏数据用）。

    chunk_id 含内容 hash，wiki 改版后新旧块并存 → 召回到旧知识。
    返回删前的该角色点数（仅日志用途，不保证与 delete 完全原子）。
    """
    s = get_settings()
    client = chromadb.PersistentClient(
        path=str(s.VECTOR_DIR / "chroma"),
        settings=Settings(anonymized_telemetry=False),
    )
    col = client.get_or_create_collection(
        name=s.CHUNK_COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    got = col.get(where={"character": character})
    ids = got["ids"]
    if ids:
        col.delete(ids=ids)
    vec_logger.info("按角色清稠密向量 %s: %d 点", character, len(ids))
    return len(ids)


async def _main(force: bool) -> None:
    ensure_dirs()
    rows = await _load_chunks()
    vec_logger.info(f"从 PG 读到 {len(rows)} 块")
    build_sparse(rows)     # 先稀疏：秒级，先验证通不通
    build_dense(rows, force=force)   # 后稠密：慢，放最后
    await close_pool()


def main() -> None:
    """命令行只认一个开关：`wuwa-index --full` 强制全量重嵌。

    未知参数直接退出而不是忽略 —— `--force` 这类笔误若被静默当成增量跑，
    使用者以为跑的是全量，差的就是「改版后旧块还在」这种最难查的故障。
    """
    args = sys.argv[1:]
    unknown = [a for a in args if a != "--full"]
    if unknown:
        raise SystemExit(f"wuwa-index 只接受 --full，未知参数：{' '.join(unknown)}")
    asyncio.run(_main("--full" in args), loop_factory=asyncio.SelectorEventLoop)


if __name__ == "__main__":
    main()
