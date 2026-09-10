"""Celery + Redis 异步流水线：爬角色→分块→入库(PG+S3)→建索引(BM25+Chroma)→建图谱(Neo4j)。

每个任务内部用 asyncio.run(loop_factory=SelectorEventLoop) 包裹现有 async 逻辑，
直接复用 ingest / retrieval / graph 里的函数，不重复造轮子。
"""
from __future__ import annotations

import asyncio
def _run(func):
    """Windows 下 psycopg/neo4j 异步必须用 SelectorEventLoop，否则连接池初始化超时。"""
    return asyncio.run(func, loop_factory=asyncio.SelectorEventLoop)

import json
from dataclasses import asdict

from celery import Celery, chain

from .config import ensure_dirs, get_settings
from .db import close_pool, get_cursor
from .ww_logger import get_logger
from .ingest.chunker import chunk_markdown
from .ingest.pipeline import ingest_one, _load_chunks as load_chunks_by_char
from .retrieval.build_index import build_sparse, build_dense, _load_chunks as load_all_chunks
from .graph.build_graph import upsert_character, ping, init_schema, close_driver
from .graph.extract import load_chunks, extract_character
from wuwa_mcp.core.container import get_container

log = get_logger("celery")
s = get_settings()
ensure_dirs()


celery_app = Celery(
    "wuwa_rag",
    broker=s.REDIS_URL,
    backend=s.REDIS_URL,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)
celery_app.conf.update(task_track_started=True, result_expires=3600)


class CharacterNotFound(Exception):
    """角色在 wiki 上不存在，链应中止（不重试）。"""


# ───────────── crawl_runs 落地（crawl 步骤用） ─────────────
async def _crawl_run_insert(character: str) -> int:
    async with get_cursor() as cur:
        await cur.execute(
            "INSERT INTO crawl_runs (status, stats) VALUES ('running', %s) RETURNING id",
            (json.dumps({"character": character}, ensure_ascii=False),),
        )
        return (await cur.fetchone())[0]


async def _crawl_run_update(run_id: int, status: str, stats: dict) -> None:
    async with get_cursor() as cur:
        await cur.execute(
            "UPDATE crawl_runs SET finished_at=now(), status=%s, stats=%s WHERE id=%s",
            (status, json.dumps(stats, ensure_ascii=False), run_id),
        )


# ───────────── 1. 爬角色 ─────────────
async def _fetch_markdown(character: str) -> str:
    svc = get_container().get_character_service()
    return await svc.get_character_info(character)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=15)
def crawl_character(self, character: str):
    run_id = _run(_crawl_run_insert(character))
    try:
        md = _run(_fetch_markdown(character))
        if md.startswith("错误"):
            _run(_crawl_run_update(run_id, "failed", {"error": "not_found"}))
            raise CharacterNotFound(f"{character} 未找到")
        (s.RAW_DIR / f"{character}.md").write_text(md, encoding="utf-8")
        _run(_crawl_run_update(run_id, "success", {"bytes": len(md)}))
        return {"character": character, "run_id": run_id}
    except CharacterNotFound:
        raise
    except Exception as exc:  # 网络/API 异常 → 重试
        _run(_crawl_run_update(run_id, "failed", {"error": f"fetch: {exc}"}))
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)
        raise
    finally:
        _run(close_pool())


# ───────────── 2. 分块（增量：只重写该角色） ─────────────
async def _chunk_character_async(character: str) -> None:
    raw = s.RAW_DIR / f"{character}.md"
    if not raw.exists():
        raise RuntimeError(f"raw/{character}.md 不存在，crawl 必须先于 chunk")
    new_chunks = chunk_markdown(raw.read_text(encoding="utf-8"), character)
    path = s.CHUNKS_JSONL
    existing = (
        [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if path.exists() else []
    )
    existing = [c for c in existing if c.get("character") != character]  # 丢掉旧版
    existing.extend(asdict(c) for c in new_chunks)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in existing:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    log.info("分块 %s: 新增 %d 块，jsonl 现 %d 块", character, len(new_chunks), len(existing))


@celery_app.task
def chunk_character(character: str):
    _run(_chunk_character_async(character))
    return {"character": character}


# ───────────── 3. 入库（PG + S3，单角色） ─────────────
async def _ingest_character_async(character: str) -> int:
    by_char = load_chunks_by_char(s.CHUNKS_JSONL)
    doc_id, n = await ingest_one(s.RAW_DIR / f"{character}.md", by_char.get(character, []))
    await close_pool()
    return n


@celery_app.task
def ingest_character(character: str):
    n = _run(_ingest_character_async(character))
    return {"character": character, "chunks": n}


# ───────────── 4. 建索引（Chroma 增量 upsert + BM25 全量重建） ─────────────
async def _index_character_async(character: str) -> None:
    all_rows = await load_all_chunks()
    build_sparse(all_rows)                       # BM25 单文件 → 必须全量重建（秒级）
    char_rows = [r for r in all_rows if r["character"] == character]
    build_dense(char_rows)                        # Chroma 按 chunk_id upsert → 只该角色
    await close_pool()


@celery_app.task
def index_character(character: str):
    _run(_index_character_async(character))
    return {"character": character}


# ───────────── 5. 建图谱（Neo4j MERGE，天然单角色） ─────────────
async def _graph_character_async(character: str) -> None:
    if not await ping():
        raise RuntimeError("Neo4j 不可用")
    await init_schema()
    chunks = load_chunks()                        # 读 chunks.jsonl 全量
    await upsert_character(extract_character(chunks, character))
    await close_driver()


@celery_app.task
def graph_character(character: str):
    _run(_graph_character_async(character))
    return {"character": character}


def build_pipeline(character: str):
    """返回整条链（不入队），由 CLI / API 直接 apply_async。
    用 .si() 让每一步都拿到 character，不被上一步返回值覆盖。"""
    return chain(
        crawl_character.si(character),
        chunk_character.si(character),
        ingest_character.si(character),
        index_character.si(character),
        graph_character.si(character),
    )


# ───────────── CLI 入口（可选） ─────────────
def ingest_cli() -> None:
    import sys

    character = sys.argv[1] if len(sys.argv) > 1 else None
    if not character:
        print("用法: wuwa-ingest-character <角色名>")
        raise SystemExit(1)
    r = build_pipeline(character).apply_async()
    print(f"已入队 {character}，chain_id={r.id}")
