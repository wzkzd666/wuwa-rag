"""Celery + Redis 异步流水线：爬角色→分块→入库(PG+S3)→建索引(BM25+Chroma)→建图谱(Neo4j)。

每个任务内部用 asyncio.run(loop_factory=SelectorEventLoop) 包裹现有 async 逻辑。
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict

import redis as redis_lib
from celery import Celery, chain
from wuwa_mcp.core.container import get_container

from .config import ensure_dirs, get_settings
from .db import close_pool, get_cursor
from .graph.build_graph import close_driver, init_schema, ping, upsert_character
from .graph.extract import extract_character, load_chunks
from .ingest.chunker import chunk_markdown
from .ingest.pipeline import _load_chunks as load_chunks_by_char
from .ingest.pipeline import ingest_one
from .retrieval.build_index import _load_chunks as load_all_chunks
from .retrieval.build_index import build_dense, build_sparse
from .ww_logger import get_logger


def _run(func):
    """Windows 下 psycopg/neo4j 异步必须用 SelectorEventLoop，否则连接池初始化超时。"""
    return asyncio.run(func, loop_factory=asyncio.SelectorEventLoop)


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


# ───────────── 入库进度打点（Redis，供 GET /ingest/status 轮询） ─────────────
# 双通道：_step_update 走 Celery backend（按 task_id），_progress_mark 写 Redis 聚合键
# （按角色）。/ingest/status 以聚合键为主数据源——chain .si 的 result.parent 链接不可靠，
# 按角色一个键最稳。步骤名与序号绑定，勿随意增删改名。
PIPELINE_STEPS = ("crawl", "chunk", "ingest", "index", "graph")
STEP_LABELS = {"crawl": "抓取", "chunk": "分块", "ingest": "入库", "index": "索引", "graph": "图谱"}
_PROGRESS_TTL = 3600
_r: redis_lib.Redis | None = None


def _redis() -> redis_lib.Redis:
    global _r
    if _r is None:
        _r = redis_lib.Redis.from_url(s.REDIS_URL, decode_responses=True, socket_connect_timeout=3)
    return _r


def progress_key(character: str) -> str:
    return f"ingest:progress:{character}"


def _progress_mark(character: str, step: str, status: str, error: str | None = None) -> None:
    """尽力而为的打点：Redis 故障绝不能把流水线任务本身带崩。"""
    try:
        key = progress_key(character)
        raw = _redis().get(key)
        data = json.loads(raw) if raw else {
            "character": character,
            "steps": {k: "pending" for k in PIPELINE_STEPS},
            "errors": {},
        }
        data["steps"][step] = status
        if error:
            data["errors"][step] = error[:300]
        else:
            data["errors"].pop(step, None)
        data["updated_at"] = int(time.time())
        _redis().setex(key, _PROGRESS_TTL, json.dumps(data, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001 —— 打点失败只记日志
        log.warning("进度打点失败（忽略）step=%s: %s", step, exc)


def get_progress(character: str) -> dict | None:
    """读某角色的聚合进度快照；未开跑（worker 还没消费）返回 None。"""
    try:
        raw = _redis().get(progress_key(character))
        return json.loads(raw) if raw else None
    except Exception as exc:  # noqa: BLE001
        log.warning("读进度失败（忽略）: %s", exc)
        return None


# 流水线五步与标签见上方 PIPELINE_STEPS / STEP_LABELS


def _step_update(task, character: str, step: str, status: str, error: str | None = None) -> None:
    """双通道步骤上报（供 GET /ingest/status 轮询）：
    1) Celery backend update_state（按 task_id，保留给外部工具）；
    2) _progress_mark 聚合键（按角色，/ingest/status 的主数据源）。
    status ∈ start|done|fail。task 未 bind 时传 None 静默跳过；
    上报抛错只记日志，绝不因观测性代码弄挂业务流水线。
    """
    if task is not None:
        try:
            task.update_state(state="PROGRESS", meta={"step": step, "status": status})
        except Exception as exc:
            log.warning("步骤上报失败 %s/%s: %s", step, status, exc)
    _progress_mark(character, step, {"start": "running", "done": "success", "fail": "failed"}[status], error)


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
def crawl_character(self, character: str, run_id: int | None = None):
    # 只在首次插入；重试通过 args 回传 run_id，避免重复 INSERT 污染 crawl_runs
    _step_update(self, character, "crawl", "start")
    if run_id is None:
        run_id = _run(_crawl_run_insert(character))
    try:
        md = _run(_fetch_markdown(character))
        if md.startswith("错误"):                     
            raise CharacterNotFound(f"{character} 未找到")
        (s.RAW_DIR / f"{character}.md").write_text(md, encoding="utf-8")
    except CharacterNotFound:
        _run(_crawl_run_update(run_id, "failed", {"error": "not_found"}))   # 补状态
        _step_update(self, character, "crawl", "fail", "wiki 上不存在该角色")
        raise
    except Exception as exc:
        _run(_crawl_run_update(run_id, "failed", {"error": f"fetch: {exc}"}))
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, args=(character, run_id))             # run_id 带回
        _step_update(self, character, "crawl", "fail", f"抓取失败(已重试): {exc}")
        raise
    else:
        _run(_crawl_run_update(run_id, "success", {"bytes": len(md)}))
        _step_update(self, character, "crawl", "done")
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
        [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if path.exists() else []
    )
    existing = [c for c in existing if c.get("character") != character]  # 丢掉旧版
    existing.extend(asdict(c) for c in new_chunks)
    # 原子写：先落临时文件再 replace，避免别处读到半截；多进程并行仍可能覆盖，
    # 所以 worker 务必 --pool=solo --concurrency=1（串行），或后续改按角色分文件
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for c in existing:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    tmp.replace(path)                                      
    log.info("分块 %s: 新增 %d 块，jsonl 现 %d 块", character, len(new_chunks), len(existing))



@celery_app.task(bind=True)
def chunk_character(self, character: str):
    _step_update(self, character, "chunk", "start")
    try:
        _run(_chunk_character_async(character))
    except Exception as exc:
        _step_update(self, character, "chunk", "fail", f"{exc}")
        raise
    _step_update(self, character, "chunk", "done")
    return {"character": character}


# ───────────── 3. 入库（PG + S3，单角色） ─────────────
async def _ingest_character_async(character: str) -> int:
    by_char = load_chunks_by_char(s.CHUNKS_JSONL)
    doc_id, n = await ingest_one(s.RAW_DIR / f"{character}.md", by_char.get(character, []))
    await close_pool()
    return n


@celery_app.task(bind=True)
def ingest_character(self, character: str):
    _step_update(self, character, "ingest", "start")
    try:
        n = _run(_ingest_character_async(character))
    except Exception as exc:
        _step_update(self, character, "ingest", "fail", f"{exc}")
        raise
    _step_update(self, character, "ingest", "done")
    return {"character": character, "chunks": n}


# ───────────── 4. 建索引（Chroma 增量 upsert + BM25 全量重建） ─────────────
async def _index_character_async(character: str) -> None:
    all_rows = await load_all_chunks()
    build_sparse(all_rows)                       # BM25 单文件 → 必须全量重建（秒级）
    char_rows = [r for r in all_rows if r["character"] == character]
    build_dense(char_rows)                        # Chroma 按 chunk_id upsert → 只该角色
    await close_pool()


@celery_app.task(bind=True)
def index_character(self, character: str):
    _step_update(self, character, "index", "start")
    try:
        _run(_index_character_async(character))
    except Exception as exc:
        _step_update(self, character, "index", "fail", f"{exc}")
        raise
    _step_update(self, character, "index", "done")
    return {"character": character}


# ───────────── 5. 建图谱（Neo4j MERGE，天然单角色） ─────────────
async def _graph_character_async(character: str) -> None:
    if not await ping():
        raise RuntimeError("Neo4j 不可用")
    await init_schema()
    chunks = load_chunks()                        # 读 chunks.jsonl 全量
    await upsert_character(extract_character(chunks, character))
    await close_driver()


@celery_app.task(bind=True)
def graph_character(self, character: str):
    _step_update(self, character, "graph", "start")
    try:
        _run(_graph_character_async(character))
    except Exception as exc:
        _step_update(self, character, "graph", "fail", f"{exc}")
        raise
    _step_update(self, character, "graph", "done")
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
