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

from wuwa_rag.config import ensure_dirs, get_settings
from wuwa_rag.core.db import close_pool, get_cursor
from wuwa_rag.knowledge import domain_terms
from wuwa_rag.knowledge.crawl.chunker import chunk_markdown
from wuwa_rag.knowledge.crawl.pipeline import _load_chunks as load_chunks_by_char
from wuwa_rag.knowledge.crawl.pipeline import ingest_one, purge_character
from wuwa_rag.knowledge.graph.build_graph import (
    close_driver,
    delete_character,
    init_schema,
    ping,
    upsert_character,
)
from wuwa_rag.knowledge.graph.extract import extract_character, load_chunks
from wuwa_rag.knowledge.index.build_index import _load_chunks as load_all_chunks
from wuwa_rag.knowledge.index.build_index import (
    build_dense,
    build_sparse,
    delete_dense_by_character,
)
from wuwa_rag.ww_logger import get_logger


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


class IngestCancelled(Exception):
    """用户手动取消了入库（pause → 等待，cancel → 掐断整条链）。"""


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


def reset_progress(character: str) -> None:
    """刷新重爬前重置五步进度（否则前端会一直看到上一轮全绿的旧快照）。"""
    try:
        data = {
            "character": character,
            "steps": {k: "pending" for k in PIPELINE_STEPS},
            "errors": {},
            "updated_at": int(time.time()),
        }
        _redis().setex(progress_key(character), _PROGRESS_TTL,
                       json.dumps(data, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001
        log.warning("进度重置失败（忽略）: %s", exc)


# ───────────── 暂停 / 继续 / 取消（控制旗标） ─────────────
#
# 为什么用 Redis 旗标而不是 `celery_control.revoke()`：
# 五步是 `chain(...)` 串联的，链里每一步各有自己的 task_id，revoke 只能停掉**某一个** id，
# 停掉链尾那一步时前面几步照跑；要停整条链得先把所有 task_id 收集齐，而
# `apply_async` 返回的 result 上取不全、随时可能漏 —— 表现为「点了取消，任务还在跑」。
# 旗标法是**每一步开头自查**：不依赖 task_id、不依赖 revoke 的语义，也不会漏。
#
# 暂停为什么用「等待」而不是「不执行」：`chain` 里某一步正常返回，下一步**照样会被调度**，
# 没有「条件链」这种原语。所以暂停 = 这一步醒来后原地等（1s 一轮），等到解除为止；
# 好处是继续之后自动往下跑，链的进度不丢。代价：暂停期间**占着一个 worker 槽**
# （本项目 worker 是 --pool=solo，本来一次也只跑一条任务，影响面很小）。
# 等待超过 _PAUSE_MAX_WAIT 秒后自动放行，避免「忘了取消」把 worker 永久占住。
_PAUSE_POLL = 1.0
_PAUSE_MAX_WAIT = 1800.0
CANCEL_REASON = "已取消"
CANCEL_ERROR = "任务被手动取消"


def ctl_key(character: str) -> str:
    return f"ingest:ctl:{character}"


def set_control(character: str, action: str) -> str:
    """写控制旗标：`pause` / `cancel` 落盘，`resume` 删键（回到「无控制」）。

    ⚠️ `cancel` **必须落盘**：早先把它也当成「删键」处理，结果 worker 只看得到 `pause`，
    取消指令根本传不出去（点了取消没反应）。取消要等 worker 在下一步开头自查到它，
    所以它得是个**能被读到的值**，而不是「键消失」。
    """
    if action == "resume":
        _redis().delete(ctl_key(character))
    else:
        _redis().setex(ctl_key(character), _PROGRESS_TTL, action)
    return action


def get_control(character: str) -> str:
    """当前旗标：`pause` / `cancel` / 空串（无控制）。读失败按「无控制」处理，别挡流水线。"""
    try:
        return _redis().get(ctl_key(character)) or ""
    except Exception as exc:  # noqa: BLE001
        log.warning("读控制旗标失败（忽略）: %s", exc)
        return ""


def wait_if_paused(character: str) -> bool:
    """任务开跑前自查控制旗标。**返回 True 表示「被取消了」，调用方应立即停止。**

    暂停时在这里循环等待；等待窗口内若收到 cancel 则立刻返回 True。
    """
    waited = 0.0
    while get_control(character) == "pause":
        if waited == 0.0:
            log.info("入库已暂停：%s（等待继续，最长 %.0f 秒）", character, _PAUSE_MAX_WAIT)
        if waited >= _PAUSE_MAX_WAIT:
            log.warning("暂停等待超时（%.0f 秒），自动继续：%s", _PAUSE_MAX_WAIT, character)
            return False
        time.sleep(_PAUSE_POLL)
        waited += _PAUSE_POLL
    if waited:
        # 循环退出说明旗标变了：要么被继续（键没了），要么被取消（键值=cancel）
        log.info("入库%s：%s", "取消" if get_control(character) == "cancel" else "继续", character)
    return get_control(character) == "cancel"


def _cancelled(character: str, step: str) -> None:
    """把当前步标记成「已取消」——状态仍是 failed，错误文案说明是手动取消。"""
    _progress_mark(character, step, "fail", CANCEL_ERROR)


def raise_if_cancelled(character: str, step: str) -> None:
    """每步开头的守卫：被取消就落账并抛 `IngestCancelled`。

    抛异常（而不是 return）才能真正**掐断整条链** —— 正常返回会让 chain 继续调度下一步。
    `IngestCancelled` 不参与重试（Celery 只对显式 `self.retry` 重试）。
    """
    if wait_if_paused(character):
        _cancelled(character, step)
        raise IngestCancelled(CANCEL_REASON)


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
        except Exception as exc:  # noqa: BLE001 —— 观测性代码（Celery backend 上报）不得弄挂业务流水线
            log.warning("步骤上报失败 %s/%s: %s", step, status, exc)
    _progress_mark(character, step, {"start": "running", "done": "success", "fail": "failed"}[status], error)


# ───────────── crawl_runs 落地（crawl 步骤用） ─────────────
async def _crawl_run_insert(character: str, submitted_by: str = "",
                           submitted_by_name: str = "") -> int:
    """建一条抓取账本。**提交人只有 API 知道**（见 crawl_runs 的建表注释）：
    正常入库由 API 先建好行、把 id 当 run_id 传进来，这里只在没有 run_id
    （自动重爬链 / CLI 直调）时自己补一行，提交人记成「自动」。"""
    async with get_cursor() as cur:
        await cur.execute(
            "INSERT INTO crawl_runs (status, stats, character, submitted_by,"
            " submitted_by_name) VALUES ('running', %s, %s, %s, %s) RETURNING id",
            (json.dumps({"character": character}, ensure_ascii=False), character,
             submitted_by, submitted_by_name or "自动"),
        )
        return (await cur.fetchone())[0]


async def _crawl_run_update(run_id: int, status: str, stats: dict) -> None:
    async with get_cursor() as cur:
        await cur.execute(
            "UPDATE crawl_runs SET finished_at=now(), status=%s, stats=%s,"
            " error=%s, updated_at=now() WHERE id=%s",
            (status, json.dumps(stats, ensure_ascii=False), stats.get("error"), run_id),
        )


# ───────────── 1. 爬角色 ─────────────
async def _fetch_markdown(character: str) -> str:
    svc = get_container().get_character_service()
    return await svc.get_character_info(character)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=15)
def crawl_character(self, character: str, run_id: int | None = None):
    raise_if_cancelled(character, "crawl")
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
    except Exception as exc:  # noqa: BLE001 —— 上面已单独接 CharacterNotFound，这里兜网络类异常交 Celery 重试
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
    raise_if_cancelled(character, "chunk")
    _step_update(self, character, "chunk", "start")
    try:
        _run(_chunk_character_async(character))
    except Exception as exc:  # noqa: BLE001 —— 必须先落账再抛，否则前端进度永远卡在 running
        _step_update(self, character, "chunk", "fail", f"{exc}")
        raise
    _step_update(self, character, "chunk", "done")
    return {"character": character}


# ───────────── 3. 入库（PG + S3，单角色） ─────────────
async def _ingest_character_async(character: str) -> int:
    by_char = load_chunks_by_char(s.CHUNKS_JSONL)
    doc_id, n = await ingest_one(s.RAW_DIR / f"{character}.md", by_char.get(character, []))
    # 新角色入库 → 领域词表的锚点集变了，标记重建（下次提问的后台任务里重算）
    domain_terms.invalidate()
    await close_pool()
    return n


@celery_app.task(bind=True)
def ingest_character(self, character: str):
    raise_if_cancelled(character, "ingest")
    _step_update(self, character, "ingest", "start")
    try:
        n = _run(_ingest_character_async(character))
    except Exception as exc:  # noqa: BLE001 —— 必须先落账再抛，否则前端进度永远卡在 running
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
    raise_if_cancelled(character, "index")
    _step_update(self, character, "index", "start")
    try:
        _run(_index_character_async(character))
    except Exception as exc:  # noqa: BLE001 —— 必须先落账再抛，否则前端进度永远卡在 running
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
    raise_if_cancelled(character, "graph")
    _step_update(self, character, "graph", "start")
    try:
        _run(_graph_character_async(character))
    except Exception as exc:  # noqa: BLE001 —— 必须先落账再抛，否则前端进度永远卡在 running
        _step_update(self, character, "graph", "fail", f"{exc}")
        raise
    _step_update(self, character, "graph", "done")
    return {"character": character}


def build_pipeline(character: str, run_id: int | None = None):
    """返回整条链（不入队），由 CLI / API 直接 apply_async。
    用 .si() 让每一步都拿到 character，不被上一步返回值覆盖。

    `run_id` 由 API 传入（它先建好 crawl_runs 行、记下提交人，再把 id 交给 crawl 步复用）。
    传 None 时 crawl 步会自己建一行，提交人记为「自动」。
    """
    return chain(
        crawl_character.si(character, run_id),
        chunk_character.si(character),
        ingest_character.si(character),
        index_character.si(character),
        graph_character.si(character),
    )


# ───────────── 刷新（verify 判资料不匹配时的重爬链：先清后爬） ─────────────
@celery_app.task(bind=True)
def purge_character_data(self, character: str):
    """刷新链第 0 步：清该角色在 PG/Chroma/Neo4j 的旧知识。

    不清就是「新旧块并存 → 召回到旧知识」的根源（chunks 表 ON CONFLICT DO
    NOTHING 只挡同 hash，wiki 改版后 hash 全变）。S3 旧对象留孤儿由存储生命周期
    回收；进度键重置让前端能看到新一轮五步。
    """
    _step_update(self, character, "crawl", "start")   # 借位：清库算重爬前置，避免前端空窗
    try:
        _run(_purge_async(character))
    except Exception as exc:  # noqa: BLE001 —— 必须先落账再抛，否则前端进度永远卡在 running
        _step_update(self, character, "crawl", "fail", f"清库: {exc}")
        raise
    return {"character": character}


async def _purge_async(character: str) -> None:
    await purge_character(character)                    # PG documents（chunks 级联）
    delete_dense_by_character(character)                # Chroma 该角色向量
    if await ping():
        await delete_character(character)               # Neo4j 私有节点+出边
        await close_driver()
    # BM25 是全量文件，index 步秒级重建，无需单删
    # 注意：本函数已在 _run 的循环内，close_pool 必须 await，不能再 _run（loop 套 loop）
    await close_pool()


def build_refresh_pipeline(character: str):
    """刷新链：清库 → 重爬 → 分块 → 入库 → 索引 → 图谱。

    与 build_pipeline 只差开头挂一步 purge_character_data；chunk 步按角色
    整写 chunks.jsonl（--pool=solo 串行前提不变）。
    """
    return chain(
        purge_character_data.si(character),
        crawl_character.si(character),
        chunk_character.si(character),
        ingest_character.si(character),
        index_character.si(character),
        graph_character.si(character),
    )


# ───────────── 删除（把该角色的知识库整个拿走，不重爬） ─────────────
@celery_app.task(bind=True)
def delete_character_knowledge(self, character: str):
    """删掉一个角色的全部知识：PG（documents 级联 chunks）+ Chroma + Neo4j + BM25。

    与刷新链第一步 `purge_character_data` 的关键差别在**最后一步**：purge 后面本来
    跟着重爬，索引会被 `index_character` 重建；纯删除没有后续步骤，所以必须自己
    重建 BM25 —— `bm25.pkl` 是**全量单文件**，不重建的话里面仍留着该角色的块，
    检索照样把它召回来，表现成「删了还在答」。

    ⚠️ 两条**刻意维持**的行为，别当成 bug 去「修」：

    1. Neo4j 里的 `Character` 节点**保留**（只 REMOVE 属性）——DETACH DELETE 会把
       **别人**指向它的 `SYNERGIZES_WITH` 入边一起毁掉，那是别人页面的数据、
       本次操作不该动。理由详见 `build_graph._C_DELETE_CHAR` 的三条铁律。
    2. 由 1 推出：删掉的角色在 Neo4j 里仍是「已知角色」，而
       `dialog.graph.ensure_characters` 判定「是否已在知识库」用的正是 Neo4j
       （`_known_characters`）→ **删除后提问不会触发自动重爬**，只会答「不知道」，
       想加回要走「收录新角色」。只图谱无实料的角色 `to_crawl=[]` 不重爬；
       真·未知角色仍正常触发爬取。
       语义上这是自洽的：**你亲手删掉的东西不该自己回来**。
    """
    try:
        _run(_delete_character_async(character))
    except Exception as exc:  # noqa: BLE001 —— 记一条可读日志后原样重抛，交 Celery 记失败状态
        log.warning("删除角色知识库失败 %s: %s", character, exc)
        raise
    reset_progress(character)          # 进度键一并清掉，免得前端看到幽灵进度
    return {"character": character}


async def _delete_character_async(character: str) -> None:
    await _purge_async(character)      # PG + Chroma + Neo4j（结尾已 close_pool）
    rows = await load_all_chunks()     # 池刚被关掉，这里会按需重开
    build_sparse(rows)                 # BM25 全量重建，剔除残留块
    await close_pool()


# ───────────── CLI 入口（可选） ─────────────
def ingest_cli() -> None:
    import sys

    character = sys.argv[1] if len(sys.argv) > 1 else None
    if not character:
        print("用法: wuwa-ingest-character <角色名>")
        raise SystemExit(1)
    r = build_pipeline(character).apply_async()
    print(f"已入队 {character}，chain_id={r.id}")
