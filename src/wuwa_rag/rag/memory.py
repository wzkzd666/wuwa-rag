"""Step 9：LangGraph checkpointer（PostgreSQL / lg schema）。"""
from __future__ import annotations

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from ..config import get_settings
from ..ww_logger import get_logger

s = get_settings()
log = get_logger("rag")

_pool: AsyncConnectionPool | None = None
_saver: AsyncPostgresSaver | None = None


async def get_checkpointer() -> AsyncPostgresSaver:
    """checkpointer 单例。
    1. schema 必须在 DSN 里指定，否则表建到 public；
    2. 必须 autocommit=True —— migration 里有 CREATE INDEX CONCURRENTLY，
       事务块里跑会报 ActiveSqlTransaction；
    3. 传连接池而不是单连接 —— 官方 from_conn_string 只给单连接，并发会打架。
    """
    global _pool, _saver
    if _saver is None:
        _pool = AsyncConnectionPool(
            conninfo=s.PG_DSN_LG,
            min_size=1,
            max_size=5,
            open=False,
            kwargs={"autocommit": True, "row_factory": dict_row},
        )
        await _pool.open(wait=True, timeout=10)
        _saver = AsyncPostgresSaver(_pool)
        await _saver.setup()
        log.info("checkpointer 就绪（schema=lg）")
    return _saver


async def close_checkpointer() -> None:
    global _pool, _saver
    if _pool is not None:
        await _pool.close()
    _pool, _saver = None, None
