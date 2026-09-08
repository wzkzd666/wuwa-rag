"""PostgreSQL 异步连接（psycopg3 AsyncConnectionPool）。
Windows 前置：入口必须先切 SelectorEventLoop，
否则 psycopg 异步在 Proactor 下报 InterfaceError。
为何不用 asyncpg：LangGraph 的 checkpointer 底层就是 psycopg，
多引一套驱动徒增平台耦合。
"""
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from psycopg import AsyncCursor
from psycopg_pool import AsyncConnectionPool

from .config import get_settings

_pool: AsyncConnectionPool | None = None

async def get_pool() -> AsyncConnectionPool:
    """获取连接池，首次调用时创建连接池"""
    global _pool
    if _pool is None:
        # open=False：连接必须在有 running loop 时才能开
        _pool = AsyncConnectionPool(
            conninfo=get_settings().PG_DSN,
            min_size=1,
            max_size=10,
            open=False,
        )
        await _pool.open(wait=True, timeout=10)
    return _pool


@asynccontextmanager
async def get_cursor(commit: bool = True) -> AsyncGenerator[AsyncCursor,None]:
    """游标管理器，自动关闭游标和归还连接"""
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            yield cur
        if commit:
            await conn.commit()


async def close_pool() -> None:
    """关闭连接池"""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def ping() -> str:
    """测试连接"""
    async with get_cursor(commit=False) as cur:
        await cur.execute("SELECT version();")
        row = await cur.fetchone()
        return row[0]
