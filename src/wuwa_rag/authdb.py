"""鉴权/画像共用的 PostgreSQL 连接池 + 建表 + admin 种子。

为什么单独一个文件：auth（users/auth_tokens）与画像（user_facts）共用一个池子；
API 启动时 ensure 一次 DDL（幂等 CREATE IF NOT EXISTS），爸爸就不用手工跑 002_auth.sql——
那份 SQL 是给手工建库/核对结构用的，两边内容保持一致。

池子参数照抄 memory.py 的经验：autocommit=True + dict_row；public schema（DSN 不带
search_path，默认 public，与 001/002 建表位置一致）。
"""
from __future__ import annotations

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from .config import get_settings
from .ww_logger import get_logger

s = get_settings()
log = get_logger("auth")

_pool: AsyncConnectionPool | None = None

# 与 pgsql/002_auth.sql 保持一致（幂等）
_DDL = [
    """
    CREATE TABLE IF NOT EXISTS users (
        id         BIGSERIAL PRIMARY KEY,
        username   TEXT        NOT NULL,
        pw_hash    TEXT        NOT NULL,
        role       TEXT        NOT NULL DEFAULT 'guest',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_users_username ON users(username)",
    """
    CREATE TABLE IF NOT EXISTS auth_tokens (
        token      TEXT        PRIMARY KEY,
        user_id    BIGINT      NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        expires_at TIMESTAMPTZ NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_tokens_user    ON auth_tokens(user_id)",
    "CREATE INDEX IF NOT EXISTS ix_tokens_expires ON auth_tokens(expires_at)",
    """
    CREATE TABLE IF NOT EXISTS user_facts (
        id         BIGSERIAL PRIMARY KEY,
        user_id    TEXT,
        session_id TEXT,
        fact       TEXT        NOT NULL,
        confidence REAL,
        source     TEXT,
        valid_from TIMESTAMPTZ NOT NULL DEFAULT now(),
        valid_to   TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_facts_user ON user_facts(user_id, valid_to)",
]


async def get_pool() -> AsyncConnectionPool:
    """鉴权/画像共享池单例。"""
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            conninfo=s.PG_DSN,
            min_size=1,
            max_size=5,
            open=False,
            kwargs={"autocommit": True, "row_factory": dict_row},
        )
        await _pool.open(wait=True, timeout=10)
        log.info("auth 池就绪 db=%s", s.PG_DB)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
    _pool = None


async def ensure_schema() -> None:
    """幂等建表 + 清过期 token + 种子 admin。API lifespan 里调用一次。"""
    pool = await get_pool()
    async with pool.connection() as conn:
        for stmt in _DDL:
            await conn.execute(stmt)
        # 清掉过期令牌（顺手机会清理，不靠定时任务）
        await conn.execute("DELETE FROM auth_tokens WHERE expires_at < now()")
        # 种子管理员：只在不存在时插入；密码固定 admin/123456（爸爸要求）
        cur = await conn.execute("SELECT 1 FROM users WHERE username = %s", ("admin",))
        if await cur.fetchone() is None:
            from .api.auth import hash_password  # 延迟导入避免环（auth → db）
            await conn.execute(
                "INSERT INTO users (username, pw_hash, role) VALUES (%s, %s, 'admin')",
                ("admin", hash_password("123456")),
            )
            log.info("已种子管理员 admin（密码见需求：123456）")
    log.info("鉴权表结构就绪")
