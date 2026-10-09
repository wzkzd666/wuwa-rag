"""鉴权/画像共用的 PostgreSQL 连接池 + 建表 + admin 种子。

为什么单独一个文件：auth（users/auth_tokens）与画像（user_facts）共用一个池子；
API 启动时 ensure 一次 DDL（幂等 CREATE IF NOT EXISTS），就不必手工跑 002_auth.sql——
那份 SQL 是给手工建库/核对结构用的，两边内容保持一致。

池子参数照抄 memory.py 的经验：autocommit=True + dict_row；public schema（DSN 不带
search_path，默认 public，与 001/002 建表位置一致）。
"""
from __future__ import annotations

import secrets

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from wuwa_rag.config import get_settings
from wuwa_rag.core.security import hash_password
from wuwa_rag.ww_logger import get_logger

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
        must_change BOOLEAN     NOT NULL DEFAULT FALSE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    # 老库补列（幂等）：must_change 标记「管理员用部署者显式配置的初始口令建号」，
    # 登录响应据此提示前端引导改密。历史用户默认 FALSE，不影响既有账号。
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS must_change BOOLEAN NOT NULL DEFAULT FALSE",
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
        category   TEXT,
        confidence REAL,
        source     TEXT,
        valid_from TIMESTAMPTZ NOT NULL DEFAULT now(),
        valid_to   TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    # 老库补列（幂等）：category 是「同类事实覆盖」的分类键（见 services/profile.py）。
    # CREATE TABLE IF NOT EXISTS 对已存在的表不生效，所以老库必须靠这句补列，
    # 否则 save_facts 的 INSERT 会报 column "category" does not exist。
    # NULL = 不参与覆盖，历史事实按叠加原样保留，不丢数据。
    "ALTER TABLE user_facts ADD COLUMN IF NOT EXISTS category TEXT",
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
    """幂等建表 + 清过期 token + 种子 admin。API lifespan 里调用一次。

    管理员口令（2026-10-09 安全收紧）：**不再内置固定弱口令**。
    - `ADMIN_PASSWORD` 留空（默认）→ 随机生成一次性口令，只在本进程日志里
      打印一次；没看到日志就无法登录，不存在「默认凭据公开」的窗口。
    - 显式设置 → 用它建号，并把 admin 标记 `must_change=TRUE`，
      登录响应带 `must_change_password` 提示前端引导改密。
    两种路径都不再依赖「记得去改默认口令」这种全靠部署者自觉的约定。
    """
    pool = await get_pool()
    async with pool.connection() as conn:
        for stmt in _DDL:
            await conn.execute(stmt)
        # 清掉过期令牌（顺手机会清理，不靠定时任务）
        await conn.execute("DELETE FROM auth_tokens WHERE expires_at < now()")
        # 种子管理员：只在不存在时插入。
        cfg_pwd = (s.ADMIN_PASSWORD or "").strip()
        if cfg_pwd:
            pwd, must_change, how = cfg_pwd, True, "ADMIN_PASSWORD 显式配置（登录后应尽快改密）"
        else:
            pwd = secrets.token_urlsafe(24)
            must_change, how = False, "首启随机生成（见本条日志，只打印这一次）"
        cur = await conn.execute(
            "INSERT INTO users (username, pw_hash, role, must_change) "
            "VALUES (%s, %s, 'admin', %s) "
            "ON CONFLICT (username) DO NOTHING RETURNING username",
            ("admin", hash_password(pwd), must_change),
        )
        if await cur.fetchone() is not None:
            log.warning(
                "已种子管理员 admin。初始口令（来源：%s）：%s"
                "——口令只显示这一次，请立即妥善保存并登录后尽快修改。",
                how, pwd,
            )
    log.info("鉴权表结构就绪")
