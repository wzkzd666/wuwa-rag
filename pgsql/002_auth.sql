-- ============================================================
-- 002_auth.sql —— 系统鉴权（2026-09-22）
-- 幂等，可重复执行。与 001_init.sql 的分工：
--   001 建知识库四表 + lg schema + user_facts（当时留了表但代码没用）；
--   002 建鉴权两表，并把 user_facts 补一遍 CREATE IF NOT EXISTS 兜底
--       （老库跑过 001 就已有；新库两份都跑也不冲突）。
-- 注意：API 启动时会自动 ensure 同样的 DDL（见 api/db.py），
--       所以这份 SQL 主要供手工建库 / 核对结构用。
-- ============================================================

-- 用户表：admin 由 API 启动时种子写入（admin / 123456），游客走 /auth/register
CREATE TABLE IF NOT EXISTS users (
    id         BIGSERIAL PRIMARY KEY,
    username   TEXT        NOT NULL,
    pw_hash    TEXT        NOT NULL,                     -- scrypt$<salt hex>$<hash hex>
    role       TEXT        NOT NULL DEFAULT 'guest',     -- 'admin' / 'guest'
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_users_username ON users(username);

-- 登录令牌：随机 256bit，30 天过期；登出即删行
CREATE TABLE IF NOT EXISTS auth_tokens (
    token      TEXT        PRIMARY KEY,
    user_id    BIGINT      NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_tokens_user    ON auth_tokens(user_id);
CREATE INDEX IF NOT EXISTS ix_tokens_expires ON auth_tokens(expires_at);

-- 用户画像事实（001 已建，这里兜底；结构保持一致）
CREATE TABLE IF NOT EXISTS user_facts (
    id         BIGSERIAL PRIMARY KEY,
    user_id    TEXT,
    session_id TEXT,
    fact       TEXT        NOT NULL,
    confidence REAL,
    source     TEXT,                               -- 'chat' / 'doc'
    valid_from TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_to   TIMESTAMPTZ,                        -- NULL = 仍有效
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_facts_user ON user_facts(user_id, valid_to);
