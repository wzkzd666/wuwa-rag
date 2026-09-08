-- LangGraph checkpointer 专用 schema（后续由其自动建表）
CREATE SCHEMA IF NOT EXISTS lg;

-- ========== 文档：一个角色一篇原文 ==========
CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    character   TEXT        NOT NULL,
    source      TEXT        NOT NULL DEFAULT 'kurobbs',
    title       TEXT,
    raw_uri     TEXT,                          -- RustFS 指针（对象）
    raw_sha256  TEXT        NOT NULL,          -- 幂等：原文对象 hash
    raw_size    BIGINT,
    mime_type   TEXT,
    version     TEXT,
    deleted_at  TIMESTAMPTZ,                   -- 软删除；对象侧绝不同步删
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_documents_raw_sha ON documents(raw_sha256);
CREATE INDEX IF NOT EXISTS ix_documents_character ON documents(character);

-- ========== 分块：真源，存原文文本 ==========
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    chunk_id    TEXT        NOT NULL,          -- 业务键 角色::模块::组件::tab::hash8
    document_id BIGINT      REFERENCES documents(id) ON DELETE CASCADE,
    character   TEXT        NOT NULL,
    element     TEXT,
    weapon      TEXT,
    rarity      SMALLINT,
    module      TEXT,                          -- H2
    component   TEXT,                          -- H3
    tab         TEXT,                          -- H4
    level       SMALLINT,                      -- 层级深度
    breadcrumb  TEXT,                          -- 角色 › 模块 › 组件 › tab
    text        TEXT        NOT NULL,          -- 块正文（含标题行，自解释）
    has_table   BOOLEAN     NOT NULL DEFAULT false,
    char_count  INTEGER     NOT NULL,
    hash        TEXT        NOT NULL,          -- 幂等 + 增量：content_sha256
    source      TEXT        NOT NULL DEFAULT 'doc',
    version     TEXT,
    meta        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_chunks_hash     ON chunks(hash);
CREATE UNIQUE INDEX IF NOT EXISTS ux_chunks_chunk_id ON chunks(chunk_id);
CREATE INDEX IF NOT EXISTS ix_chunks_character ON chunks(character);
CREATE INDEX IF NOT EXISTS ix_chunks_char_mod   ON chunks(character, module);

-- ========== 立绘：图存 RustFS，描述走向量 ==========
CREATE TABLE IF NOT EXISTS images (
    id                   BIGSERIAL PRIMARY KEY,
    character            TEXT        NOT NULL,
    image_uri            TEXT        NOT NULL,   -- RustFS 指针
    image_sha256         TEXT        NOT NULL,
    width                INTEGER,
    height               INTEGER,
    mime_type            TEXT,
    visual_desc_chunk_id TEXT,                   -- 关联 chunks.chunk_id（VLM 描述）
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_images_sha ON images(image_sha256);
CREATE INDEX IF NOT EXISTS ix_images_character  ON images(character);

-- ========== 工具调用审计 ==========
CREATE TABLE IF NOT EXISTS tool_calls (
    id          BIGSERIAL PRIMARY KEY,
    run_id      TEXT,
    tool_name   TEXT        NOT NULL,
    args        JSONB,
    result_hash TEXT,
    status      TEXT,
    duration_ms INTEGER,
    error       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_tool_calls_run     ON tool_calls(run_id);
CREATE INDEX IF NOT EXISTS ix_tool_calls_created ON tool_calls(created_at);

-- ========== 采集运行记录 ==========
CREATE TABLE IF NOT EXISTS crawl_runs (
    id          BIGSERIAL PRIMARY KEY,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    status      TEXT,
    stats       JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- ========== 记忆四表 ==========
CREATE TABLE IF NOT EXISTS messages (              -- 只追加，永不删改
    id         BIGSERIAL PRIMARY KEY,
    session_id TEXT        NOT NULL,
    role       TEXT        NOT NULL,
    content    TEXT        NOT NULL,
    meta       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_messages_session ON messages(session_id, created_at);

CREATE TABLE IF NOT EXISTS session_summaries (     -- 派生，可重算
    id            BIGSERIAL PRIMARY KEY,
    session_id    TEXT        NOT NULL,
    summary       TEXT        NOT NULL,
    message_count INTEGER,
    model         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_summaries_session ON session_summaries(session_id, created_at);

CREATE TABLE IF NOT EXISTS user_facts (            -- 软删除，不物理删除
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

CREATE TABLE IF NOT EXISTS sessions (
    id         BIGSERIAL PRIMARY KEY,
    session_id TEXT        NOT NULL,
    title      TEXT,
    meta       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_sessions_id ON sessions(session_id);
