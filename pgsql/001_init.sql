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
-- ⚠️ 2026-09-22：hash 是「纯正文」sha256，跨角色会大量复用（突破材料表这类小表格的正文
-- 不含角色名，57 个角色逐字相同）。曾写成 UNIQUE 当幂等键 → 第 2 个角色起的同文本块被
-- `ON CONFLICT (hash) DO NOTHING` 静默吞掉（实测丢 124 块，「含一阶突破」的角色 9→57）。
-- 幂等键是 chunk_id（含角色，见下面 ux_chunks_chunk_id）；hash 只作查询索引，必须非唯一。
DROP INDEX IF EXISTS ux_chunks_hash;              -- 清掉历史库里遗留的错误唯一索引
CREATE INDEX IF NOT EXISTS ix_chunks_hash     ON chunks(hash);
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
-- crawl_runs 同时兼作「入库提交账本」：谁在什么时候提交了哪个角色的抓取。
-- 原来角色名只塞在 stats.character 里、且行是 **worker 抓取时**才建的 —— 于是提交人无处可记
-- （worker 只知道自己跑了什么，不知道是谁点的），前端那份提交列表只能放浏览器内存里：
-- 刷新即失、换设备看不到，也没法回答「这条是谁提交的」。
-- 现在：行由 **API 在提交时**先建（只有它知道提交人），把 id 当 run_id 传给 crawl 步，
-- worker 复用同一行更新状态、不再另插。character 提升为独立列，并从 stats 回填老数据。
CREATE TABLE IF NOT EXISTS crawl_runs (
    id          BIGSERIAL PRIMARY KEY,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    status      TEXT,
    stats       JSONB NOT NULL DEFAULT '{}'::jsonb
);
ALTER TABLE crawl_runs ADD COLUMN IF NOT EXISTS character TEXT;
ALTER TABLE crawl_runs ADD COLUMN IF NOT EXISTS submitted_by TEXT;        -- users.id
ALTER TABLE crawl_runs ADD COLUMN IF NOT EXISTS submitted_by_name TEXT;   -- 冗余名，改名后历史仍可读
ALTER TABLE crawl_runs ADD COLUMN IF NOT EXISTS chain_id TEXT;
ALTER TABLE crawl_runs ADD COLUMN IF NOT EXISTS error TEXT;
ALTER TABLE crawl_runs ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now();
CREATE INDEX IF NOT EXISTS ix_crawl_runs_started ON crawl_runs(started_at DESC);
-- 老库回填：角色名原来只存在 stats 里
UPDATE crawl_runs SET character = stats ->> 'character'
WHERE character IS NULL AND stats ? 'character';

-- ========== LLM 用量与答案反馈（管理员看板的数据源）==========
-- 用量：每次 LLM 调用一行。provider 分 local/cloud —— 本地 aemeath/qwen3 不烧钱、
-- 云端按 token 计价，两者的成本口径完全不同，混在一起算等于没有。
-- scene 标出这次调用是干什么的：chat 最终作答 / tool 工具任务 / verify 校验 / emotion 情绪…
CREATE TABLE IF NOT EXISTS llm_usage (
    id                BIGSERIAL PRIMARY KEY,
    user_id           TEXT,                    -- 未登录/无归属时为空（如 CLI）
    username          TEXT,
    thread_id         TEXT,
    scene             TEXT        NOT NULL,   -- chat | tool | verify | emotion | other
    provider          TEXT        NOT NULL,   -- local | cloud
    model             TEXT        NOT NULL,
    prompt_tokens     INTEGER     NOT NULL DEFAULT 0,
    completion_tokens INTEGER     NOT NULL DEFAULT 0,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_llm_usage_created ON llm_usage(created_at DESC);
CREATE INDEX IF NOT EXISTS ix_llm_usage_user    ON llm_usage(user_id, created_at DESC);

-- 反馈：点赞/点踩 + 可选文字 + 当时的问答快照。
-- 为什么存快照而不只存 thread_id：反馈是**事后**分析用的，问答原文可能已被清理或
-- 随会话滚动出窗口；只留 id 的话管理员点开时可能已经看不到当时答了什么。
CREATE TABLE IF NOT EXISTS answer_feedback (
    id         BIGSERIAL PRIMARY KEY,
    user_id    TEXT        NOT NULL,
    username   TEXT        NOT NULL,
    thread_id  TEXT        NOT NULL,
    target_id  TEXT        NOT NULL,           -- 被评价的那条回答（前端消息 id）
    rating     SMALLINT    NOT NULL,          -- 1 满意 / -1 不满意
    comment    TEXT,                          -- 可选补充说明
    question   TEXT,
    answer     TEXT,
    provider   TEXT,
    model      TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_feedback_created ON answer_feedback(created_at DESC);
CREATE INDEX IF NOT EXISTS ix_feedback_user    ON answer_feedback(user_id, created_at DESC);
-- 同一回答重复提交时覆盖（用户改主意、或前端重试），避免看板被连点灌水
CREATE UNIQUE INDEX IF NOT EXISTS ux_feedback_target
    ON answer_feedback(user_id, target_id);

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
    category   TEXT,                               -- 同类覆盖的分类键；NULL = 不参与覆盖（叠加保留）
    confidence REAL,
    source     TEXT,                               -- 'chat' / 'doc'
    valid_from TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_to   TIMESTAMPTZ,                        -- NULL = 仍有效
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_facts_user ON user_facts(user_id, valid_to);
-- 老库补列（幂等）。category 为 NULL 的历史事实不参与同类覆盖，仍按叠加保留 —— 不丢数据。
ALTER TABLE user_facts ADD COLUMN IF NOT EXISTS category TEXT;

CREATE TABLE IF NOT EXISTS sessions (
    id         BIGSERIAL PRIMARY KEY,
    session_id TEXT        NOT NULL,
    title      TEXT,
    meta       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_sessions_id ON sessions(session_id);
