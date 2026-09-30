-- ============================================================
-- 003_conversation_scope.sql
-- 会话转录的「按用户隔离」增量 DDL
--
-- 背景：sessions / messages 两张表在 001_init.sql 里已经建好（注释叫「记忆四表」），
-- 但当时没考虑多用户——sessions 没有归属用户这一列，导致「列会话」这件事没法按用户做。
--
-- 这里**不改 001**（那是部署者手工建库的基线，改了会让老库和新库不一致），
-- 只做幂等的增量补齐：加一列 + 建索引。可重复执行。
--
-- 与 src/wuwa_rag/conversations.py 的 _DDL 内容保持一致：
-- 那份是 API 启动时自动跑的（ensure_schema），这份是给手工建库 / 核对结构用的。
-- ============================================================

-- sessions：会话行。session_id 存的是**规范化 key** `u<user_id>:<前端短 id>`，
-- 与 LangGraph checkpointer 的 thread_id 完全相同，所以天然全局唯一（ux_sessions_id）。
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS user_id BIGINT;

CREATE INDEX IF NOT EXISTS ix_sessions_user ON sessions(user_id, created_at DESC);

-- messages 不用改：它通过 session_id 关联，而 session_id 里已经带了用户前缀。
-- 这里只是把索引补全（001 建的是 (session_id, created_at)，与代码里的 ORDER BY id 一致可复用）。
CREATE INDEX IF NOT EXISTS ix_messages_session ON messages(session_id, created_at);
