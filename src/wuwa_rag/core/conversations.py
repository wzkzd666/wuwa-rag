"""会话转录：把每轮问答落到 PG 的 sessions / messages（001_init.sql 的「记忆四表」）。

**为什么需要这个模块**：会话历史原来只存在浏览器 localStorage 里，而且**不按用户分**——
同一台机器换个账号登录，就能看到上一个人的会话列表和消息。转录落库后：
  - 历史跟着**账号**走（换浏览器/换设备都在），不再跟着浏览器走；
  - 每次查询都带 `user_id` 条件，别的用户查不到。

**与 checkpointer 的分工**（两条线都挂在同一个 key 上，否则会错位）：
  - `lg` schema（`memory.py` 的 checkpointer）：**模型侧**的滚动记忆——
    `history` 窗口 + `context_summary`，只按 thread_id 建键，**没有用户维度**，
    所以它不能当「用户可见的历史来源」：谁拿到别人的 thread_id 就能读到别人的多轮记忆。
  - `public.sessions` / `public.messages`（本模块）：**用户侧**可翻阅的转录，按 user_id 隔离。

两条线共用的 key 是 `thread_key(user_id, client_thread_id)` = `u<user_id>:<client_tid>`，
即写进 `sessions.session_id` 的就是**checkpointer 的 thread_id**。这样：
  - 001 里 `ux_sessions_id` 是全局 UNIQUE，加用户前缀后天然不会撞车
    （否则 alice 和 bob 各自新建的会话可能都是 `abc123`，直接唯一键冲突）；
  - 前端拿到的仍是自己的短 id（存在 `meta.client_thread_id`），前缀是服务端内部约定。
"""
from __future__ import annotations

from psycopg.types.json import Jsonb

from wuwa_rag.core.authdb import get_pool
from wuwa_rag.ww_logger import get_logger

log = get_logger("conv")

# 标题长度上限（侧栏一行放不下，超出截断加省略号）
MAX_TITLE_CHARS = 22

# 与 pgsql/003_conversation_scope.sql 保持一致（幂等）。
# sessions / messages 两张表在 001_init.sql 里已有，但那里**没有 user_id 列**——
# 001 的 DDL 不能改（那是部署者手工建库的基线），所以这里做**增量 ALTER**：
# 幂等、可重复执行、不改既有列语义。
_DDL = [
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id         BIGSERIAL PRIMARY KEY,
        session_id TEXT        NOT NULL,
        title      TEXT,
        meta       JSONB       NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_sessions_id ON sessions(session_id)",
    """
    CREATE TABLE IF NOT EXISTS messages (
        id         BIGSERIAL PRIMARY KEY,
        session_id TEXT        NOT NULL,
        role       TEXT        NOT NULL,
        content    TEXT        NOT NULL,
        meta       JSONB       NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_messages_session ON messages(session_id, created_at)",
    # ↓ 本模块新增：归属用户。历史会话必须能按用户列出来，否则没法隔离。
    "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS user_id BIGINT",
    "CREATE INDEX IF NOT EXISTS ix_sessions_user ON sessions(user_id, created_at DESC)",
]


def thread_key(user_id: int, thread_id: str) -> str:
    """会话的规范 key（= checkpointer 的 thread_id = sessions.session_id）。

    前缀里的 user_id 用的是 `users.id`（BIGINT），不含冒号，所以反向解析安全。
    """
    return f"u{user_id}:{thread_id}"


def client_thread_id(session_id: str) -> str:
    """从规范 key 反解出前端用的短 id（`u3:abc123` → `abc123`）。"""
    head, sep, tail = session_id.partition(":")
    return tail if sep and head.startswith("u") else session_id


def title_from(text: str) -> str:
    """用首条用户消息生成会话标题。"""
    clean = " ".join(text.split()).strip()
    if not clean:
        return "新对话"
    return clean[:MAX_TITLE_CHARS] + "…" if len(clean) > MAX_TITLE_CHARS else clean


async def ensure_schema() -> None:
    """幂等建表 + 加列。API lifespan 里调用一次。"""
    pool = await get_pool()
    async with pool.connection() as conn:
        for stmt in _DDL:
            await conn.execute(stmt)
    log.info("会话转录表结构就绪")


async def ensure_conversation(user_id: int, thread_id: str, question: str = "") -> str:
    """取或建会话行，返回 session_id。

    已存在则**只补空标题**（用户改过的不覆盖）。
    """
    sid = thread_key(user_id, thread_id)
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            "INSERT INTO sessions (session_id, title, meta, user_id)"
            " VALUES (%s, %s, %s, %s) ON CONFLICT (session_id) DO NOTHING",
            (sid, title_from(question), Jsonb({"client_thread_id": thread_id}), user_id),
        )
        # 首句为空（例如 regenerate 时已删光）时不覆盖既有标题
        if question:
            await conn.execute(
                "UPDATE sessions SET title = %s"
                " WHERE session_id = %s AND user_id = %s"
                "   AND (title IS NULL OR title = '' OR title = %s)",
                (title_from(question), sid, user_id, "新对话"),
            )
    return sid


async def list_conversations(user_id: int, limit: int = 200, query: str = "") -> list[dict]:
    """该用户的会话列表（不含消息正文，只带最后一条预览）。新的在前。

    `query` 非空时按「标题**或任意一条消息正文**」模糊匹配 —— 转录在服务端，
    不在服务端搜就等于把「搜消息内容」这个功能弄丢了。匹配用的是 ILIKE，
    中文没有分词问题，够用。
    """
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            SELECT s.session_id,
                   coalesce(nullif(s.title, ''), '新对话')          AS title,
                   s.created_at,
                   coalesce(agg.last_at, s.created_at)              AS updated_at,
                   coalesce(agg.n, 0)                               AS message_count,
                   coalesce(last.role, '')                          AS last_role,
                   coalesce(last.content, '')                       AS last_content
              FROM sessions s
              LEFT JOIN LATERAL (
                  SELECT count(*)::int AS n, max(m.created_at) AS last_at
                    FROM messages m WHERE m.session_id = s.session_id
              ) agg ON true
              LEFT JOIN LATERAL (
                  SELECT m.role, m.content FROM messages m
                   WHERE m.session_id = s.session_id ORDER BY m.id DESC LIMIT 1
              ) last ON true
             WHERE s.user_id = %s
               AND (%s = '' OR s.title ILIKE %s
                    OR EXISTS (SELECT 1 FROM messages m2
                                WHERE m2.session_id = s.session_id AND m2.content ILIKE %s))
             ORDER BY coalesce(agg.last_at, s.created_at) DESC
             LIMIT %s
            """,
            (user_id, query, f"%{query}%", f"%{query}%", limit),
        )
        return list(await cur.fetchall())


async def get_conversation(user_id: int, thread_id: str) -> dict | None:
    """单个会话 + 全部消息。**不属于该用户一律返回 None**（不是 404 的 403）。"""
    sid = thread_key(user_id, thread_id)
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT session_id, coalesce(nullif(title, ''), '新对话') AS title, created_at"
            "  FROM sessions WHERE session_id = %s AND user_id = %s",
            (sid, user_id),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        cur = await conn.execute(
            "SELECT id, role, content, meta, created_at FROM messages"
            " WHERE session_id = %s ORDER BY id",
            (sid,),
        )
        msgs = list(await cur.fetchall())
    return {**row, "messages": msgs}


async def rename_conversation(user_id: int, thread_id: str, title: str) -> bool:
    sid = thread_key(user_id, thread_id)
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "UPDATE sessions SET title = %s WHERE session_id = %s AND user_id = %s",
            (title.strip()[:MAX_TITLE_CHARS] or "新对话", sid, user_id),
        )
        return cur.rowcount > 0


async def delete_conversation(user_id: int, thread_id: str) -> bool:
    """删除会话及其消息。

    消息**物理删除**（不是软删）：既然用户点了「删除会话」，转录就该真的消失；
    留着孤儿消息等于「界面上删了、库里还在」，这与隔离的初衷相悖。
    （001 里 messages 的注释写的是「只追加，永不删改」——那条描述的是正常问答路径的
    追加语义，删除会话属于用户主动行使的删除权，不冲突。）
    """
    sid = thread_key(user_id, thread_id)
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "DELETE FROM sessions WHERE session_id = %s AND user_id = %s", (sid, user_id)
        )
        if cur.rowcount == 0:
            return False
        await conn.execute("DELETE FROM messages WHERE session_id = %s", (sid,))
        return True


async def clear_conversations(user_id: int) -> int:
    """清空该用户全部会话，返回删除的会话数。"""
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT session_id FROM sessions WHERE user_id = %s", (user_id,)
        )
        sids = [r["session_id"] for r in await cur.fetchall()]
        if not sids:
            return 0
        await conn.execute("DELETE FROM sessions WHERE user_id = %s", (user_id,))
        await conn.execute("DELETE FROM messages WHERE session_id = ANY(%s)", (sids,))
        return len(sids)


async def append_message(user_id: int, thread_id: str, role: str,
                         content: str, meta: dict | None = None) -> int:
    """追加一条转录。返回消息 id。

    ⚠️ 逐条 INSERT，不做 upsert：正常路径下同一轮问答只会走到这里一次
    （用户消息在流式开始前、助手消息在流结束的 finally 里）。
    """
    sid = thread_key(user_id, thread_id)
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "INSERT INTO messages (session_id, role, content, meta)"
            " VALUES (%s, %s, %s, %s) RETURNING id",
            (sid, role, content, Jsonb(meta or {})),
        )
        row = await cur.fetchone()
    return int(row["id"])


async def get_history(user_id: int, thread_id: str, limit: int = 6) -> list[dict]:
    """给模型回放用的最近若干条（老→新）。仅 role/content，不含 meta。"""
    sid = thread_key(user_id, thread_id)
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT role, content FROM ("
            "  SELECT id, role, content FROM messages"
            "   WHERE session_id = %s ORDER BY id DESC LIMIT %s"
            ") t ORDER BY id",
            (sid, limit),
        )
        return [{"role": r["role"], "content": r["content"]} for r in await cur.fetchall()]


async def last_user_message(user_id: int, thread_id: str) -> dict | None:
    """最后一条用户消息（重新生成时要知道重答哪一句）。"""
    sid = thread_key(user_id, thread_id)
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT id, content FROM messages"
            " WHERE session_id = %s AND role = 'user' ORDER BY id DESC LIMIT 1",
            (sid,),
        )
        return await cur.fetchone()


async def truncate_from(user_id: int, thread_id: str, message_id: int) -> int:
    """删除 id >= message_id 的转录（重新生成：把「最后一句问 + 它的答」整段抹掉重来）。

    先校验会话归属，避免拿别人的 thread_id 删别人的转录。
    """
    sid = thread_key(user_id, thread_id)
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = %s AND user_id = %s", (sid, user_id)
        )
        if await cur.fetchone() is None:
            return 0
        cur = await conn.execute(
            "DELETE FROM messages WHERE session_id = %s AND id >= %s", (sid, message_id)
        )
        return cur.rowcount
