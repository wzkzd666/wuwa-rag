"""答案反馈：点赞/点踩 + 可选文字，落 `answer_feedback`。

为什么连问答快照一起存：反馈是**事后**分析用的，而问答原文可能已被清理、或随会话
滚动出上下文窗口。只留 thread_id 的话，管理员点开一条差评时可能已经看不到当时答了什么。
"""
from __future__ import annotations

from wuwa_rag.core import conversations as conv
from wuwa_rag.core.db import get_cursor
from wuwa_rag.ww_logger import get_logger

log = get_logger("feedback")

RATING_UP = 1
RATING_DOWN = -1

# 各列的长度帽。**入口（`api/usage.FeedbackIn`）与这里引用同一组常量**，
# 不在两处各写一遍数字——那正是本项目反复出问题的「同一语义两处判据」。
# 为什么两处都要用：pydantic 那层负责「早失败 + 422 给对的人」，
# 这里这层负责「无论如何都不会把超长串写进库」（脚本/将来别的调用方未必过 pydantic）。
COMMENT_MAX = 1000
QUESTION_MAX = 2000
ANSWER_MAX = 4000
TARGET_ID_MAX = 32
NAME_MAX = 64            # provider / model
USERNAME_MAX = 64


async def save_feedback(*, user_id: str, username: str, thread_id: str, target_id: str,
                        rating: int, comment: str = "", question: str = "", answer: str = "",
                        provider: str = "", model: str = "") -> int:
    """写一条反馈，返回行 id。`rating` 只能是 1 / -1。

    同一回答重复提交 = **改主意**（踩改赞），所以 upsert 覆盖而不是插第二条 ——
    否则连点几下就能把满意度看板灌成一片差评。
    """
    if rating not in (RATING_UP, RATING_DOWN):
        raise ValueError("rating 只能是 1（满意）或 -1（不满意）")
    # thread_id / target_id 是**键的一部分**（upsert 靠 (user_id, target_id)），
    # 绝不能像快照那样截断——截断会把两个不同的 id 合成一个，
    # 表现为「给 A 点的赞跑到 B 上」。所以这里校验并拒绝，交给 API 层转 4xx。
    if not conv.THREAD_ID_RE.match(thread_id or ""):
        raise ValueError("thread_id 只能是字母/数字/下划线/连字符，最长 64 字符")
    if not target_id or len(target_id) > TARGET_ID_MAX:
        raise ValueError(f"target_id 不能为空且不得长于 {TARGET_ID_MAX} 字符")
    async with get_cursor() as cur:
        await cur.execute(
            "INSERT INTO answer_feedback (user_id, username, thread_id, target_id, rating,"
            " comment, question, answer, provider, model)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            " ON CONFLICT (user_id, target_id) DO UPDATE SET"
            "   rating = EXCLUDED.rating, comment = EXCLUDED.comment, model = EXCLUDED.model,"
            "   provider = EXCLUDED.provider, created_at = now()"
            " RETURNING id",
            (user_id, username[:USERNAME_MAX] or None, thread_id, target_id,
             rating, comment[:COMMENT_MAX] or None,
             question[:QUESTION_MAX] or None, answer[:ANSWER_MAX] or None,
             provider[:NAME_MAX] or None, model[:NAME_MAX] or None),
        )
        return (await cur.fetchone())[0]


async def list_feedback(days: int = 7, only_user: str | None = None) -> dict:
    """近 N 天的反馈列表 + 满意度汇总。`only_user` 非空时只看该用户（普通用户）。"""
    sql = (
        "SELECT id, user_id, username, thread_id, target_id, rating, comment, question, answer,"
        " provider, model, created_at FROM answer_feedback"
        " WHERE created_at >= now() - make_interval(days => %s)"
    )
    args: list = [days]
    if only_user is not None:
        sql += " AND user_id = %s"
        args.append(only_user)
    sql += " ORDER BY created_at DESC LIMIT 200"
    async with get_cursor() as cur:
        await cur.execute(sql, tuple(args))
        cols = [d.name for d in cur.description]
        items = []
        for r in await cur.fetchall():
            d = dict(zip(cols, r))
            d["created_at"] = d["created_at"].isoformat() if d["created_at"] else None
            items.append(d)
    up = sum(1 for i in items if i["rating"] == RATING_UP)
    down = sum(1 for i in items if i["rating"] == RATING_DOWN)
    return {
        "days": days,
        "items": items,
        "summary": {
            "up": up,
            "down": down,
            "total": up + down,
            # 差评占比：管理员最该盯的数字（点赞多不代表答得对）
            "down_ratio": round(down / (up + down), 4) if (up + down) else None,
        },
    }


async def my_feedback(user_id: str, target_ids: list[str]) -> dict[str, int]:
    """这些回答我点过什么（前端据此标成「已赞/已踩」，避免重复弹窗）。按回答维度，不是会话。"""
    if not target_ids:
        return {}
    async with get_cursor() as cur:
        await cur.execute(
            "SELECT target_id, rating FROM answer_feedback"
            " WHERE user_id = %s AND target_id = ANY(%s)",
            (user_id, target_ids),
        )
        return {r[0]: r[1] for r in await cur.fetchall()}


async def get_feedback(record_id: int) -> dict | None:
    """按 id 取一条反馈（删除前的归属判定用）。不存在返回 None。"""
    async with get_cursor() as cur:
        await cur.execute(
            "SELECT id, user_id, username, thread_id, rating FROM answer_feedback WHERE id = %s",
            (record_id,),
        )
        row = await cur.fetchone()
    if row is None:
        return None
    return {"id": row[0], "user_id": row[1], "username": row[2],
            "thread_id": row[3], "rating": row[4]}


async def delete_feedback(record_id: int) -> str | None:
    """删掉一条反馈，返回它的 thread_id（不存在返回 None）。只删账本，不碰会话原文。"""
    async with get_cursor() as cur:
        await cur.execute("DELETE FROM answer_feedback WHERE id = %s RETURNING thread_id",
                          (record_id,))
        row = await cur.fetchone()
    return row[0] if row else None


async def delete_all_for_user(user_id: str) -> int:
    """删掉该用户的全部评价（账号被移除时回收）。

    与 `delete_feedback(record_id)` 不是一回事：那条是「撤一条评价」，
    这条是「数据主体消失了」。两个动作都存在，谁都不许顺手替谁。
    """
    async with get_cursor() as cur:
        await cur.execute("DELETE FROM answer_feedback WHERE user_id = %s RETURNING id", (str(user_id),))
        return len(await cur.fetchall())
