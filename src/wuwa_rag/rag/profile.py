"""用户画像：把 001_init.sql 里闲置的 user_facts 表用起来（2026-09-22）。

数据流：
  用户提问 → (异步、不阻塞回答) tool 模型从消息里抽「稳定偏好事实」→ 去重入库
  下次提问 → 取该用户仍有效的事实 → 拼成 user_context 注入生成 prompt（个性化）

事实示例：「主玩角色是守岸人」「萌新，需要基础讲解」「偏爱湮灭队」。
约束：
- 只抽**稳定的长期偏好**，不抽一次性问题（「今汐几级突破」这种不能进画像）；
- 每条 ≤40 字，单次最多 3 条，防倒垃圾；
- 同文去重：活跃事实（valid_to IS NULL）里已有完全相同的就跳过；
- 删除是软删（valid_to=now），与表设计的「不物理删除」一致；
- 抽取走 get_tool_llm()（qwen3:8b，temperature=0）；模型挂了/解析失败一律回落 []，
  画像失败绝不影响问答主链路。
"""
from __future__ import annotations

import json
import re

from ..authdb import get_pool
from ..ww_logger import get_logger
from .llm import get_tool_llm

# 事实长度 / 条数硬帽
FACT_MIN_CHARS = 4
FACT_MAX_CHARS = 40
MAX_FACTS_PER_MSG = 3
# 注入 prompt 的事实条数上限（太多会挤占注意力）
MAX_FACTS_IN_PROMPT = 8

log = get_logger("profile")

_EXTRACT_SYSTEM = (
    "你从用户与游戏助手的对话里提取该用户的长期偏好事实。"
    "只提取稳定、可复用的信息（主玩角色/常用配队、新手还是老手、喜欢的玩法、称呼偏好等）；"
    "一次性问题（具体数值、某次突破）不要提取。"
    "没有可提取的就返回空数组。只输出 JSON 数组，每条是不超过40字的短句，"
    "形如：[\"主玩角色是守岸人\"]，不要任何解释。"
)


async def extract_facts_safe(question: str) -> list[str]:
    """从一条用户消息抽事实；任何异常回落 []（画像绝不挡问答）。"""
    if not question or len(question.strip()) < 4:
        return []
    try:
        rsp = await get_tool_llm().ainvoke(
            [("system", _EXTRACT_SYSTEM), ("user", question[:500])]
        )
        content = rsp.content
        if isinstance(content, list):
            # 某些返回形态是 [{"type":"text","text":...}] 分片
            content = "".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        text = str(content or "")
        m = re.search(r"\[.*\]", text, re.S)
        if not m:
            return []
        arr = json.loads(m.group(0))
        if not isinstance(arr, list):
            return []
        out: list[str] = []
        for item in arr[:MAX_FACTS_PER_MSG]:
            if not isinstance(item, str):
                continue
            fact = re.sub(r"\s+", " ", item).strip()
            if FACT_MIN_CHARS <= len(fact) <= FACT_MAX_CHARS and fact not in out:
                out.append(fact)
        return out
    except Exception as exc:
        log.warning("画像抽取失败（不影响问答）：%s", exc)
        return []


async def save_facts(user_id: str, session_id: str, facts: list[str]) -> int:
    """去重入库。返回实际新增条数。同文活跃事实已存在则跳过。"""
    if not facts:
        return 0
    pool = await get_pool()
    added = 0
    async with pool.connection() as conn:
        for fact in facts:
            cur = await conn.execute(
                "SELECT 1 FROM user_facts WHERE user_id = %s AND fact = %s AND valid_to IS NULL",
                (user_id, fact),
            )
            if await cur.fetchone() is not None:
                continue
            await conn.execute(
                "INSERT INTO user_facts (user_id, session_id, fact, confidence, source)"
                " VALUES (%s, %s, %s, %s, 'chat')",
                (user_id, session_id, fact, 0.8),
            )
            added += 1
    if added:
        log.info("画像入库 user=%s 新增 %d 条", user_id, added)
    return added


async def get_facts(user_id: str, limit: int = 100) -> list[dict]:
    """某用户的活跃画像事实，新的在前。"""
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT id, fact, confidence, source, created_at FROM user_facts"
            " WHERE user_id = %s AND valid_to IS NULL"
            " ORDER BY created_at DESC LIMIT %s",
            (user_id, limit),
        )
        return list(await cur.fetchall())


async def soft_delete_fact(user_id: str, fact_id: int) -> bool:
    """软删（valid_to=now）。只能删自己的：带 user_id 条件。"""
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "UPDATE user_facts SET valid_to = now()"
            " WHERE id = %s AND user_id = %s AND valid_to IS NULL",
            (fact_id, user_id),
        )
        return cur.rowcount > 0


async def all_users_stats() -> list[dict]:
    """管理员视角：每个用户的画像条数（users 左连 facts，0 条也在列）。"""
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            SELECT u.username, u.role, u.created_at,
                   COUNT(f.id) FILTER (WHERE f.valid_to IS NULL) AS facts
            FROM users u LEFT JOIN user_facts f
              ON f.user_id = u.username
            GROUP BY u.id ORDER BY u.created_at
            """
        )
        return list(await cur.fetchall())


def facts_to_context(facts: list[dict]) -> str:
    """把画像事实拼成注入生成 prompt 的一段话（供 chain._build_prompt 使用）。"""
    lines = [str(f["fact"]) for f in facts[:MAX_FACTS_IN_PROMPT]]
    if not lines:
        return ""
    return "；".join(lines)
