"""Neo4j 异步驱动封装 + 约束初始化。

选型：项目全链路 async（psycopg / LangGraph / FastAPI），用官方 AsyncGraphDatabase
避免同步 IO 阻塞事件循环。Neo4j 的 async driver 是纯 asyncio 实现，
不像 psycopg 那样受 Windows Proactor/Selector 差异影响。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession

from ..config import get_settings
from ..ww_logger import get_logger

neo4j_log = get_logger("neo4j")

_driver: AsyncDriver | None = None
_driver_loop = None


def get_driver() -> AsyncDriver:
    """进程内单例 driver（惰性创建，按当前 running loop 绑定）。

    celery 每个任务都新建事件循环，旧的 driver 若绑在已关闭的 loop 上会触发
    'Future attached to a different loop'。这里记录创建它的 loop，loop 变了就重建。
    """
    global _driver, _driver_loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if _driver is None or _driver_loop is not loop:
        if _driver is not None:
            neo4j_log.warning("Neo4j driver 绑定的 loop 已变更，丢弃旧 driver 重建")
        s = get_settings()
        _driver = AsyncGraphDatabase.driver(
            s.NEO4J_URI,
            auth=(s.NEO4J_USER, s.NEO4J_PASSWORD),
            max_connection_pool_size=20,
            connection_acquisition_timeout=30.0,
        )
        _driver_loop = loop
        neo4j_log.info("Neo4j driver 已创建 | uri=%s", s.NEO4J_URI)
    return _driver


async def close_driver() -> None:
    """关闭 driver 重置状态"""
    global _driver, _driver_loop
    if _driver is not None:
        await _driver.close()
        neo4j_log.info("Neo4j driver 已关闭")
    _driver = None
    _driver_loop = None


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession]:
    """获取会话"""
    async with get_driver().session() as s:
        yield s


async def ping() -> bool:
    try:
        await get_driver().verify_connectivity()
        neo4j_log.info("Neo4j 连通正常")
        return True
    except Exception as exc:  # noqa: BLE001
        neo4j_log.error("Neo4j 连接失败: %s", exc)
        return False


# ── 唯一约束：MERGE 幂等的前提 ──────────────────────
_CONSTRAINTS: tuple[str, ...] = (
    "CREATE CONSTRAINT character_name IF NOT EXISTS "
    "FOR (n:Character) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT skill_name IF NOT EXISTS "
    "FOR (n:Skill) REQUIRE (n.name, n.character) IS UNIQUE",
    "CREATE CONSTRAINT chain_name IF NOT EXISTS "
    "FOR (n:ChainNode) REQUIRE (n.name, n.character) IS UNIQUE",
    "CREATE CONSTRAINT material_name IF NOT EXISTS "
    "FOR (n:Material) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT weapon_name IF NOT EXISTS "
    "FOR (n:Weapon) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT echoset_name IF NOT EXISTS "
    "FOR (n:EchoSet) REQUIRE n.name IS UNIQUE",
)


async def init_schema() -> None:
    """建唯一约束。重复执行安全（IF NOT EXISTS）。"""
    async with get_session() as s:
        for cypher in _CONSTRAINTS:
            await s.execute_write(_run_one, cypher)
    neo4j_log.info("Neo4j 约束初始化完成 | %d 条", len(_CONSTRAINTS))


async def _run_one(tx, cypher: str) -> None:
    """执行一条 Cypher 语句"""
    await (await tx.run(cypher)).consume()
