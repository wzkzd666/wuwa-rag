"""把结构化事实写入 Neo4j。

幂等设计：
  1. 全部用 MERGE，重复执行不产生重复节点/关系。
  2. 关系 MERGE 的键刻意不含 qty / rank 这类易变属性，改为 MERGE 后 SET，
     否则属性一变就会新建一条重复关系。
  3. 角色属性用 `SET c += $attrs` 且 attrs 已过滤空值，
     保证「队友空壳节点」不会被后来的空属性覆盖。
"""
from __future__ import annotations

import asyncio

from ..config import ensure_dirs
from ..ww_logger import get_logger
from .extract import CharacterFacts, all_characters, extract_character, load_chunks
from .neo4j_client import close_driver, get_session, init_schema, ping

log = get_logger("neo4j")

_C_CHAR = """
UNWIND $rows AS r
MERGE (c:Character {name: r.name})
SET c += r.attrs
"""

_C_SKILL = """
UNWIND $rows AS r
MATCH (c:Character {name: r.character})
MERGE (s:Skill {name: r.name, character: r.character})
SET s.kind = r.kind
MERGE (c)-[:HAS_SKILL]->(s)
"""

_C_CHAIN = """
UNWIND $rows AS r
MATCH (c:Character {name: r.character})
MERGE (n:ChainNode {name: r.name, character: r.character})
SET n.seq = r.seq, n.effect = r.effect
MERGE (c)-[:HAS_CHAIN]->(n)
"""

_C_MAT = """
UNWIND $rows AS r
MATCH (c:Character {name: r.character})
MERGE (m:Material {name: r.name})
MERGE (c)-[rel:NEEDS_MATERIAL {stage: r.stage, kind: r.kind}]->(m)
SET rel.qty = r.qty
"""

_C_WEAPON = """
UNWIND $rows AS r
MATCH (c:Character {name: r.character})
MERGE (w:Weapon {name: r.name})
MERGE (c)-[rel:RECOMMENDS_WEAPON]->(w)
SET rel.rank = r.rank
"""

_C_ECHO = """
UNWIND $rows AS r
MATCH (c:Character {name: r.character})
MERGE (e:EchoSet {name: r.name})
MERGE (c)-[:RECOMMENDS_ECHO]->(e)
"""

_C_BUILD = """
UNWIND $rows AS r
MATCH (c:Character {name: r.character})
MERGE (e:EchoSet {name: r.name})
MERGE (c)-[rel:HAS_BUILD {stage: r.stage}]->(e)
SET rel.cost = r.cost, rel.pieces = r.pieces
"""

# 队友：一个队友可能出现在多支队伍里，teams 去重累加
_C_TEAM = """
UNWIND $rows AS r
MATCH (c:Character {name: r.character})
MERGE (t:Character {name: r.name})
MERGE (c)-[rel:SYNERGIZES_WITH]->(t)
SET rel.teams = CASE
  WHEN r.team IN coalesce(rel.teams, []) THEN rel.teams
  ELSE coalesce(rel.teams, []) + r.team END
"""


async def _run(cypher: str, rows: list[dict]) -> None:
    if not rows:
        return
    async with get_session() as s:
        await s.execute_write(_write, cypher, rows)


async def _write(tx, cypher: str, rows: list[dict]) -> None:
    await (await tx.run(cypher, rows=rows)).consume()


async def upsert_character(f: CharacterFacts) -> None:
    attrs = {
        "element": f.attrs.get("属性", ""),
        "weapon": f.attrs.get("武器", ""),
        "gender": f.attrs.get("性别", ""),
        "birthplace": f.attrs.get("出生", ""),
        "echo_main": f.echo_main,
        "echo_main_stats": f.echo_main_stats,
        "echo_sub_stats": f.echo_sub_stats,
    }
    # 过滤空值：避免队友空壳节点被空属性覆盖
    await _run(_C_CHAR, [{"name": f.name,
                          "attrs": {k: v for k, v in attrs.items() if v}}])

    await _run(_C_SKILL, [{"character": f.name, **s} for s in f.skills])
    await _run(_C_CHAIN, [{"character": f.name, **c} for c in f.chains])
    await _run(_C_MAT, [{"character": f.name, **m} for m in f.materials])
    await _run(_C_ECHO, [{"character": f.name, "name": e} for e in f.echoes])
    await _run(_C_BUILD, [
        {"character": f.name, "name": s, "stage": b["stage"],
         "cost": b["cost"], "pieces": p}
        for b in f.builds for s, p in b["sets"]
    ])
    await _run(_C_WEAPON, [{"character": f.name, "name": w, "rank": i + 1}
                           for i, w in enumerate(f.weapons)])
    await _run(_C_TEAM, [{"character": f.name, **t} for t in f.teammates])

    log.info(
        "写入 %s | 技能%d 共鸣链%d 材料%d 声骸%d 配装%d 武器%d 队友%d | 属性%s",
        f.name, len(f.skills), len(f.chains), len(f.materials), len(f.echoes),
        len(f.builds), len(f.weapons), len(f.teammates), f.attrs or "-",
    )


async def _main() -> None:
    ensure_dirs()
    if not await ping():
        return
    await init_schema()

    chunks = load_chunks()
    names = all_characters(chunks)
    log.info("待处理角色: %s", names)

    for n in names:
        await upsert_character(extract_character(chunks, n))

    await close_driver()
    log.info("图谱构建完成")


def main() -> None:
    # 与 build_index.py 保持一致：Windows 下显式 Selector 事件循环
    asyncio.run(_main(), loop_factory=asyncio.SelectorEventLoop)


if __name__ == "__main__":
    main()
