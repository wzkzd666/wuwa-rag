"""两路检索：图谱（Cypher 模板）+ 向量/稀疏（Chroma + BM25，RRF 融合）。"""
from __future__ import annotations

from functools import lru_cache

import chromadb
from chromadb.config import Settings

from ..config import get_settings
from ..graph.neo4j_client import get_session
from ..retrieval.bm25 import BM25Index
from ..retrieval.embeddings import BgeM3Embeddings
from ..ww_logger import get_logger

log = get_logger("rag")

# 用模板而不是 Text2Cypher：固定领域下模板 100% 可控，
# qwen3:8b 生成 Cypher 幻觉率高，而且慢。
CYPHER: dict[str, str] = {
    "属性": """MATCH (c:Character {name:$n}) RETURN c.element AS 属性, c.weapon AS 武器,
        c.gender AS 性别, c.birthplace AS 出生, c.echo_main AS 首位声骸,
        c.echo_main_stats AS 主词条, c.echo_sub_stats AS 副词条""",
    "属性反查": """MATCH (c:Character) WHERE c.element = $e
        RETURN c.element AS 属性, c.name AS 角色, c.weapon AS 武器, c.gender AS 性别 ORDER BY 角色""",
    "技能": """MATCH (:Character {name:$n})-[:HAS_SKILL]->(s)
        RETURN s.kind AS 类型, s.name AS 名称 ORDER BY 名称""",
    "共鸣链": """MATCH (:Character {name:$n})-[:HAS_CHAIN]->(x)
        RETURN x.seq AS 序号, x.name AS 名称, x.effect AS 效果 ORDER BY 序号""",
    "突破材料": """MATCH (:Character {name:$n})-[r:NEEDS_MATERIAL]->(m)
        WHERE $stage = '' OR r.stage = $stage
        RETURN r.kind AS 类别, r.stage AS 阶段, m.name AS 材料, r.qty AS 数量
        ORDER BY 类别, 阶段, 材料""",
    "声骸": """MATCH (c:Character {name:$n})
        OPTIONAL MATCH (c)-[r:RECOMMENDS_ECHO]->(e:EchoSet)
        WITH c, collect(DISTINCT e.name) AS 推荐套装,
             collect(DISTINCT {stage:r.stage, cost:r.cost, pieces:r.pieces, set:e.name}) AS 配装方案原
        RETURN c.echo_main AS 首位声骸,
               c.echo_main_stats AS 主词条,
               c.echo_sub_stats AS 副词条,
               推荐套装,
               [x IN 配装方案原 WHERE x.cost IS NOT NULL] AS 配装方案""",
    "武器": """MATCH (:Character {name:$n})-[r:RECOMMENDS_WEAPON]->(w)
        RETURN r.rank AS 优先级, w.name AS 武器 ORDER BY 优先级""",
    "队友": """MATCH (:Character {name:$n})-[r:SYNERGIZES_WITH]->(t)
        RETURN t.name AS 队友, r.teams AS 队伍, r.effect AS 推荐理由 ORDER BY 队友""",
}

# 图谱字段是 schema 名，用户说的是游戏术语，必须显式映射
SLOT_LABEL: dict[str, str] = {
    "配装":    "声骸配装（套装字段即声骸套装名，COST 是声骸费用组合）",
    "共鸣链":  "共鸣链（序号字段即第几链，相当于命座）",
    "突破材料": "突破材料（类别区分角色突破/技能突破）",
    "武器":    "武器推荐（优先级字段：1 为首选）",
    "队友":    "队友推荐（推荐理由字段是队友提供的增益效果，如伤害加深百分比）",
    "声骸": "声骸配装（套装字段即声骸套装名，COST 是声骸费用组合，主/副词条为推荐词条）",
}


def _fmt_val(v):
    """graph_search 展示：把 list(推荐套装/配装方案) 与 dict 列表格式化为可读串。"""
    if isinstance(v, list):
        if not v:
            return None
        if isinstance(v[0], dict):
            seen, items = set(), []
            for x in v:
                key = (x.get("cost"), x.get("set"))
                if key in seen:
                    continue
                seen.add(key)
                items.append("/".join(
                    f"{k}={x[k]}" for k in ("stage", "cost", "set", "pieces") if x.get(k) is not None))
            return "; ".join(items) if items else None
        return "、".join(str(x) for x in v)
    return v


async def graph_search(
    characters: list[str], slots: list[str], element: str = "", stage: str = ""
) -> str:
    """属性反查问题提前解决。支持多角色：每个角色各查一遍，各自带小标题。"""
    if not slots:
        return ""
    # 配装语义 == 声骸：把"配装"槽位映射到"声骸"检索通道，
    # 避免只查 HAS_BUILD 漏掉节点属性(主/副词条)与推荐套装集合。
    _remap = []
    for sl in slots:
        target = "声骸" if sl == "配装" else sl
        if target not in _remap:
            _remap.append(target)
    slots = _remap

    if not characters and "属性反查" not in slots:
        return ""
    blocks: list[str] = []
    async with get_session() as s:
        if element and "属性反查" in slots:
            rows = await (await s.run(CYPHER["属性反查"], e=element)).data()
            if rows:
                blocks.append("【属性反查】\n" + "\n".join(
                    "  " + " / ".join(f"{k}={v}" for k, v in r.items() if v not in (None, ""))
                    for r in rows
                ))
        for char in characters:
            lines: list[str] = []
            for slot in slots:
                if slot == "属性反查":
                    continue       
                cy = CYPHER.get(slot)
                if not cy:
                    continue
                rows = await (await s.run(cy, n=char , stage=stage)).data()
                if not rows:
                    continue
                lines.append(f"【{SLOT_LABEL.get(slot, slot)}】")
                for r in rows:
                    lines.append("  " + " / ".join(
                        f"{k}={_fmt_val(v)}" for k, v in r.items()
                        if _fmt_val(v) not in (None, "", [])
                    ))
                log.info("图谱命中 %s/%s: %d 行", char, slot, len(rows))
            if lines:
                blocks.append(f"## {char}\n" + "\n".join(lines))
    return "\n\n".join(blocks)


@lru_cache(maxsize=1)
def _collection():
    """返回chromadb连接"""
    s = get_settings()
    client = chromadb.PersistentClient(
        path=str(s.VECTOR_DIR / "chroma"),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(name=s.CHUNK_COLLECTION)


@lru_cache(maxsize=1)
def _bm25() -> BM25Index:
    return BM25Index.load()


@lru_cache(maxsize=1)
def _embedder() -> BgeM3Embeddings:
    """返回唯一BgeM3"""
    return BgeM3Embeddings()


def _rrf(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    """RRF 融合：dense 余弦分和 BM25 分值量纲不同，不能直接加权。"""
    score: dict[str, float] = {}
    for ids in rank_lists:
        for rank, cid in enumerate(ids):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + rank + 1)                   # rrf核心，都有就叠加，根据排名打分
    return [cid for cid, _ in sorted(score.items(), key=lambda x: x[1], reverse=True)]


def vector_search(question: str, topk: int | None = None) -> list[dict]:
    s = get_settings()
    col = _collection()

    dense = col.query(
        query_embeddings=[_embedder().embed_query(question)],
        n_results=topk or s.TOPK_DENSE,
    )["ids"][0]
    sparse = [h.chunk_id for h in _bm25().search(question, s.TOPK_SPARSE)]

    fused = _rrf([dense, sparse])[: s.TOPK_RERANK_IN]
    if not fused:
        return []

    got = col.get(ids=fused)
    by_id = {
        i: (d, m or {})
        for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])
    }
    out = [{"chunk_id": c, "text": by_id[c][0], **by_id[c][1]} for c in fused if c in by_id]
    log.info("向量召回 dense=%d sparse=%d -> 融合 %d", len(dense), len(sparse), len(out))
    return out
