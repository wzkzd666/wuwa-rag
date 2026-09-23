"""Step 4：把 chunks.jsonl 落进 RustFS + PostgreSQL。

顺序：先 documents（父），再 chunks（子，外键引用）。
幂等：documents 靠 raw_sha256、chunks 靠 chunk_id，均用 ON CONFLICT，重跑安全。
      documents 用 DO UPDATE（冲突时也能 RETURNING id），chunks 用 DO NOTHING。

⚠️⚠️ 2026-09-22 修（重要）：冲突目标必须是 **chunk_id**，曾经写成 `ON CONFLICT (hash)`。
`hash` 是**纯正文** sha256，而 chunk_id = `角色::H2::H3::H4::hash8`（含角色）。
两者差在「跨角色同文本」上：一阶突破材料表这类小表格的正文**不含角色名**，
57 个角色里正文**逐字相同** → 只保留第一份，其余全被 DO NOTHING 静默丢掉。

实测（2026-09-22）：chunks.jsonl 6572 行、不同 hash **6448** 个、跨角色重复 **124** 处
—— 而 PG 里正好 **6448** 行，缺的 124 块与这个数字分毫不差。
症状：卡卡罗问「一阶突破材料」永远答不出（该行落在别的角色名下，`fetch_chunks` 按
character 过滤取不到），而 Chroma/BM25 三方一致，**看上去完全不像缺数据**，极难排查。

⚠️ 配套：`pgsql/001_init.sql` 里 `CREATE UNIQUE INDEX ux_chunks_hash ON chunks(hash)`
必须放宽（改非唯一），否则跨角色同文本会撞唯一约束**直接报错**而不是被跳过。
`chunk_id` 自带角色名，`ux_chunks_chunk_id` 已能保证每个角色的块唯一，够用。
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from ..config import ensure_dirs, get_settings
from ..db import close_pool, get_cursor
from ..storage import s3
from ..ww_logger import get_logger

upload_logger=get_logger('upload')

def _load_chunks(path: Path) -> dict[str, list[dict]]:
    """按角色分组读回分块结果。"""
    by_char: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_char.setdefault(r["character"], []).append(r)
    return by_char


async def ingest_one(md_path: Path, rows: list[dict]) -> tuple[int, int]:
    character = md_path.stem
    raw = md_path.read_bytes()                    # 直接拿 bytes，绕开编码问题
    uri, sha = s3.put_raw(character, raw)

    async with get_cursor() as cur:
        await cur.execute(
            """
            INSERT INTO documents (character, source, title, raw_uri, raw_sha256, raw_size, mime_type)
            VALUES (%s, 'kurobbs', %s, %s, %s, %s, 'text/markdown')
            ON CONFLICT (raw_sha256) DO UPDATE SET updated_at = now()
            RETURNING id
            """,
            (character, character, uri, sha, len(raw)),
        )
        doc_id = (await cur.fetchone())[0]

        payload = [
            (
                r["chunk_id"], doc_id, character,
                r.get("meta", {}).get("element"),
                r.get("meta", {}).get("weapon"),
                r.get("meta", {}).get("rarity"),
                r["module"], r["component"], r["tab"], r["level"],
                r["breadcrumb"], r["text"], r["has_table"], r["char_count"],
                r["hash"], "doc",
                json.dumps(r.get("meta", {}), ensure_ascii=False),
            )
            for r in rows
        ]
        await cur.executemany(
            """
            INSERT INTO chunks (chunk_id, document_id, character, element, weapon, rarity,
                                module, component, tab, level, breadcrumb, text,
                                has_table, char_count, hash, source, meta)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (chunk_id) DO NOTHING
            """,
            payload,
        )
    return doc_id, len(payload)


async def purge_character(character: str) -> int:
    """刷新重爬前清该角色旧数据（PG 真源：documents 级联删 chunks）。

    chunks 表外键是 ON DELETE CASCADE，删 documents 即连带删块；
    不删的话 wiki 改版产生的新旧块会并存（hash 各不同，DO NOTHING 挡不住），
    召回到旧知识——这正是「缓存知识不匹配」要重爬前清掉的东西。
    S3 的旧对象不清：documents 重建后 raw_uri 指向新对象，旧对象成孤儿，
    靠对象存储生命周期策略回收（清理属运维范畴，不混进业务刷新链）。
    """
    async with get_cursor() as cur:
        await cur.execute(
            "DELETE FROM documents WHERE character = %s", (character,)
        )
        n = cur.rowcount
    upload_logger.info(f"清库 {character}: documents 删 {n} 行（chunks 级联）")
    return n


async def _main() -> None:
    s=get_settings()
    ensure_dirs()

    by_char = _load_chunks(s.CHUNKS_JSONL)
    total = 0
    for md_path in sorted(s.RAW_DIR.glob("*.md")):
        doc_id, n = await ingest_one(md_path, by_char.get(md_path.stem, []))
        total += n
        upload_logger.info(f"{md_path.stem}: document_id={doc_id}, {n} 块")
    upload_logger.info(f"pg总计写入 {total} 块")
    await close_pool()


def main() -> None:
    asyncio.run(_main(), loop_factory=asyncio.SelectorEventLoop)


if __name__ == "__main__":
    main()
