"""Step 4：把 chunks.jsonl 落进 RustFS + PostgreSQL。

顺序：先 documents（父），再 chunks（子，外键引用）。
幂等：documents 靠 raw_sha256、chunks 靠 hash，均用 ON CONFLICT，重跑安全。
      documents 用 DO UPDATE（冲突时也能 RETURNING id），chunks 用 DO NOTHING。
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from ..db import close_pool, get_cursor
from ..storage import s3
from ..config import get_settings, ensure_dirs


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
            ON CONFLICT (hash) DO NOTHING
            """,
            payload,
        )
    return doc_id, len(payload)


async def _main() -> None:
    s=get_settings()
    ensure_dirs()

    by_char = _load_chunks(s.CHUNKS_JSONL)
    total = 0
    for md_path in sorted(s.RAW_DIR.glob("*.md")):
        doc_id, n = await ingest_one(md_path, by_char.get(md_path.stem, []))
        total += n
        print(f"{md_path.stem}: document_id={doc_id}, {n} 块")
    print(f"总计写入 {total} 块")
    await close_pool()


def main() -> None:
    asyncio.run(_main(), loop_factory=asyncio.SelectorEventLoop)


if __name__ == "__main__":
    main()
