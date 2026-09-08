"""Markdown 结构感知分块器（混合方案）。

第一层：LangChain MarkdownHeaderTextSplitter 按 H1-H4 切段 + 注入层级元数据。
        （通用逻辑，工具比手写稳）
第二层：自写——超长块切块（无表格行叠切块）、过小块合并、生成稳定 chunk_id。
        （工具不做二次切分，巨块原样返回；也没有 id/hash/增量）

不切表格的原理：Markdown 表格行与行之间无空行，
按空行切分时整表必然完整留在同一片，无需为表格写特殊逻辑。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from langchain_text_splitters import MarkdownHeaderTextSplitter

from ..config import get_settings, ensure_dirs

HEADERS_TO_SPLIT_ON = [("#", "H1"), ("##", "H2"), ("###", "H3"), ("####", "H4")]
TABLE_RE = re.compile(r"^\s*\|")
s=get_settings()


@dataclass
class Chunk:
    chunk_id: str
    character: str
    module: str | None = None      # H2
    component: str | None = None   # H3
    tab: str | None = None         # H4
    level: int = 2
    breadcrumb: str = ""
    text: str = ""
    has_table: bool = False
    char_count: int = 0
    hash: str = ""
    meta: dict = field(default_factory=dict)


@lru_cache(maxsize=1)  # 懒加载、测试友好、可扩展、明确表达是可缓存的构造器
def _splitter() -> MarkdownHeaderTextSplitter:
    """懒加载返回 MarkdownHeaderTextSplitter 实例"""
    return MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,        
    )


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _clean(lines: list[str]) -> str:
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()


def _has_table(text: str) -> bool:
    return any(TABLE_RE.match(x) for x in text.splitlines())


def _split_table(text: str, breadcrumb: str, max_chars: int) -> list[str]:
    """无空行可切时的降级切分：按表格行切，每片重复表头 + 分隔行 + breadcrumb。"""
    lines = text.splitlines()

    # 1) 拆出开头的 # 标题行
    head_lines: list[str] = []
    i = 0
    while i < len(lines) and lines[i].lstrip().startswith("#"):
        head_lines.append(lines[i])
        i += 1
    rest = lines[i:]

    # 2) 按「连续表格行 / 连续非表格行」分组
    blocks: list[tuple[bool, list[str]]] = []
    for line in rest:
        is_tbl = bool(TABLE_RE.match(line))
        if blocks and blocks[-1][0] == is_tbl:
            blocks[-1][1].append(line)
        else:
            blocks.append((is_tbl, [line]))

    prefix = "\n".join([breadcrumb, *head_lines]).strip()
    out: list[str] = []

    for is_tbl, blk in blocks:
        # 非表格行 / 不成表（缺分隔行）-> 原样带前缀输出
        if not is_tbl or len(blk) < 2:
            budget = max_chars - len(prefix) - 120
            batch, used = [], 0
            for line in blk:
                n = len(line) + 1
                if batch and used + n > budget:
                    out.append("\n".join([prefix, *batch]).strip())
                    # 关键：保留末尾几行作为下一片的开头
                    tail = batch[-s.OVERLAP_LINES:]
                    batch = list(tail)
                    used = sum(len(x) + 1 for x in tail)
                batch.append(line)
                used += n
            if batch:
                out.append("\n".join([prefix, *batch]).strip())
            continue

        table_head = blk[:2]                       # 表头 + |---|
        budget = max_chars - len(prefix) - len("\n".join(table_head)) - 2
        if budget <= 0:                            # 前缀本身太长，切了也超，放弃
            out.append("\n".join([prefix, *blk]).strip())
            continue

        batch: list[str] = []
        used = 0
        for row in blk[2:]:
            n = len(row) + 1
            if batch and used + n > budget:        # batch 为空时强制放入：绝不切行内部
                out.append("\n".join([prefix, *table_head, *batch]).strip())
                batch, used = [], 0
            batch.append(row)
            used += n
        if batch:
            out.append("\n".join([prefix, *table_head, *batch]).strip())

    return out


def _split_oversize(text: str, breadcrumb: str = "") -> list[str]:
    if len(text) <= s.MAX_CHARS:
        return [text]
    parts, buf = [], []
    for line in text.splitlines():
        buf.append(line)
        if not line.strip():                 # 只在空行处切，表格天然安全
            piece = _clean(list(buf))
            if piece:
                parts.append(piece)
            buf = []
    if _clean(list(buf)):
        parts.append(_clean(list(buf)))

    refined: list[str] = []
    for p in parts:
        if len(p) > s.MAX_CHARS:
            refined.extend(_split_table(p, breadcrumb, s.MAX_CHARS))
        else:
            refined.append(p)
    parts = refined

    return parts or [text]


def chunk_markdown(
    md: str,
    character: str,
    element: str | None = None,
    weapon: str | None = None,
    rarity: int | None = None,
) -> list[Chunk]:
    """character 用文件名传入（比 H1 可靠）。"""
    chunks: list[Chunk] = []

    for d in _splitter().split_text(md):
        m = d.metadata
        module, component, tab = m.get("H2"), m.get("H3"), m.get("H4")
        level = 4 if tab else 3 if component else 2 if module else 1
        breadcrumb = " › ".join([x for x in [character, module, component, tab] if x])

        pieces = _split_oversize(d.page_content.strip(), breadcrumb)
        for i, piece in enumerate(pieces):
            # 单块时 page_content 已含标题行；切出多片时每片都要补面包屑，否则成孤儿
            text = piece if len(pieces) == 1 or piece.startswith(breadcrumb) else f"{breadcrumb}\n\n{piece}"

            # 过小块并入上一块
            if len(text) < s.MIN_CHARS and chunks:
                prev = chunks[-1]
                prev.text = f"{prev.text}\n\n{text}".strip()
                prev.char_count = len(prev.text)
                prev.hash = _sha(prev.text)
                prev.has_table = prev.has_table or _has_table(text)
                prev.chunk_id = prev.chunk_id.rsplit("::", 1)[0] + "::" + prev.hash[:8]
                continue

            h = _sha(text)
            cid = "::".join([character, module or "-", component or "-", tab or "-", h[:8]])
            if i:
                cid = f"{cid}#{i}"

            meta = {
                "H1": m.get("H1"),
                "element": element,
                "weapon": weapon,
                "rarity": rarity,
            }
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    character=character,
                    module=module,
                    component=component,
                    tab=tab,
                    level=level,
                    breadcrumb=breadcrumb,
                    text=text,
                    has_table=_has_table(text),
                    char_count=len(text),
                    hash=h,
                    meta={k: v for k, v in meta.items() if v is not None},
                )
            )
    return chunks


def main() -> None:
    import json
    import pathlib
    import sys

    src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else s.RAW_DIR)
    out = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else s.CHUNKS_JSONL)
    out.parent.mkdir(parents=True, exist_ok=True)
    ensure_dirs()
    
    total = 0
    with out.open("w", encoding="utf-8") as f:
        for md_file in sorted(src.glob("*.md")):
            chunks = chunk_markdown(md_file.read_text(encoding="utf-8"), md_file.stem)
            for c in chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
            total += len(chunks)
            print(f"{md_file.stem}: {len(chunks)} 块")
    print(f"总计 {total} 块 -> {out}")


if __name__ == "__main__":
    main()
