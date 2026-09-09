"""文本清洗与拼装：写入端（建索引）和读取端（重排/生成）共用一份，避免两边逻辑漂移。"""
from __future__ import annotations

import re

# wiki 表格没有表头时，导出工具会填「列1 / 列2」占位符，实测 56/102 个 chunk 中招。
# 只删「占位表头 + 紧跟的分隔行」这个组合，正常表格（表头是「身份/名称」）不受影响。
_RE_PH_BLOCK = re.compile(
    r"^[ \t]*\|(?:[ \t]*列\d+[ \t]*\|)+[ \t]*\r?\n"      # | 列1 | 列2 |
    r"[ \t]*\|(?:[ \t]*:?-{3,}:?[ \t]*\|)+[ \t]*\r?$",   # | --- | --- |
    re.M,
)
_RE_PH_ROW = re.compile(r"^[ \t]*\|(?:[ \t]*列\d+[ \t]*\|)+[ \t]*\r?$", re.M)
_RE_BLANKS = re.compile(r"\n{3,}")


def strip_placeholder_table(text: str) -> str:
    """删掉「| 列1 | 列2 |」占位表头及其分隔行。幂等，重复调用无害。"""
    text = _RE_PH_BLOCK.sub("", text)
    text = _RE_PH_ROW.sub("", text)
    return _RE_BLANKS.sub("\n\n", text).strip()


def join_breadcrumb(breadcrumb: str, text: str) -> str:
    """面包屑 + 正文，只补不重复。

    实测 102 个 chunk 里 74 个的 text 本身就以 breadcrumb 开头
    （MarkdownHeaderTextSplitter 开了 strip_headers=False，标题行留在正文里），
    无条件拼接会让面包屑出现两遍——build_index 和 rerank 都踩过这个坑。
    """
    bc = (breadcrumb or "").strip()
    tx = (text or "").strip()
    if not bc or tx.startswith(bc):
        return tx
    return f"{bc}\n{tx}"


def embed_input(breadcrumb: str, text: str) -> str:
    """写入端：建 BM25 / Chroma 之前的最终文本。"""
    return strip_placeholder_table(join_breadcrumb(breadcrumb, text))


def chunk_text(d: dict) -> str:
    """读取端：从检索结果 dict 取干净文本。幂等——写入端已洗过也不会二次伤害。"""
    return strip_placeholder_table(join_breadcrumb(d.get("breadcrumb", ""), d.get("text", "")))
