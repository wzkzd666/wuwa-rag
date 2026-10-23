"""语料对账：PG / chunks.jsonl / Chroma / bm25.pkl / Neo4j / data/raw 一次跑清。

为什么要有这条命令（三条都是本仓真出事之后记下来的，不是假想的运维美化）：

  · 2026-09-22 —— `ON CONFLICT (hash)` 把 124 个「跨角色同文本」的块静默丢弃。
    当时 PG / Chroma / BM25 **三方彼此完全一致**（都 6448），逐角色核对也对得上，
    肉眼看不出缺数据；最后是靠「chunks.jsonl 6572 行 ≠ PG 6448 行」这个差值才翻案。
  · 2026-10-23 —— `chunks.jsonl` 被写成 0 字节，而 PG 里 6741 块完好。
    单角色增量链的「保留别人」是从**当前文件**里取的，文件空 = 别人全没了，
    而五步链照样报 success。
  · 2026-10-24 —— 6741 块里有 36 块在 Chroma 里存的还是旧口径文本（配队块那一长串 `-`），
    而查询侧早已换成顿号：派生索引「有点数、对得上行数」，文本却和真源不同口径。

三次同一个形态：**派生索引看起来没事，真源其实对不上**。而每次排查都是临时手写一遍脚本，
下次从头想。这条命令把那个口径固化下来 —— **只读，不写任何数据**；
退出码非 0 表示有 fail 级不一致（每条直接给对应的补救命令）。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass

from wuwa_rag.config import get_settings
from wuwa_rag.core.db import close_pool, get_cursor
from wuwa_rag.knowledge.graph import neo4j_client
from wuwa_rag.text import embed_input

FAIL = "fail"
WARN = "warn"


@dataclass(frozen=True)
class Finding:
    """一条对账结论。`fix` 是直接能抄进终端的补救命令（没有就说明无需动作）。"""
    level: str
    check: str
    detail: str
    fix: str = ""


# ---------------- 纯判据（可离线单测，不碰任何服务） ----------------
def compare_by_character(jsonl: dict[str, int], pg: dict[str, int]) -> list[Finding]:
    """逐角色比「产物块数 vs 库里块数」。

    必须逐角色：总数相等照样能错 —— A 角色多插 10 块、B 角色少 10 块，
    全局计数完全对齐，而 B 的答案永远召不回那 10 块（上面 124 块那起事故就是这个形状）。
    """
    out: list[Finding] = []
    for ch in sorted(set(jsonl) | set(pg)):
        a, b = jsonl.get(ch, 0), pg.get(ch, 0)
        if a == b:
            continue
        if a == 0:
            out.append(Finding(FAIL, "逐角色块数", f"{ch}: 产物里没有、库里有 {b} 块", "uv run wuwa-chunk"))
        elif b == 0:
            out.append(Finding(FAIL, "逐角色块数", f"{ch}: 产物里 {a} 块、库里没有", "uv run wuwa-ingest"))
        else:
            out.append(Finding(FAIL, "逐角色块数", f"{ch}: 产物 {a} ≠ 库里 {b}", "uv run wuwa-chunk && uv run wuwa-ingest"))
    return out


def compare_id_sets(pg_ids: set[str], other: set[str], check: str, fix: str) -> list[Finding]:
    """集合级对账：缺（库里有、索引没有）与多（索引里有、库已经删掉）是两个方向的故障。"""
    out: list[Finding] = []
    missing = sorted(pg_ids - other)
    extra = sorted(other - pg_ids)
    if missing:
        out.append(Finding(FAIL, check,
                           f"缺 {len(missing)} 个（前 3：{missing[:3]}）", fix))
    if extra:
        out.append(Finding(FAIL, check,
                           f"多 {len(extra)} 个（PG 里已没有的孤儿，前 3：{extra[:3]}）", fix))
    return out


def compare_doc_texts(pg: dict[str, tuple[str, str]], stored: dict[str, str | None]) -> list[Finding]:
    """稠密索引里存的**文本**是否等于当前口径算出来的那一份。

    只比 id 集合抓不到「2026-10-24 那 36 块」：点数对得上、行数对得上，
    但 documents 里还是改版前的旧文本，与查询侧不同口径 → 相似度算错而无人报错。
    """
    bad = [cid for cid, (bc, text) in pg.items() if stored.get(cid) != embed_input(bc, text)]
    if not bad:
        return []
    return [Finding(FAIL, "稠密文本口径",
                    f"{len(bad)} 块存的不是当前口径（前 3：{bad[:3]}）", "uv run wuwa-index")]


def find_raw_drift(doc_sha: dict[str, str], raw_sha: dict[str, str]) -> list[Finding]:
    """`documents.raw_sha256` 与磁盘上 `data/raw/<角色>.md` 的当前哈希是否一致。

    不一致只有两种成因，都不是好事：手工改过 raw（绕过流水线），
    或者重爬/重分块了但没走完入库 —— 后者会让「库里内容」与「产物」各说各话。

    缺席（库里有、raw 里没有）单独归成**一条** warn：`data/raw` 整个目录不在时
    逐角色刷屏会把真正要看的 fail 埋掉（对账工具的输出是给人一眼读完的）。
    """
    out: list[Finding] = []
    drift = [ch for ch in sorted(set(doc_sha) & set(raw_sha)) if doc_sha[ch] != raw_sha[ch]]
    if drift:
        out.append(Finding(FAIL, "raw 与库里同源",
                           f"{len(drift)} 个角色的磁盘 raw 已变、库里还是旧哈希（前 5：{drift[:5]}）",
                           "uv run wuwa-chunk && uv run wuwa-ingest"))
    missing = sorted(set(doc_sha) - set(raw_sha))
    if missing:
        out.append(Finding(WARN, "raw 与库里同源",
                           f"{len(missing)} 个角色在 data/raw 里没有文件（前 5：{missing[:5]}）"
                           "—— 只影响能不能重跑，不影响检索"))
    return out


def compare_graph_coverage(pg_chars: set[str], graph_chars: set[str]) -> list[Finding]:
    """名册一致性，两个方向都要看。

    缺（PG 有、图谱没有）：`graph_search` 开头 `if not characters ... return ""`，
    对这个角色只剩向量路 —— 技能/配队这类要靠图谱的回答直接退化。
    多（图谱有、PG 没有）：**2026-09-22 那 4 个垃圾节点（`主输出`/`副输出`/`卡提`/`暗主`）
    就是这么来的** —— 抽取把 wiki 的位置标签当成了角色名。它不影响当前问答（没人会问「主输出是谁」），
    但会污染名册与「主流队友」计数，所以是 warn：看得见、不拦路。
    """
    out: list[Finding] = []
    missing = sorted(pg_chars - graph_chars)
    if missing:
        out.append(Finding(FAIL, "图谱角色覆盖",
                           f"图谱里少了 {len(missing)} 个：{missing[:5]}", "uv run wuwa-graph"))
    extra = sorted(graph_chars - pg_chars)
    if extra:
        out.append(Finding(WARN, "图谱角色覆盖",
                           f"图谱里有 {len(extra)} 个库里没有的角色名（疑似抽取把位置标签当角色）：{extra[:5]}"))
    return out


# ---------------- 取数（要真服务，手动跑这条命令时才走） ----------------
def _jsonl_counts(path) -> tuple[dict[str, int], dict[str, str]]:
    """读 chunks.jsonl → (逐角色块数, {chunk_id: (breadcrumb, text)})。"""
    counts: dict[str, int] = {}
    rows: dict[str, tuple[str, str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cid = d["chunk_id"]
            counts[d["character"]] = counts.get(d["character"], 0) + 1
            rows[cid] = (d.get("breadcrumb", ""), d.get("text", ""))
    return counts, rows


def _raw_digests(raw_dir) -> dict[str, str]:
    out: dict[str, str] = {}
    if not raw_dir.is_dir():
        return out
    for md in raw_dir.glob("*.md"):
        out[md.stem] = hashlib.sha256(md.read_bytes()).hexdigest()
    return out


def _chroma_docs() -> dict[str, str | None]:
    from chromadb import PersistentClient
    from chromadb import Settings as ChromaSettings

    s = get_settings()
    col = PersistentClient(path=str(s.VECTOR_DIR / "chroma"),
                           settings=ChromaSettings(anonymized_telemetry=False)
                           ).get_or_create_collection(name=s.CHUNK_COLLECTION)
    got = col.get(include=["documents"])
    return dict(zip(got["ids"], got["documents"]))


def _bm25_ids() -> set[str]:
    from wuwa_rag.knowledge.index.bm25 import BM25Index

    return set(BM25Index.load().chunk_ids)


async def _collect() -> list[Finding]:
    s = get_settings()
    out: list[Finding] = []

    async with get_cursor(commit=False) as cur:
        await cur.execute("SELECT chunk_id, character, breadcrumb, text FROM chunks")
        pg_rows = [(r[0], r[1], r[2] or "", r[3] or "") for r in await cur.fetchall()]
        await cur.execute("SELECT character, raw_sha256 FROM documents")
        doc_sha: dict[str, str] = {}
        for r in await cur.fetchall():
            doc_sha[r[0]] = r[1]
        await cur.execute("SELECT DISTINCT character FROM chunks")
        pg_chars = {r[0] for r in await cur.fetchall()}
    await close_pool()

    pg_ids = {r[0] for r in pg_rows}
    pg_by_char: dict[str, int] = {}
    pg_texts: dict[str, tuple[str, str]] = {}
    for cid, ch, bc, text in pg_rows:
        pg_by_char[ch] = pg_by_char.get(ch, 0) + 1
        pg_texts[cid] = (bc, text)

    sizes: dict[str, str] = {}

    if not s.CHUNKS_JSONL.exists():
        # 只报这一条就够：产物整个不在时再逐角色刷 58 行「库里有多少块」，
        # 会把真正要看的其它 fail 埋掉（对账输出是给人一眼读完的）
        out.append(Finding(FAIL, "产物存在",
                           f"{s.CHUNKS_JSONL} 不存在", "uv run wuwa-chunk"))
        sizes["chunks.jsonl"] = "缺失"
    else:
        jsonl_counts, jsonl_rows = _jsonl_counts(s.CHUNKS_JSONL)
        # ① 产物 vs 真源（逐角色 + 集合）
        out += compare_by_character(jsonl_counts, pg_by_char)
        out += compare_id_sets(pg_ids, set(jsonl_rows), "产物 chunk_id", "uv run wuwa-chunk")
        sizes["chunks.jsonl"] = str(len(jsonl_rows))
    # ② 稠密索引：先比集合再比文本口径
    try:
        chroma = _chroma_docs()
        sizes["Chroma"] = str(len(chroma))
    except Exception as exc:  # noqa: BLE001 —— 对账工具不该因为一路索引读不了就整个哑掉
        out.append(Finding(WARN, "稠密索引可读", f"读 Chroma 失败（跳过这条对账）：{type(exc).__name__}: {exc}"))
        chroma = {}
        sizes["Chroma"] = "读不到"
    if chroma:
        out += compare_id_sets(pg_ids, set(chroma), "稠密索引 chunk_id", "uv run wuwa-index")
        out += compare_doc_texts(pg_texts, chroma)
    # ③ 稀疏索引
    try:
        bm25_ids = _bm25_ids()
        sizes["BM25"] = str(len(bm25_ids))
        out += compare_id_sets(pg_ids, bm25_ids, "BM25 chunk_id", "uv run wuwa-index")
    except Exception as exc:  # noqa: BLE001
        sizes["BM25"] = "读不到"
        out.append(Finding(WARN, "BM25 可读", f"读 bm25.pkl 失败（跳过这条对账）：{type(exc).__name__}: {exc}"))
    # ④ 图谱覆盖
    try:
        async with neo4j_client.get_session() as sess:
            res = await sess.run("MATCH (c:Character) RETURN c.name AS n")
            graph_chars = {rec["n"] async for rec in res}
        sizes["图谱角色"] = str(len(graph_chars))
        out += compare_graph_coverage(pg_chars, graph_chars)
    except Exception as exc:  # noqa: BLE001
        sizes["图谱角色"] = "读不到"
        out.append(Finding(WARN, "图谱可读", f"读 Neo4j 失败（跳过这条对账）：{type(exc).__name__}: {exc}"))
    finally:
        await neo4j_client.close_driver()
    # ⑤ raw 与库里同源
    raw = _raw_digests(s.RAW_DIR)
    sizes["raw 文件"] = str(len(raw))
    out += find_raw_drift(doc_sha, raw)

    # 汇总行放最前面：三次事故的共同指纹就是「这几个数应该相等而不等」
    sizes_out = {"PG": str(len(pg_ids)), **sizes}
    print("[对账] " + "  ".join(f"{k}={v}" for k, v in sizes_out.items()))
    return out


def _report(findings: list[Finding]) -> int:
    fails = [f for f in findings if f.level == FAIL]
    warns = [f for f in findings if f.level == WARN]
    for f in findings:
        mark = "✗" if f.level == FAIL else "!"
        fix = f"   → {f.fix}" if f.fix else ""
        print(f"{mark} [{f.check}] {f.detail}{fix}")
    if not findings:
        print("✓ 语料与三路索引、图谱、raw 全部一致")
    else:
        print(f"合计：fail {len(fails)} 条 / warn {len(warns)} 条")
    return 1 if fails else 0


def main() -> None:
    findings = asyncio.run(_collect(), loop_factory=asyncio.SelectorEventLoop)
    raise SystemExit(_report(findings))
