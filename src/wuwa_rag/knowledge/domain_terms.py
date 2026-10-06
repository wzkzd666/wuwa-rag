"""领域词表：从语料自动派生，供角色名消歧使用。

要解决的问题
------------
角色名本身可能是常用词（心、椿…）。`knowledge/entities.find_mentions` 的判据是
「单字名必须独立成词」，而 `心声` 是 jieba 默认词典里的真词、最长匹配会把它整体切走，
于是「心声骸推荐」里 `心` 这个 token 根本不出现 → 角色识别不到 → `characters` 为空 →
`dialog/graph.verify_node` 的「按角色清库重爬」分支永不触发 → 落到联网兜底。

本模块提供**第二道证据**：判定「这个单字名是不是在谈角色」，依据是它周围的
**游戏领域词**。领域词表不是手写的，而是从语料里长出来的：

① **分类法词表** —— wiki 自己的面包屑/模块/组件（`丹瑾 › 角色养成 › 技能介绍 › …`）。
   干净、可枚举，负样本词（开心/心情/派生）一个都不在里面；
   缺点是**词形是标签不是问句用词**（「武器推荐」而不是「武器」）——
   所以只拿它当**过滤器**（子词），不当词表本身。
② **锚点共现词表** —— 以**无歧义的多字角色名**为锚，收集其 ±14 字窗口内的词，
   按「出现在多少个不同角色的窗口里」计分。含声骸/配队/养成/技能/武器这类问句用词；
   缺点是混进通用词（查看/点击/可以/自己/资料）。

**词表 = ①的子词 ∩ ②，再按 jieba 词性滤掉虚词**（什么/一个/不会/所以）。
三类过滤各司其职，缺一不可（每条都是实测踩出来的）：
- **角色覆盖度 ≥ 4**：剧情标题（焰光之影/童年远去）只在单个角色页出现 → 自动剔除；
- **分类法子词**：通用词（查看/可以）不在任何 wiki 分类标签里 → 剔除；
- **词性白名单**：虚词（什么/一个）既在标签文本里、又高頻 → 用 jieba 词性滤掉。

⚠️ 词表里**会**留下「核心/机制/资料」这类 wiki 结构词（角色页正文自己就写「核心机制」，
它们确实是高频共现词）。它们不会造成误判，因为兜底判据还要求**单字名左边是边界** ——
「核心玩法」里 `心` 在「核心」内部，直接被边界规则挡掉。用语义筛掉它们反而需要手写停用词表，
不划算；靠边界规则兜更稳。

为什么不用别的办法（都实测过，别再走一遍）
----------------------------------------
- **逐角色名的语料共现画像**（NER 里的 gathering context）：对**歧义**名字无效。
  `心` 的高关联词是「心眼/核心/派生/机制」—— 角色页正文自己就混用常用义，
  而「核心」恰好是负样本词，**信号方向会反**。`椿`（非常用词）反而干净。
- **分类法段与共现词取「段级」交集**：也不行 —— 交集里全是「武器推荐/技能介绍」这种
  复合标签，问句写的是「武器/技能」，一个都对不上。必须展开成**子词**再取交集。

运行方式
--------
构建是「异步取数 + CPU 密集分词」，所以分成三段，问答主链不等待：
- `current()`：**同步**、只读内存缓存，零等待；
- 缓存过期时由调用方触发 `warmup()`，它 fire-and-forget 地重建，**本轮仍用旧词表**；
- 结果落盘 `data/derived/domain_terms.json`（可打印、可人工审阅），
  版本键 = `chunks` 行数 + 名册大小。新角色入库/删除都会让行数变化 → 缓存自动失效，
  不需要人工清。
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import jieba

from wuwa_rag.config import get_settings
from wuwa_rag.core.db import get_cursor
from wuwa_rag.ww_logger import get_logger

log = get_logger("rag")

# ---------- 调参（都是「派生数据」的形状参数，不是部署配置）----------
# 词表多久重算一次。角色入库/删除走 invalidate() 会立刻失效，这里只是兜底。
_TTL = 1800.0
# 一个词至少要出现在这么多个**不同角色**的窗口里，才算「游戏术语」而非某页的特色用词。
_MIN_ANCHORS = 4
# 全语料至少要在这么多个分块里出现过（太低会把只出现两三页的怪词收进来）。
_MIN_DOCS = 40
# 一个词至少要在这么多个分类法标签的**子词**里出现过（词表必须是 wiki 认可的术语）。
_MIN_TAXONOMY = 5
# 词表上限。实测候选 540 取前 120 就够覆盖「心」的常见问法，留余量。
_MAX_TERMS = 150
# 锚点窗口半径（字）。14 字能覆盖「X 的声骸怎么配」这类问法。
_WINDOW = 14

_CJK = re.compile(r"^[\u4e00-\u9fff]+$")
_SEP = re.compile(r"[›>/]")

# (构建时刻的 monotonic, 词表)。空词表 = 还没构建成功，
# 调用方据此回落到 entities._DOMAIN_SUFFIXES 那份手写表。
_cache: tuple[float, frozenset[str]] = (0.0, frozenset())
# 上次重建的版本键（chunks 行数, 名册大小），用于跳过无意义的重算
_version: tuple[int, int] | None = None


def current() -> frozenset[str]:
    """当前词表（同步、零等待）。空 = 尚未构建成功，调用方应回落到手写表。"""
    return _cache[1]


def is_stale() -> bool:
    return (time.monotonic() - _cache[0]) > _TTL


def invalidate() -> None:
    """标记为需重算。**旧词表继续可用**，等新的建好再替换 —— 避免重建期间判据失效。"""
    global _cache
    _cache = (0.0, _cache[1])


def _words(text: str) -> list[str]:
    """中文词（≥2 字、纯汉字）。

    ⚠️ 词性未知的（`cut` 对未登录词直接 yield 字符串，没有词性）**要留着** ——
    「声骸」这类游戏术语在 jieba 词典里就是未登录词，一刀切掉会让派生表直接归零
    （实测：严格按词性过滤后 106 词 → 0 词）。通用词靠「去重段计数 ≥5」这层挡，
    不靠词性。
    """
    out: list[str] = []
    for item in jieba.cut(text, HMM=False):
        word = item[0] if isinstance(item, tuple) else item
        if len(word) >= 2 and _CJK.match(word):
            out.append(word)
    return out


def _segments(row: tuple) -> list[str]:
    """一个分块的结构化标签段（面包屑按 `›`/`/`/`>` 切分，另加 module/component/tab）。"""
    _text, bc, mod, comp, tab = row
    segs = [s.strip() for s in _SEP.split(str(bc or "")) if s.strip()]
    segs += [str(x).strip() for x in (mod, comp, tab) if x and str(x).strip()]
    return segs


def _fetch_rows() -> list[tuple]:
    """同步取语料（跑在 `to_thread` 里）。

    ⚠️ 这里刻意用**同步**驱动而不是项目的异步连接池：psycopg 的异步实现要求
    SelectorEventLoop，而 `to_thread` 给的正是普通工作线程。读的是同一张只读的
    chunks 表，不碰连接池、不写任何东西。
    """
    import psycopg

    with psycopg.connect(get_settings().PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT text, breadcrumb, module, component, tab FROM chunks")
            return cur.fetchall()


def _derive(rows: list[tuple], roster: set[str]) -> frozenset[str]:
    """从语料派生领域词表（CPU 密集，调用方放线程里跑）。

    ⚠️ 窗口按**分词结果的累计偏移**近似，不回原文定位：jieba 会丢掉标点与空白，
    累计偏移与原文位置有偏差。窗口是个 ±14 字的粗判据，这点偏差不影响结论。
    """
    # ⚠️ 必须按**去重后的段**统计，不能按行统计：同一个段（尤其 `tab`）会在几千个
    # 分块里反复出现，按行计数时「一个/什么/不会」这类通用词也会攒到几百次而混进词表。
    # 按去重段统计后实测：通用词 ≤4 次（一个 4 / 什么 2 / 不会 2），领域词 ≥5
    # （武器 13 / 技能 17 / 突破 24 / 推荐 21），`_MIN_TAXONOMY=5` 正好把两者分开。
    seg_texts: set[str] = set()
    for row in rows:
        seg_texts.update(_segments(row))
    taxonomy: Counter[str] = Counter()
    for seg in seg_texts:
        taxonomy.update(_words(seg))          # ← 段展开成**子词**，不收段本身
    tax_words = {w for w, c in taxonomy.items() if c >= _MIN_TAXONOMY}

    anchors = sorted(n for n in roster if len(n) >= 2)
    # 角色名的**碎片**也要排除：`坎特` 不是角色名，但它是「坎特蕾拉」的一部分，
    # 留着会让词表混进角色名残片。用「是任一角色名的子串」判定，光靠 `in roster` 不够。
    fragments = {w for n in roster for w in _words(n)}
    anchor_hits: dict[str, set[str]] = defaultdict(set)
    base_docs: Counter[str] = Counter()
    for row in rows:
        text = row[0] or ""
        toks = _words(text)
        if not toks:
            continue
        base_docs.update(set(toks))
        starts = [0] * (len(toks) + 1)
        for i, w in enumerate(toks):
            starts[i + 1] = starts[i] + len(w)
        for a in anchors:
            if a not in text:
                continue
            for m in re.finditer(re.escape(a), text):
                lo, hi = m.start() - _WINDOW, m.end() + _WINDOW
                for i, w in enumerate(toks):
                    if w in fragments:
                        continue
                    if starts[i] >= lo and starts[i] + len(w) <= hi:
                        anchor_hits[w].add(a)

    cooc = {
        w for w, s in anchor_hits.items()
        if len(s) >= _MIN_ANCHORS and base_docs[w] >= _MIN_DOCS
    }
    # 子词交集是核心：分类法压掉通用词，共现提供问句用词
    terms = (tax_words & cooc) - fragments
    if not terms:
        # 交集为空（语料还很小 / 新库）时退回「只用分类法」——
        # 它更严，绝不会把通用词当领域词，误报风险最低。
        terms = tax_words - fragments
    if not terms:
        return frozenset()
    ranked = sorted(terms, key=lambda w: (-len(anchor_hits.get(w, ())), -base_docs[w]))
    return frozenset(ranked[:_MAX_TERMS])


def _cache_file() -> Path:
    return get_settings().DATA_DIR / "derived" / "domain_terms.json"


def _load(version: tuple[int, int]) -> frozenset[str] | None:
    try:
        data = json.loads(_cache_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if tuple(data.get("version") or ()) != version:
        return None
    log.info("领域词表命中磁盘缓存：%d 词（chunks=%d）", len(data.get("terms") or []),
             data.get("chunks", -1))
    return frozenset(data.get("terms") or ())


def _save(terms: frozenset[str], version: tuple[int, int], n_chunks: int, n_anchor: int) -> None:
    try:
        path = _cache_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "version": list(version),
            "chunks": n_chunks,
            "anchors": n_anchor,
            "count": len(terms),
            "terms": sorted(terms),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError as exc:
        log.warning("领域词表落盘失败（只影响下次启动的重算速度）: %s", exc)


async def build(force: bool = False) -> frozenset[str]:
    """重建词表。`force=True` 无视版本键与 TTL。

    磁盘缓存命中就直接返回、不重算 —— 稳态下问答主链**零额外开销**。
    """
    global _cache, _version
    if not force and _version is not None and not is_stale():
        return _cache[1]
    async with get_cursor() as cur:
        await cur.execute("SELECT count(*) FROM chunks")
        n_chunks = (await cur.fetchone())[0]
        await cur.execute("SELECT DISTINCT character FROM documents WHERE deleted_at IS NULL")
        roster = {r[0] for r in await cur.fetchall() if r[0]}
    if not roster or not n_chunks:
        return _cache[1]
    version = (n_chunks, len(roster))
    if not force:
        cached = _load(version)
        if cached is not None:
            _version = version
            _cache = (time.monotonic(), cached)
            return cached
    rows = await asyncio.to_thread(_fetch_rows)
    terms = await asyncio.to_thread(_derive, rows, roster)
    if not terms:
        log.warning("派生出的领域词表为空，沿用旧词表（%d 词）", len(_cache[1]))
        _cache = (time.monotonic(), _cache[1])
        return _cache[1]
    n_anchor = sum(1 for n in roster if len(n) >= 2)
    _version = version
    _cache = (time.monotonic(), terms)
    log.info("领域词表已重建：%d 词（chunks=%d，锚点角色 %d 个）", len(terms), n_chunks, n_anchor)
    _save(terms, version, n_chunks, n_anchor)
    return terms


def schedule_warmup() -> None:
    """过期时丢一个后台重建任务，**不阻塞本轮问答**。

    供问答主链调用：建好之前本轮仍用旧词表（首轮为空 → 走手写表兜底），
    下一轮开始生效。这是「宁可稍晚生效，也不让用户等」取舍。
    """
    if _version is None or not is_stale():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:          # 不在事件循环里（CLI/脚本）：直接不后台化
        return
    task = loop.create_task(build())
    task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
