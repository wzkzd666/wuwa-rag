"""架构守卫：分层方向、循环依赖、空包清理。

这条测试是**把 `scripts/check_layers.py` 从「手动跑一次看看」变成提交门禁**。
项目文档里写着「改完结构跑一次；接 CI 后可以作为提交门禁」，本文件就是那一步。

守卫在守什么
----------
`wuwa_rag` 分 7 层，只允许**向下**依赖（L6 api → L5 dialog → … → L0 config）。
向上依赖或成环，意味着「改一处炸一片」和「导入顺序决定能不能跑」。
历史上 `api/auth.py` 下沉口令哈希是唯一被批准的例外，已在 ARCHITECTURE.md 单独说明。

⚠️ 两个实测口径，别按直觉改断言
------------------------------
① **`modules()` 的键名不带 `wuwa_rag.` 前缀**（是 `dialog.prompt`，不是
   `wuwa_rag.dialog.prompt`）。这不是风格问题：写成带前缀的断言会**恒真通过**，
   空包回来了测试也不会红——典型假绿。本文件所有键名都是无前缀形态。
② 计数口径：
   · `modules()` 返回 **56**：`__init__.py` 归到它所属的包名，所以顶层包与各子包
     （`core`、`knowledge.graph` …）都算一个模块，不是「纯 .py 文件数」；
   · `build_graph()` 只有 **41** 个键：它用 defaultdict，**只收录有出边的模块**，
     没有任何内部依赖的叶子模块不会出现在键里；
   · 依赖边 **164** 条、违规 **0**、环 **0**。
2026-09-30 重构遗留的 5 个空包（rag/graph/ingest/retrieval/storage）已于
2026-10-06 清理；在此之前它们让模块计数虚增 5（55 vs 50）。
同日新增 `knowledge/domain_terms.py`（单字角色名消歧用的领域词表派生），
+1 模块 / +7 边：它自己依赖 config+core.db+ww_logger（3 条出边），
被 entities / dialog.graph / api.app / tasks.worker 引用（4 条入边）。
"""
from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_GUARD = _ROOT / "scripts" / "check_layers.py"

# 实测基线（2026-10-06）。这些数字会随正常开发变化——
# 变化时应当**读懂为什么变了**再更新，而不是为了让测试变绿而随手改数。
EXPECTED_MODULES = 56
EXPECTED_EDGES = 164
EXPECTED_GRAPH_KEYS = 41
EXPECTED_LAYER_COUNTS = {0: 3, 1: 10, 2: 17, 3: 2, 4: 8, 5: 9, 6: 7}

# 2026-09-30 重构后遗留、2026-10-06 清理掉的空壳包（**无前缀**键名，见模块 docstring ①）。
REMOVED_EMPTY_PACKAGES = ("rag", "graph", "ingest", "retrieval", "storage")

# 现役子包及其子模块数下限（实测值）。与上面那条配对：删空包时绝不能连带删掉同名现役包。
ACTIVE_SUBPACKAGES = (
    ("knowledge.graph", 3),   # build_graph / extract / neo4j_client
    ("knowledge.index", 4),   # bm25 / build_index / embeddings / rerank
    ("knowledge.crawl", 2),   # chunker / pipeline
)


@pytest.fixture(scope="module")
def guard():
    """把守卫脚本作为模块加载（它在 scripts/ 下，不是包成员，只能按路径加载）。"""
    assert _GUARD.exists(), f"架构守卫脚本不存在：{_GUARD}"
    spec = importlib.util.spec_from_file_location("check_layers", _GUARD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def modules(guard) -> dict:
    return guard.modules()


@pytest.fixture(scope="module")
def graph(guard, modules):
    return guard.build_graph(modules)


# ---------- 核心不变量 ----------

def test_无向上依赖(guard, graph) -> None:
    """🔴 分层方向：任何模块都不得依赖比它更高的层。

    这是守卫存在的**唯一理由**。一条向上依赖就足以让「底层可独立测试/复用」失效。
    """
    violations = []
    for src, dsts in graph.items():
        src_layer = guard.layer_of(src)
        for dst in dsts:
            dst_layer = guard.layer_of(dst)
            if src_layer is None or dst_layer is None:
                continue
            if dst_layer > src_layer:
                violations.append((src, dst, src_layer, dst_layer))
    assert not violations, f"发现向上依赖：{violations}"


def test_无循环依赖(guard, graph) -> None:
    cycles = guard.find_cycles(graph)
    assert not cycles, f"发现循环依赖：{cycles}"


def test_模块与依赖边计数(modules, graph) -> None:
    """计数回归桩：数字变了说明结构动了，必须确认是**预期内**的改动。

    模块数减少要特别警惕——可能误删了活跃包（顶层 `graph` 与 `knowledge.graph`
    同名不同物，删空包时最容易连带删错）。
    """
    assert len(modules) == EXPECTED_MODULES, (
        f"模块数 {len(modules)} != {EXPECTED_MODULES}。"
        f"若减少：确认没误删活跃包；若增加：确认新模块归入了正确的层。"
    )
    edges = sum(len(v) for v in graph.values())
    assert edges == EXPECTED_EDGES, f"依赖边 {edges} != {EXPECTED_EDGES}"
    assert len(graph) == EXPECTED_GRAPH_KEYS, (
        f"有出边的模块数 {len(graph)} != {EXPECTED_GRAPH_KEYS}"
        f"（build_graph 用 defaultdict，叶子模块不计入键，别按 modules() 的数断言）"
    )


def test_各层模块分布(guard, modules) -> None:
    """分层分布回归桩。L2 knowledge 最厚（16）、L0 最薄（3）是预期形态。"""
    counts = Counter(guard.layer_of(m) for m in modules)
    assert dict(sorted(counts.items())) == EXPECTED_LAYER_COUNTS


def test_每个模块都能归入某一层(guard, modules) -> None:
    """layer_of 返回 None = 该模块的顶层包不在 LAYERS 表里，守卫会**静默跳过**它。

    那等于「新加的包不受分层约束」——比违规更糟，因为看不见。
    新建顶层包时必须同步更新 `check_layers.py` 的 LAYERS 表。
    """
    unlayered = [m for m in modules if guard.layer_of(m) is None]
    assert not unlayered, f"这些模块未被分层表覆盖（守卫会静默跳过）：{unlayered}"


# ---------- 空包清理回归 ----------

@pytest.mark.parametrize("pkg", REMOVED_EMPTY_PACKAGES)
def test_重构遗留空包已清理(modules, pkg: str) -> None:
    """2026-09-30 重构（rag/ → dialog/ knowledge/ services/）留下的空壳必须保持删除状态。

    它们只含 `__init__.py`、零引用，却让守卫的模块计数虚增 5，掩盖真实的结构规模。

    ⚠️ 键名**不带** `wuwa_rag.` 前缀（实测确认）。带前缀会让本断言恒真——
    空包回来了也测不出来。`graph` 这条尤其要小心：顶层 `graph`（旧空壳，已删）
    与 `knowledge.graph`（现役 L2 子包）同名不同物，无前缀键名正好只查顶层那个。
    """
    assert pkg not in modules, f"空壳包 {pkg}/ 又出现了（应只存在于 knowledge/ 等现役目录下）"


@pytest.mark.parametrize("pkg,min_children", ACTIVE_SUBPACKAGES)
def test_现役子包未被误删(modules, pkg: str, min_children: int) -> None:
    """与上一条配对：删空包时**绝不能**连带删掉同名的现役子包。"""
    assert pkg in modules, f"现役包 {pkg} 不见了"
    children = [m for m in modules if m.startswith(pkg + ".")]
    assert len(children) >= min_children, (
        f"{pkg} 只剩 {len(children)} 个子模块，少于实测的 {min_children}"
    )


# ---------- 守卫脚本自身的健全性 ----------

def test_resolve_能区分子模块与符号(guard, modules) -> None:
    """`from wuwa_rag.core import llmstore` → core.llmstore（子模块）；
    `from wuwa_rag.config import get_settings` → config（符号，回退到 head）。

    分不清就会把「用了某个符号」误记成「依赖某个模块」，依赖图会虚胖。
    返回值同样是**无前缀**形态。
    """
    known = set(modules)
    assert guard.resolve("core", "llmstore", known) == "core.llmstore"
    assert guard.resolve("config", "get_settings", known) == "config"
    assert guard.resolve("core", "不存在的符号", known) == "core"


def test_守卫能识别新增的两个模块(guard, modules) -> None:
    """2026-10-06 新增 `dialog/prompt.py`（L5）与 `api/ratelimit.py`（L6）。

    钉这条是因为它们曾被守卫的**噪声**掩盖过：当时报 54 模块（含 5 个空包），
    新模块带来的变化被虚增计数盖住，看不出守卫到底扫没扫到。
    守卫若扫不到新模块，分层约束就是假的。
    """
    assert "dialog.prompt" in modules
    assert "api.ratelimit" in modules
    assert guard.layer_of("dialog.prompt") == 5
    assert guard.layer_of("api.ratelimit") == 6
