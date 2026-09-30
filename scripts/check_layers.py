"""架构守卫：校验 `wuwa_rag` 内部依赖方向，违规即非零退出。

为什么要有这个脚本
------------------
「分层」如果只写在文档里，三个月后必然烂掉。这里把约束变成可执行检查：
CI 或提交前跑一次，任何「上层被下层依赖」「循环依赖」立刻暴露。

分层（数字越小越底层；同层可互相依赖，跨层只允许**上层 → 下层**）
    L0 基础   config / ww_logger / text
    L1 内核   core      配置读取、DB 连接池、鉴权表、口令哈希、LLM 客户端与凭据库
    L2 知识   knowledge 语料爬取、图谱、检索索引、实体词典
    L3 任务   tasks     Celery worker（编排 L2 的批处理）
    L4 服务   services 人格、情绪、TTS、校验、联网检索、用户画像
    L5 对话   dialog    LangGraph 编排：NLU → 路由 → 检索 → 生成
    L6 接口   api       FastAPI 路由与鉴权

用法
----
    python scripts/check_layers.py          # 人类可读报告，违规时 exit 1
    python scripts/check_layers.py --dot    # 额外输出 Graphviz DOT
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "src" / "wuwa_rag"

LAYERS: dict[str, int] = {
    "config": 0, "ww_logger": 0, "text": 0,
    "core": 1, "knowledge": 2, "tasks": 3,
    "services": 4, "dialog": 5, "api": 6,
}
LAYER_NAMES = ["L0 基础", "L1 内核", "L2 知识", "L3 任务", "L4 服务", "L5 对话", "L6 接口"]

FROM_RE = re.compile(r"^\s*from wuwa_rag\.([\w\.]*) import (.+)$")
IMPORT_RE = re.compile(r"^\s*import wuwa_rag\.([\w\.]+)")


def modules() -> dict[str, Path]:
    """点号模块名 -> 文件路径。`__init__.py` 归到它所属的包名上。"""
    out: dict[str, Path] = {}
    for p in PKG.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        rel = p.relative_to(PKG)
        name = ((".".join(rel.parts[:-1])) if rel.name == "__init__.py"
                else ".".join(rel.with_suffix("").parts))
        if name:
            out[name] = p
    return out


def resolve(head: str, item: str, known: set[str]) -> str | None:
    """`from wuwa_rag.<head> import <item>` 究竟依赖哪个模块。

    - `from wuwa_rag.core import llmstore`   -> item 是**子模块** -> core.llmstore
    - `from wuwa_rag.config import settings` -> item 是**符号**  -> 回退到 head
    - item 为空（括号换行导入）              -> 只认 head
    """
    if item:
        cand = f"{head}.{item}" if head else item
        if cand in known:
            return cand
    return head if head in known else None


def build_graph(mods: dict[str, Path]) -> dict[str, set[str]]:
    known = set(mods)
    graph: dict[str, set[str]] = defaultdict(set)
    for name, path in mods.items():
        for line in path.read_text(encoding="utf-8").split("\n"):
            m = FROM_RE.match(line)
            if not m:
                im = IMPORT_RE.match(line)
                if im and im.group(1) in known and im.group(1) != name:
                    graph[name].add(im.group(1))
                continue
            head, rest = m.group(1), m.group(2)
            if rest.lstrip().startswith("(") or "#" in rest:
                hits = [resolve(head, "", known)]
            else:
                hits = [resolve(head, it.split(" as ")[0].strip(), known)
                        for it in rest.split(",") if it.strip()]
            for hit in hits:
                if hit and hit != name:
                    graph[name].add(hit)
    return graph


def layer_of(mod: str) -> int | None:
    return LAYERS.get(mod.split(".")[0])


def find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    color: dict[str, int] = {}
    cycles: list[list[str]] = []

    def dfs(u: str, stack: list[str]) -> None:
        color[u] = 1
        stack.append(u)
        for v in sorted(graph.get(u, ())):
            if color.get(v, 0) == 0:
                dfs(v, stack)
            elif color.get(v) == 1 and len(cycles) < 12:
                cycles.append(stack[stack.index(v):] + [v])
        color[u] = 2
        stack.pop()

    for m in sorted(graph):
        if color.get(m, 0) == 0:
            dfs(m, [])
    return cycles


def main() -> int:
    mods = modules()
    graph = build_graph(mods)

    violations = [(s, d) for s, ds in graph.items() for d in ds
                  if layer_of(s) is not None and layer_of(d) is not None
                  and layer_of(d) > layer_of(s)]
    cycles = find_cycles(graph)

    by_layer: dict[int, list[str]] = defaultdict(list)
    unclassified: list[str] = []
    for m in sorted(mods):
        lv = layer_of(m)
        (by_layer[lv] if lv is not None else by_layer.setdefault(-1, [])).append(m)

    print(f"模块 {len(mods)} 个，内部依赖边 {sum(len(v) for v in graph.values())} 条\n")
    print("=== 分层概览 ===")
    for lv in sorted(k for k in by_layer if k >= 0):
        outs: dict[str, int] = defaultdict(int)
        for m in by_layer[lv]:
            for d in graph.get(m, ()):
                outs[f"L{layer_of(d)}"] += 1
        print(f"  {LAYER_NAMES[lv]:<8} {len(by_layer[lv]):>2} 模块   出边 {dict(sorted(outs.items()))}")
    if -1 in by_layer:
        print(f"  ⚠ 未归类 {len(by_layer[-1])}: {by_layer[-1]}")

    print("\n=== 跨包依赖（包级去重）===")
    pkg_edges: dict[tuple[str, str], int] = defaultdict(int)
    for s, ds in graph.items():
        for d in ds:
            pkg_edges[(s.split(".")[0], d.split(".")[0])] += 1
    for (a, b), n in sorted(pkg_edges.items()):
        if a == b:
            continue
        mark = "   <-- 逆流!" if (layer_of(b) or 0) > (layer_of(a) or 0) else ""
        print(f"  {a:<10} -> {b:<10} {n:>2}{mark}")

    print("\n=== 结论 ===")
    print("  分层方向违规：" + ("无" if not violations else ""))
    for s, d in violations:
        print(f"    ✗ {s} (L{layer_of(s)}) -> {d} (L{layer_of(d)})")
    print("  循环依赖：" + ("无" if not cycles else ""))
    for c in cycles:
        print("    ✗ " + " -> ".join(c))

    if "--dot" in sys.argv:
        print("\ndigraph wuwa_rag {")
        print('  rankdir=BT; node [shape=box, fontname="Consolas"];')
        for lv in sorted(k for k in by_layer if k >= 0):
            print(f'  subgraph cluster_{lv} {{ label="{LAYER_NAMES[lv]}";')
            print("    " + " ".join(f'"{m}";' for m in by_layer[lv]) + " }")
        for s, ds in sorted(graph.items()):
            for d in sorted(ds):
                print(f'  "{s}" -> "{d}";')
        print("}")

    return 1 if (violations or cycles) else 0


if __name__ == "__main__":
    sys.exit(main())
