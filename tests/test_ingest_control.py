"""入库的暂停 / 继续 / 取消（worker 侧控制旗标）。

为什么不用 `celery revoke`
--------------------------
五步是 `chain(...)` 串联的，每一步各有自己的 task_id。revoke 只能停掉**某一个** id：
停链尾那一步时前面几步照跑；要停整条链得先把所有 task_id 收集齐，而 `apply_async`
返回的 result 上取不全、随时可能漏 —— 表现为「点了取消，任务还在跑」。
旗标法是**每一步开头自查**（`raise_if_cancelled`），不依赖 task_id、不依赖 revoke 语义。

为什么暂停是「原地等待」
------------------------
`chain` 里某一步正常返回，下一步**照样会被调度**，Celery 没有「条件链」原语。
所以暂停 = 这一步醒来后原地等（1s 一轮），继续后自动往下跑，进度不丢。
代价是暂停期间占着一个 worker 槽 —— 本项目 worker 是 `--pool=solo`，一次只跑一条，
影响面很小；另设 `_PAUSE_MAX_WAIT` 兜底，避免「忘了取消」把 worker 永久占住。

这些用例把 `get_control` 换成脚本返回值，因此**不碰 Redis**、可离线跑。
"""
from __future__ import annotations

import pytest

from wuwa_rag.tasks import worker

NL = chr(10)   # 用例里的换行一律显式拼，避开源码里反斜杠被工具链吃掉



def _scripted(monkeypatch, values: list[str], *, poll: float = 0.01, max_wait: float = 0.05):
    """把 get_control 换成「按调用次数依次返回值」，并把等待参数压到毫秒级。"""
    seq = list(values)

    def fake_get_control(character: str) -> str:
        return seq.pop(0) if len(seq) > 1 else seq[0]

    monkeypatch.setattr(worker, "get_control", fake_get_control)
    monkeypatch.setattr(worker, "_PAUSE_POLL", poll)
    monkeypatch.setattr(worker, "_PAUSE_MAX_WAIT", max_wait)
    return fake_get_control


def test_无旗标时放行(monkeypatch) -> None:
    _scripted(monkeypatch, [""])
    assert worker.wait_if_paused("心") is False


def test_取消旗标立即判定中止(monkeypatch) -> None:
    """cancel 必须是**能被读到的值**而不是「键消失」——
    早先 `set_control('cancel')` 走的是删键分支，worker 只看得到 pause，取消根本传不出去。"""
    _scripted(monkeypatch, ["cancel"])
    assert worker.wait_if_paused("心") is True


def test_暂停时等待_取消后放行并判中止(monkeypatch) -> None:
    """先 pause（循环等待），随后 cancel —— 必须立刻退出等待并判定为中止。"""
    _scripted(monkeypatch, ["pause", "cancel"], poll=0.01, max_wait=5)
    assert worker.wait_if_paused("心") is True


def test_暂停等待超时自动放行(monkeypatch) -> None:
    """一直 pause：超过 _PAUSE_MAX_WAIT 后自动继续，不能把 worker 永久占住。"""
    _scripted(monkeypatch, ["pause"], poll=0.01, max_wait=0.03)
    assert worker.wait_if_paused("心") is False


def test_守卫在被取消时落账并抛异常(monkeypatch) -> None:
    """raise_if_cancelled 必须**抛异常**才能掐断整条链；正常返回会让 chain 继续调度下一步。"""
    _scripted(monkeypatch, ["cancel"])
    marks: list[tuple[str, str, str | None]] = []
    monkeypatch.setattr(worker, "_progress_mark",
                        lambda ch, st, status, error=None: marks.append((ch, st, error)))
    with pytest.raises(worker.IngestCancelled):
        worker.raise_if_cancelled("心", "crawl")
    assert marks == [("心", "crawl", worker.CANCEL_ERROR)]


def test_守卫在暂停后继续放行不抛(monkeypatch) -> None:
    _scripted(monkeypatch, ["pause", ""], poll=0.01, max_wait=5)
    monkeypatch.setattr(worker, "_progress_mark", lambda *a, **k: None)
    worker.raise_if_cancelled("心", "chunk")     # 不抛即通过


def test_控制旗标键名按角色隔离() -> None:
    assert worker.ctl_key("心") == "ingest:ctl:心"
    assert worker.progress_key("心") == "ingest:progress:心"
    assert worker.ctl_key("心") != worker.progress_key("心")


# ---------- 分块步：0 块与「产物缺角色」不能静默通过 ----------

def test_分块零结果必须响亮失败(tmp_path, monkeypatch) -> None:
    """0 块当成功 → jsonl 被写成只剩这个角色，而五步链全绿（真丢数据的前夜）。"""
    import asyncio
    from types import SimpleNamespace

    raw = tmp_path / "空角色.md"
    raw.write_text("", encoding="utf-8")                     # 空 raw 必然分不出块
    # Settings 的 CHUNKS_JSONL 是只读 property，改不了实例 —— 直接换掉模块里的 s 引用
    monkeypatch.setattr(worker, "s", SimpleNamespace(
        RAW_DIR=tmp_path, CHUNKS_JSONL=tmp_path / "chunks.jsonl"))

    with pytest.raises(RuntimeError, match="0 块"):
        asyncio.run(worker._chunk_character_async("空角色"),
                    loop_factory=asyncio.SelectorEventLoop)
    # 关键：失败时**不许**已经把文件写坏
    assert not (tmp_path / "chunks.jsonl").exists()


def test_产物缺角色只告警不拦路(tmp_path, monkeypatch) -> None:
    """空 jsonl 对全新安装是合法起点，所以覆盖不全只 warning，不 raise。"""
    import asyncio
    from types import SimpleNamespace

    calls: dict = {}

    class _Boom:                     # get_cursor 直接抛：诊断必须不影响主链
        async def __aenter__(self): raise RuntimeError("PG 没起")
        async def __aexit__(self, *a): return False

    def _fake_cursor(**_k):
        calls["n"] = calls.get("n", 0) + 1
        return _Boom()

    monkeypatch.setattr(worker, "get_cursor", _fake_cursor)
    monkeypatch.setattr(worker.log, "warning", lambda *a, **k: None)
    md = tmp_path / "新角色.md"
    md.write_text("## 基础资料" + NL + NL + "- 属性：导电" + NL + NL + "### 概述" + NL + "正文一段。" + NL, encoding="utf-8")
    monkeypatch.setattr(worker, "s", SimpleNamespace(
        RAW_DIR=tmp_path, CHUNKS_JSONL=tmp_path / "chunks.jsonl"))

    asyncio.run(worker._chunk_character_async("新角色"),
                loop_factory=asyncio.SelectorEventLoop)
    assert (tmp_path / "chunks.jsonl").read_text(encoding="utf-8").strip()
    assert calls.get("n") == 1                              # 诊断确实跑了，但被吞掉了


def test_bm25缓存必须随磁盘指纹失效(tmp_path, monkeypatch) -> None:
    """worker 是**另一个进程**：它重写 bm25.pkl，API 进程里的缓存不会自己知道。

    `@lru_cache(maxsize=1)` 的形态是「新角色入库了、五步链也报成功，但稀疏检索
    永远召不回它」，只有重启 API 才好——dense 那条实时查 Chroma 所以看起来像偶发召回问题。
    这里用替身数 `load()` 的调用次数，验两件事：文件没变不许重载（那才是缓存的意义）、
    文件变了必须重载（那才是这个 bug 的修法）。
    """
    from types import SimpleNamespace

    from wuwa_rag.knowledge import retrieve

    loads: list[int] = []

    class _Idx:
        def __init__(self, tag: int) -> None:
            self.tag = tag

    def _load() -> _Idx:
        loads.append(1)
        return _Idx(len(loads))

    monkeypatch.setattr(retrieve, "BM25Index", type("BM25Index", (), {"load": staticmethod(_load)}))
    pkl = tmp_path / "bm25.pkl"
    pkl.write_bytes(b"a")
    monkeypatch.setattr(retrieve, "get_settings",
                        lambda: SimpleNamespace(VECTOR_DIR=tmp_path))
    monkeypatch.setattr(retrieve, "_BM25_CACHE", None)

    assert retrieve._bm25().tag == 1
    assert retrieve._bm25().tag == 1
    assert len(loads) == 1, "文件没变却重载了 —— 缓存白写"

    pkl.write_bytes(b"changed-content-longer")     # 大小与 mtime 都变
    assert retrieve._bm25().tag == 2, "worker 重写了 pkl，API 还在用旧快照"
    assert len(loads) == 2

def test_覆盖不全时真的写出warning(tmp_path, monkeypatch) -> None:
    """上一条只验了「不拦路」，这条验「能 warn」—— 缺了它，`r["character"]` 那种
    取列写法错误会被诊断自己的宽 except 吞掉，函数变成永远不会出声的死代码（真实踩过：
    `core/db.py` 的池没有 dict_row，行是**元组**）。"""
    import asyncio
    from types import SimpleNamespace

    class _Rows:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def execute(self, *_a): return None
        async def fetchall(self):
            # 真库口径：元组行，不是 dict
            return [("守岸人",), ("卡卡罗",), ("忌炎",)]

    monkeypatch.setattr(worker, "get_cursor", lambda **_k: _Rows())
    warned: list[str] = []
    monkeypatch.setattr(worker.log, "warning",
                        lambda fmt, *a, **k: warned.append(fmt % a))

    md = tmp_path / "新角色.md"
    md.write_text("## 基础资料" + NL + NL + "- 属性：导电" + NL + NL + "### 概述" + NL + "正文一段。" + NL, encoding="utf-8")
    monkeypatch.setattr(worker, "s", SimpleNamespace(
        RAW_DIR=tmp_path, CHUNKS_JSONL=tmp_path / "chunks.jsonl"))

    asyncio.run(worker._chunk_character_async("新角色"),
                loop_factory=asyncio.SelectorEventLoop)

    assert len(warned) == 1, f"该 warn 的没 warn（诊断被吞了）：{warned}"
    assert "守岸人" in warned[0] and "wuwa-chunk" in warned[0]
