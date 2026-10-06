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
