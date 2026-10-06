"""LLM 用量计量：每次调用落一行 `llm_usage`，供管理员看板按人/按 provider 统计。

怎么量的（为什么不用「改每个调用点」的路）
----------------------------------------
实测（ChatOllama + 本地 aemeath）：**流式与非流式都能从 `on_llm_end` 回调里拿到精确
用量**（`generations[0][0].message.usage_metadata`）。而 LangChain 把原始 chunk 的
`prompt_eval_count/eval_count` 丢掉了（`response_metadata` 是空的），所以
**不能**靠读 chunk 字段。

于是走两个钩子，都不用改调用点：
1. **客户端级回调**：`_build_ollama` / `_build_openai` 构造时挂上 `UsageCollector`，
   于是这个客户端的**每一次** invoke/astream 都会自动上报；
2. **contextvar 传归属**：`scope()` 在 API 入口把 user/thread 塞进去，collector 读出来
   写进那一行。调用点（nlu / verify / emotion / graph…）完全不知道自己被计量了。

provider/model 在构造时就知道（`local`/`cloud` + 模型名），scene 由构造它的入口决定。

⚠️ 计量**绝不能影响主链**：落库用 `asyncio.create_task` fire-and-forget，
任何异常只记日志。上报失败就当没发生 —— 看板是管理功能，不是业务。
"""
from __future__ import annotations

import asyncio
import contextvars
from contextlib import contextmanager
from datetime import datetime, timedelta

from langchain_core.callbacks import BaseCallbackHandler

from wuwa_rag.core.db import get_cursor
from wuwa_rag.ww_logger import get_logger

log = get_logger("usage")

# ── 本轮归属（user / thread）────────────────────────────────────────────
# 用 contextvar 而不是逐层传参：调用点分散在 nlu/verify/emotion/graph 四处，
# 逐个加参数等于把计量逻辑糊进业务代码。contextvar 是 asyncio 任务级隔离的，
# 天然适配「每轮问答一个任务」的并发模型。
_current: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar(
    "llm_usage_scope", default={}
)


@contextmanager
def scope(**fields):
    """声明本轮归属（user_id / username / thread_id）。API 入口调用一次即可。"""
    token = _current.set({**_current.get(), **{k: v for k, v in fields.items() if v is not None}})
    try:
        yield
    finally:
        _current.reset(token)


def current_scope() -> dict[str, str]:
    return _current.get()


class UsageCollector(BaseCallbackHandler):
    """把一次 LLM 调用的用量写进 `llm_usage`。挂在客户端上，全调用点自动生效。"""

    def __init__(self, scene: str, provider: str, model: str) -> None:
        self.scene = scene
        self.provider = provider
        self.model = model

    def on_llm_end(self, response, **_kwargs) -> None:  # noqa: ANN001
        try:
            usage = None
            for gen in (getattr(response, "generations", None) or [])[:1]:
                for g in (gen or [])[:1]:
                    usage = getattr(getattr(g, "message", None), "usage_metadata", None)
            if not usage:
                return
            scope_ = _current.get()
            row = {
                "user_id": scope_.get("user_id"),
                "username": scope_.get("username"),
                "thread_id": scope_.get("thread_id"),
                "scene": self.scene,
                "provider": self.provider,
                "model": self.model,
                "prompt_tokens": int(usage.get("input_tokens") or 0),
                "completion_tokens": int(usage.get("output_tokens") or 0),
            }
            _spawn(_insert_usage, row)
        except Exception as exc:  # noqa: BLE001 —— 计量是观测性功能，挂了不能影响问答
            log.debug("用量计量失败（忽略）: %s", exc)


def _spawn(fn, *args) -> None:
    """把落库丢到后台。事件循环没在跑时（CLI/脚本）直接同步跑完。

    注意：传的是**函数 + 参数**而不是已建好的协程 —— 没有运行中的事件循环时
    协程对象无法被执行，那样同步兜底分支就废了。
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001
            log.debug("用量落库失败（忽略）: %s", exc)
        return
    task = loop.create_task(asyncio.to_thread(fn, *args))
    task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)


def _insert_usage(row: dict) -> None:
    """同步驱动写一行用量（跑在 `to_thread` 的工作线程里）。

    ⚠️ **刻意不复用 `core.db` 的异步连接池**：那个池跟应用生命周期绑定（启动时 open、
    关闭时 close），而计量是 fire-and-forget 的，**必然与池的开关竞态** ——
    实测踩到 `pool initialization incomplete` 与 `pool is already closed`，
    失败还被静默吞掉，表现为「用量统计莫名其妙少几条」。自己开一条短连接，
    多花几毫秒换一个不会丢统计。
    另：psycopg 的异步实现要求 SelectorEventLoop，而 `to_thread` 给的是普通工作线程，
    所以这里只能用同步驱动（与 api/app.py 里的 `_create_crawl_run` 同一套写法）。
    """
    import psycopg

    from wuwa_rag.config import get_settings
    try:
        with psycopg.connect(get_settings().PG_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO llm_usage (user_id, username, thread_id, scene, provider,"
                    " model, prompt_tokens, completion_tokens) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                    (row["user_id"], row["username"], row["thread_id"], row["scene"],
                     row["provider"], row["model"], row["prompt_tokens"],
                     row["completion_tokens"]),
                )
            conn.commit()
    except Exception as exc:  # noqa: BLE001
        log.warning("用量落库失败（忽略）: %s", exc)


async def usage_summary(days: int = 7, only_user: str | None = None) -> dict:
    """近 N 天用量汇总：按人 × provider(local/cloud) 聚合。

    `only_user` 非空时只统计该用户 —— **普通用户只能看自己的**，由 API 层强制传，
    不依赖前端藏按钮（越权必须后端拦）。
    """
    sql = (
        "SELECT user_id, username,"
        " sum(prompt_tokens) FILTER (WHERE provider = 'local') AS local_prompt,"
        " sum(completion_tokens) FILTER (WHERE provider = 'local') AS local_completion,"
        " sum(prompt_tokens) FILTER (WHERE provider = 'cloud') AS cloud_prompt,"
        " sum(completion_tokens) FILTER (WHERE provider = 'cloud') AS cloud_completion,"
        " count(*) AS calls,"
        " max(created_at) AS last_at"
        " FROM llm_usage WHERE created_at >= now() - make_interval(days => %s)"
    )
    args: list = [days]
    if only_user is not None:
        sql += " AND user_id = %s"
        args.append(only_user)
    sql += " GROUP BY user_id, username ORDER BY (sum(prompt_tokens) + sum(completion_tokens)) DESC"
    async with get_cursor() as cur:
        await cur.execute(sql, tuple(args))
        cols = [d.name for d in cur.description]
        rows = [dict(zip(cols, r)) for r in await cur.fetchall()]
    return {"days": days, "rows": rows, "daily": await daily_usage(days, only_user)}


async def daily_usage(days: int = 7, only_user: str | None = None) -> list[dict]:
    """近 N 天**按天**的用量（画图用）。

    按会话时区分日（`AT TIME ZONE current_setting('TimeZone')`）而不是 UTC ——
    否则每根柱子会整体偏 8 小时，「今天的用量」看着像昨天的。
    """
    sql = (
        "SELECT (created_at AT TIME ZONE current_setting('TimeZone'))::date AS d,"
        " sum(prompt_tokens) FILTER (WHERE provider = 'local') AS local_prompt,"
        " sum(completion_tokens) FILTER (WHERE provider = 'local') AS local_completion,"
        " sum(prompt_tokens) FILTER (WHERE provider = 'cloud') AS cloud_prompt,"
        " sum(completion_tokens) FILTER (WHERE provider = 'cloud') AS cloud_completion,"
        " count(*) AS calls"
        " FROM llm_usage WHERE created_at >= (now() - make_interval(days => %s))::date"
    )
    args: list = [days]
    if only_user is not None:
        sql += " AND user_id = %s"
        args.append(only_user)
    sql += " GROUP BY 1 ORDER BY 1"
    async with get_cursor() as cur:
        await cur.execute(sql, tuple(args))
        cols = [d.name for d in cur.description]
        got = {str(r[0]): dict(zip(cols, r)) for r in await cur.fetchall()}
    # 补齐没有记录的日期：折线图断点会被画成直线插值，看着像「那天有量」
    out: list[dict] = []
    today = datetime.now().astimezone().date()
    for i in range(days - 1, -1, -1):
        key = str(today - timedelta(days=i))
        row = got.get(key) or {
            "d": key, "local_prompt": 0, "local_completion": 0,
            "cloud_prompt": 0, "cloud_completion": 0, "calls": 0,
        }
        out.append(row)
    return out
