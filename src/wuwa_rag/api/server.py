"""Step 9 启动入口。

必须自己控 loop：uvicorn 在 Windows 上默认起 ProactorEventLoop，
psycopg 异步在 Proactor 下报 InterfaceError。
所以用 asyncio.run(..., loop_factory=SelectorEventLoop) 包住 server.serve()。
"""
from __future__ import annotations

import asyncio

import uvicorn

from ..config import ensure_dirs, get_settings


def main() -> None:
    ensure_dirs()
    s = get_settings()
    cfg = uvicorn.Config(
        "wuwa_rag.api.app:app", host=s.API_HOST, port=s.API_PORT, reload=False
    )
    asyncio.run(uvicorn.Server(cfg).serve(), loop_factory=asyncio.SelectorEventLoop)


if __name__ == "__main__":
    main()
