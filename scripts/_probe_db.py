# scripts/_probe_db.py 或直接 python -c
import asyncio
from wuwa_rag.db import ping, close_pool

async def main():
    print(await ping())
    await close_pool()

asyncio.run(main(), loop_factory=asyncio.SelectorEventLoop)