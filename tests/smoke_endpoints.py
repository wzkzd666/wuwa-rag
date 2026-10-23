"""全栈冒烟（**手动跑的维护脚本，pytest 不收集**）：把今天的改动可能碰到的每个端点走一遍。

为什么值得有：今天动过 `ww_logger` 的名单、`conversations` 的删除口径、`api/app` 的回收清单、
`retrieve` 的重构与画像判据 —— 这些都是「单测全绿但接线断了也不会红」的地方（AGENTS 记过好几起）。
单测守判据、`eval_retrieval` 守检索效果、`eval_answers` 守答案质量，而**端点接线**这一层此前
每次都是临时手写 curl。

前置：`.\\dev.bat` 起服务（要 PG，因为令牌从本地库里取）。
用法：`uv run python tests/smoke_endpoints.py`
退出码 0 = 全通过；非 0 = 有端点异常，最后一行会报数量。

⚠️ 两条实测踩过的形状：
  · `/feedback/mine` 的查询参数叫 **`target_ids`**，写成 `ids` 会得到 422（那是校验对了，不是缺陷）；
  · `/conversations` 返回的每一项**没有 `session_id`**，对外键叫 **`thread_id`**（前端的删除
    与重命名都用它）。
⚠️ 令牌只在进程内使用，绝不打印；流式那一问会在结束时**自己删掉自己产生的会话**，
   不给库留 residue。
"""
from __future__ import annotations

import asyncio
import json
import sys
import time

import httpx

from wuwa_rag.config import get_settings

# (路径, 期望在响应里出现的顶层键)。/health 例外，它只有 {"ok": true}。
READS = [
    ("/health", ("ok",)),
    ("/auth/me", ("username", "role")),
    ("/profile", ("username", "facts")),
    ("/conversations", ("conversations",)),
    ("/llm/providers", ("providers",)),
    ("/llm/config", ("configured", "enabled", "key_hint")),
    ("/tts/status", ("enabled", "ready", "source")),
    ("/music/status", ("available",)),
    ("/usage/summary?days=7", ("days", "rows", "daily")),
    ("/feedback/mine?target_ids=1", ()),
    ("/knowledge/characters", ("items", "total", "seeded_only")),
    ("/ingest/records?limit=3", ("items", "total")),
    ("/admin/users", ("users",)),
    ("/debug/retrieval?question=%E5%AE%88%E5%B2%B8%E4%BA%BA%E9%85%8D%E9%98%9F",
     ("question", "dense", "sparse", "fused", "reranked", "gate")),
]
# SSE `done` 事件必须带的键（少一个就是前端某个面板拿到 undefined）
DONE_KEYS = ("answer", "intent", "docs", "sources", "truncated", "message_id",
             "emotion", "web_reason", "thread_id", "characters", "slots")


def _admin_token() -> str:
    """从**本机库**里取一条仍有效的 admin 令牌（冒烟要覆盖 admin 专属端点）。绝不打印。"""
    from wuwa_rag.core.authdb import get_pool

    async def _get():
        pool = await get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                """SELECT t.token FROM auth_tokens t JOIN users u ON u.id = t.user_id
                   WHERE u.role = 'admin' AND t.expires_at > now()
                   ORDER BY t.created_at DESC LIMIT 1""")
            row = await cur.fetchone()
        await pool.close()
        if row is None:
            raise SystemExit("库里没有仍在有效的 admin 令牌；先用 .\\dev.bat 起服务并登录一次")
        return row["token"]

    return asyncio.run(_get(), loop_factory=asyncio.SelectorEventLoop)


def _stream_once(base: str, headers: dict) -> int:
    """跑一轮流式问答，检查 token/stage/done 三类事件与 done 的键集合；返回异常数。"""
    tid = f"smoke-{int(time.time())}"
    bad = 0
    ev = {"token": 0, "stage": 0, "done": 0, "error": 0}
    keys: list[str] = []
    with httpx.Client(timeout=300) as c:
        with c.stream("POST", f"{base}/ask/stream",
                      json={"question": "守岸人配队", "thread_id": tid},
                      headers=headers) as r:
            if r.status_code != 200:
                print(f"BAD {r.status_code} /ask/stream")
                return 1
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                try:
                    d = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                if "token" in d:
                    ev["token"] += 1
                if "stage" in d:
                    ev["stage"] += 1
                if "error" in d:
                    ev["error"] += 1
                    print("  流内错误:", str(d)[:200])
                if d.get("done"):
                    ev["done"] += 1
                    keys = sorted(d)
        print(f"OK  200 /ask/stream{''.ljust(30)} token={ev['token']} stage={ev['stage']} "
              f"done={ev['done']} error={ev['error']}")
        missing = [k for k in DONE_KEYS if k not in keys]
        if missing:
            print(f"    !! done 缺键: {missing}")
            bad += 1
        if ev["done"] == 0 or ev["token"] == 0:
            print("    !! 没收到 done 或一个 token 都没有")
            bad += 1
        # 自己造的数据自己收走（会话删除会连带清模型侧记忆）
        rd = c.delete(f"{base}/conversations/{tid}", headers=headers)
        print(f"    清理本次会话 -> {rd.status_code}")
        if rd.status_code != 200:
            bad += 1
    return bad


def main() -> int:
    base = f"http://127.0.0.1:{get_settings().API_PORT}"
    headers = {"Authorization": f"Bearer {_admin_token()}"}
    bad = 0
    with httpx.Client(timeout=90) as c:
        for path, want in READS:
            r = c.get(f"{base}{path}", headers=headers)
            if r.status_code != 200:
                bad += 1
                print(f"BAD {r.status_code} {path}\n     {r.text[:160]}")
                continue
            try:
                j = r.json()
            except json.JSONDecodeError:
                bad += 1
                print(f"BAD 200 {path} 但响应不是 JSON")
                continue
            got = tuple(j) if isinstance(j, dict) else ()
            miss = [k for k in want if k not in got]
            if miss:
                bad += 1
                print(f"BAD 200 {path} 缺键 {miss}（实得 {list(got)[:6]}）")
            else:
                print(f"OK  200 {path}")
    bad += _stream_once(base, headers)
    print("\n冒烟:", "全部通过" if bad == 0 else f"{bad} 项异常")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
