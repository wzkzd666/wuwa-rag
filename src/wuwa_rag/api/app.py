"""Step 9：FastAPI 端点。"""
from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..rag.chain import ask, ask_stream
from ..rag.memory import close_checkpointer, get_checkpointer
from ..worker import PIPELINE_STEPS, STEP_LABELS, build_pipeline, get_progress
from ..ww_logger import get_logger

log = get_logger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await get_checkpointer()
    yield
    await close_checkpointer()


app = FastAPI(title="wuwa-rag", version="0.9", lifespan=lifespan)


# ── 全局异常处理 ─────────────────────────────
# 未捕获异常不再裸 500（前端只能看到 "HTTP 500"，无从下手）：
# 统一 {error: 简述} JSON + 服务端完整堆栈日志。request_id 串起两端。
@app.exception_handler(Exception)
async def on_unhandled(request: Request, exc: Exception) -> JSONResponse:
    rid = uuid.uuid4().hex[:8]
    log.exception("[%s] 未捕获异常 %s %s", rid, request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": f"服务内部错误（编号 {rid}，详情见服务端日志）", "detail": str(exc)[:300]},
    )


class AskIn(BaseModel):
    question: str = Field(..., description="问题")
    thread_id: str | None = Field(None, description="会话 id；不传则新建")


class AskOut(BaseModel):
    answer: str
    thread_id: str
    intent: str = ""
    slots: list[str] = []
    characters: list[str] = []
    docs: int = 0
    truncated: bool = False   # 复读兜底触发、答案被截断过


class IngestIn(BaseModel):
    character: str = Field(..., description="角色中文名，如 忌炎")


@ app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.post("/ask", response_model=AskOut)
async def api_ask(body: AskIn) -> AskOut:
    tid = body.thread_id or uuid.uuid4().hex[:12]
    r = await ask(body.question, tid)
    log.info("ask thread=%s 意图=%s 角色=%s", tid, r.get("intent"), r.get("characters"))
    return AskOut(
        answer=r.get("answer") or "",
        thread_id=tid,
        intent=r.get("intent", ""),
        slots=r.get("slots") or [],
        characters=r.get("characters") or [],
        docs=len(r.get("docs") or []),
        truncated=bool(r.get("truncated")),
    )

@app.post("/ask/stream")
async def api_ask_stream(body: AskIn):
    tid = body.thread_id or uuid.uuid4().hex[:12]

    async def event_gen():
        # SSE 一旦开始流式，响应头已发出，全局异常处理器接不住这里的异常——
        # 必须就地捕获并转成 {'error'} 事件下发（前端 store 有对应处理）。
        try:
            async for evt in ask_stream(body.question, tid):
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
        except Exception as exc:
            rid = uuid.uuid4().hex[:8]
            log.exception("[%s] SSE 流中途异常 thread=%s", rid, tid)
            payload = {"error": f"生成中断（编号 {rid}）", "detail": str(exc)[:300]}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/ingest")
async def api_ingest(body: IngestIn) -> dict:
    """把一个角色塞进异步流水线，立即返回 chain_id。"""
    r = build_pipeline(body.character).apply_async()
    return {"character": body.character, "chain_id": r.id, "state": r.state}


@app.get("/ingest/status")
async def api_ingest_status(character: str) -> dict:
    """按角色查入库进度（worker 侧 _progress_mark 写 Redis 聚合键）。

    不依赖 chain_id：/ingest 返回的 chain_id 刷新页面就丢了，而进度键按角色
    天然可查。steps 恒为五步数组（含中文标签），整体 status：
    pending(还没开跑/无记录) | running | success | failed。
    """
    snap = await asyncio.to_thread(get_progress, character)
    if snap is None:
        return {"character": character, "status": "pending", "found": False,
                "steps": [{"key": k, "label": STEP_LABELS[k], "status": "pending",
                           "error": None} for k in PIPELINE_STEPS],
                "updated_at": None}
    steps = []
    overall = "success"
    for k in PIPELINE_STEPS:
        st = snap["steps"].get(k, "pending")
        if st == "failed":
            overall = "failed"
        elif st in ("pending", "running"):
            overall = "running"
        steps.append({"key": k, "label": STEP_LABELS[k], "status": st,
                      "error": snap.get("errors", {}).get(k)})
    return {"character": character, "status": overall, "found": True,
            "steps": steps, "updated_at": snap.get("updated_at")}
