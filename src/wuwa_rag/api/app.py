"""Step 9：FastAPI 端点。"""
from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..rag.chain import ask, ask_stream
from ..rag.memory import close_checkpointer, get_checkpointer
from ..worker import build_pipeline
from ..ww_logger import get_logger

log = get_logger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await get_checkpointer()
    yield
    await close_checkpointer()


app = FastAPI(title="wuwa-rag", version="0.9", lifespan=lifespan)


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
        async for evt in ask_stream(body.question, tid):
            yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/ingest")
async def api_ingest(body: IngestIn) -> dict:
    """把一个角色塞进异步流水线，立即返回 chain_id。"""
    r = build_pipeline(body.character).apply_async()
    return {"character": body.character, "chain_id": r.id, "state": r.state}
