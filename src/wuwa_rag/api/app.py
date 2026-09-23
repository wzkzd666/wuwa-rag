"""Step 9：FastAPI 端点。

2026-09-22 加系统鉴权 + 用户画像：
- 鉴权（api/auth.py + db.py）：admin 种子（admin/123456），游客 /auth/register 注册后登录；
  请求带 `Authorization: Bearer <token>`。/ask、/ask/stream 登录即可；/ingest* 管理员专属。
- 画像（rag/profile.py）：user_facts 表落地——问答后异步抽「稳定偏好事实」入库，
  下次提问取活跃事实注入生成 prompt（个性化）；/profile 查自己的、可软删单条。
"""
from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..authdb import close_pool, ensure_schema
from ..rag.chain import ask, ask_stream, doc_sources
from ..rag.memory import close_checkpointer, get_checkpointer
from ..rag.profile import (
    all_users_stats,
    extract_facts_safe,
    facts_to_context,
    get_facts,
    save_facts,
    soft_delete_fact,
)
from ..worker import PIPELINE_STEPS, STEP_LABELS, build_pipeline, get_progress
from ..ww_logger import get_logger
from . import auth as authn

log = get_logger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await get_checkpointer()
    await ensure_schema()   # 幂等建鉴权表 + 种子 admin/123456（见 pgsql/002_auth.sql）
    yield
    await close_checkpointer()
    await close_pool()


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
    sources: list[str] = []   # 引用来源面包屑（角色 › 模块 › 组件），去重保序
    truncated: bool = False   # 复读兜底触发、答案被截断过


class IngestIn(BaseModel):
    character: str = Field(..., description="角色中文名，如 忌炎")


class CredentialsIn(BaseModel):
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


# ── 鉴权（无需登录） ─────────────────────────────

@app.post("/auth/register")
async def api_register(body: CredentialsIn) -> dict:
    """游客注册（注册即登录，直接发 token）。"""
    return await authn.register(body.username, body.password)


@app.post("/auth/login")
async def api_login(body: CredentialsIn) -> dict:
    return await authn.login(body.username, body.password)


@app.get("/auth/me")
async def api_me(user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    return {"username": user.username, "role": user.role}


@app.post("/auth/logout")
async def api_logout(user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    await authn.revoke_token(user.token)
    return {"ok": True}


# ── 用户画像 ─────────────────────────────

@app.get("/profile")
async def api_profile(user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """当前用户的画像事实（新的在前）。"""
    facts = await get_facts(user.username)
    return {"username": user.username, "facts": facts}


@app.delete("/profile/fact/{fact_id}")
async def api_profile_delete(fact_id: int,
                             user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """软删自己的一条画像事实（valid_to=now，表设计为不物理删除）。"""
    ok = await soft_delete_fact(user.username, fact_id)
    if not ok:
        raise HTTPException(status_code=404, detail="事实不存在或不属于你")
    return {"ok": True}


@app.get("/admin/users")
async def api_admin_users(user: authn.AuthUser = Depends(authn.require_admin)) -> dict:
    """管理员：全部用户 + 各自画像条数。"""
    return {"users": await all_users_stats()}


# ── 问答（登录即可） ─────────────────────────────

async def _user_context(user: authn.AuthUser) -> str:
    """取该用户的画像事实拼注入串；查库失败回落空（画像只增益、绝不挡问答）。"""
    try:
        return facts_to_context(await get_facts(user.username))
    except Exception as exc:
        log.warning("读取画像失败（忽略）：%s", exc)
        return ""


def _spawn_profile_task(username: str, thread_id: str, question: str) -> None:
    """问答后异步抽画像事实（fire-and-forget，不阻塞响应）。"""
    async def job():
        try:
            facts = await extract_facts_safe(question)
            if facts:
                await save_facts(username, thread_id, facts)
        except Exception:
            log.exception("画像任务异常（忽略）")
    asyncio.create_task(job())


@app.post("/ask", response_model=AskOut)
async def api_ask(body: AskIn, user: authn.AuthUser = Depends(authn.get_current_user)) -> AskOut:
    tid = body.thread_id or uuid.uuid4().hex[:12]
    r = await ask(body.question, tid, user_context=await _user_context(user))
    log.info("ask thread=%s 意图=%s 角色=%s", tid, r.get("intent"), r.get("characters"))
    _spawn_profile_task(user.username, tid, body.question)
    return AskOut(
        answer=r.get("answer") or "",
        thread_id=tid,
        intent=r.get("intent", ""),
        slots=r.get("slots") or [],
        characters=r.get("characters") or [],
        docs=len(r.get("docs") or []),
        sources=doc_sources(r.get("docs") or []),
        truncated=bool(r.get("truncated")),
    )

@app.post("/ask/stream")
async def api_ask_stream(body: AskIn, user: authn.AuthUser = Depends(authn.get_current_user)):
    tid = body.thread_id or uuid.uuid4().hex[:12]
    user_ctx = await _user_context(user)

    async def event_gen():
        # SSE 一旦开始流式，响应头已发出，全局异常处理器接不住这里的异常——
        # 必须就地捕获并转成 {'error'} 事件下发（前端 store 有对应处理）。
        try:
            async for evt in ask_stream(body.question, tid, user_context=user_ctx):
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
        except Exception as exc:
            rid = uuid.uuid4().hex[:8]
            log.exception("[%s] SSE 流中途异常 thread=%s", rid, tid)
            payload = {"error": f"生成中断（编号 {rid}）", "detail": str(exc)[:300]}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        finally:
            # 流结束后再抽画像：不与生成抢 LLM（OLLAMA_NUM_PARALLEL=1）
            _spawn_profile_task(user.username, tid, body.question)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ── 知识库入库（管理员专属） ─────────────────────────────

@app.post("/ingest")
async def api_ingest(body: IngestIn,
                     user: authn.AuthUser = Depends(authn.require_admin)) -> dict:
    """把一个角色塞进异步流水线，立即返回 chain_id。"""
    r = build_pipeline(body.character).apply_async()
    return {"character": body.character, "chain_id": r.id, "state": r.state}


@app.get("/ingest/status")
async def api_ingest_status(character: str,
                            user: authn.AuthUser = Depends(authn.require_admin)) -> dict:
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
