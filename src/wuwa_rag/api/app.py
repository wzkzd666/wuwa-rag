"""Step 9：FastAPI 端点。

系统鉴权与用户画像：
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
from ..config import get_settings
from ..rag import llmstore
from ..rag.chain import ask, ask_stream, doc_sources
from ..rag.emotion import EMOTION_TAGS
from ..rag.llmstore import PROVIDER_PRESETS
from ..rag.memory import close_checkpointer, get_checkpointer
from ..rag.profile import (
    all_users_stats,
    extract_facts_safe,
    facts_to_context,
    get_facts,
    save_facts,
    soft_delete_fact,
)
from ..rag.tts import synthesize, tts_ready
from ..worker import PIPELINE_STEPS, STEP_LABELS, build_pipeline, get_progress
from ..ww_logger import get_logger
from . import auth as authn

log = get_logger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await get_checkpointer()
    await ensure_schema()   # 幂等建鉴权表 + 种子 admin/123456（见 pgsql/002_auth.sql）
    await llmstore.ensure_schema()   # 幂等建 user_llm_configs（云端模型配置，含加密 key）
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
    emotion: str = ""         # 情绪标签（TTS 用；TTS 未开启时为空串）


class IngestIn(BaseModel):
    character: str = Field(..., description="角色中文名，如 忌炎")


class TtsIn(BaseModel):
    text: str = Field(..., description="待合成文本（通常是某条 AI 回答）")
    emotion: str = Field("", description="情绪标签名；空则用默认 cheerful")


class LlmConfigIn(BaseModel):
    """用户自定义云端模型配置。api_key 留空表示「保留原 key 不改」。"""
    base_url: str = Field(..., description="OpenAI 兼容 base_url，到 /v1 为止")
    model: str = Field(..., description="模型 id，如 gpt-4o-mini / deepseek-chat")
    api_key: str = Field("", description="API Key；留空表示保留已存的 key")
    provider: str = Field("openai", description="预设名，仅用于前端归类，不影响调用")
    enabled: bool = Field(True, description="停用则回落本地默认模型")
    emotion_enabled: bool = Field(
        False, description="是否用该模型兼任情绪判定；false 则走本地 qwen3:8b")


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
    r = await ask(body.question, tid, user_context=await _user_context(user),
                  user_id=user.id)
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
        emotion=r.get("emotion", ""),
    )

@app.post("/ask/stream")
async def api_ask_stream(body: AskIn, user: authn.AuthUser = Depends(authn.get_current_user)):
    tid = body.thread_id or uuid.uuid4().hex[:12]
    user_ctx = await _user_context(user)

    async def event_gen():
        # SSE 一旦开始流式，响应头已发出，全局异常处理器接不住这里的异常——
        # 必须就地捕获并转成 {'error'} 事件下发（前端 store 有对应处理）。
        try:
            async for evt in ask_stream(body.question, tid, user_context=user_ctx,
                                        user_id=user.id):
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


# ── 语音合成（TTS，登录即可）─────────────────────────────

@app.get("/tts/status")
async def api_tts_status(user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """TTS 是否可用 + 当前模型/音色（不暴露 key）。前端据此决定是否显示播放按钮。"""
    ready, why = tts_ready()
    s = get_settings()
    return {"enabled": s.TTS_ENABLED, "ready": ready, "reason": why,
            "model": s.TTS_MODEL if ready else "", "voice": s.TTS_VOICE if ready else "",
            "emotions": list(EMOTION_TAGS)}


@app.post("/tts")
async def api_tts(body: TtsIn, user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """把文本合成语音，返回音频 URL（24h 有效）。

    未开启或配置不全时返回 200 + ok=false + 可读原因，而不是 503：
    这是"功能预留"而非服务故障，前端只需隐藏按钮或提示未开启，不该弹错误。
    """
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text 不能为空")
    s = get_settings()
    if len(text) > s.TTS_MAX_CHARS * 2:
        raise HTTPException(status_code=400,
                            detail=f"文本过长（>{s.TTS_MAX_CHARS * 2} 字），请缩短后再合成")
    r = await synthesize(text, body.emotion or None)
    if not r.ok:
        log.info("TTS 未合成 user=%s 原因=%s", user.username, r.error)
        return {"ok": False, "error": r.error, "url": "", "emotion": "", "elapsed_ms": 0}
    return {"ok": True, "url": r.url, "error": "", "emotion": r.emotion,
            "model": r.model, "voice": r.voice, "elapsed_ms": r.elapsed_ms}


# ── 用户自定义云端模型（登录即可，配置只属于自己）─────────────────

@app.get("/llm/providers")
async def api_llm_providers(user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """provider 预设列表（含默认 base_url），前端做下拉与自动填充。"""
    return {"providers": [
        {"key": k, "label": v["label"], "base_url": v["base_url"]}
        for k, v in PROVIDER_PRESETS.items()
    ]}


@app.get("/llm/config")
async def api_llm_config_get(user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """读自己的云端模型配置。**只返回掩码**，任何情况都不回显明文 key。"""
    cfg = await llmstore.get_config_masked(user.id)
    cfg["default_provider"] = get_settings().CHAT_PROVIDER_DEFAULT
    cfg["default_model"] = get_settings().LLM_MODEL   # 本地默认 agent（回落时用）
    return cfg


@app.put("/llm/config")
async def api_llm_config_put(body: LlmConfigIn,
                             user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """保存云端模型配置（api_key 加密落库）。留空 api_key = 保留原 key。"""
    try:
        return await llmstore.save_config(
            user.id, base_url=body.base_url, model=body.model,
            api_key=body.api_key, provider=body.provider, enabled=body.enabled,
            emotion_enabled=body.emotion_enabled,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/llm/config/test")
async def api_llm_config_test(body: LlmConfigIn,
                              user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """保存前连通性测试：拉一次 /models 验证 base_url + key 是否可用。

    api_key 留空时用已存的 key 测（前端"只改地址不改 key"的场景）。
    """
    key = (body.api_key or "").strip()
    if not key:
        cfg = await llmstore.get_runtime(user.id)
        key = (cfg or {}).get("api_key", "")
    if not key:
        raise HTTPException(status_code=400, detail="请填写 API Key 后再测试")
    try:
        models = await llmstore.list_models(body.base_url, key)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "models": []}
    return {"ok": True, "error": "", "models": models[:100]}


@app.get("/llm/models")
async def api_llm_models(base_url: str,
                         user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """拉取指定 base_url 的可选模型列表（**模型自选**）。

    key 取自该用户已保存的配置——不接受前端传 key，避免明文 key 出现在 URL/query
    里被日志、代理、浏览器历史记录下来。
    """
    cfg = await llmstore.get_runtime(user.id)
    if not cfg:
        raise HTTPException(status_code=400,
                            detail="请先保存 API Key，再拉取模型列表")
    try:
        models = await llmstore.list_models(base_url, cfg["api_key"])
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"models": models[:200]}


@app.delete("/llm/config")
async def api_llm_config_delete(user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """删除自己的云端配置（含密文），之后回落本地默认 agent。"""
    ok = await llmstore.delete_config(user.id)
    return {"ok": ok, "deleted": ok}


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
