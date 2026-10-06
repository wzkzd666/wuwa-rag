"""Step 9：FastAPI 端点。

系统鉴权与用户画像：
- 鉴权（api/auth.py + db.py）：admin 种子（admin/123456），游客 /auth/register 注册后登录；
  请求带 `Authorization: Bearer <token>`。/ask、/ask/stream 登录即可；/ingest* 管理员专属。
- 画像（rag/profile.py）：user_facts 表落地——问答后异步抽「稳定偏好事实」入库，
  下次提问取活跃事实注入生成 prompt（个性化）；/profile 查自己的、可软删单条。
- 会话历史（conversations.py + /conversations*）：转录落 PG，**按用户隔离**。
  thread_id 是客户端传的，落到 checkpointer 前一律加 `u<user_id>:` 前缀——checkpointer
  只按 thread_id 建键、没有用户维度，不加前缀就等于「谁猜到别人的会话 id
  就能读到别人的多轮记忆」。
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from contextlib import asynccontextmanager

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from wuwa_rag.api import auth as authn
from wuwa_rag.api import ratelimit as rl
from wuwa_rag.api import usage as usage_api
from wuwa_rag.config import get_settings
from wuwa_rag.core import conversations as conv
from wuwa_rag.core import llmstore, usage
from wuwa_rag.core.authdb import close_pool, ensure_schema
from wuwa_rag.core.db import get_cursor
from wuwa_rag.core.llmstore import PROVIDER_PRESETS
from wuwa_rag.dialog.graph import ask, ask_stream, doc_sources
from wuwa_rag.dialog.memory import close_checkpointer, get_checkpointer
from wuwa_rag.knowledge import domain_terms
from wuwa_rag.knowledge import entities as kb
from wuwa_rag.services.emotion import EMOTION_TAGS
from wuwa_rag.services.profile import (
    all_users_stats,
    extract_facts_safe,
    facts_to_context,
    get_facts,
    save_facts,
    soft_delete_fact,
)
from wuwa_rag.services.tts import MODEL_LABEL, VOICE_LABELS, synthesize, voice_label
from wuwa_rag.services.tts import resolve as tts_resolve
from wuwa_rag.tasks.worker import (
    CANCEL_ERROR,
    PIPELINE_STEPS,
    STEP_LABELS,
    build_pipeline,
    build_refresh_pipeline,
    delete_character_knowledge,
    get_control,
    get_progress,
    set_control,
)
from wuwa_rag.ww_logger import get_logger

log = get_logger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await get_checkpointer()
    await ensure_schema()   # 幂等建鉴权表 + 种子 admin/123456（见 pgsql/002_auth.sql）
    await conv.ensure_schema()       # 幂等建会话转录表（见 pgsql/003_conversation_scope.sql）
    await llmstore.ensure_schema()   # 幂等建 user_llm_configs（云端模型配置，含加密 key）
    # 领域词表（单字角色名消歧用）：启动时建好，之后问答主链零额外开销。
    # 失败不阻断启动 —— 判据会回落到 entities 的手写兜底表。
    try:
        await domain_terms.build()
    except Exception as exc:  # noqa: BLE001 —— 派生词表是增益，挂了不该让服务起不来
        log.warning("领域词表预热失败（回落到手写兜底表）: %s", exc)
    yield
    await close_checkpointer()
    await close_pool()


app = FastAPI(title="wuwa-rag", version="0.9", lifespan=lifespan)
# 用量看板 / 答案反馈（见 api/usage.py：普通用户只能看自己的用量）
usage_api.install(app)

# ── 限流 ─────────────────────────────
# ⚠️ 必须在 CORS **之前**安装：Starlette 里后添加的中间件位于**最外层**，
# 而限流产生的 429 需要再经过 CORS 中间件才能带上跨域响应头 —— 顺序反了，
# 前端在浏览器里看到的就是一个没有 CORS 头的模糊跨域错误，而不是真正的「请求过于频繁」。
rl.install(app)

# CORS 配置：支持前后端分端口开发（Vite 5173 / FastAPI 8000）。
# 生产环境用环境变量 CORS_ORIGINS 收窄允许的源（逗号分隔）：
#   CORS_ORIGINS=http://localhost:5173,https://yourdomain.com
#
# ⚠️ allow_credentials 只在**显式配了源**时才开：通配符 `*` + credentials=True 等于
# 允许任意站点带凭据跨域调用本 API（浏览器规范也拒绝这种组合，Starlette 会退化成回显
# 请求 Origin，效果同样是全放行）。本项目鉴权走 `Authorization: Bearer <token>`
# 请求头，不依赖 Cookie，通配符场景下根本不需要 credentials。
_cors_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
_wildcard = "*" in _cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=not _wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    emotion: str = ""         # 情绪标签（TTS 用；语音服务不可用时为空串）


class IngestIn(BaseModel):
    character: str = Field(..., description="角色中文名，如 忌炎")


class IngestControlIn(BaseModel):
    """暂停/继续/取消的请求体。`action` 三选一，语义见 worker.set_control。"""
    character: str = Field(..., description="角色中文名；控制按角色生效，不按 chain_id")
    action: str = Field(..., pattern="^(pause|resume|cancel)$",
                        description="pause 暂停（worker 原地等待）｜resume 继续｜cancel 取消并掐断整条链")


class TitleIn(BaseModel):
    """会话重命名。长度与 conversations.MAX_TITLE_CHARS 同量级，前端已截断。"""
    title: str = Field(..., max_length=60, description="新标题")


class TtsIn(BaseModel):
    text: str = Field(..., description="待合成文本（通常是某条 AI 回答）")
    emotion: str = Field("", description="情绪标签名；空则用默认 cheerful")


class TtsConfigIn(BaseModel):
    """用户自持的语音合成凭据（阿里云百炼 · 北京地域）。

    与云端模型配置同一套加密体系（同一个 DEK）：`passphrase` 是用户自持的加密口令，
    本会话尚未解锁时必填（首次用于建立口令，之后用于解锁）。服务端只在内存里派生密钥，
    不保存、不写日志。

    `api_key` 留空表示「保留原 key 不改」——避免用户只改音色时被迫重贴密钥。
    """

    api_key: str = Field("", description="百炼 API Key（北京地域）；留空表示保留已存的 key")
    workspace_id: str = Field(..., description="百炼业务空间 ID，端点域名的一部分")
    model: str = Field("", description="模型 id；空则用默认 qwen-audio-3.1-tts-flash")
    voice: str = Field("", description="音色 id；3.1 只认 `_v3.1` 后缀音色")
    instruction: str = Field("", description="指令控制文本（音色性格/语速基调），≤100 字符口径")
    passphrase: str = Field("", description="加密口令（兜底解锁通道）")
    password: str = Field("", description="登录密码（自动解锁通道，会绑定到账号）")


class LlmConfigIn(BaseModel):
    """用户自定义云端模型配置。api_key 留空表示「保留原 key 不改」。

    `passphrase` 是用户自持的加密口令：本会话尚未解锁时必填（首次用于建立口令，
    之后用于解锁）。服务端只在内存里用它派生密钥，不保存、不写日志。
    """
    base_url: str = Field(..., description="OpenAI 兼容 base_url，到 /v1 为止")
    model: str = Field(..., description="模型 id，如 gpt-4o-mini / deepseek-chat")
    api_key: str = Field("", description="API Key；留空表示保留已存的 key")
    provider: str = Field("openai", description="预设名，仅用于前端归类，不影响调用")
    enabled: bool = Field(True, description="停用则回落本地默认模型")
    emotion_enabled: bool = Field(
        False, description="是否用该模型兼任情绪判定；false 则走本地 qwen3:8b")
    password: str = Field("", description="登录密码；已解锁时可留空，传了则补建自动解锁通道")
    passphrase: str = Field("", description="加密口令；已解锁时可留空")


class PassphraseIn(BaseModel):
    passphrase: str = Field(..., description="用户自持的加密口令；服务端不保存")


class UnlockIn(BaseModel):
    """解锁云端密钥：两条通道任选（服务端不保存任何一个）。"""
    password: str = Field("", description="登录密码（自动解锁通道，登录时已自动尝试）")
    passphrase: str = Field("", description="加密口令（兜底通道）")


class ChangePassphraseIn(BaseModel):
    old: str = Field(..., description="原加密口令")
    new: str = Field(..., description="新加密口令（至少 8 位）")


class LlmEnabledIn(BaseModel):
    """只切换「用本地默认 / 用云端自定义」，**不触碰任何凭据**。

    存在的意义：这是唯一能把「本地默认」这个选择**落库**的入口。
    原先前端只有「保存」会把 enabled 写进库，而保存按钮长在云端配置表单里——
    切到本地后表单收起，用户就再没有任何途径持久化这个选择；刷新页面时
    GET /llm/config 读回 enabled=true，界面又跳回「云端自定义」（实测表现：
    「切回本地模型后会自动跳回云端配置」）。而当时唯一的持久化办法是「删除配置」，
    那会把 api_key 密文一起抹掉（该行还同时承载语音凭据，连 TTS 也一起没了）。
    """
    enabled: bool = Field(..., description="True=用云端自定义模型；False=回落本地默认")


class ChangePasswordIn(BaseModel):
    old: str = Field(..., description="原登录密码")
    new: str = Field(..., description="新登录密码（至少 6 位）")


class CredentialsIn(BaseModel):
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")


@app.get("/health")
@rl.limiter.exempt
async def health() -> dict:
    return {"ok": True}


# ── 鉴权（无需登录） ─────────────────────────────

@app.post("/auth/register")
@rl.limit_auth()
async def api_register(request: Request, body: CredentialsIn) -> dict:
    """游客注册（注册即登录，直接发 token）。"""
    return await authn.register(body.username, body.password)


@app.post("/auth/login")
@rl.limit_auth()
async def api_login(request: Request, body: CredentialsIn) -> dict:
    return await authn.login(body.username, body.password)


@app.get("/auth/me")
async def api_me(user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    return {"username": user.username, "role": user.role}


@app.post("/auth/logout")
async def api_logout(user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """登出：吊销 token，并丢弃内存里该用户的云端密钥（DEK）。"""
    await authn.revoke_token(user.token)
    llmstore.lock(user.id)
    return {"ok": True}


@app.post("/auth/password")
async def api_change_password(body: ChangePasswordIn,
                              user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """改登录密码。后端会**先用旧密码重绑**云端密钥，再更新密码哈希，
    否则改完密码就再也解不开已存的 API Key（只能删配置重填）。"""
    try:
        await authn.change_password(user.id, body.old, body.new)
    except authn.AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
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
    except Exception as exc:  # noqa: BLE001 —— 画像只增益，查库失败回落空串，绝不挡问答
        log.warning("读取画像失败（忽略）：%s", exc)
        return ""


# 后台画像任务必须**持引用**：asyncio 官方文档明确要求「保存 create_task 的返回值，
# 否则任务可能在运行途中被 GC 回收、静默不执行」。这里全是 fire-and-forget，一旦被回收，
# 表现就是「聊了半天画像也长不出来，而且日志里什么都没有」—— 静默失败最难查，
# 所以宁可多留一个 set。done_callback 里 discard，集合不会无限增长。
_PROFILE_TASKS: set[asyncio.Task] = set()


def _spawn_profile_task(username: str, session_id: str, question: str) -> None:
    """问答后异步抽画像事实（fire-and-forget，不阻塞响应）。

    `session_id` 传的是**规范 key**（`u<id>:<短 id>`，见 conversations.thread_key），
    与转录里的 session_id 对齐，方便按会话回溯画像来源。
    """
    async def job():
        try:
            facts = await extract_facts_safe(question)
            if facts:
                await save_facts(username, session_id, facts)
        except Exception:  # noqa: BLE001 —— fire-and-forget 后台任务，异常只记日志不外抛
            log.exception("画像任务异常（忽略）")

    task = asyncio.create_task(job())
    _PROFILE_TASKS.add(task)
    task.add_done_callback(_PROFILE_TASKS.discard)


# ── 会话隔离 + 转录落库（本轮新增）─────────────────────────────

# thread_id 是**客户端传来的字符串**，会进 PG 主键、进日志、进 checkpointer 的键，
# 必须在入口收口。前端生成的是 8 位 base36（Math.random().toString(36).slice(2,10)），
# 天然满足；限长是为了防「超长随机串把索引撑爆」。
_THREAD_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _client_thread_id(raw: str | None) -> str:
    """校验并归一化客户端的 thread_id；为空则新生成一个。"""
    tid = (raw or "").strip() or uuid.uuid4().hex[:12]
    if not _THREAD_RE.match(tid):
        raise HTTPException(
            status_code=400,
            detail="thread_id 只允许字母/数字/下划线/连字符，最长 64 字符",
        )
    return tid


def _meta_of(evt: dict) -> dict:
    """done 事件 → messages.meta。历史回看时要能复原「引用了几条、来自哪。"""
    return {
        "intent": evt.get("intent", ""),
        "slots": evt.get("slots") or [],
        "characters": evt.get("characters") or [],
        "docs": evt.get("docs", 0),
        "sources": evt.get("sources") or [],
        "truncated": bool(evt.get("truncated")),
        "emotion": evt.get("emotion", ""),
    }


async def _drop_memory(key: str) -> None:
    """删掉 LangGraph 里该 thread 的记忆（lg.checkpoints / checkpoint_writes）。

    删会话必须连记忆一起删：checkpointer 的表**不参与** /conversations 的查询，
    只删转录的话，一旦同一个 thread_id 再被复用（前端本地 id 撞车、或用户手填），
    旧记忆会原地复活。清理失败只告警——不能因为清记忆失败就让用户删不掉会话。
    """
    try:
        saver = await get_checkpointer()
        await saver.adelete_thread(key)
    except Exception as exc:  # noqa: BLE001 —— 清记忆失败不该让用户删不掉会话
        log.warning("清理会话记忆失败（忽略）：%s", exc)


async def _sse(events):
    """把事件字典编成 SSE 帧。"""
    async for evt in events:
        yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"


async def _stream_answer(user: authn.AuthUser, tid: str, question: str,
                         history: list[dict] | None = None):
    """一次问答的 SSE 事件流：落库用户消息 → 流式生成 → 落库助手消息。

    `/ask/stream` 与 `/conversations/{tid}/regenerate` 共用这一条路径 ——
    两条入口各写一遍落库逻辑，迟早会走偏（比如只有一边补了 meta）。
    """
    key = conv.thread_key(user.id, tid)
    # 计量归属（见 api_ask）。SSE 是一次性生成器，scope 必须包住整个迭代过程：
    # 只在「建生成器时」设一次是无效的（contextvar 不会跨任务继承给新任务）。
    with usage.scope(user_id=str(user.id), username=user.username, thread_id=key):
        async for evt in _stream_answer_inner(user, tid, question, key, history):
            yield evt


async def _stream_answer_inner(user: authn.AuthUser, tid: str, question: str, key: str,
                               history: list[dict] | None = None):
    user_ctx = await _user_context(user)
    await conv.ensure_conversation(user.id, tid, question)
    await conv.append_message(user.id, tid, "user", question)

    try:
        async for evt in ask_stream(question, key, user_context=user_ctx,
                                    user_id=user.id, history=history):
            if evt.get("done"):
                # 必须在把 done 交给前端**之前**写完库：前端收到 done 会立刻重拉会话
                # 列表，晚一步就是「刚聊完一刷新答案没了」的竞态。
                answer = evt.get("answer") or ""
                if answer:
                    await conv.append_message(user.id, tid, "assistant", answer, _meta_of(evt))
                # 回给前端的必须是**短 id**（它只认自己的 id，不认 u<id>: 前缀）
                evt["thread_id"] = tid
            yield evt
    except Exception as exc:  # noqa: BLE001 —— 见下：SSE 响应头已发出，此处**必须**宽捕获
        # SSE 一旦开始流式，响应头已发出，全局异常处理器接不住这里的异常——
        # 必须就地捕获并转成 {'error'} 事件下发（前端 store 有对应处理）。
        # ⚠️ 绝不能收窄成具体异常类型：这条流里可能出任何错（LLM 断连、检索异常、
        # 编码错误……），漏掉一种就等于让用户的流式回答中途裸崩、前端收不到收口事件，
        # 界面永远停在「生成中」。宁可宽捕获把一切转成 error 事件。
        rid = uuid.uuid4().hex[:8]
        log.exception("[%s] SSE 流中途异常 thread=%s", rid, key)
        yield {"error": f"生成中断（编号 {rid}）", "detail": str(exc)[:300]}
    finally:
        # 流结束后再抽画像：不与生成抢 LLM（OLLAMA_NUM_PARALLEL=1）
        _spawn_profile_task(user.username, key, question)


@app.post("/ask", response_model=AskOut)
@rl.limit_ask()
async def api_ask(request: Request, body: AskIn,
                  user: authn.AuthUser = Depends(authn.get_current_user)) -> AskOut:
    tid = _client_thread_id(body.thread_id)
    key = conv.thread_key(user.id, tid)
    await conv.ensure_conversation(user.id, tid, body.question)
    await conv.append_message(user.id, tid, "user", body.question)

    # 计量归属：contextvar 一声明，本次问答里所有 LLM 调用（作答 + 工具任务）自动记到这个人名下
    with usage.scope(user_id=str(user.id), username=user.username, thread_id=key):
        r = await ask(body.question, key, user_context=await _user_context(user),
                      user_id=user.id)
    log.info("ask thread=%s 意图=%s 角色=%s", key, r.get("intent"), r.get("characters"))
    answer = r.get("answer") or ""
    if answer:
        await conv.append_message(user.id, tid, "assistant", answer, {
            "intent": r.get("intent", ""),
            "slots": r.get("slots") or [],
            "characters": r.get("characters") or [],
            "docs": len(r.get("docs") or []),
            "sources": doc_sources(r.get("docs") or []),
            "truncated": bool(r.get("truncated")),
            "emotion": r.get("emotion", ""),
        })
    _spawn_profile_task(user.username, key, body.question)
    return AskOut(
        answer=answer,
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
@rl.limit_ask()
async def api_ask_stream(request: Request, body: AskIn,
                         user: authn.AuthUser = Depends(authn.get_current_user)):
    tid = _client_thread_id(body.thread_id)
    return StreamingResponse(
        _sse(_stream_answer(user, tid, body.question)),
        media_type="text/event-stream",
    )


# ── 会话历史（登录即可；一律按登录用户过滤）─────────────────────
# 别人的会话一律按「不存在」处理（404），不回 403 —— 403 等于告诉对方「这个 id 真有」。

def _conv_out(row: dict) -> dict:
    """会话行 → 前端结构（thread_id 回**短 id**，不带 u<id>: 前缀）。"""
    return {
        "thread_id": conv.client_thread_id(row["session_id"]),
        "title": row["title"],
        "message_count": row["message_count"],
        "created_at": row["created_at"].isoformat(),
        "updated_at": row["updated_at"].isoformat(),
        "last_role": row["last_role"],
        "last_content": row["last_content"],
    }


@app.get("/conversations")
async def api_conversations(q: str = "",
                            user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """当前用户的会话列表（新的在前，不含正文，只带最后一条预览）。

    `q` 非空时按标题或任意一条消息正文模糊搜索（转录在服务端，只能在这边搜）。
    """
    rows = await conv.list_conversations(user.id, query=q.strip()[:100])
    return {"conversations": [_conv_out(r) for r in rows]}


@app.get("/conversations/{thread_id}")
async def api_conversation(thread_id: str,
                           user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """单个会话 + 全部消息。"""
    tid = _client_thread_id(thread_id)
    row = await conv.get_conversation(user.id, tid)
    if row is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {
        "thread_id": tid,
        "title": row["title"],
        "created_at": row["created_at"].isoformat(),
        "messages": [
            {
                "id": m["id"],
                "role": m["role"],
                "content": m["content"],
                "meta": m["meta"] or {},
                "created_at": m["created_at"].isoformat(),
            }
            for m in row["messages"]
        ],
    }


@app.patch("/conversations/{thread_id}")
async def api_conversation_rename(body: TitleIn, thread_id: str,
                                  user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    tid = _client_thread_id(thread_id)
    if not await conv.rename_conversation(user.id, tid, body.title):
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"ok": True}


@app.delete("/conversations/{thread_id}")
async def api_conversation_delete(thread_id: str,
                                  user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """删除会话：转录 + 模型侧记忆一起清（见 _drop_memory 的说明）。"""
    tid = _client_thread_id(thread_id)
    if not await conv.delete_conversation(user.id, tid):
        raise HTTPException(status_code=404, detail="会话不存在")
    await _drop_memory(conv.thread_key(user.id, tid))
    return {"ok": True}


@app.delete("/conversations")
async def api_conversations_clear(user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """清空当前用户全部会话（连带各自的 LangGraph 记忆）。"""
    keys = [conv.thread_key(user.id, conv.client_thread_id(r["session_id"]))
            for r in await conv.list_conversations(user.id, limit=1000)]
    n = await conv.clear_conversations(user.id)
    for k in keys:
        await _drop_memory(k)
    return {"ok": True, "deleted": n}


@app.post("/conversations/{thread_id}/regenerate")
async def api_conversation_regenerate(thread_id: str,
                                      user: authn.AuthUser = Depends(authn.get_current_user)):
    """重新生成最后一条回答。

    必须做三件事，少一件都会出现「重新生成但还是同一段话」：
      ① 把「最后一句问 + 它的答」从转录里删掉（truncate_from）；
      ② 把模型侧记忆**替换**成删完之后的窗口（history 回放）——
         checkpointer 里的 history 不会因为我们删了转录就自动变；
      ③ 清掉该 thread 的 checkpoint（否则 context_summary 等旧字段会残留）。
    """
    tid = _client_thread_id(thread_id)
    last = await conv.last_user_message(user.id, tid)
    if last is None:
        raise HTTPException(status_code=404, detail="这个会话还没有可以重新生成的问题")

    key = conv.thread_key(user.id, tid)
    await conv.truncate_from(user.id, tid, last["id"])
    history = await conv.get_history(user.id, tid,
                                     limit=get_settings().MAX_HISTORY_TURNS * 2)
    await _drop_memory(key)
    return StreamingResponse(
        _sse(_stream_answer(user, tid, last["content"], history=history)),
        media_type="text/event-stream",
    )


# ── 语音合成（TTS，登录即可；密钥由用户自持）─────────────────
# 与云端模型同构：密钥加密落库、同一把 DEK、同一套解锁通道。
# 优先级「用户自持凭据 > 全局 .env 兜底」——开源分发下部署者不该替用户垫额度。

@app.get("/tts/status")
async def api_tts_status(user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """语音能力状态：是否可用、不可用原因、当前生效的模型与音色（不暴露密钥）。

    返回的 `model` / `voice` 是**当前实际生效值**：用户配了就回显用户值，没配则回显
    代码默认值，而不是空串。同时附中文展示名（`*_label`），避免前端直接渲染
    `longanlingxi_v3.1` 这类内部 id。

    `source` 标明生效来源：`user`（用户自持）/ `global`（部署者全局兜底）/ `none`。
    `defaults` 单独给出代码默认值，供前端把输入框占位提示写成「留空则用默认：xxx」
    （不给具体值，避免把「用户没填」和「用户填了默认值」在表单里混为一谈）。
    """
    s = get_settings()
    rt, why = await tts_resolve(user.id)
    # 生效值优先取 resolve 结果；不可用时回落到「用户存的配置」再回落代码默认，
    # 这样设置页在「配了但没解锁」时仍能正确预填用户之前填过的内容。
    cfg = await llmstore.get_tts_masked(user.id)
    model = (rt or {}).get("model") or cfg.get("model") or s.TTS_MODEL
    voice = (rt or {}).get("voice") or cfg.get("voice") or s.TTS_VOICE
    return {
        "enabled": s.TTS_ENABLED,
        "ready": rt is not None,
        "reason": why,
        "source": (rt or {}).get("source", "none"),
        "model": model,
        "model_label": MODEL_LABEL,
        "voice": voice,
        "voice_label": voice_label(voice),
        "emotions": list(EMOTION_TAGS),
        "configured": cfg.get("configured", False),   # 用户是否已存过自己的凭据
        "unlocked": cfg.get("unlocked", False),
        "voices": [{"id": k, "label": v} for k, v in VOICE_LABELS.items()],
        "defaults": {
            "model": s.TTS_MODEL,
            "voice": s.TTS_VOICE,
            "voice_label": voice_label(s.TTS_VOICE),
            "instruction": s.TTS_INSTRUCTION,
        },
    }


@app.post("/tts")
@rl.limit_tts()
async def api_tts(request: Request, body: TtsIn,
                  user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
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
    r = await synthesize(text, body.emotion or None, user_id=user.id)
    if not r.ok:
        log.info("TTS 未合成 user=%s 原因=%s", user.username, r.error)
        return {"ok": False, "error": r.error, "url": "", "emotion": "", "elapsed_ms": 0}
    return {"ok": True, "url": r.url, "error": "", "emotion": r.emotion,
            "model": r.model, "voice": r.voice, "elapsed_ms": r.elapsed_ms}


@app.get("/tts/config")
async def api_tts_config_get(user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """读自己的语音凭据。**只返回掩码**，任何情况都不回显明文 key。"""
    return await llmstore.get_tts_masked(user.id)


@app.put("/tts/config")
async def api_tts_config_put(body: TtsConfigIn,
                             user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """保存语音凭据（api_key 加密落库）。留空 api_key = 保留原 key。

    未解锁时 `password` 与 `passphrase` 至少给一个：首次用来建立 DEK 与解锁通道，
    之后用来解锁。传 `password` 会顺带绑定「登录自动解锁」，下次登录直接生效。
    """
    try:
        return await llmstore.save_tts_config(
            user.id, api_key=body.api_key, workspace_id=body.workspace_id,
            model=body.model, voice=body.voice, instruction=body.instruction,
            passphrase=body.passphrase, password=body.password,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/tts/config")
async def api_tts_config_delete(user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """删除自己的语音凭据（含密文），之后回落全局兜底配置或不可用。"""
    ok = await llmstore.delete_tts_config(user.id)
    return {"ok": ok, "deleted": ok}


# ── 用户自定义云端模型（登录即可，配置只属于自己）─────────────────
# 密钥体系：加密密钥由用户自持的「加密口令」派生，只活在服务端进程内存里
# （详见 rag/llmstore.py）。进程重启 = 上锁，需重新 unlock；未解锁期间云模型
# 自动回落本地默认 agent。

@app.get("/llm/providers")
async def api_llm_providers(user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """provider 预设列表（含默认 base_url），前端做下拉与自动填充。"""
    return {"providers": [
        {"key": k, "label": v["label"], "base_url": v["base_url"]}
        for k, v in PROVIDER_PRESETS.items()
    ]}


@app.get("/llm/config")
async def api_llm_config_get(user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """读自己的云端模型配置。**只返回掩码**，任何情况都不回显明文 key。

    `unlocked=false` 表示本会话尚未输入加密口令，云端模型不会生效。
    """
    cfg = await llmstore.get_config_masked(user.id)
    cfg["default_provider"] = get_settings().CHAT_PROVIDER_DEFAULT
    cfg["default_model"] = get_settings().LLM_MODEL   # 本地默认 agent（回落时用）
    return cfg


@app.post("/llm/unlock")
async def api_llm_unlock(body: UnlockIn,
                         user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """解锁自己的云端密钥：登录密码（自动通道）或加密口令（兜底）任选其一。

    正常登录时后端已经自动解锁过了；只有进程重启后仍持有旧 token、
    或当初只用加密口令建立密钥时才需要手动调它。
    """
    ok = await llmstore.unlock(user.id, password=body.password, passphrase=body.passphrase)
    if not ok:
        raise HTTPException(status_code=400,
                            detail="登录密码或加密口令不正确（或尚未配置云端模型）")
    return {"ok": True, "unlocked": True}


@app.post("/llm/lock")
async def api_llm_lock(user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """丢弃内存里的派生密钥（之后云端模型回落本地，需重新输入口令）。"""
    llmstore.lock(user.id)
    return {"ok": True, "unlocked": False}


@app.post("/llm/passphrase")
async def api_llm_passphrase(body: ChangePassphraseIn,
                             user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """更换加密口令：必须提供**原口令**，服务端用它解出 key 再用新口令重新加密。"""
    try:
        await llmstore.change_passphrase(user.id, body.old, body.new)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "unlocked": True}


@app.post("/llm/enabled")
async def api_llm_enabled(body: LlmEnabledIn,
                          user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """切换作答模型来源（本地默认 ↔ 云端自定义），**凭据原样保留**。

    与 `PUT /llm/config` 职责分开，别合并：
      · 本端点 = 只翻一个开关。不需要口令、不要求已解锁、不改 key；
      · `PUT /llm/config` = 改地址/模型/key，需要口令来建立或解开 DEK。

    这样「切回本地」不必再走「删除配置」，云端配置与语音凭据都留着随时切回来。

    关掉（enabled=False）对没有配置行的用户是**幂等成功**——本来就在本地，
    没必要为一次空操作报 400 把前端卡住。打开时才要求配置真的存在。
    """
    ok = await llmstore.set_enabled(user.id, body.enabled)
    if body.enabled and not ok:
        raise HTTPException(status_code=400,
                            detail="还没有保存过云端配置，请先填写 API 信息并保存")
    cfg = await llmstore.get_config_masked(user.id)
    return {"ok": True, "enabled": bool(cfg["enabled"]),
            "configured": bool(cfg["configured"]),
            # 已启用但没解锁时云端并不生效（get_runtime 回落本地），前端据此给提示
            "unlocked": bool(cfg["unlocked"])}


@app.put("/llm/config")
async def api_llm_config_put(body: LlmConfigIn,
                             user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """保存云端模型配置（api_key 加密落库）。留空 api_key = 保留原 key。

    未解锁时 `password` 与 `passphrase` 至少给一个：首次用来建立 DEK 与解锁通道，
    之后用来解锁。传 `password` 会顺带绑定「登录自动解锁」，下次登录直接生效。
    """
    try:
        return await llmstore.save_config(
            user.id, base_url=body.base_url, model=body.model,
            api_key=body.api_key, provider=body.provider, enabled=body.enabled,
            emotion_enabled=body.emotion_enabled, passphrase=body.passphrase,
            password=body.password,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/llm/config/test")
@rl.limit_outbound()
async def api_llm_config_test(request: Request, body: LlmConfigIn,
                              user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """保存前连通性测试：拉一次 /models 验证 base_url + key 是否可用。

    api_key 留空时用已存的 key 测（前端"只改地址不改 key"的场景）；
    此时若尚未解锁，会先用 `password` / `passphrase` 解锁。
    """
    key = (body.api_key or "").strip()
    if not key:
        if not llmstore.is_unlocked(user.id):
            await llmstore.unlock(user.id, password=body.password,
                                  passphrase=body.passphrase)
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
@rl.limit_outbound()
async def api_llm_models(request: Request, base_url: str,
                         user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """拉取指定 base_url 的可选模型列表（**模型自选**）。

    key 取自该用户已保存的配置——不接受前端传 key，避免明文 key 出现在 URL/query
    里被日志、代理、浏览器历史记录下来。
    """
    if not llmstore.is_unlocked(user.id):
        raise HTTPException(status_code=400, detail="请先输入加密口令解锁，再拉取模型列表")
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
                     user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """把一个角色塞进异步流水线，立即返回 chain_id。

    先建 `crawl_runs` 行再入队：**提交人只有 API 知道**（worker 只知道自己跑了什么），
    行 id 作为 run_id 交给 crawl 步复用同一行，所以这份账本刷新页面/换设备都还在。

    权限：**任何登录用户**都可提交（用户四项权限之一）。
    """
    run_id = await asyncio.to_thread(_create_crawl_run, body.character, user)
    r = build_pipeline(body.character, run_id).apply_async()
    await asyncio.to_thread(_set_chain_id, run_id, r.id)
    return {"character": body.character, "chain_id": r.id, "state": r.state, "run_id": run_id}


def _create_crawl_run(character: str, user: authn.AuthUser) -> int:
    """提交时建一行 crawl_runs（同步驱动，跑在线程里）。返回行 id。

    `submitted_by` 存 user.id（授权判定用它），`submitted_by_name` 存用户名（只给人看）。
    """
    with psycopg.connect(get_settings().PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO crawl_runs (status, stats, character, submitted_by,"
                " submitted_by_name) VALUES ('running', %s, %s, %s, %s) RETURNING id",
                (json.dumps({"character": character}, ensure_ascii=False), character,
                 str(user.id), user.username),
            )
            return cur.fetchone()[0]


def _may_control(user: authn.AuthUser, submitted_by: str | None) -> bool:
    """能不能暂停/继续/取消/删除这条提交：admin 全可以；普通用户只限**自己提交的**。"""
    return user.is_admin or (submitted_by is not None and submitted_by == str(user.id))


def _require_may_control(user: authn.AuthUser, submitted_by: str | None) -> None:
    if not _may_control(user, submitted_by):
        raise HTTPException(status_code=403, detail="只能操作自己提交的入库任务")


def _latest_submitter(character: str) -> str | None:
    """该角色最近一次提交的 `submitted_by`（控制是按角色的，授权要跟着它走）。"""
    with psycopg.connect(get_settings().PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT submitted_by FROM crawl_runs WHERE character = %s"
                " ORDER BY started_at DESC LIMIT 1",
                (character,),
            )
            row = cur.fetchone()
    return row[0] if row else None


def _set_chain_id(run_id: int, chain_id: str) -> None:
    with psycopg.connect(get_settings().PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE crawl_runs SET chain_id = %s WHERE id = %s", (chain_id, run_id))


@app.get("/ingest/records")
async def api_ingest_records(limit: int = 20,
                             user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """最近的抓取/入库提交记录（服务端真值，含提交人），按提交时间倒序。

    `status` 是**实时**流水线状态：从 Redis 进度键按角色读出来叠加。
    记录本身只存「谁在什么时候提了什么」，活的状态那份在 Redis。
    """
    limit = max(1, min(limit, 100))

    def _load() -> list[dict]:
        with psycopg.connect(get_settings().PG_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, character, submitted_by, submitted_by_name, chain_id,"
                    " status, error, started_at, finished_at FROM crawl_runs"
                    " WHERE character IS NOT NULL"
                    " ORDER BY started_at DESC LIMIT %s",
                    (limit,),
                )
                cols = [d.name for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]

    rows = await asyncio.to_thread(_load)
    snaps = await asyncio.gather(
        *(asyncio.to_thread(get_progress, r["character"]) for r in rows)
    )
    items = []
    for r, snap in zip(rows, snaps):
        live = _live_status(snap)
        items.append({
            "id": r["id"],
            "character": r["character"],
            "submitted_by": r["submitted_by"],
            "submitted_by_name": r["submitted_by_name"] or "自动",
            "chain_id": r["chain_id"],
            "state": r["status"],
            "error": r["error"],
            "created_at": r["started_at"].isoformat() if r["started_at"] else None,
            "finished_at": r["finished_at"].isoformat() if r["finished_at"] else None,
            "status": live,
            # 前端不自己判身份（它拿不到 user.id），一律照服务端给的 can_control 显示按钮
            "can_control": _may_control(user, r["submitted_by"]),
        })
    return {"items": items, "total": len(items)}


@app.delete("/ingest/records/{record_id}")
async def api_ingest_record_delete(record_id: int,
                                   user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """删掉一条提交记录（只删账本，不碰后台正在跑的流水线；要停请先 POST /ingest/control）。

    权限：admin 随便删；普通用户只能删**自己提交的**那几条。
    """
    with psycopg.connect(get_settings().PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT character, submitted_by FROM crawl_runs WHERE id = %s", (record_id,))
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="记录不存在")
            _require_may_control(user, row[1])
            cur.execute("DELETE FROM crawl_runs WHERE id = %s", (record_id,))
    return {"id": record_id, "character": row[0]}


def _live_status(snap: dict | None) -> str:
    """五步进度快照 -> 整体状态。`/ingest/status` 与 `/ingest/records` 共用这一份判定，
    免得两处口径漂移（一个说完成一个说失败最难查）。无快照 = `unknown`。"""
    if not snap:
        return "unknown"
    live = "success"
    for k in PIPELINE_STEPS:
        st = snap["steps"].get(k, "pending")
        if st == "failed":
            return "failed"
        if st in ("pending", "running"):
            live = "running"
            break
    return live


@app.get("/ingest/status")
async def api_ingest_status(character: str,
                            user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """按角色查入库进度（worker 侧 _progress_mark 写 Redis 聚合键）。

    权限：**任何登录用户**可读（前端要靠它轮询五步进度；提交本身已对所有登录用户开放，
    读不到自己的进度就没法用）。它只是进度快照，不含任何管理动作。

    不依赖 chain_id：/ingest 返回的 chain_id 刷新页面就丢了，而进度键按角色
    天然可查。steps 恒为五步数组（含中文标签），整体 status：
    pending(还没开跑/无记录) | running | success | failed。

    另外带两个控制旗标（`paused` / `cancelled`）：它们**不改变** steps 里的状态 ——
    暂停时那一步保持 pending（没在跑），取消时那一步是 failed + `已取消`。
    单看 status 区分不出「失败」与「被取消」，所以显式给前端两个布尔。
    """
    ctl = await asyncio.to_thread(get_control, character)
    snap = await asyncio.to_thread(get_progress, character)
    if snap is None:
        return {"character": character, "status": "pending", "found": False,
                "paused": ctl == "pause", "cancelled": ctl == "cancel",
                "steps": [{"key": k, "label": STEP_LABELS[k], "status": "pending",
                           "error": None} for k in PIPELINE_STEPS],
                "updated_at": None}
    steps = []
    overall = _live_status(snap)
    for k in PIPELINE_STEPS:
        steps.append({"key": k, "label": STEP_LABELS[k],
                      "status": snap["steps"].get(k, "pending"),
                      "error": snap.get("errors", {}).get(k)})
    # 被取消时把整体状态也标成 cancelled：失败原因落在 error 里，前端据此显示「已取消」
    cancelled = ctl == "cancel" or any(s["error"] == CANCEL_ERROR for s in steps)
    return {"character": character,
            "status": "cancelled" if cancelled else overall,
            "found": True,
            "paused": ctl == "pause",
            "cancelled": cancelled,
            "steps": steps, "updated_at": snap.get("updated_at")}


@app.post("/ingest/control")
async def api_ingest_control(body: IngestControlIn,
                            user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """暂停 / 继续 / 取消一条正在跑的入库链（按角色，不是按 chain_id）。

    实现是**每步开头自查 Redis 旗标**（见 worker.wait_if_paused），不用
    `celery revoke`：五步是 chain 串联，revoke 只能停掉单个 task_id，停链尾那一步时
    前面几步照跑，表现为「点了取消还在跑」。
    """
    submitter = await asyncio.to_thread(_latest_submitter, body.character)
    _require_may_control(user, submitter)
    await asyncio.to_thread(set_control, body.character, body.action)
    ctl = await asyncio.to_thread(get_control, body.character)
    return {"character": body.character, "action": body.action,
            "paused": ctl == "pause", "cancelled": ctl == "cancel"}


# ── 知识库视图（列出「已拥有什么」；重爬/删除仅管理员） ─────────────

@app.get("/knowledge/characters")
async def api_knowledge_characters(
        user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """列出知识库里**实际拥有**的角色 —— 回答「我现在有什么」。

    与 `/ingest` 那条「往流水线里塞一个角色」是两件事。列表对**所有登录用户**
    开放（知识库覆盖度是产品信息，游客也该看得到），写操作才是 admin。

    字段含义：
    - `source`  入库时写的来源标识（当前恒为 `kurobbs`，即鸣潮 WIKI）；
    - `raw_uri` RustFS 里的原文对象指针，用来核对「这份知识是从哪份原文来的」；
    - `chunks`  实际块数 —— 为 0 说明入了库但没跑索引，是个有用的健康信号；
    - `seeded`  是否属于内置种子名册，**False = 靠自动爬取发现并入库的新角色**，
                这也是「名册随增随变」在界面上的可见证据。
    """
    async with get_cursor() as cur:
        await cur.execute(
            """
            SELECT d.character, d.source, d.title, d.raw_uri, d.raw_size,
                   d.created_at, d.updated_at, count(c.id) AS chunks
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            WHERE d.deleted_at IS NULL
            GROUP BY d.id
            ORDER BY d.updated_at DESC, d.character
            """
        )
        rows = await cur.fetchall()
    items = [
        {
            "character": r[0],
            "source": r[1],
            "title": r[2],
            "raw_uri": r[3],
            "raw_size": r[4],
            "created_at": r[5].isoformat() if r[5] else None,
            "updated_at": r[6].isoformat() if r[6] else None,
            "chunks": int(r[7] or 0),
            "seeded": r[0] in kb.SEED_CHARACTER_NAMES,
        }
        for r in rows
    ]
    in_db = {r[0] for r in rows}
    return {
        "items": items,
        "total": len(items),
        # 种子名册里还没入库的角色 —— 前端拿去渲染「可一键收录」的候选区。
        # 由后端下发而不是前端硬编码：候选名册只允许有一个来源，
        # 否则加新角色时前后端两份清单必然不同步。
        "seeded_only": sorted(kb.SEED_CHARACTER_NAMES - in_db),
    }


@app.post("/knowledge/refresh")
async def api_knowledge_refresh(body: IngestIn,
                                user: authn.AuthUser = Depends(authn.require_admin)) -> dict:
    """重爬更新：**先清该角色的旧知识**，再重跑五步链。

    为什么必须先清而不是直接重跑：wiki 改版后块内容全变、hash 也全变，
    `chunks` 表的 `ON CONFLICT (chunk_id) DO NOTHING` 只能挡同 hash，
    挡不住新旧并存 —— 不清就会召回到旧知识（这正是 verify 判不匹配时的同一套刷新链）。
    """
    r = build_refresh_pipeline(body.character).apply_async()
    return {"character": body.character, "chain_id": r.id, "state": r.state}


@app.delete("/knowledge/characters/{character}")
async def api_knowledge_delete(
        character: str,
        user: authn.AuthUser = Depends(authn.require_admin)) -> dict:
    """删除该角色的知识库：PG（级联 chunks）+ Chroma + Neo4j + BM25 全清。

    异步执行 —— Chroma 与 Neo4j 的清理不是毫秒级的事，同步做会把接口挂住。
    前端拿到 task_id 后轮询 `GET /knowledge/characters`，看它从列表里消失即可。

    ⚠️ 删除后**不会**被自动重爬：Neo4j 的 `Character` 节点刻意保留（理由见
    `build_graph._C_DELETE_CHAR`），于是它仍算「已知角色」，提问只会答「不知道」。
    要加回需走 `POST /ingest`。这是刻意维持的语义，别改。
    """
    r = delete_character_knowledge.apply_async(args=[character])
    kb.invalidate_roster()      # 名册立刻不再包含它，不必等 60s TTL
    domain_terms.invalidate()   # 领域词表同理：角色少了一个，锚点集变了、重建即可
    return {"character": character, "task_id": r.id}
