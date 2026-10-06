"""请求限流：slowapi（底层 limits 库），计数存储可选进程内存或 Redis。

为什么需要限流
--------------
① `/ask` 一轮要跑检索+重排+生成（实测 20s 量级），且与 Ollama 抢同一块 GPU
   （`OLLAMA_NUM_PARALLEL=1`）——并发打进来互相拖慢，谁都拿不到结果；
② `/tts` 每次合成都是真实费用（按 token 计价），必须防刷；
③ `/auth/login` 面对的是**种子管理员 admin/123456** 这种公开弱口令，
   不限速等于把暴力枚举的门敞开；
④ `/llm/models`、`/llm/config/test` 会向用户填的地址发起**出站请求**，
   是 SSRF 面，频率要收紧。

两种计数主体
------------
- **按 IP**：登录/注册时还没有用户身份，只能按来源 IP（`get_remote_address`）。
- **按登录用户**：问答/语音按 `u<user_id>` 计。身份由 `api/auth.get_current_user`
  写进 `request.state.user_id`；取不到时回落 IP，所以依赖顺序写错也只是退化成
  按 IP 限速，不会变成「完全不限流」。

⚠️ **slowapi 的硬要求**：被 `@limiter.limit(...)` 装饰的端点函数**必须有
`request: Request` 形参**，装饰器靠它取 key。漏写会在请求期报错，不是启动期。

两层限额
--------
- `default_limits`：**所有**端点的兜底（含未显式加限额的）。由 `SlowAPIMiddleware` 生效，
  所以 `install()` 里的中间件不能省——省了就只有装饰器那几处生效。
- `@limiter.limit(...)`：单端点覆盖。`limit()` 的 `override_defaults=True` 是默认值，
  即显式限额**替换**兜底而非叠加（别指望「默认 200/min + 显式 20/min」会取严的那个）。

豁免
----
`/health` 必须豁免：`scripts/start.ps1` 靠轮询它判就绪，被限流会让启动脚本误判失败。
`/docs`、`/redoc`、`/openapi.json` 同样豁免（文档面，不是攻击面）。
用 `@limiter.exempt` 装饰器实现——它按 `module.qualname` 注册，`wraps` 保留了名字，
所以放在 `@app.get(...)` 内层（紧贴函数）即可。

存储与容错
----------
默认 `memory://`：单人开发、uvicorn 单进程，够用且零外部依赖。
多实例部署把 `RATE_LIMIT_STORAGE` 设成 `redis://…`（项目已有 Redis）即可跨实例共享计数。

⚠️ 两个容错开关是刻意打开的，别关：
  - `in_memory_fallback_enabled=True`：Redis 不可用时自动回落进程内存，而不是让每个
    请求都失败——限流是**保护性**设施，它自己挂了不该反过来把问答弄挂。
  - `swallow_errors=True`：限额检查本身出错时放行并记日志，同上理由。
代价是 Redis 故障期间跨实例计数会各自为政（限流变松），这是可接受的降级。
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from wuwa_rag.config import get_settings
from wuwa_rag.ww_logger import get_logger

log = get_logger("app")


def key_by_user_or_ip(request: Request) -> str:
    """限流主体：优先登录用户（`u<id>`），取不到回落客户端 IP。

    `request.state.user_id` 由 `api/auth.get_current_user` 写入。
    """
    uid = getattr(request.state, "user_id", None)
    return f"u{uid}" if uid else get_remote_address(request)


def _storage_uri() -> str:
    """计数存储 DSN。留空 = 进程内存。"""
    return (get_settings().RATE_LIMIT_STORAGE or "").strip() or "memory://"


# 全进程共享一个 Limiter：限流的意义就在于跨请求累计计数。
limiter = Limiter(
    key_func=key_by_user_or_ip,
    storage_uri=_storage_uri(),
    default_limits=[get_settings().RATE_LIMIT_DEFAULT],
    enabled=get_settings().RATE_LIMIT_ENABLED,
    # ⚠️ headers_enabled 必须是 False（实测踩过，别改回 True）
    # ------------------------------------------------------------
    # 它为 True 时，slowapi 会在端点**成功返回后**调 `_inject_headers(kwargs["response"], ...)`
    # 往响应里写 X-RateLimit-* 头；而这要求**每个被限流的端点都声明 `response: Response` 形参**。
    # 少了这个形参，业务逻辑明明成功、却会在写头时抛
    # `Exception: parameter response must be an instance of starlette.responses.Response`
    # → 用户看到的是 **500**（登录成功也 500）。
    # 实测对照：headers_enabled=True + 无 response 形参 → 抛异常；有 response 形参 → 200。
    # 之所以选「关掉」而不是「给 7 个端点都加形参」：
    #   ① 我们想要的 `Retry-After` 由下方 `_on_rate_limit_exceeded` 自己给，
    #      关掉后**依然存在**（实测 429 响应仍带 Retry-After: 60）——没有损失；
    #   ② `api_ask_stream` 返回的是 StreamingResponse，加形参反而更容易出错。
    headers_enabled=False,
    # 两个容错开关，理由见模块 docstring「存储与容错」。
    in_memory_fallback_enabled=True,
    swallow_errors=True,
)


# 单端点限额装饰器。⚠️ 每个被装饰的端点都必须有 `request: Request` 形参。
def limit_auth():
    """登录/注册：**按 IP** 限速（此时无用户身份）。"""
    return limiter.limit(get_settings().RATE_LIMIT_AUTH, key_func=get_remote_address)


def limit_ask():
    """问答：按登录用户限速。重计算 + 抢 GPU，是最该限的一档。

    ⚠️ 必须用 `shared_limit` 而不是 `limit`，且 `scope` 固定为 `"ask"`：
    `limit()` 的计数键含路由名，`/ask` 与 `/ask/stream` 会各算一份 —— 那等于给
    同一个能力开了双倍配额，换个端点就能绕过限流。`shared_limit` 让两者共用
    同一个计数器，两条入口合起来才受这一档约束。
    """
    return limiter.shared_limit(get_settings().RATE_LIMIT_ASK, "ask")


def limit_tts():
    """语音合成：按登录用户限速（有真实费用）。"""
    return limiter.limit(get_settings().RATE_LIMIT_TTS)


def limit_outbound():
    """会向外部地址发请求的端点（模型列表探测 / 连通性测试）：SSRF 面，频率收紧。"""
    return limiter.limit(get_settings().RATE_LIMIT_OUTBOUND)


async def _on_rate_limit_exceeded(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """429 响应：与 `app.on_unhandled` 同一风格（中文 error + detail）。

    ⚠️ 日志只记主体与路径，不记请求内容——限流日志是高频噪声，
    不该把用户提问抄进日志文件。
    """
    log.warning("限流拦截 %s %s subject=%s limit=%s",
                request.method, request.url.path, key_by_user_or_ip(request), exc.detail)
    return JSONResponse(
        status_code=429,
        content={"error": "请求过于频繁，请稍后再试", "detail": str(exc.detail)},
        headers={"Retry-After": "60"},
    )


def install(app: FastAPI) -> None:
    """把限流接到 FastAPI 上。必须在 `app` 创建后调用一次。

    三件事缺一不可：
      ① `app.state.limiter` —— 装饰器与中间件都从这里取实例；
      ② 异常处理器 —— 否则超限会裸 500（slowapi 抛 RateLimitExceeded，不是 HTTPException）；
      ③ 中间件 —— `default_limits` 靠它生效，装饰器只管显式加限额的端点。
    """
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _on_rate_limit_exceeded)
    app.add_middleware(SlowAPIMiddleware)
    s = get_settings()
    log.info("限流 enabled=%s 存储=%s default=%s auth=%s ask=%s tts=%s outbound=%s",
             s.RATE_LIMIT_ENABLED,
             "redis" if s.RATE_LIMIT_STORAGE else "memory",
             s.RATE_LIMIT_DEFAULT, s.RATE_LIMIT_AUTH, s.RATE_LIMIT_ASK,
             s.RATE_LIMIT_TTS, s.RATE_LIMIT_OUTBOUND)
