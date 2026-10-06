"""限流与 CORS。

⚠️ 为什么用「合成 app」而不打真实端点
------------------------------------
真实端点（/auth/login 等）会连 PG，而 TestClient 的 anyio portal 与 psycopg 连接池
不兼容（实测 `PoolTimeout: pool initialization incomplete after 10 sec`，
即使容器 healthy、用 asyncio.run 直连是通的）。
合成 app 复用**同一套真实接线代码**（`rl.install` / `rl.limit_auth` / `rl.limit_ask` /
`rl.limiter.exempt`），只把端点函数换成空实现——测的仍是限流语义本身，且完全离线。

⚠️ 合成 app 必须**全模块只建一次**（module 级 fixture）
----------------------------------------------------
`rl.limiter` 是模块级单例，`@rl.limit_auth()` 会把端点名注册进 `limiter._route_limits`。
每个用例都新建一个合成 app，就会**反复注册同一个路由名**，导致：
  · `_route_limits` 被合成端点污染（`test_真实_app_的限额接线完整` 会看到多余项）；
  · 同一路由挂上多份限额，429 比预期来得更早。
所以这里用 module 级 fixture 建一次、所有用例共用，只靠 `limiter.reset()` 清计数。

⚠️ 限额主体是 IP，不是用户
------------------------
TestClient 的请求都来自同一客户端 IP，按 IP 计数即可稳定复现，不需要伪造登录态。
「优先用户、回落 IP」由 `test_限流主体_优先用户回落IP` 单独覆盖。

本文件钉住四件曾出过问题的事：
  ① `headers_enabled` 必须为 False（True 会让**成功请求** 500）；
  ② 显式限额真的会触发 429（不是配了但不生效的摆设）；
  ③ `/ask` 与 `/ask/stream` **共享**计数器（否则换个端点就能绕过限流）；
  ④ 429 带 Retry-After 与 CORS 头（中间件顺序反了前端只看到模糊跨域错误）。
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from wuwa_rag.api import ratelimit as rl

ORIGIN = "http://localhost:5173"
AUTH_LIMIT = int(rl.get_settings().RATE_LIMIT_AUTH.split("/")[0])   # 10
ASK_LIMIT = int(rl.get_settings().RATE_LIMIT_ASK.split("/")[0])     # 20


def _build_app() -> FastAPI:
    """合成 app：真实接线（install / 限额装饰器 / exempt）+ 空端点实现。

    ⚠️ 中间件顺序是硬的：CORS 必须在 `rl.install`（它 add 了 SlowAPIMiddleware）
    **之后** add。Starlette 后 add 的在**最外层**，这样 429 响应才会被 CORS 加上头；
    反过来则 429 裸奔，浏览器只报跨域错误，排查方向会被彻底带偏。
    """
    app = FastAPI()
    rl.install(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,   # 通配符下带凭据是无效且不安全的组合
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 按 IP 限速（与 /auth/login、/auth/register 同档）
    @app.post("/auth/login")
    @rl.limit_auth()
    async def ep_auth(request: Request):
        return {"ok": "auth"}

    # 带 response 形参的对照组：headers_enabled=True 时只有它不炸
    @app.post("/needs_response")
    @rl.limit_auth()
    async def ep_needs_response(request: Request, response: Response):
        return {"ok": "needs_response"}

    # 两个端点共用 limit_ask() —— 对应真实的 /ask 与 /ask/stream
    @app.post("/ask")
    @rl.limit_ask()
    async def ep_ask(request: Request):
        return {"ok": "ask"}

    @app.post("/ask/stream")
    @rl.limit_ask()
    async def ep_stream(request: Request):
        return {"ok": "stream"}

    # 豁免端点（对应 /health）
    @app.get("/free")
    @rl.limiter.exempt
    async def ep_free():
        return {"ok": "free"}

    return app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(_build_app())


@pytest.fixture(autouse=True)
def _reset_counts():
    """每个用例前后清空**计数**（不清注册表——注册表是 module 级建一次的结果）。

    limiter 是模块级单例 + MemoryStorage，不重置的话前一个用例打满的配额
    会让后一个用例一上来就 429 —— 那会变成「测试顺序决定结果」的假失败。
    """
    rl.limiter.reset()
    yield
    rl.limiter.reset()


# ---------- ① headers_enabled 回归桩 ----------

def test_headers_enabled_必须为_False() -> None:
    """🔴 生产级回归桩：为 True 时，**成功通过限流的请求也会 500**。

    slowapi 在端点成功返回后调 `_inject_headers(kwargs["response"], ...)` 写
    X-RateLimit-* 头，这要求每个被限流端点都声明 `response: Response` 形参。
    真实端点（api_login / api_ask / api_tts …）都没有这个形参 →
    业务明明成功，却在写头时抛异常，用户看到 500（**登录成功也 500**）。

    实测对照：True + 无形参 → 抛异常；True + 有形参 → 200；False + 无形参 → 200。
    我们想要的 Retry-After 由 `_on_rate_limit_exceeded` 自己给，关掉它没有损失
    （见 test_429_带_RetryAfter_与_CORS_头）。
    """
    assert rl.limiter._headers_enabled is False


def test_成功请求不因限流写头而_500(client: TestClient) -> None:
    """端点不带 response 形参（与真实端点一致）时，成功请求必须是 200。"""
    assert client.post("/auth/login").status_code == 200


def test_带_response_形参的端点同样正常(client: TestClient) -> None:
    """关掉 headers_enabled 后，带不带 response 形参都应正常——不必为限流改端点签名。"""
    assert client.post("/needs_response").status_code == 200


# ---------- ② 限额真的生效 ----------

def test_超过限额返回_429(client: TestClient) -> None:
    """auth 档是 10/minute：前 10 次放行，第 11 次起 429。

    这条守的是「配了但不生效」——限流最大的风险不是拦得太狠，而是静默不拦。
    """
    codes = [client.post("/auth/login").status_code for _ in range(AUTH_LIMIT + 2)]
    assert codes[:AUTH_LIMIT] == [200] * AUTH_LIMIT, f"前 {AUTH_LIMIT} 次应全放行，实际 {codes}"
    assert codes[AUTH_LIMIT:] == [429, 429], f"超限应 429，实际 {codes}"


def test_exempt_端点不受限(client: TestClient) -> None:
    """/health 必须豁免：`scripts/start.ps1` 靠轮询它判就绪，被限流会误判启动失败。"""
    codes = {client.get("/free").status_code for _ in range(30)}
    assert codes == {200}, f"豁免端点出现非 200：{codes}"


def test_打满一个端点不影响另一个(client: TestClient) -> None:
    """/auth/login 打满后，豁免端点与另一档端点仍应正常。

    与 test_ask_两个端点共享计数器 正好相反——两者的区别就是
    `limit()`（计数键含路由名）与 `shared_limit()`（含 scope）的区别，
    这两条用例一起钉住这个语义。
    """
    for _ in range(AUTH_LIMIT + 2):
        client.post("/auth/login")
    assert client.post("/auth/login").status_code == 429
    assert client.get("/free").status_code == 200


# ---------- ③ 共享计数器防绕过 ----------

def test_ask_两个端点共享计数器(client: TestClient) -> None:
    """🔴 `/ask` 与 `/ask/stream` 必须共用一个计数器，否则换个端点就能绕过限流。

    若误用 `limit()`（计数键含路由名），两者会**各算一份** → 同一个能力开了双倍配额，
    攻击者交替打两个端点即可把 GPU 打满。`shared_limit(scope="ask")` 才是对的。

    断言方式：交替打两个端点，合计放行次数必须正好等于单档限额（20），而不是 2×20。
    这个断言对「共享 / 不共享」的区分是决定性的。
    """
    codes = []
    for i in range(ASK_LIMIT * 2 + 5):
        url = "/ask" if i % 2 == 0 else "/ask/stream"
        codes.append(client.post(url).status_code)
    assert codes.count(200) == ASK_LIMIT, (
        f"两个端点应共享 {ASK_LIMIT} 配额，实际放行 {codes.count(200)} 次"
        f"（若约等于 {ASK_LIMIT * 2} 说明各算一份，限流可被换端点绕过）"
    )
    assert 429 in codes


# ---------- ④ 429 的响应头 ----------

def test_429_带_RetryAfter_与_CORS_头(client: TestClient) -> None:
    """429 必须同时带 Retry-After（调用方据此退避）与 CORS 头（前端能读到错误）。

    Retry-After 来自 `_on_rate_limit_exceeded` 自己，与 headers_enabled 无关——
    这正是敢把 headers_enabled 关掉的前提。
    CORS 头依赖中间件顺序（CORS 在外层），顺序写反这里会红。
    """
    for _ in range(AUTH_LIMIT + 2):
        client.post("/auth/login")
    resp = client.post("/auth/login", headers={"Origin": ORIGIN})
    assert resp.status_code == 429
    assert resp.headers.get("retry-after") == "60"
    assert resp.headers.get("access-control-allow-origin") == "*"
    body = resp.json()
    assert body["error"] == "请求过于频繁，请稍后再试"
    assert f"{AUTH_LIMIT} per 1 minute" in body["detail"]


def test_正常请求也带_CORS_头(client: TestClient) -> None:
    resp = client.post("/auth/login", headers={"Origin": ORIGIN})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "*"


def test_通配符来源下不带凭据(client: TestClient) -> None:
    """allow_origins=['*'] 与 allow_credentials=True 是无效且不安全的组合。

    本项目用 Bearer token（不依赖 Cookie），通配符下不需要 credentials。
    """
    for mw in client.app.user_middleware:
        if mw.cls is CORSMiddleware:
            assert mw.kwargs.get("allow_credentials") is not True


# ---------- ⑤ 接线完整性（读真实 app 的注册表，不发请求 → 不触 PG）----------

def test_真实_app_的限额接线完整() -> None:
    """导入真实 app，断言 7 个端点都挂上了限额、且 /health 在豁免名单里。

    只读 `limiter` 的注册表，**不发任何请求**，所以不触 PG / 不联网。
    ⚠️ 必须按模块名过滤：本文件的合成端点也注册进了同一个单例，
    不过滤会把 ep_auth / ep_ask 之类当成真实端点（实测踩过）。

    漏挂一个 = 那个端点完全不限流，而这种「静默失效」靠人工看代码很难发现。
    """
    from wuwa_rag.api.app import app  # noqa: PLC0415 —— 延迟导入，避免收集期就加载整个 app

    assert app is not None
    real = {
        name.rsplit(".", 1)[-1]
        for name in rl.limiter._route_limits
        if name.startswith("wuwa_rag.api.app.")
    }
    assert real == {
        "api_login", "api_register",              # 按 IP：挡暴力枚举 / 批量注册
        "api_ask", "api_ask_stream",              # 按用户：重计算 + 抢 GPU
        "api_tts",                                # 按用户：有真实费用
        "api_llm_models", "api_llm_config_test",  # 出站请求：SSRF 面收窄
    }
    exempt = {n for n in rl.limiter._exempt_routes if n.startswith("wuwa_rag.api.app.")}
    assert "wuwa_rag.api.app.health" in exempt, "/health 未豁免，启动脚本轮询会被限流"


def test_限流主体_优先用户回落IP() -> None:
    """key_func 语义：有 request.state.user_id 用 u<id>，否则回落客户端 IP。

    身份由 `api/auth.get_current_user` 写入 state。取不到时回落 IP 是**刻意**的——
    依赖顺序写错最多退化成按 IP 限速，绝不会变成「完全不限流」。
    """

    class _Req:
        def __init__(self, uid=None, host="1.2.3.4"):
            self.state = type("S", (), {})()
            if uid is not None:
                self.state.user_id = uid
            self.client = type("C", (), {"host": host})()

    assert rl.key_by_user_or_ip(_Req(uid=7)) == "u7"
    assert rl.key_by_user_or_ip(_Req()) == "1.2.3.4"
