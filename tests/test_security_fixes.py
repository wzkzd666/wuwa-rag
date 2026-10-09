"""2026-10-09 安全修复的回归桩。

钉住八条修复里**能在纯离线层验证**的判据（全量离线约定见 conftest.py）：
① 配置默认值：S3 凭据留空（不再默认公开弱凭据）、ADMIN_PASSWORD 留空（不再内置弱口令）、
   TRUST_PROXY_HEADERS 默认 False、CORS_ORIGINS 默认不含通配符；
② ratelimit.client_ip：默认**不信任** X-Forwarded-For（客户端自报头即可绕过按 IP 的
   登录限速）；显式开 TRUST_PROXY_HEADERS 后才取最左项；
③ services/verify：非字符串 content 不得把 TypeError 漏进 fail-open 之外——
   分片 list 要能拼、拼不出要走降级（审查静默失效比报错难查）；
④ services/tts._global_ready：全局兜底的 workspace_id 必须过 validate_workspace_id
   （该值会拼进出站 URL 的 host 段，SSRF 面）；
⑤ 全局异常/SSE 错误体不再回传 str(exc)——内部信息（路径/SQL/版本）不出机器；
⑥ users.must_change 列：authdb._DDL 与 pgsql/002_auth.sql 两处口径一致。

断言口径与 test_config_and_prompt.py 同：默认值取 `model_fields[...].default`，
不读 `.env`（实例值随机器变，断言它会「本机绿、CI 红」两头不可信）。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from wuwa_rag.api import ratelimit as rl
from wuwa_rag.config import Settings

_ROOT = Path(__file__).resolve().parents[1]


def _default(name: str):
    return Settings.model_fields[name].default


# ---------- ① 配置默认值 ----------

@pytest.mark.parametrize("field", ["S3_ACCESS_KEY", "S3_SECRET_KEY"])
def test_S3_凭据默认留空(field: str) -> None:
    """留空 = 连不上就响亮报错；默认 rustfsadmin 等于静默用公开弱凭据开桶。"""
    assert _default(field) == ""


def test_管理员口令不再内置弱默认() -> None:
    assert _default("ADMIN_PASSWORD") == ""
    # 代码默认值里不得再出现历史种子口令
    src = (_ROOT / "src/wuwa_rag/core/authdb.py").read_text(encoding="utf-8")
    assert "123456" not in src, "authdb 又退回内置固定口令了"


def test_反代信任默认关闭() -> None:
    assert _default("TRUST_PROXY_HEADERS") is False


def test_CORS_默认不放通配符() -> None:
    assert "*" not in _default("CORS_ORIGINS")


# ---------- ② client_ip 的自报头防御 ----------

class _Req:
    def __init__(self, headers: dict[str, str] | None = None, host: str = "1.2.3.4"):
        self.headers = headers or {}
        self.client = type("C", (), {"host": host})()


def test_client_ip_默认无视_XFF() -> None:
    """直连部署：X-Forwarded-For 是客户端可自报的头，信了就能换 IP 绕开登录限速。"""
    req = _Req(headers={"x-forwarded-for": "9.9.9.9"})
    assert rl.client_ip(req) == "1.2.3.4"


def test_client_ip_显式信任后取最左项(monkeypatch) -> None:
    """反代场景：只认代理覆写的最左项（最早写入的真实客户端）。"""
    fake = Settings(TRUST_PROXY_HEADERS=True)
    monkeypatch.setattr(rl, "get_settings", lambda: fake)
    req = _Req(headers={"x-forwarded-for": "9.9.9.9, 10.0.0.1, 10.0.0.2"})
    assert rl.client_ip(req) == "9.9.9.9"


def test_client_ip_信任开启但无XFF仍回落连接IP(monkeypatch) -> None:
    fake = Settings(TRUST_PROXY_HEADERS=True)
    monkeypatch.setattr(rl, "get_settings", lambda: fake)
    assert rl.client_ip(_Req()) == "1.2.3.4"


# ---------- ③ verify 对分片响应的处理 ----------

@pytest.mark.asyncio
async def test_verify_分片list响应能拼出结果(monkeypatch) -> None:
    from wuwa_rag.services import verify as vf

    # 真实形态：提示词末尾的 AIMessage 以 `{"` 截停，模型**续写**不带 `{"` 前缀，
    # 函数自己拼回 `{"` 再解析。分片模拟 provider 返回 list 的情况。
    class _Resp:
        content = ['match": ', 'false, "refined": "卡卡罗 声骸"}']

    class _LLM:
        async def ainvoke(self, msgs, config=None):
            return _Resp()

    monkeypatch.setattr(vf, "get_tool_llm", lambda: _LLM())
    ok, refined = await vf.verify_knowledge("卡卡罗用什么声骸", "【长离】突破材料", [])
    assert ok is False
    assert refined == "卡卡罗 声骸"


@pytest.mark.asyncio
async def test_verify_纯非字符串响应走降级放行(monkeypatch) -> None:
    """拼不出文本时不得抛穿 fail-open：降级为放行，但根因留在日志里。"""
    from wuwa_rag.services import verify as vf

    class _Resp:
        content = [123, None]

    class _LLM:
        async def ainvoke(self, msgs, config=None):
            return _Resp()

    monkeypatch.setattr(vf, "get_tool_llm", lambda: _LLM())
    ok, refined = await vf.verify_knowledge("任意问题", "图谱事实", [{"breadcrumb": "b", "text": "t"}])
    assert (ok, refined) == (True, "")


@pytest.mark.asyncio
async def test_verify_空材料仍本地判不匹配(monkeypatch) -> None:
    """VERIFY 的空材料短路不经过模型（回归：改动不得把这条路径带偏）。"""
    from wuwa_rag.services import verify as vf

    def _boom():
        raise AssertionError("空材料不该调模型")

    monkeypatch.setattr(vf, "get_tool_llm", _boom)
    ok, refined = await vf.verify_knowledge("问题X", "", [])
    assert ok is False and refined == "问题X"


# ---------- ④ TTS 全局兜底的 workspace_id 校验 ----------

def test_TTS_全局兜底拒绝非法workspace(monkeypatch) -> None:
    """workspace_id 会拼进 URL host：非法值必须判「不可用」，绝不发出站请求。"""
    from wuwa_rag.services import tts as tts_mod

    fake = Settings(
        TTS_ENABLED=True, DASHSCOPE_API_KEY="sk-x",
        TTS_WORKSPACE_ID="bad id/../evil",   # 空格 + 路径片段，都不是合法 id
    )
    monkeypatch.setattr(tts_mod, "get_settings", lambda: fake)
    ok, why = tts_mod._global_ready()
    assert ok is False and "格式" in why


def test_TTS_全局兜底接受合法workspace(monkeypatch) -> None:
    from wuwa_rag.services import tts as tts_mod

    fake = Settings(TTS_ENABLED=True, DASHSCOPE_API_KEY="sk-x", TTS_WORKSPACE_ID="ws-12345")
    monkeypatch.setattr(tts_mod, "get_settings", lambda: fake)
    ok, why = tts_mod._global_ready()
    assert ok is True, why


# ---------- ⑤ 异常体不回传内部信息 ----------

def test_全局异常响应不含异常原文() -> None:
    """复用真实的 on_unhandled：客户端只见编号，str(exc) 留在服务端日志。"""
    from wuwa_rag.api.app import on_unhandled

    app = FastAPI()
    app.add_exception_handler(Exception, on_unhandled)

    @app.get("/boom")
    async def _boom():
        raise RuntimeError("SELECT * FROM pg_credentials -- 内部细节")

    # raise_server_exceptions=False：让异常走注册的处理器而不是重抛到测试进程
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/boom")
    assert resp.status_code == 500
    body = resp.json()
    assert "编号" in body["error"]
    assert "detail" not in body, "异常原文又回传客户端了"
    assert "pg_credentials" not in resp.text


# ---------- ⑥ must_change：两处 DDL 口径一致 ----------

def test_must_change_两处DDL一致() -> None:
    from wuwa_rag.core.authdb import _DDL

    joined = "\n".join(_DDL)
    sql = (_ROOT / "pgsql/002_auth.sql").read_text(encoding="utf-8")
    for stmt in ("must_change BOOLEAN", "ADD COLUMN IF NOT EXISTS must_change"):
        assert stmt in joined, f"authdb._DDL 缺 {stmt}"
        assert stmt in sql, f"002_auth.sql 缺 {stmt}"
    # 手工 SQL 与运行时 DDL 都不得残留种子口令
    assert "123456" not in sql


def test_登录响应带改密提示字段() -> None:
    """auth.login 的 SELECT 与返回字典都要带 must_change（前端据此提示）。"""
    src = (_ROOT / "src/wuwa_rag/api/auth.py").read_text(encoding="utf-8")
    assert "must_change FROM users" in src
    assert '"must_change_password": bool(row.get("must_change"))' in src
    # 改密必须清标记，否则提示永远不消失
    assert "must_change = FALSE" in src


def test_生命周期收尾关闭音乐会话() -> None:
    """高危修复⑤：lifespan 收尾必须显式 aclose 音乐 MCP（常驻子进程不靠 GC）。"""
    src = (_ROOT / "src/wuwa_rag/api/app.py").read_text(encoding="utf-8")
    start = src.index("async def lifespan")
    end = src.index("app = FastAPI(")
    lifespan_src = src[start:end]
    assert "music_svc.aclose()" in lifespan_src
