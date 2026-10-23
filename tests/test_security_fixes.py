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


# ---------- llmstore：InvalidToken 不是 ValueError 子类（实测推翻的想当然假设） ----------

def test_解密异常口径必须接住_InvalidToken() -> None:
    """`_DECRYPT_ERRORS` 若只捕 ValueError，「口令/密码不对」会裸冲出去弄挂登录。"""
    from cryptography.fernet import Fernet, InvalidToken

    from wuwa_rag.core.llmstore import _DECRYPT_ERRORS

    # 钉住那个反直觉事实本身：InvalidToken 直接继承 Exception
    assert not issubclass(InvalidToken, ValueError)
    k1, k2 = Fernet.generate_key(), Fernet.generate_key()
    cipher = Fernet(k1).encrypt(b"sk-secret")
    with pytest.raises(_DECRYPT_ERRORS):
        Fernet(k2).decrypt(cipher)
    # 合法密钥必须能解回——防止把判据写成「什么都算解密失败」的假绿
    assert Fernet(k1).decrypt(cipher) == b"sk-secret"


# ---------- 云端 chat 客户端：重定向必须关（SSRF 的第二跳） ----------

def test_云端客户端不得跟随重定向() -> None:
    """openai SDK 建 httpx 客户端时写死 `follow_redirects` 默认 True。

    `llmstore.validate_base_url` 只核**第一跳**的地址：留着跟随重定向，用户填的
    base_url 就能用一条 302 把服务端请求甩进内网/云元数据端点。
    httpx 跨源重定向会剥掉 Authorization（key 不泄），但请求仍打出去、响应仍进上下文。
    这里断言的是「我们塞进去的两个客户端都是关的」——不依赖 SDK 的默认值。
    """
    from langchain_openai import ChatOpenAI

    from wuwa_rag.core.llm import _build_openai

    cfg = {"model": "gpt-test", "api_key": "sk-none", "base_url": "https://example.invalid/v1"}
    llm: ChatOpenAI = _build_openai(cfg, strict=False)
    for name in ("http_async_client", "http_client"):
        client = getattr(llm, name)
        assert client is not None, f"ChatOpenAI 没用上我们注入的 {name}"
        assert client.follow_redirects is False, f"{name} 仍在跟随重定向"
    # 光看字段还不够：必须确认 SDK 真拿这两个对象发请求
    # （不注入时 `http_async_client` 是 None，客户端由 SDK 自己建，实测 follow_redirects=True）
    assert llm.root_client._client is llm.http_client
    assert llm.async_client._client._client is llm.http_async_client


# ---------- 反馈输入：长度帽与 thread_id 判据只能有一处来源 ----------

def test_反馈入口按帽拒绝而不静默截断() -> None:
    """`/feedback` 的 `thread_id`/`target_id` 是 upsert 键的一部分，**截断会把两个 id 合成一个**。

    表现是「给 A 点的赞跑到 B 上」——数据不报错、只是悄悄错位，最难发现的一类。
    所以两列都必须「拒绝而不是截断」，而快照三列（comment/question/answer）才允许截断。
    """
    import asyncio

    from pydantic import ValidationError

    from wuwa_rag.api.usage import FeedbackIn
    from wuwa_rag.core import feedback as fb

    base = {"thread_id": "abc123", "target_id": "168", "rating": 1}
    FeedbackIn(**base)                                     # 正常输入必须过
    with pytest.raises(ValidationError):
        FeedbackIn(**{**base, "thread_id": "x" * 65})      # 超长的 thread 串
    with pytest.raises(ValidationError):
        FeedbackIn(**{**base, "thread_id": "../../etc/passwd"})   # 模式不符
    with pytest.raises(ValidationError):
        FeedbackIn(**{**base, "target_id": ""})            # 空键会 upsert 到同一行

    # core 侧是第二道闸（脚本/别的调用方未必过 pydantic），同样只拒不断
    with pytest.raises(ValueError):
        asyncio.run(fb.save_feedback(user_id="1", username="u", thread_id="y" * 100,
                                     target_id="168", rating=1),
                     loop_factory=asyncio.SelectorEventLoop)
    with pytest.raises(ValueError):
        asyncio.run(fb.save_feedback(user_id="1", username="u", thread_id="ok",
                                     target_id="", rating=1),
                     loop_factory=asyncio.SelectorEventLoop)


def test_反馈帽只定义在core一处() -> None:
    """入口模型用的数字必须**引用** core 的常量，而不是再抄一遍字面量。"""
    from wuwa_rag.api.usage import FeedbackIn
    from wuwa_rag.core import conversations as conv
    from wuwa_rag.core import feedback as fb

    meta = {name: {type(m).__name__: getattr(m, "max_length", None) or getattr(m, "min_length", None)
                   for m in field.metadata}
            for name, field in FeedbackIn.model_fields.items()}
    assert meta["comment"].get("MaxLen") == fb.COMMENT_MAX
    assert meta["question"].get("MaxLen") == fb.QUESTION_MAX
    assert meta["answer"].get("MaxLen") == fb.ANSWER_MAX
    assert meta["target_id"].get("MaxLen") == fb.TARGET_ID_MAX
    # thread_id 的模式来自 conversations（入口校验与反馈落库共用同一个正则对象）
    pats = [getattr(m, "pattern", None) for m in FeedbackIn.model_fields["thread_id"].metadata]
    assert conv.THREAD_ID_RE.pattern in pats


def test_会话搜索转义_like_元字符() -> None:
    """`%` / `_` 是 LIKE 的通配符：不转义时用户搜「100%」会匹配到全部会话。

    不是注入（参数化已经挡住了注入），是**结果不对** + 「随便打个百分号就全表扫」。
    """
    from wuwa_rag.core.conversations import _like_pattern

    assert _like_pattern("守岸人") == "%守岸人%"
    B = chr(92)          # 反斜杠：这里一律用 chr(92) 拼，避开源码里 `\` 被转义层级吃掉
    assert _like_pattern("100%") == f"%100{B}%%"                 # 百分号变字面量
    assert _like_pattern("a_b") == f"%a{B}_b%"                   # 下划线不再匹配任意字符
    assert _like_pattern(f"C:{B}x") == f"%C:{B}{B}x%"            # 反斜杠自己先转义

def test_入库失败原因对非管理员只给一句话() -> None:
    """措辞只有一处（`_public_step_error`），两个出口共用；管理员照旧看原文。"""
    from wuwa_rag.api.app import CANCEL_ERROR, _public_step_error

    raw = "fetch: ConnectionError to https://wiki.internal/…（trace 之类）"
    assert _public_step_error("crawl", raw, is_admin=True) == raw
    got = _public_step_error("crawl", raw, is_admin=False)
    assert got is not None and "抓取" in got and "ConnectionError" not in got
    # 没有失败步骤时不许输出「「」这种空引号」
    assert _public_step_error("", raw, is_admin=False) == "入库失败，详情请联系管理员"
    # 取消哨兵必须原样透出（下游靠它显示「已取消」）
    assert _public_step_error("chunk", CANCEL_ERROR, is_admin=False) == CANCEL_ERROR
    assert _public_step_error("chunk", None, is_admin=False) is None


# ---------- 账号移除（数据回收） ----------
def _user_column_tables() -> set[str]:
    """从**建表语句**里反推「有用户身份列的表」。

    两处都要扫：`pgsql/*.sql`（部署时幂等应用的那份）和 `src/**/*.py` 里的 `_DDL`
    字符串（`user_llm_configs` 就只活在 llmstore 里）。只扫一处会漏表 —— 而漏掉的
    那张表在删账号时**不会报错**，只会把数据留在库里当孤儿。
    """
    import re

    marker = "CREATE TABLE IF NOT EXISTS "
    # `session_id` 也要算：转录与衍生行是按会话归属的，而 `messages` 连 user 列都没有。
    # 只匹配 user 列的后果实测过：手写 SQL 删掉 sessions 行之后，孤儿转录既没被回收、
    # 这条守卫也**看不见它**（库里现成就躺着 4 条 u40/u42 的转录）。
    has_user_col = re.compile("user_id|submitted_by|session_id")
    out: set[str] = set()
    paths = sorted((_ROOT / "pgsql").glob("*.sql")) + sorted((_ROOT / "src" / "wuwa_rag").rglob("*.py"))
    for path in paths:
        text = path.read_text(encoding="utf-8")
        # 按标记切开逐段看，不用带反斜杠的正则（本仓的工具链吃掉过太多次）
        for chunk in text.split(marker)[1:]:
            body = chunk.split(");", 1)[0]            # 建表语句体
            head = body.split("(", 1)[0].strip()      # 表名
            if len(head.split()) == 1 and has_user_col.search(body):
                out.add(head)
    return out


def test_用户回收清单覆盖每张有用户列的表() -> None:
    """新增一张用户表却忘了回收 = 静默留孤儿数据。这条把它变成当场红。

    本仓实测已经躺着这种孤儿：5 位「authdb 里已不存在的主人」的会话与画像
    —— 正因为以前「删用户」只能手写一条 DELETE FROM users。
    """
    from wuwa_rag.api.app import EXCLUDED_TABLES, PURGED_TABLES

    tables = _user_column_tables()
    assert tables, "反推没拿到任何表：多半是上面的建表语句正则没匹配上，这条会恒真"
    accounted = set(PURGED_TABLES) | set(EXCLUDED_TABLES)
    missing = tables - accounted
    assert not missing, f"这些表带用户列却没进回收清单（既不回收也不说明理由）：{sorted(missing)}"
    assert all(str(r).strip() for r in EXCLUDED_TABLES.values()), "不回收必须写清理由"


def test_删账号端点是admin专属且挡住自删与末位admin() -> None:
    """不可恢复的管理操作：权限与两道护栏都要在，缺一就能把整套鉴权删没。"""
    from wuwa_rag.api import app as app_mod
    from wuwa_rag.api import auth as authn

    routes = [r for r in app_mod.app.routes
              if getattr(r, "path", "") == "/admin/users/{user_id}"]
    assert len(routes) == 1, f"应当只挂一条，实际 {len(routes)}"
    assert "DELETE" in routes[0].methods
    def _walk(dep):
        for sub in getattr(dep, 'dependencies', []) or []:
            yield sub
            yield from _walk(sub)

    calls = [d.call for d in _walk(routes[0].dependant)]
    assert authn.require_admin in calls, "删账号必须是 admin 专属（get_current_user 不够）"

    src = (_ROOT / "src/wuwa_rag/api/app.py").read_text(encoding="utf-8")
    body = src[src.index('api_admin_user_delete'):]
    assert "不能删除自己这个账号" in body, "少了「不许自删」护栏"
    assert "最后一个管理员" in body, "少了「末位 admin」护栏"


def test_删账号的记忆回收取两处并集() -> None:
    """只从 `sessions` 反查 key 会漏：**会话行被手写 SQL 删掉而 checkpointer 还在**。
    实测库里躺着 3 条这种孤儿记忆（u10 / u40 / u42，主人早已不在 users 里；
    这 3 条已由端点自己跑通后手工清掉，今天实测 `u40/u42` 的 lg 线程 = 0）。
    所以端点必须并上 `conv.memory_keys(user_id)`（按 `u{id}:` 前缀直查 lg.checkpoints）。
    """
    src = (_ROOT / "src/wuwa_rag/api/app.py").read_text(encoding="utf-8")
    body = src[src.index('api_admin_user_delete'):src.index('api_admin_user_delete') + 3000]
    assert "await conv.memory_keys(user_id)" in body, "少了按前缀直查 checkpointer 的那一路"


def test_删账号的转录回收也要按前缀补孤儿() -> None:
    """`clear_conversations` 原先只删 `sessions` 里**还在**的那批 session_id。
    手写 SQL 删过 sessions 行之后，转录就成了谁够不着的孤儿 —— 实测库里现成躺着 4 条
    （`u40:ffe8ddde3ee7` 两条 + `u42:438ca97f4e2f` 两条），而删账号端点对它们一无所知。
    `session_id` 的前缀就是 `u{users.id}:`，所以必须按前缀补一刀；
    `session_summaries` 同属这批 session_id，一并进来（今天 0 行，但它归会话归属）。
    """
    import inspect

    from wuwa_rag.core.conversations import clear_conversations

    body = inspect.getsource(clear_conversations)
    assert "session_id LIKE" in body, "没按 u{id}: 前缀补删孤儿转录"
    assert "session_summaries" in body, "衍生表没跟着会话回收"
    assert 'f"u{int(user_id)}:%"' in body, "前缀没锚住冒号（u1:% 会匹到 u10: 就删错人了）"


def test_回收条数只报真的删成的() -> None:
    """不可恢复操作的回执行必须如实：`memories: len(keys)` 在 saver 整批失败时仍写 N 条，
    而实际一条没删 —— 这比少报危险（操作员以为清干净了）。
    """
    src = (_ROOT / "src/wuwa_rag/api/app.py").read_text(encoding="utf-8")
    body = src[src.index('api_admin_user_delete'):src.index('api_admin_user_delete') + 3000]
    assert '"memories": len(keys)' not in body, "把「打算删的条数」当成「删成的条数」"
    assert "if await _drop_memory(k)" in body, "没按 _drop_memory 的返回值计数"
    drop = src[src.index("async def _drop_memory"):]
    assert "return True" in drop and "return False" in drop, "_drop_memory 没出声失败"
    assert "dict.fromkeys" in body, "两路 key 必须并集去重（同一条记忆重复 drop 会白跑一次）"

    # 前缀本身也要钉住：LIKE 写成 `%{id}%` 会把 u10 的记忆错删到 u1 名下
    csrc = (_ROOT / "src/wuwa_rag/core/conversations.py").read_text(encoding="utf-8")
    assert 'f"u{int(user_id)}:%"' in csrc, "memory_keys 的 LIKE 前缀口径变了"
