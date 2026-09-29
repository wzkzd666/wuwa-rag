"""用户自定义云端 LLM 配置的加密存储（api_key 用 Fernet 加密落库）。

隐私保护设计（需求：接收 API-KEY 且保护隐私，便于共享项目给他人）
----------------------------------------------------------------
- **加密落库**：api_key 用 Fernet 对称加密后存 `user_llm_configs.api_key_enc`，
  密钥由 `SECRET_KEY + SECRET_KEY_SALT` 经 PBKDF2 派生（stdlib hashlib，无新依赖）。
  → 数据库被拖库/备份外流时，key 不可直接读取。
- **永不回显明文**：读接口只返回掩码（`sk-****abcd`）与是否已配置，
  连管理员也拿不到别人的 key 原文（`get_config_masked`）。
- **日志只记指纹**：出错日志打 `sha256[:8]`，够定位"哪个 key"又不泄漏内容。
- **SECRET_KEY 未配置 = 功能关闭**：`available()` 返回 False，写配置直接拒绝、
  读配置返回未配置——**安全降级，绝不退化成明文存储**（明文存 key 比不存更糟）。
- **密钥不可轮换**：改 SECRET_KEY 会导致旧密文解不开（InvalidToken），此时按
  "未配置"处理并提示重填，不崩、不静默用错 key。

为什么不用 bcrypt/scrypt 哈希：key 必须**可逆**（要拿去调云端 API），
auth.py 那套单向哈希适用于口令、不适用于此。
"""
from __future__ import annotations

import base64
import hashlib
import ipaddress
import socket
from typing import Any
from urllib.parse import urlparse

import httpx

from ..authdb import get_pool
from ..config import get_settings
from ..ww_logger import get_logger

log = get_logger("llmstore")

# ── provider 预设：选完自动带出 base_url，模型列表再拉给用户选 ──────────
# 「模型也支持自选」：不要求用户手抄 model id，从服务商 /models 拉真实列表。
# base_url 始终允许用户修改，预设只是便捷项而非限制。
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "openai":     {"label": "OpenAI",         "base_url": "https://api.openai.com/v1"},
    "deepseek":   {"label": "DeepSeek",       "base_url": "https://api.deepseek.com/v1"},
    "siliconflow": {"label": "硅基流动",       "base_url": "https://api.siliconflow.cn/v1"},
    "moonshot":   {"label": "Moonshot 月之暗面", "base_url": "https://api.moonshot.cn/v1"},
    "dashscope":  {"label": "阿里云百炼",      "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
    "qianfan":    {"label": "百度千帆 v2",     "base_url": "https://qianfan.baidubce.com/v2"},
    "zhipu":      {"label": "智谱 GLM",        "base_url": "https://open.bigmodel.cn/api/paas/v4"},
    "ollama":     {"label": "本地 Ollama (/v1)", "base_url": "http://localhost:11434/v1"},
    "vllm":       {"label": "本地 vLLM",       "base_url": "http://localhost:8000/v1"},
    "custom":     {"label": "自定义（OpenAI 兼容）", "base_url": ""},
}

# SSRF 防护：base_url 由用户提供、服务端去请求它，必须限制可达范围。
# 云元数据端点无任何合法用途，**无条件**拦截。
_METADATA_HOSTS = {"169.254.169.254", "metadata.google.internal", "metadata"}
_HTTP_TIMEOUT = 20.0
_MAX_MODELS = 200          # 列表长度帽（有些服务商返回上千条）

# 与 authdb._DDL 一致（幂等）；此处单独一份供 ensure 时补建，避免跨模块改表结构
_DDL = [
    """
    CREATE TABLE IF NOT EXISTS user_llm_configs (
        user_id    BIGINT      PRIMARY KEY,
        provider   TEXT        NOT NULL DEFAULT 'openai',
        base_url   TEXT        NOT NULL DEFAULT '',
        model      TEXT        NOT NULL DEFAULT '',
        api_key_enc TEXT       NOT NULL DEFAULT '',
        key_hint   TEXT        NOT NULL DEFAULT '',
        enabled    BOOLEAN     NOT NULL DEFAULT TRUE,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    # 是否让**用户自己的云端模型**兼任情绪判定（分工见 rag/emotion.py）。
    # 默认为否：情绪判定一律走本地 qwen3:8b，省一次远程调用，也不把个人的额度
    # 花在内部任务上。存量库补列，幂等可重跑。
    "ALTER TABLE user_llm_configs "
    "ADD COLUMN IF NOT EXISTS emotion_enabled BOOLEAN NOT NULL DEFAULT FALSE",
]

# 掩码保留的头尾长度：够用户认出是哪把 key，又不足以还原
_HINT_HEAD = 3
_HINT_TAIL = 4
MAX_KEY_LEN = 512          # 超长视为误填（粘贴了整段文本），拒绝入库
MAX_TEXT_LEN = 512         # base_url / model 长度帽


# ---------- SSRF 防护（base_url 用户可填、服务端去请求） ----------

def validate_base_url(url: str) -> None:
    """校验用户提交的 base_url，不合法直接抛 ValueError（API 层转 400）。

    拦三类：① 非 http(s)；② 云元数据端点（169.254.169.254 等，无任何合法用途，
    无条件封）；③ 私网/环回地址——**默认允许**（预设里就有本地 Ollama/vLLM，
    开发者自用是主流场景），公网部署可用 `CLOUD_ALLOW_PRIVATE_NET=False` 关掉，
    否则等于把内网探测口开给所有注册用户。

    需先解析主机名到 IP 再判断：仅比对字符串会被 http://内网IP.nip.io 这类
    DNS 重绑定/通配域名绕过。
    """
    u = (url or "").strip().rstrip("/")
    if not u:
        raise ValueError("base_url 不能为空")
    if not u.startswith(("http://", "https://")):
        raise ValueError("base_url 必须以 http:// 或 https:// 开头")
    if len(u) > MAX_TEXT_LEN:
        raise ValueError(f"base_url 过长（>{MAX_TEXT_LEN}）")

    parsed = urlparse(u)
    host = (parsed.hostname or "").strip().lower().strip("[]")
    if not host:
        raise ValueError("base_url 缺少主机名")
    if host in _METADATA_HOSTS:
        raise ValueError("该地址被禁止访问（云元数据端点）")

    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80),
                                   proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ValueError(f"base_url 主机名无法解析：{host}") from exc

    if not get_settings().CLOUD_ALLOW_PRIVATE_NET:
        for info in infos:
            try:
                ip = ipaddress.ip_address(info[4][0])
            except ValueError:
                continue
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                raise ValueError("服务端已禁用内网地址（CLOUD_ALLOW_PRIVATE_NET=False）")


async def list_models(base_url: str, api_key: str) -> list[str]:
    """从服务商 `/models` 拉可选模型名（OpenAI 兼容约定），供前端下拉自选。

    失败抛 ValueError（带可读原因），由 API 层转 4xx/5xx——不静默返回空列表，
    否则用户分不清"服务商没模型"和"我 key/地址填错了"。
    """
    validate_base_url(base_url)
    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=False) as cli:
            rsp = await cli.get(url, headers=headers)
    except httpx.HTTPError as exc:
        raise ValueError(f"连接模型列表失败：{type(exc).__name__}") from exc

    if rsp.status_code in (401, 403):
        raise ValueError("API-KEY 无效或无权限（401/403）")
    if rsp.status_code == 404:
        raise ValueError("该地址没有 /models 接口（404），请确认 base_url 是否到 /v1")
    if rsp.status_code >= 400:
        raise ValueError(f"服务商返回 {rsp.status_code}")

    try:
        payload = rsp.json()
    except ValueError as exc:
        raise ValueError("模型列表响应不是合法 JSON") from exc

    # OpenAI 约定：{"data":[{"id":...}, ...]}；部分服务商直接给 ["a","b"] 或 {"models":[...]}
    raw = payload.get("data") if isinstance(payload, dict) else payload
    if raw is None and isinstance(payload, dict):
        raw = payload.get("models")
    if not isinstance(raw, list):
        raise ValueError("模型列表结构不符合 OpenAI 约定")

    out: list[str] = []
    for item in raw[:_MAX_MODELS]:
        name = item.get("id") if isinstance(item, dict) else item
        if isinstance(name, str) and name.strip() and name.strip() not in out:
            out.append(name.strip())
    if not out:
        raise ValueError("服务商返回了空的模型列表")
    log.info("拉取模型列表 base_url=%s 共 %d 个", base_url, len(out))
    return out


# ---------- Fernet（延迟导入：未配 SECRET_KEY 时不该因为缺包而崩） ----------

_FERNET_CACHE: dict[str, Any] = {}   # SECRET_KEY → Fernet | None（派生很贵，必须缓存）


def _fernet():
    """由 SECRET_KEY 派生 Fernet 实例；SECRET_KEY 空 → None（功能关闭）。

    PBKDF2 十万次迭代不便宜（约 50~100ms），而问答路径每次都要解密，
    故按 SECRET_KEY 值缓存实例（key 变了自然重新派生，旧缓存留着无害）。
    """
    s = get_settings()
    secret = s.SECRET_KEY.strip()
    if not secret:
        return None
    if secret in _FERNET_CACHE:
        return _FERNET_CACHE[secret]
    try:
        from cryptography.fernet import Fernet

        raw = hashlib.pbkdf2_hmac(
            "sha256", s.SECRET_KEY.encode(), s.SECRET_KEY_SALT.encode(), 100_000, dklen=32
        )
        inst: Any = Fernet(base64.urlsafe_b64encode(raw))
    except Exception as exc:                       # cryptography 缺失等
        log.error("凭证加密不可用（云模型配置将被禁用）：%s", exc)
        inst = None
    _FERNET_CACHE[secret] = inst
    return inst


def available() -> bool:
    """云模型配置功能是否可用（SECRET_KEY 已配 + 加密库可用）。"""
    return _fernet() is not None


def _mask(key: str) -> str:
    """生成掩码：`sk-a****wxyz`。短 key 全掩。"""
    k = (key or "").strip()
    if len(k) <= _HINT_HEAD + _HINT_TAIL + 1:
        return "*" * len(k)
    return f"{k[:_HINT_HEAD]}****{k[-_HINT_TAIL:]}"


def fingerprint(key: str) -> str:
    """key 指纹（sha256 前 8 位）：仅用于日志定位，不可逆推。"""
    return hashlib.sha256((key or "").encode()).hexdigest()[:8]


async def ensure_schema() -> None:
    """幂等建表。API lifespan 里随 authdb.ensure_schema 一起调。"""
    pool = await get_pool()
    async with pool.connection() as conn:
        for stmt in _DDL:
            await conn.execute(stmt)


async def get_config_masked(user_id: int) -> dict[str, Any]:
    """读配置（**只返回掩码，绝不返回明文**）。未配置/禁用/密钥不可用 → enabled=False。"""
    row = await _fetch(user_id)
    if row is None:
        return {"configured": False, "enabled": False, "provider": "", "base_url": "",
                "model": "", "key_hint": "", "emotion_enabled": False,
                "crypto_available": available()}
    return {
        "configured": True,
        "enabled": bool(row["enabled"]),
        "provider": row["provider"],
        "base_url": row["base_url"],
        "model": row["model"],
        "key_hint": row["key_hint"] or _mask(""),
        # 情绪判定是否由该用户的云端模型兼任（false = 走本地 qwen3:8b）
        "emotion_enabled": bool(row.get("emotion_enabled", False)),
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        "crypto_available": available(),
    }


async def _fetch(user_id: int) -> dict | None:
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT provider, base_url, model, api_key_enc, key_hint, enabled, "
            "emotion_enabled, updated_at"
            " FROM user_llm_configs WHERE user_id = %s", (user_id,)
        )
        return await cur.fetchone()


async def get_runtime(user_id: int) -> dict | None:
    """取**可调用**的运行时配置（含解密后的明文 key，仅在内存中短暂存在）。

    返回 None 表示应回落本地默认模型：未配置 / 未启用 / 加密不可用 / 解密失败。
    明文 key 只传入 ChatOpenAI 实例，不写日志、不进 state、不返回给前端。
    """
    if not available():
        return None
    row = await _fetch(user_id)
    if row is None or not row["enabled"]:
        return None
    f = _fernet()
    try:
        key = f.decrypt(row["api_key_enc"].encode()).decode()
    except Exception as exc:
        # 典型原因：SECRET_KEY 被改过 → 旧密文解不开。按未配置处理，不崩。
        log.warning("user=%s 云端 key 解密失败（SECRET_KEY 可能已变更，需重填）：%s",
                    user_id, type(exc).__name__)
        return None
    if not key.strip():
        return None
    return {
        "provider": row["provider"] or "openai",
        "base_url": row["base_url"],
        "model": row["model"],
        "api_key": key,
        # 由调用链决定是否让该云端模型兼任情绪判定（rag/chain.py）
        "emotion_enabled": bool(row.get("emotion_enabled", False)),
    }


async def save_config(user_id: int, *, base_url: str, model: str, api_key: str,
                      provider: str = "openai", enabled: bool = True,
                      emotion_enabled: bool = False) -> dict:
    """写入/更新配置。api_key 传空串表示**保留原 key 不改**（前端只改了 url/model 时）。

    校验失败抛 ValueError（由 API 层转 400），不静默吞。
    """
    base_url = (base_url or "").strip().rstrip("/")
    model = (model or "").strip()
    api_key = (api_key or "").strip()

    if not available():
        raise ValueError("服务端未配置 SECRET_KEY，云模型功能已关闭（不会明文存储密钥）")
    validate_base_url(base_url)     # 含 http(s) / 长度 / 元数据端点 / 私网策略
    if not model:
        raise ValueError("model 不能为空")
    if len(model) > MAX_TEXT_LEN:
        raise ValueError(f"model 长度不得超过 {MAX_TEXT_LEN}")

    f = _fernet()
    row = await _fetch(user_id)
    if api_key:
        if len(api_key) > MAX_KEY_LEN:
            raise ValueError(f"api_key 过长（>{MAX_KEY_LEN}），疑似粘贴了多余内容")
        enc = f.encrypt(api_key.encode()).decode()
        hint = _mask(api_key)
    elif row and row["api_key_enc"]:
        enc, hint = row["api_key_enc"], row["key_hint"]   # 保留原 key
    else:
        raise ValueError("api_key 不能为空")

    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO user_llm_configs
                (user_id, provider, base_url, model, api_key_enc, key_hint, enabled,
                 emotion_enabled, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (user_id) DO UPDATE SET
                provider = EXCLUDED.provider, base_url = EXCLUDED.base_url,
                model = EXCLUDED.model, api_key_enc = EXCLUDED.api_key_enc,
                key_hint = EXCLUDED.key_hint, enabled = EXCLUDED.enabled,
                emotion_enabled = EXCLUDED.emotion_enabled, updated_at = now()
            """,
            (user_id, provider, base_url, model, enc, hint, enabled, emotion_enabled),
        )
    log.info("user=%s 保存云端模型配置 base_url=%s model=%s key=%s",
             user_id, base_url, model, hint)          # 只记掩码，不记明文
    return {"configured": True, "enabled": enabled, "provider": provider,
            "base_url": base_url, "model": model, "key_hint": hint}


async def set_enabled(user_id: int, enabled: bool) -> bool:
    """启用/停用云端配置（停用即回落本地默认模型，key 保留）。"""
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "UPDATE user_llm_configs SET enabled = %s, updated_at = now() WHERE user_id = %s",
            (enabled, user_id),
        )
        return cur.rowcount > 0


async def delete_config(user_id: int) -> bool:
    """删除配置（含密文），之后回落本地默认模型。"""
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "DELETE FROM user_llm_configs WHERE user_id = %s", (user_id,)
        )
    if cur.rowcount:
        log.info("user=%s 已删除云端模型配置", user_id)
    return cur.rowcount > 0
