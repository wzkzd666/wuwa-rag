"""用户自定义云端 LLM 配置的加密存储（加密落盘 + 登录自动解锁 + 不泄密）。

需求三条必须同时成立：① **加密落盘**；② **下次登录自动连接**；③ **不泄密**。
只用「一个用户口令」做不到这三条的交点——口令只有用户知道，服务端没有它就
无法自动解密；而服务端要自行解密，就必然得持有某样东西。故采用**双层密钥**
（DEK / KEK，与 Bitwarden、1Password 等密码管理器同构）：

    api_key ──DEK─────> api_key_enc     DEK = 随机数据密钥，只用来加密数据
    DEK     ──KEK_pwd─> dek_by_pwd      KEK_pwd = **登录密码**派生
    DEK     ──KEK_pp──> dek_by_pp       KEK_pp  = 用户自设**加密口令**派生（兜底）

- **落盘只有密文**：库里三样全是密文，没有任何一把明文密钥。拖库/备份外流时，
  攻击者既拿不到登录密码（库里只有 scrypt **单向**哈希）也拿不到加密口令，
  解不开。开源分发时 clone 即用，无需任何人先配一个共享主密钥。
- **登录即自动解锁**：登录请求带明文密码，服务端当场派生 KEK_pwd 解出 DEK
  放进内存 —— 用户无需任何额外输入，下次登录云端模型直接生效。
- **服务端零持久密钥材料**：解出的 DEK 只在本进程内存里，不落库、不写日志、
  不进 LangGraph state（checkpointer 会把 state 持久化进 PG，进 state = 等于
  落库）。进程重启会清空，**重新登录即可恢复**；也可用加密口令兜底解锁。
- **改密码必须重绑**：KEK_pwd 由密码派生，改密码后旧的 `dek_by_pwd` 立即失效。
  `rebind_password` 在改密码时用**旧密码**解出 DEK、用新密码重新加密，
  否则用户改一次密码就把自己的 Key 永久锁死（只能删配置重填）。
- **加密口令是独立兜底通道**：不依赖登录密码，用于「服务重启后暂不重新登录」
  「不想让登录密码触碰密钥」等场景；忘记它不影响密码通道，反之亦然。
- **永不回显明文**：读接口只返回掩码；日志只记 sha256 前 8 位指纹。
- **解不开 = 回落本地**：未解锁 / 密文解不开时静默走本地默认 agent，
  绝不退化成明文存储，也不阻塞问答。
- **派生开销**：PBKDF2 二十万轮约 0.1~0.4s，只在登录与解锁时各付一次，
  之后走内存缓存；问答路径解密不再派生。

为什么不用 bcrypt/scrypt 单向哈希：key 必须**可逆**（要拿去调云端 API），
单向哈希适用于校验口令、不适用于此。
"""
from __future__ import annotations

import base64
import hashlib
import ipaddress
import re
import secrets
import socket
from typing import Any
from urllib.parse import urlparse

import httpx

from wuwa_rag.config import get_settings
from wuwa_rag.core.authdb import get_pool
from wuwa_rag.ww_logger import get_logger

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
    # 加密口令（兜底通道）派生的两个材料（见模块 docstring）：
    # key_salt  = 该用户专属随机盐（首次设置口令时生成，之后不变）
    # key_check = 用 KEK_pp 加密的固定串，用于**校验口令对错**——否则只能等真正
    #             解密 dek_by_pp 失败才知道错了，无法给用户明确反馈
    "ALTER TABLE user_llm_configs ADD COLUMN IF NOT EXISTS key_salt TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE user_llm_configs ADD COLUMN IF NOT EXISTS key_check TEXT NOT NULL DEFAULT ''",
    # 登录密码（自动解锁通道）派生的两个材料：
    # pwd_salt   = 密码派生 KEK 用的随机盐，与 auth 的密码哈希盐**相互独立**
    #              （两者用途不同，共用一把盐等于把两处安全性绑死）
    # dek_by_pwd = DEK 用 KEK_pwd 加密后的密文，登录时用它自动解锁
    "ALTER TABLE user_llm_configs ADD COLUMN IF NOT EXISTS pwd_salt TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE user_llm_configs ADD COLUMN IF NOT EXISTS dek_by_pwd TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE user_llm_configs ADD COLUMN IF NOT EXISTS dek_by_pp TEXT NOT NULL DEFAULT ''",
    # ---------- 语音合成（TTS）凭据（2026-09-30）----------
    # 为什么与云端 LLM 塞进同一张表、共用同一把 DEK，而不是另起一张：
    #   · DEK 的建立 / 绑定密码 / 换口令逻辑全部锚在这张表（`_PENDING` → `_bootstrap`
    #     → `_take_pending`）。另建表会遇到「用户只配了 TTS，DEK 该存哪」的两难；
    #     若两张表各存一份 DEK，换密码 / 换口令就得同步改两处，多一个坏点。
    #   · 同一把 DEK 加密两类凭据 = 用户只维护一套口令，登录解锁一次两边都可用。
    # 表名仍是 user_llm_configs（历史命名），实际语义已是「**用户自持的外部服务凭据**」。
    # ⚠️ 因此 `get_config_masked().configured` / `get_tts_masked().configured` 必须按
    # **各自的密文字段是否非空**判断，不能只看行是否存在——否则「只配了 TTS」的用户
    # 会被前端显示成「已配置云端模型」。
    #
    # 刻意**不设** per-user 的「语音总开关」：前端已有一个本地偏好开关负责「我不想听」，
    # 后端再来一个「停用后回落全局兜底」的开关语义不直观、且两者重名。想不用自己这份
    # 凭据就直接删除（`delete_tts_config`），少一个状态就少一类说不清的边界。
    "ALTER TABLE user_llm_configs ADD COLUMN IF NOT EXISTS tts_api_key_enc TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE user_llm_configs ADD COLUMN IF NOT EXISTS tts_key_hint TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE user_llm_configs ADD COLUMN IF NOT EXISTS tts_workspace_id TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE user_llm_configs ADD COLUMN IF NOT EXISTS tts_model TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE user_llm_configs ADD COLUMN IF NOT EXISTS tts_voice TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE user_llm_configs ADD COLUMN IF NOT EXISTS tts_instruction TEXT NOT NULL DEFAULT ''",
    # 一次性清理：`tts_enabled` 是本次开发中途加过又撤掉的 per-user 语音开关
    # （理由见上）。它**从未发布**，故直接 DROP；保留此语句仅为让本地已建过的库
    # 与服务端 schema 保持同步。`IF EXISTS` 保证对全新库是空操作，可长期留着。
    "ALTER TABLE user_llm_configs DROP COLUMN IF EXISTS tts_enabled",
]

# 掩码保留的头尾长度：够用户认出是哪把 key，又不足以还原
_HINT_HEAD = 3
_HINT_TAIL = 4
MAX_KEY_LEN = 512          # 超长视为误填（粘贴了整段文本），拒绝入库
MAX_TEXT_LEN = 512         # base_url / model 长度帽

# 口令派生参数
_PBKDF2_ROUNDS = 200_000
_SALT_BYTES = 16           # 每用户独立随机盐（32 个 hex 字符）
MIN_PASSPHRASE = 8         # 口令最短长度：太短等于没加密
_CHECK_PLAIN = b"wuwa-rag-llm-key-check"   # 口令校验用固定明文（不含任何秘密）


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


# ---------- 双层密钥（DEK / KEK）----------
# 延迟导入 cryptography：未启用云模型时不该因为缺这个包而崩。

# user_id → DEK 的原始密钥字节。**仅进程内存**：不落库、不写日志、
# 不进 LangGraph state（checkpointer 会把 state 持久化进 PostgreSQL）。
# 进程重启即清空 —— 用户重新登录即可自动恢复（KEK_pwd 通道）。
# 存原始字节而非 Fernet 实例：补建第二条通道时要拿它重新加密落盘。
_DEK_RAW: dict[int, bytes] = {}

# 首次建立密钥时先算出盐与两份 DEK 密文，但要等 save_config 的事务里才落库；
# 用这个小暂存区传递，避免"密钥建了、key 没存成"留下半成品行。
_PENDING: dict[int, dict[str, str]] = {}


def crypto_ready() -> bool:
    """加密库是否可用（`cryptography` 为可选依赖，缺失则云模型功能整体降级）。"""
    try:
        from cryptography.fernet import Fernet  # noqa: F401
    except Exception as exc:
        log.error("凭证加密不可用（云模型配置将被禁用）：%s", exc)
        return False
    return True


def _fernet(raw: bytes) -> Any:
    """原始密钥字节 → Fernet 实例。构造极廉价，可以随用随建。"""
    from cryptography.fernet import Fernet

    return Fernet(raw)


def _derive(secret: str, salt: str) -> Any:
    """密码/口令 + 用户专属盐 → KEK（PBKDF2-SHA256，二十万轮）。"""
    from cryptography.fernet import Fernet

    raw = hashlib.pbkdf2_hmac(
        "sha256", secret.encode("utf-8"), salt.encode("utf-8"),
        _PBKDF2_ROUNDS, dklen=32,
    )
    return Fernet(base64.urlsafe_b64encode(raw))


def _dek(user_id: int) -> Any:
    """取该用户已解锁的 DEK（Fernet）；未解锁 → None。"""
    raw = _DEK_RAW.get(user_id)
    return _fernet(raw) if raw else None


def is_unlocked(user_id: int) -> bool:
    """该用户的 DEK 是否已在本进程内解锁。"""
    return user_id in _DEK_RAW


def lock(user_id: int) -> None:
    """丢弃该用户的 DEK（主动锁定 / 退出登录）。"""
    _DEK_RAW.pop(user_id, None)


async def unlock_with_password(user_id: int, password: str) -> bool:
    """用**登录密码**解锁（登录成功后自动调用，用户无需任何额外输入）。

    失败返回的三种常见情形都不影响登录本身：尚未配置云模型、只启用了加密口令
    通道、或密码与建立密钥时不一致（改过密码且未重绑）。
    """
    pw = password or ""
    row = await _fetch(user_id)
    if not pw or row is None:
        return False
    salt, enc = row.get("pwd_salt") or "", row.get("dek_by_pwd") or ""
    if not salt or not enc:
        return False                       # 未启用密码通道
    try:
        raw = _derive(pw, salt).decrypt(enc.encode())
    except Exception:
        return False                       # 密码不一致（InvalidToken）
    _DEK_RAW[user_id] = raw
    return True


async def unlock_with_passphrase(user_id: int, passphrase: str) -> bool:
    """用**加密口令**解锁（兜底通道，独立于登录密码）。

    先用 `key_check` 校验口令对错（便宜且能给明确反馈），再解出 DEK。
    """
    pp = (passphrase or "").strip()
    row = await _fetch(user_id)
    if not pp or row is None:
        return False
    salt, check, enc = (row.get("key_salt") or "", row.get("key_check") or "",
                        row.get("dek_by_pp") or "")
    if not salt or not check or not enc:
        return False                       # 未启用口令通道
    try:
        kek = _derive(pp, salt)
        kek.decrypt(check.encode())        # 解不开 = 口令错
        raw = kek.decrypt(enc.encode())
    except Exception:
        return False
    _DEK_RAW[user_id] = raw
    return True


async def unlock(user_id: int, password: str = "", passphrase: str = "") -> bool:
    """任选一条通道解锁：先试登录密码（自动通道），再试加密口令（兜底）。"""
    if password and await unlock_with_password(user_id, password):
        return True
    if passphrase and await unlock_with_passphrase(user_id, passphrase):
        return True
    return False


def _bootstrap(user_id: int, *, password: str = "", passphrase: str = "") -> Any:
    """首次建立 DEK 与两条解锁通道（至少给一个，两个都给则都能解锁）。

    调用方负责把 `_take_pending(user_id)` 的结果随本次 save 一起落库。
    """
    pp = (passphrase or "").strip()
    if pp and len(pp) < MIN_PASSPHRASE:
        raise ValueError(f"加密口令至少 {MIN_PASSPHRASE} 位（服务端不保存它，忘了无法找回）")
    if not password and not pp:
        raise ValueError("首次保存需要登录密码或加密口令来建立加密密钥")

    from cryptography.fernet import Fernet

    raw = Fernet.generate_key()
    pending: dict[str, str] = {}

    if password:
        salt = secrets.token_hex(_SALT_BYTES)
        kek = _derive(password, salt)
        pending["pwd_salt"] = salt
        pending["dek_by_pwd"] = kek.encrypt(raw).decode()
    if pp:
        salt = secrets.token_hex(_SALT_BYTES)
        kek = _derive(pp, salt)
        pending["key_salt"] = salt
        pending["key_check"] = kek.encrypt(_CHECK_PLAIN).decode()
        pending["dek_by_pp"] = kek.encrypt(raw).decode()

    _PENDING[user_id] = pending
    _DEK_RAW[user_id] = raw
    return _fernet(raw)


def _take_pending(user_id: int) -> dict[str, str]:
    """取出并清空「待落库的盐与 DEK 密文」（随本次 save 一起写进表里）。"""
    return _PENDING.pop(user_id, {})


async def _ensure_dek(user_id: int, password: str, passphrase: str) -> Any:
    """确保拿到该用户的 DEK（Fernet 实例）；失败抛 ValueError。

    三条分支，与「加密落盘 + 登录自动解锁」的语义一一对应：
      ① 已解锁 → 直接用；若还缺登录密码通道而本轮给了密码，顺手补建；
      ② 有历史密文但本进程未解锁 → 用密码或口令解锁，凭证错则**拒绝**
         （用错误密钥写入会把原数据变成永久解不开的密文）；
      ③ 全新用户 → 建立 DEK 与解锁通道。

    抽成函数是为了让**云端 LLM 与 TTS 两条保存路径共用同一套逻辑** —— 这类分支
    各写一份时，改一处忘另一处几乎是必然的。
    """
    row = await _fetch(user_id)
    f = _dek(user_id)
    if f is not None:
        if password and not (row or {}).get("dek_by_pwd"):
            # 已解锁但缺登录密码通道（例如当初只用加密口令建的）→ 顺手补建，
            # 之后登录即可自动解锁，不必再手输口令。
            await bind_password(user_id, password)
        return f
    if row is not None and (row.get("dek_by_pwd") or row.get("dek_by_pp")):
        if not password and not passphrase:
            # 区分「没填」与「填错」：前者是操作没做完，后者是凭据不对，用户要做的事完全不同
            raise ValueError("请填写登录密码或加密口令来解锁已保存的密钥")
        if not await unlock(user_id, password=password, passphrase=passphrase):
            raise ValueError("登录密码或加密口令不正确")
        return _dek(user_id)
    return _bootstrap(user_id, password=password, passphrase=passphrase)


async def rebind_password(user_id: int, old_password: str, new_password: str) -> bool:
    """改密码后重绑密码通道：旧 KEK 解出 DEK，新 KEK 重新加密。

    必须在 `users.pw_hash` 更新**之前**调用（要用旧密码解）。
    返回 False 表示无需重绑（未配置云模型 / 只启用了口令通道）—— 这不是错误，
    调用方不应因此中断改密码流程。
    """
    row = await _fetch(user_id)
    if row is None or not (row.get("pwd_salt") and row.get("dek_by_pwd")):
        return False
    raw = _DEK_RAW.get(user_id)
    if raw is None:
        try:
            raw = _derive(old_password, row["pwd_salt"]).decrypt(
                row["dek_by_pwd"].encode())
        except Exception as exc:
            log.warning("user=%s 改密码时旧密码解不开 DEK（需重新登录或用加密口令解锁）：%s",
                        user_id, type(exc).__name__)
            return False
    salt = secrets.token_hex(_SALT_BYTES)
    enc = _derive(new_password, salt).encrypt(raw).decode()
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE user_llm_configs SET dek_by_pwd = %s, pwd_salt = %s, "
            "updated_at = now() WHERE user_id = %s",
            (enc, salt, user_id),
        )
    _DEK_RAW[user_id] = raw
    return True


async def bind_password(user_id: int, password: str) -> bool:
    """为已解锁的用户**补建**登录密码通道（原本只有加密口令，现在想自动解锁）。"""
    raw = _DEK_RAW.get(user_id)
    if raw is None:
        return False
    salt = secrets.token_hex(_SALT_BYTES)
    enc = _derive(password, salt).encrypt(raw).decode()
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE user_llm_configs SET dek_by_pwd = %s, pwd_salt = %s, "
            "updated_at = now() WHERE user_id = %s",
            (enc, salt, user_id),
        )
    return True


def _mask(key: str) -> str:
    """生成掩码：`sk-a****wxyz`。短 key 全掩。"""
    k = (key or "").strip()
    if len(k) <= _HINT_HEAD + _HINT_TAIL + 1:
        return "*" * len(k)
    return f"{k[:_HINT_HEAD]}****{k[-_HINT_TAIL:]}"


def fingerprint(key: str) -> str:
    """key 指纹（sha256 前 8 位）：仅用于日志定位，不可逆推。"""
    return hashlib.sha256((key or "").encode()).hexdigest()[:8]


# ---------- 语音合成（TTS）凭据 ----------
# ⚠️ 安全要点：`workspace_id` 会被**拼进请求 URL 的 host 段**
# （`https://{workspace_id}.cn-beijing.maas.aliyuncs.com/...`）。用户可填意味着
# 必须防注入：只允许字母/数字/连字符，`.` `/` `@` `:` 等一律拒绝——否则填
# `evil.com/x` 就能把服务端请求引到任意主机（SSRF），性质与 base_url 同类。
_RE_WORKSPACE_ID = re.compile(r"^[A-Za-z0-9-]{1,64}$")
# 音色 / 模型 id 只进 JSON 请求体，不参与 URL 拼接，但仍限字符与长度，
# 避免用超长串或换行撑爆请求体。
_RE_TTS_TOKEN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def validate_workspace_id(wid: str) -> str:
    """校验业务空间 ID（会被拼进 URL host）。不合法抛 ValueError。"""
    w = (wid or "").strip()
    if not w:
        raise ValueError("业务空间 ID 不能为空")
    if not _RE_WORKSPACE_ID.match(w):
        raise ValueError("业务空间 ID 只能包含字母、数字与连字符")
    return w


def _validate_tts_token(value: str, name: str) -> str:
    v = (value or "").strip()
    if not v:
        return ""
    if not _RE_TTS_TOKEN.match(v):
        raise ValueError(f"{name} 只能包含字母、数字、点、下划线与连字符")
    return v


async def ensure_schema() -> None:
    """幂等建表。API lifespan 里随 authdb.ensure_schema 一起调。"""
    pool = await get_pool()
    async with pool.connection() as conn:
        for stmt in _DDL:
            await conn.execute(stmt)


async def get_config_masked(user_id: int) -> dict[str, Any]:
    """读云端模型配置（**只返回掩码，绝不返回明文**）。

    - `unlocked`：本进程内是否已解锁；False 时云端模型不会生效（回落本地默认）。
    - `auto_unlock`：是否已绑定登录密码通道 —— 为真表示**下次登录会自动连上**，
      当前未解锁只是因为进程重启，重新登录即可（界面据此给出不同提示）。
    - `pp_bound`：是否已绑定加密口令通道（独立于登录密码的兜底通道）。

    ⚠️ `configured` 按 `api_key_enc` 是否非空判断，**不能只看行是否存在**：这张表
    现在同时承载 TTS 凭据，只配了语音的用户也有行、但并没有云端模型。
    """
    row = await _fetch(user_id)
    if row is None:
        return {"configured": False, "enabled": False, "provider": "", "base_url": "",
                "model": "", "key_hint": "", "emotion_enabled": False,
                "crypto_ready": crypto_ready(), "unlocked": False,
                "auto_unlock": False, "pp_bound": False}
    return {
        "configured": bool(row["api_key_enc"]),
        "enabled": bool(row["enabled"]),
        "provider": row["provider"],
        "base_url": row["base_url"],
        "model": row["model"],
        "key_hint": row["key_hint"] or _mask(""),
        # 情绪判定是否由该用户的云端模型兼任（false = 走本地 qwen3:8b）
        "emotion_enabled": bool(row.get("emotion_enabled", False)),
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        "crypto_ready": crypto_ready(),
        "unlocked": is_unlocked(user_id),
        "auto_unlock": bool(row.get("pwd_salt") and row.get("dek_by_pwd")),
        "pp_bound": bool(row.get("key_salt") and row.get("dek_by_pp")),
    }


async def _fetch(user_id: int) -> dict | None:
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT provider, base_url, model, api_key_enc, key_hint, enabled, "
            "emotion_enabled, key_salt, key_check, pwd_salt, dek_by_pwd, dek_by_pp, "
            "tts_api_key_enc, tts_key_hint, tts_workspace_id, tts_model, tts_voice, "
            "tts_instruction, "
            "updated_at"
            " FROM user_llm_configs WHERE user_id = %s", (user_id,)
        )
        return await cur.fetchone()


async def get_runtime(user_id: int) -> dict | None:
    """取**可调用**的运行时配置（含解密后的明文 key，仅在内存中短暂存在）。

    返回 None 表示应回落本地默认模型：未解锁 / 未配置 / 未启用 / 解密失败。
    明文 key 只传入 ChatOpenAI 实例，不写日志、不进 state、不返回给前端。
    """
    f = _dek(user_id)
    if f is None:
        return None                     # 未解锁：静默回落本地模型（正常态，不打日志刷屏）
    row = await _fetch(user_id)
    if row is None or not row["enabled"]:
        return None
    # ⚠️ 必须先判空再解密：该行可能只承载了 TTS 凭据（用户没配云端模型），
    # 直接 decrypt("") 会抛异常、走进下面的 except 打出误导性的「解密失败」日志。
    if not row["api_key_enc"]:
        return None
    try:
        key = f.decrypt(row["api_key_enc"].encode()).decode()
    except Exception as exc:
        log.warning("user=%s 云端 key 解密失败（需重新填写）：%s",
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
                      emotion_enabled: bool = False, passphrase: str = "",
                      password: str = "") -> dict:
    """写入/更新配置。api_key 传空串表示**保留原 key 不改**（前端只改了 url/model 时）。

    `password` / `passphrase`：本进程尚未解锁时至少给一个 —— 首次用它建立 DEK 与
    解锁通道，之后用它解锁。已解锁时可留空；已解锁但还没绑定登录密码通道时，
    传 `password` 会**补建**该通道（之后登录就能自动连上）。

    校验失败抛 ValueError（由 API 层转 400），不静默吞。
    """
    base_url = (base_url or "").strip().rstrip("/")
    model = (model or "").strip()
    api_key = (api_key or "").strip()

    if not crypto_ready():
        raise ValueError("服务端缺少 cryptography 依赖，云模型功能已关闭（不会明文存储密钥）")
    validate_base_url(base_url)     # 含 http(s) / 长度 / 元数据端点 / 私网策略
    if not model:
        raise ValueError("model 不能为空")
    if len(model) > MAX_TEXT_LEN:
        raise ValueError(f"model 长度不得超过 {MAX_TEXT_LEN}")

    row = await _fetch(user_id)
    # 未解锁时用登录密码 / 加密口令建立或解锁 DEK，凭证错一律拒绝（见 _ensure_dek）
    f = await _ensure_dek(user_id, password, passphrase)

    if api_key:
        if len(api_key) > MAX_KEY_LEN:
            raise ValueError(f"api_key 过长（>{MAX_KEY_LEN}），疑似粘贴了多余内容")
        enc = f.encrypt(api_key.encode()).decode()
        hint = _mask(api_key)
    elif row and row["api_key_enc"]:
        enc, hint = row["api_key_enc"], row["key_hint"]   # 保留原 key
    else:
        raise ValueError("api_key 不能为空")

    # 首次建立时盐与 DEK 密文在暂存区；其余沿用库里的旧值（换凭证走 change_*）
    pending = _take_pending(user_id)
    prev = row or {}

    def _col(name: str) -> str:
        return pending.get(name) or prev.get(name) or ""

    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO user_llm_configs
                (user_id, provider, base_url, model, api_key_enc, key_hint, enabled,
                 emotion_enabled, key_salt, key_check, pwd_salt, dek_by_pwd, dek_by_pp,
                 updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (user_id) DO UPDATE SET
                provider = EXCLUDED.provider, base_url = EXCLUDED.base_url,
                model = EXCLUDED.model, api_key_enc = EXCLUDED.api_key_enc,
                key_hint = EXCLUDED.key_hint, enabled = EXCLUDED.enabled,
                emotion_enabled = EXCLUDED.emotion_enabled,
                key_salt = EXCLUDED.key_salt, key_check = EXCLUDED.key_check,
                pwd_salt = EXCLUDED.pwd_salt, dek_by_pwd = EXCLUDED.dek_by_pwd,
                dek_by_pp = EXCLUDED.dek_by_pp, updated_at = now()
            """,
            (user_id, provider, base_url, model, enc, hint, enabled, emotion_enabled,
             _col("key_salt"), _col("key_check"), _col("pwd_salt"),
             _col("dek_by_pwd"), _col("dek_by_pp")),
        )
    log.info("user=%s 保存云端模型配置 base_url=%s model=%s key=%s",
             user_id, base_url, model, hint)          # 只记掩码，不记明文
    return {"configured": True, "enabled": enabled, "provider": provider,
            "base_url": base_url, "model": model, "key_hint": hint,
            "unlocked": True,
            "auto_unlock": bool(_col("pwd_salt") and _col("dek_by_pwd"))}


async def change_passphrase(user_id: int, old: str, new: str) -> None:
    """换加密口令：用**原口令**解出 DEK，再用新口令重新加密 `dek_by_pp`（换盐）。

    `api_key_enc` **不用动** —— 它始终由同一把 DEK 加密，这正是双层密钥的好处：
    换口令 / 换密码都不必重新加密数据本身。失败抛 ValueError（API 层转 400）。
    """
    raw = _DEK_RAW.get(user_id)
    if raw is None:
        if not await unlock_with_passphrase(user_id, old):
            raise ValueError("原口令不正确（或尚未设置加密口令）")
        raw = _DEK_RAW[user_id]

    np = (new or "").strip()
    if len(np) < MIN_PASSPHRASE:
        raise ValueError(f"新口令至少 {MIN_PASSPHRASE} 位")
    salt = secrets.token_hex(_SALT_BYTES)
    kek = _derive(np, salt)
    enc = kek.encrypt(raw).decode()
    check = kek.encrypt(_CHECK_PLAIN).decode()

    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE user_llm_configs SET dek_by_pp = %s, key_salt = %s, "
            "key_check = %s, updated_at = now() WHERE user_id = %s",
            (enc, salt, check, user_id),
        )
    log.info("user=%s 已更换云端配置的加密口令", user_id)


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
    """删除配置（含密文），之后回落本地默认模型；同时丢弃内存里的派生密钥。"""
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "DELETE FROM user_llm_configs WHERE user_id = %s", (user_id,)
        )
    if cur.rowcount:
        lock(user_id)
        log.info("user=%s 已删除云端模型配置", user_id)


# ---------- 语音合成（TTS）凭据：加密落盘 + 与云端 LLM 共用同一把 DEK ----------
# 设计要点：**不新增密钥体系**。用户的 TTS key 与云端 LLM key 由同一把 DEK 加密，
# 因此「登录即自动解锁」「改密码自动重绑」「加密口令兜底」三条能力全部自动继承，
# 用户只需维护一套口令。见模块 docstring 与 _DDL 里的说明。

async def get_tts_masked(user_id: int) -> dict[str, Any]:
    """读 TTS 配置（**只返回掩码**；workspace/model/voice 是标识不是密钥，可回显）。

    `configured` 按 `tts_api_key_enc` 是否非空判断——与 `get_config_masked` 同理，
    不能只看行是否存在（这张表同时承载云端 LLM 凭据）。
    """
    row = await _fetch(user_id)
    if row is None:
        return {"configured": False, "key_hint": "",
                "workspace_id": "", "model": "", "voice": "", "instruction": "",
                "crypto_ready": crypto_ready(), "unlocked": False,
                "auto_unlock": False, "pp_bound": False}
    return {
        "configured": bool(row["tts_api_key_enc"]),
        "key_hint": row.get("tts_key_hint") or "",
        "workspace_id": row.get("tts_workspace_id") or "",
        "model": row.get("tts_model") or "",
        "voice": row.get("tts_voice") or "",
        "instruction": row.get("tts_instruction") or "",
        "crypto_ready": crypto_ready(),
        "unlocked": is_unlocked(user_id),
        "auto_unlock": bool(row.get("pwd_salt") and row.get("dek_by_pwd")),
        "pp_bound": bool(row.get("key_salt") and row.get("dek_by_pp")),
    }


async def get_tts_runtime(user_id: int) -> dict | None:
    """取该用户**可调用**的 TTS 配置（含解密后的明文 key，仅内存中短暂存在）。

    返回 None 表示该用户未配置 / 未解锁 —— 调用方据此回落全局 `.env` 配置
    （自部署者可以统一配一份，给所有用户兜底），再不行才判定为不可用。
    明文 key 只传进 httpx 请求头，不写日志、不进 state、不返回前端。
    """
    f = _dek(user_id)
    if f is None:
        return None
    row = await _fetch(user_id)
    if row is None:
        return None
    if not row["tts_api_key_enc"]:
        return None                     # 只配了云端 LLM、没配语音
    try:
        key = f.decrypt(row["tts_api_key_enc"].encode()).decode()
    except Exception as exc:
        log.warning("user=%s TTS key 解密失败（需重新填写）：%s", user_id, type(exc).__name__)
        return None
    if not key.strip():
        return None
    return {
        "api_key": key,
        "workspace_id": row.get("tts_workspace_id") or "",
        "model": row.get("tts_model") or "",
        "voice": row.get("tts_voice") or "",
        "instruction": row.get("tts_instruction") or "",
    }


async def save_tts_config(user_id: int, *, api_key: str, workspace_id: str,
                          model: str = "", voice: str = "", instruction: str = "",
                          passphrase: str = "", password: str = "") -> dict:
    """写入/更新该用户的语音合成凭据。api_key 传空串表示**保留原 key 不改**。

    **整体替换语义**：`workspace_id` / `model` / `voice` / `instruction` 以本次提交为准
    （留空即回落代码默认值，见 rag/tts.py 的 `_fill`）；只有 `api_key` 留空是「保留」，
    因为明文 key 永不回显，前端没法把它填回输入框。

    `workspace_id` 会被拼进请求 URL 的 host 段，故走 `validate_workspace_id`
    严格校验（防 SSRF）；`model`/`voice`/`instruction` 只进请求体，做字符与长度限制。
    未解锁时 `password` / `passphrase` 至少给一个（与 `save_config` 同一套语义）。
    校验失败抛 ValueError（API 层转 400）。
    """
    api_key = (api_key or "").strip()
    ws = validate_workspace_id(workspace_id)
    model = _validate_tts_token(model, "模型名") if model else ""
    voice = _validate_tts_token(voice, "音色名") if voice else ""
    # 指令可能含中文与标点，不走 token 正则，只限长度
    instruction = (instruction or "").strip()
    if len(instruction) > MAX_TEXT_LEN:
        raise ValueError(f"指令长度不得超过 {MAX_TEXT_LEN}")

    if not crypto_ready():
        raise ValueError("服务端缺少 cryptography 依赖，语音配置已禁用（不会明文存储密钥）")

    row = await _fetch(user_id)
    f = await _ensure_dek(user_id, password, passphrase)   # 共用同一套建/解锁逻辑

    if api_key:
        if len(api_key) > MAX_KEY_LEN:
            raise ValueError(f"api_key 过长（>{MAX_KEY_LEN}），疑似粘贴了多余内容")
        enc = f.encrypt(api_key.encode()).decode()
        hint = _mask(api_key)
    elif row and row["tts_api_key_enc"]:
        enc, hint = row["tts_api_key_enc"], row["tts_key_hint"]   # 保留原 key
    else:
        raise ValueError("api_key 不能为空")

    # 首次建立 DEK 时，盐与 DEK 密文在暂存区；其余沿用库里的旧值
    pending = _take_pending(user_id)
    prev = row or {}

    def _col(name: str) -> str:
        return pending.get(name) or prev.get(name) or ""

    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO user_llm_configs
                (user_id, tts_api_key_enc, tts_key_hint, tts_workspace_id, tts_model,
                 tts_voice, tts_instruction,
                 key_salt, key_check, pwd_salt, dek_by_pwd, dek_by_pp, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (user_id) DO UPDATE SET
                tts_api_key_enc = EXCLUDED.tts_api_key_enc,
                tts_key_hint = EXCLUDED.tts_key_hint,
                tts_workspace_id = EXCLUDED.tts_workspace_id,
                tts_model = EXCLUDED.tts_model,
                tts_voice = EXCLUDED.tts_voice,
                tts_instruction = EXCLUDED.tts_instruction,
                key_salt = EXCLUDED.key_salt, key_check = EXCLUDED.key_check,
                pwd_salt = EXCLUDED.pwd_salt, dek_by_pwd = EXCLUDED.dek_by_pwd,
                dek_by_pp = EXCLUDED.dek_by_pp, updated_at = now()
            """,
            (user_id, enc, hint, ws, model, voice, instruction,
             _col("key_salt"), _col("key_check"), _col("pwd_salt"),
             _col("dek_by_pwd"), _col("dek_by_pp")),
        )
    log.info("user=%s 保存语音配置 workspace=%s model=%s voice=%s key=%s",
             user_id, ws, model or "-", voice or "-", hint)      # 只记掩码
    return await get_tts_masked(user_id)


async def delete_tts_config(user_id: int) -> bool:
    """删除语音凭据（含密文）。

    只清 TTS 字段而**不删整行**——同一行可能还承载着云端 LLM 凭据。仅当该行
    已无任何凭据（`api_key_enc` 也为空）时才整行删除并丢弃内存 DEK，
    避免留下无法解锁的空壳行。
    """
    row = await _fetch(user_id)
    if row is None or not row["tts_api_key_enc"]:
        return False
    pool = await get_pool()
    if not row["api_key_enc"]:
        # 没有任何其他凭据了 → 整行删掉并上锁
        async with pool.connection() as conn:
            await conn.execute("DELETE FROM user_llm_configs WHERE user_id = %s", (user_id,))
        lock(user_id)
    else:
        async with pool.connection() as conn:
            await conn.execute(
                "UPDATE user_llm_configs SET tts_api_key_enc = '', tts_key_hint = '', "
                "tts_workspace_id = '', tts_model = '', tts_voice = '', "
                "tts_instruction = '', updated_at = now() "
                "WHERE user_id = %s",
                (user_id,),
            )
    log.info("user=%s 已删除语音配置", user_id)
    return True
