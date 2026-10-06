"""口令哈希（项目最底层：零第三方依赖、零项目内依赖）。

为什么单独拆一个模块，而不是留在 `api/auth.py`
------------------------------------------------
`core.authdb.ensure_schema()` 要种子管理员账号，也得算一次口令哈希；如果继续
从 `api.auth` 导入，就形成 `core → api` 的**反向依赖**——`api` 是最外层，
`core` 是内核，方向反了，且一旦 api 层再 import 别的东西就容易成环。

把这两个纯函数下沉到 `core`，两边都从 `core.security` 拿，依赖方向就正了，
原先那句「延迟导入避免环（auth → db）」也可以退休。

格式
----
`scrypt$<salt hex>$<hash hex>`，salt 为 16 字节随机值，`n=2**14, r=8, p=1`。
校验用 `hmac.compare_digest` 做**常量时间比较**，避免按字节提前返回泄露信息。
"""
from __future__ import annotations

import hashlib
import hmac
import secrets


def hash_password(password: str) -> str:
    """生成 `scrypt$<salt hex>$<hash hex>` 形式的口令摘要（每次调用盐都不同）。"""
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt${salt.hex()}${h.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """格式不对/参数不对一律返回 False，不抛异常（老数据兼容）。

    只吞「stored 字符串格式不合法」这类可预期异常：
      - `ValueError`：`split("$")` 解包数量不对（不是 3 段）、`bytes.fromhex` 遇非法十六进制；
      - `AttributeError` / `TypeError`：老数据里 stored 为 None 或非字符串。
    其余异常（如 hashlib 本身故障）属于编程/环境错误，应该冒出来而不是被当成「密码错」。
    """
    try:
        algo, salt_hex, hash_hex = stored.split("$")
        if algo != "scrypt":
            return False
        h = hashlib.scrypt(
            password.encode("utf-8"), salt=bytes.fromhex(salt_hex), n=2**14, r=8, p=1
        )
        return hmac.compare_digest(h.hex(), hash_hex)
    except (ValueError, AttributeError, TypeError):
        return False
