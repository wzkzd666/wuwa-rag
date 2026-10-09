"""注册 / 登录 / token 校验（零第三方依赖：scrypt + secrets，不上 bcrypt 不上 JWT）。

设计取舍（保持简单）：
- 口令哈希用 stdlib hashlib.scrypt（自带盐），格式 `scrypt$<salt hex>$<hash hex>`；
  **实现已下沉到 `wuwa_rag.core.security`**——`core.authdb` 种子管理员时也要算哈希，
  留在本层会让 core 反向依赖 api（层次倒置，且容易成环）；
- token 是 secrets.token_hex(32)（256bit），存 auth_tokens 表、30 天过期——
  服务重启不丢登录态（内存 dict 方案重启全员掉线，pass）；
- 请求带 `Authorization: Bearer <token>`，FastAPI 依赖里解析。

与 rag/llmstore.py 的协作（用户云端 API-KEY 的加密存储）：
- **登录即自动解锁**：登录请求带明文密码，这里顺手派生 KEK 解开该用户的 DEK
  放进内存，下次登录云端模型直接生效（用户无需额外输入任何口令）。
- **改密码必须重绑**：DEK 的密码通道由密码派生，改密码后旧密文立即失效，
  `change_password` 因此**先用旧密码重绑、再更新 pw_hash**（顺序颠倒会让用户
  把自己的 Key 永久锁死）。
- 两者都只做"尽力而为"：解锁/重绑失败只记日志，**绝不影响登录本身**。
"""
from __future__ import annotations

import secrets
from datetime import UTC, timedelta
from datetime import datetime as dt

from fastapi import Depends, HTTPException, Request
from pydantic import BaseModel

from wuwa_rag.core.authdb import get_pool
from wuwa_rag.core.llmstore import rebind_password, unlock_with_password
from wuwa_rag.core.security import hash_password, verify_password
from wuwa_rag.ww_logger import get_logger

log = get_logger("auth")

TOKEN_TTL_DAYS = 30
_ERR_UNAUTHORIZED = HTTPException(status_code=401, detail="未登录或登录已过期")


# ---------- 注册 / 登录 / token ----------

class AuthError(HTTPException):
    pass


async def register(username: str, password: str) -> dict:
    """注册游客账号。用户名重复 → 400。成功后直接发 token（注册即登录）。"""
    username = username.strip()
    if not (2 <= len(username) <= 24):
        raise AuthError(status_code=400, detail="用户名需 2~24 个字符")
    if len(password) < 6:
        raise AuthError(status_code=400, detail="密码至少 6 位")
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT 1 FROM users WHERE username = %s", (username,)
        )
        if await cur.fetchone() is not None:
            raise AuthError(status_code=400, detail=f"用户名「{username}」已被占用")
        cur = await conn.execute(
            "INSERT INTO users (username, pw_hash, role) VALUES (%s, %s, 'guest') RETURNING id, username, role",
            (username, hash_password(password)),
        )
        user = await cur.fetchone()
        assert user is not None
    log.info("注册新用户 %s", username)
    return {**user, "token": await create_token(user["id"])}


async def login(username: str, password: str) -> dict:
    """登录。成功发新 token；失败一律「用户名或密码错误」（不泄露哪个错）。

    响应带 `must_change_password`：为 true 时前端引导改密（见 AuthPage 的 toast）。
    它对应「部署者用 ADMIN_PASSWORD 显式配了初始口令」那一路——初始口令属于配置文件
    里的明文，不该长期使用。
    """
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT id, username, role, pw_hash, must_change FROM users WHERE username = %s",
            (username.strip(),),
        )
        row = await cur.fetchone()
        if row is None or not verify_password(password, row["pw_hash"]):
            raise AuthError(status_code=401, detail="用户名或密码错误")

    # 登录即解锁该用户的云端模型密钥（DEK），实现"下次登录自动连接"。
    # 失败只记日志：尚未配置云模型 / 只用加密口令通道 / 密码与建密钥时不一致，
    # 都不是登录失败的理由。
    try:
        await unlock_with_password(row["id"], password)
    except Exception as exc:  # noqa: BLE001 —— 解锁失败不影响登录本身
        log.warning("user=%s 登录时自动解锁云端密钥失败（忽略）：%s", row["id"], exc)
    return {"id": row["id"], "username": row["username"], "role": row["role"],
            "must_change_password": bool(row.get("must_change")),
            "token": await create_token(row["id"])}


async def change_password(user_id: int, old: str, new: str) -> None:
    """改密码。**顺序敏感**：先用旧密码重绑云端密钥，再更新 pw_hash。

    重绑失败不中断改密码（用户还能用加密口令通道解锁），但要留日志——
    否则用户会困惑"为什么改完密码云端模型不生效了"。
    """
    if len(new) < 6:
        raise AuthError(status_code=400, detail="新密码至少 6 位")
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute("SELECT pw_hash FROM users WHERE id = %s", (user_id,))
        row = await cur.fetchone()
        if row is None or not verify_password(old, row["pw_hash"]):
            raise AuthError(status_code=400, detail="原密码不正确")
        try:
            await rebind_password(user_id, old, new)
        except Exception as exc:  # noqa: BLE001 —— 重绑失败不影响改密码本身
            log.warning("user=%s 改密码时重绑云端密钥失败（改用加密口令解锁即可）：%s",
                        user_id, exc)
        await conn.execute(
            "UPDATE users SET pw_hash = %s, must_change = FALSE WHERE id = %s",
            (hash_password(new), user_id),
        )
    log.info("user=%s 已修改密码", user_id)


async def create_token(user_id: int) -> str:
    token = secrets.token_hex(32)
    expires = dt.now(UTC) + timedelta(days=TOKEN_TTL_DAYS)
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            "INSERT INTO auth_tokens (token, user_id, expires_at) VALUES (%s, %s, %s)",
            (token, user_id, expires),
        )
    return token


async def resolve_token(token: str) -> dict | None:
    """token → {id, username, role}；过期/不存在 → None。顺带惰性删过期行。"""
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            SELECT u.id, u.username, u.role, t.expires_at
            FROM auth_tokens t JOIN users u ON u.id = t.user_id
            WHERE t.token = %s
            """,
            (token,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        if row["expires_at"] < dt.now(UTC):
            await conn.execute("DELETE FROM auth_tokens WHERE token = %s", (token,))
            return None
    return {"id": row["id"], "username": row["username"], "role": row["role"]}


async def revoke_token(token: str) -> None:
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute("DELETE FROM auth_tokens WHERE token = %s", (token,))


# ---------- FastAPI 依赖 ----------

class AuthUser(BaseModel):
    """依赖注入的当前用户（Pydantic 模型：FastAPI 才能拿它建依赖/响应字段）。"""

    id: int
    username: str
    role: str
    token: str

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


async def get_current_user(request: Request) -> AuthUser:
    """从 Authorization: Bearer <token> 解析当前用户；缺失/无效 → 401。"""
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise _ERR_UNAUTHORIZED
    token = auth[7:].strip()
    if not token:
        raise _ERR_UNAUTHORIZED
    user = await resolve_token(token)
    if user is None:
        raise _ERR_UNAUTHORIZED
    auth_user = AuthUser(**user, token=token)
    # 把主体身份挂到 request.state，供限流按用户计数（见 api/ratelimit.py）。
    # ⚠️ 必须写在这里而不是端点函数里：slowapi 的限额装饰器包在端点外层，
    # 而 FastAPI 的依赖在端点之前解析，写在这儿装饰器才取得到。
    request.state.user_id = auth_user.id
    return auth_user


async def require_admin(user: AuthUser = Depends(get_current_user)) -> AuthUser:
    """管理员守卫：挂在 admin 专用端点上（先过 get_current_user 再查角色）。

    依赖参数必须写成 `= Depends(get_current_user)`：省略时 FastAPI 会把 AuthUser
    当成**请求体字段**解析，端点直接 422（实测）。
    """
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user
