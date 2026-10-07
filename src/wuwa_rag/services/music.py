r"""音乐播放：QQ音乐 MCP server（tools/qqmusic_mcp）的 stdio 客户端封装。

为什么走 MCP 而不是直接 import
------------------------------
`tools/qqmusic_mcp/qqmusic_local_mcp.py` 本身就是 stdio MCP server（用 FastMCP 写的），
按 MCP 规范以**子进程 + stdio** 通信。好处是彼此解耦：server 可以单独挂到任何 MCP 客户端
（Claude Desktop / Cursor）上调试，项目这边只是它的一个 Host；server 崩了或改版都不影响
主链路 —— 调用失败一律回落成一句人话，不阻断问答。

生命周期
--------
子进程**懒启动**（第一次真要放歌时才拉），全局只活一个实例，进程内复用；
用 `AsyncExitStack` 持有会话，`aclose()` 显式收尾（不依赖 GC —— 官方文档明确要求
保存引用，否则可能在运行途中被回收、静默失效）。

开关：`config.MUSIC_ENABLED` 默认 **False**。没开 / 没装 QQ音乐 / 没装 music 依赖时
一律「不可用」，绝不抛异常打断问答。
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from wuwa_rag.config import get_settings
from wuwa_rag.ww_logger import get_logger

log = get_logger("music")

_SERVER = Path(__file__).resolve().parents[3] / "tools" / "qqmusic_mcp" / "qqmusic_local_mcp.py"

# 工具调用预算：本地 stdio + 两次 HTTP 搜索，10 秒足够；超了就当失败，不挂住问答
_CALL_TIMEOUT = 20.0
_lock = asyncio.Lock()
_stack: AsyncExitStack | None = None
_session: Any | None = None
# MCP 会话是**全局单例**（与用户无关），但「能不能用」依赖当前用户的开关。
# 用 contextvar 把当前请求的用户带进同步的 available()。
_USER_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("music_user", default=None)


@contextlib.contextmanager
def as_user(user_id: int | str | None):
    """把当前用户标到这个上下文里（API 层每个入口包一次）。"""
    token = _USER_ID.set(str(user_id) if user_id is not None else None)
    try:
        yield
    finally:
        _USER_ID.reset(token)


class MusicUnavailable(RuntimeError):
    """音乐功能不可用（未启用 / 缺依赖 / 找不到 server / 客户端没跑）。"""


# 设置项名（存 user_settings 表）
SETTING_KEY = "music_enabled"
EXE_KEY = "music_exe"          # 用户自定义的 QQMusic.exe 路径


async def is_enabled(user_id: int | str | None = None) -> bool:
    """这个用户的音乐开关是开还是关。

    **两个来源任一为开就算开**：
    - `user_settings.music_enabled`（设置页里改，**立即生效，不用重启**）；
    - `config.MUSIC_ENABLED`（.env 里的**部署级默认**，给还没开过设置的人用）。

    取「或」而不是「个人设置覆盖部署默认」：这是**本机能力**而不是账号数据 ——
    机器上装了 QQ音乐就能放，个人设置只当他想显式关掉时用。
    """
    if get_settings().MUSIC_ENABLED:
        return True
    if user_id is None:
        return False
    try:
        from wuwa_rag.core.settings import get_setting

        return bool(await get_setting(str(user_id), SETTING_KEY, False))
    except Exception as exc:  # noqa: BLE001 —— 读不到设置就当没开，不能因此报错
        log.warning("读音乐开关失败（按关闭处理）: %s", exc)
        return False


async def set_enabled(user_id: int | str, enabled: bool | None) -> None:
    """设置页开关。`None` 表示恢复部署默认（删掉这条设置）。"""
    await _put(user_id, SETTING_KEY, None if enabled is None else bool(enabled))


async def get_exe(user_id: int | str | None = None) -> str:
    """QQMusic.exe 路径：**个人设置 > .env 的 MUSIC_EXE > 环境变量 QQMUSIC_EXE**。

    留空的语义是「**用环境变量**」，不是「我不设就自己找」—— 路径是机器/部署级的事，
    应当由环境变量声明；因此留空时**不覆盖**子进程继承到的 QQMUSIC_EXE。
    """
    if user_id is not None:
        try:
            v = await _get(str(user_id), EXE_KEY, "")
            if v:
                return str(v)
        except Exception as exc:  # noqa: BLE001
            log.warning("读音乐路径设置失败（回落 .env）: %s", exc)
    return get_settings().MUSIC_EXE


async def set_exe(user_id: int | str, path: str | None) -> None:
    """设置自定义播放器路径。改完要**重启 MCP 会话**才会用新路径（env 是启动时注入的）。"""
    value = (path or "").strip()
    await _put(user_id, EXE_KEY, value or None)
    # 会话是单例且 env 已固化，只能重建才能让新路径生效
    async with _lock:
        await _close_stack()


async def _get(user_id: str, key: str, default=None):
    from wuwa_rag.core.settings import get_setting

    return await get_setting(user_id, key, default)


async def _put(user_id: int | str, key: str, value) -> None:
    from wuwa_rag.core.settings import delete_setting, set_setting

    if value is None:
        await delete_setting(str(user_id), key)
    else:
        await set_setting(str(user_id), key, value)


def available() -> tuple[bool, str]:
    """同步可用性判定：**只看部署默认**（.env 的 MUSIC_ENABLED）与依赖是否齐。

    ⚠️ 这里**故意不接 user_id**：个人设置在 `user_settings` 表里，是**异步**查询，
    同步函数里查不了。曾经的写法是「传了 user_id 就跳过开关检查」，那是个**真 bug** ——
    结果是所有带 user_id 的调用（`now_playing(user.id)` 等）都认为「可用」并真的去
    拉起 MCP 子进程，在某些环境下直接挂死。带个人设置的判定一律走 `available_async`。
    """
    s = get_settings()
    if not s.MUSIC_ENABLED:
        return False, "音乐功能未启用（可在设置页「音乐播放」里打开）"
    if not _SERVER.is_file():
        return False, f"找不到 MCP server：{_SERVER}"
    try:
        import mcp  # noqa: F401
    except ImportError:
        return False, "缺 mcp 依赖（uv sync --extra music）"
    return True, ""


async def available_async(user_id: int | str | None = None) -> tuple[bool, str]:
    """真正可用性判定（含个人开关）。"""
    if await is_enabled(user_id):
        ok, why = available()
        return (True, "") if ok else (True, why)
    return available()


async def _ensure_session() -> Any:
    """懒启动并复用 MCP 会话。已在启动中则排队等它（锁粒度是「建会话」）。"""
    global _stack, _session
    if _session is not None:
        return _session
    ok, why = available()                   # 同步版只看部署默认；个人设置在 available_async
    if not ok:
        raise MusicUnavailable(why)
    async with _lock:
        if _session is not None:
            return _session
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise MusicUnavailable(f"缺 mcp 依赖（uv sync --extra music）：{exc}") from exc
        env = dict(os.environ)
        # 个人设置 > 环境变量：留空**不是**「不许用」，而是「不覆盖」——
        # 子进程默认就继承本进程 env，QQMUSIC_EXE 会照常生效。
        # （曾经在这里 pop 掉它是反的：那会让用户设了环境变量却完全不生效。）
        exe = await get_exe(_USER_ID.get())
        if exe:
            env["QQMUSIC_EXE"] = exe
        params = StdioServerParameters(command=sys.executable,
                                       args=["-X", "utf8", str(_SERVER)],
                                       env=env)
        try:
            _stack = AsyncExitStack()
            read, write = await _stack.enter_async_context(stdio_client(params))
            session = await _stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except Exception as exc:  # noqa: BLE001 —— 起不来就如实说，不打断问答
            await _close_stack()
            raise MusicUnavailable(f"启动 QQ音乐 MCP server 失败：{exc}") from exc
        _session = session
        log.info("音乐 MCP server 已就绪：%s", _SERVER.name)
        return _session


async def _close_stack() -> None:
    global _stack, _session
    stack, _stack, _session = _stack, None, None
    if stack is not None:
        try:
            await stack.aclose()
        except Exception as exc:  # noqa: BLE001
            log.debug("关闭音乐 MCP 会话失败（忽略）: %s", exc)


async def aclose() -> None:
    """显式收尾（进程退出 / 测试用）。"""
    async with _lock:
        await _close_stack()


async def call(tool: str, **kwargs: Any) -> str:
    """调一个音乐工具，返回给用户看的一句话。失败一律变成可读原因，不抛。"""
    try:
        session = await _ensure_session()
        result = await asyncio.wait_for(session.call_tool(tool, kwargs), timeout=_CALL_TIMEOUT)
    except MusicUnavailable as exc:
        return f"音乐功能不可用：{exc}"
    except TimeoutError:
        return f"音乐工具 {tool} 超时（{_CALL_TIMEOUT:.0f}s）"
    except Exception as exc:  # noqa: BLE001
        log.warning("音乐工具 %s 失败: %s", tool, exc, exc_info=True)
        return f"音乐工具 {tool} 失败：{type(exc).__name__}: {exc}"
    texts = [c.text for c in result.content if getattr(c, "type", "") == "text"]
    return " ".join(t.strip() for t in texts if t and t.strip()) or f"{tool} 已执行"


# ---------- 给 graph 用的薄封装（带语义，别让上层拼工具名）----------

async def play(keyword: str, user_id: int | str | None = None) -> str:
    with as_user(user_id):
        return await call("play_music", keyword=keyword)


async def control(action: str, user_id: int | str | None = None) -> str:
    """控制播放（音量类动作在 MCP server 进程里走 Core Audio，不占本服务事件循环）。"""
    with as_user(user_id):
        return await call("player_control", action=action)


# 音量动作在 **MCP server 进程里**执行（tools/qqmusic_mcp/volume.py 用 Core Audio COM），
# 那里是同步阻塞的，与事件循环无关。这里只做转发 —— 绝不能把 comtypes 的同步 COM
# 调用直接放进本进程的 asyncio 循环：COM 属于 STA，在事件循环线程里调会**卡死整个服务**
# （实测 /music/status 挂住直到超时）。这层只负责把动作名送过去。
_VOLUME_PREFIXES = ("volume_", "mute", "unmute")


async def status() -> str:
    return await call("player_status")


async def now_playing(user_id: int | str | None = None) -> dict:
    """当前播放状态（结构化，给前端播放条用）。

    读不到 / 未启用都返回 `{"playing": false, "available": False, ...}` ——
    前端据此隐藏播放条，而不是显示一个永远空的条子。
    """
    out = {"available": False, "playing": False, "paused": False, "title": "",
           "artist": "", "volume": None, "muted": False, "app": ""}
    ok, why = await available_async(user_id)
    if not ok:
        out["reason"] = why
        return out
    if str(_SERVER.parent) not in sys.path:
        sys.path.insert(0, str(_SERVER.parent))
    try:
        import smtc
    except ImportError as exc:
        out["reason"] = f"缺少依赖：{exc}"
        return out
    try:
        st = await smtc.read_state()
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"读取失败：{exc}"
        return out
    if st is None:
        out["reason"] = "QQ音乐未运行"
        return out
    out.update({"available": True, "app": "QQ音乐", "title": st["title"],
                "artist": st["artist"], "playing": st["playing"],
                "paused": st["state"] == "PAUSED"})
    # ⚠️ 这里**刻意不读音量**。音量走 Core Audio（comtypes，同步阻塞、COM 属 STA），
    # 在请求路径上读它会把 /music/status 拖住（实测挂到超时，且**无任何报错**——
    # 最难查的一类故障）。播放条需要的音量改为**按需查**（点音量按钮/拖滑块时走
    # /music/control 的 volume_status），不在 5 秒轮询里带上。
    return out

