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
# 建会话预算：首次要起一个 python 子进程并跑完 MCP initialize，比单次调用宽松些
_START_TIMEOUT = 30.0
# MCP 会话是**全局单例**（与用户无关），但「能不能用」依赖当前用户的开关。
# 用 contextvar 把当前请求的用户带进同步的 available()。
_USER_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("music_user", default=None)

# ---------- MCP 会话：关进一个**长驻 task** ----------
#
# ⚠️ 为什么不能把 AsyncExitStack 存成模块级全局（旧写法，已踩过）：
# MCP 的 stdio_client 内部用 anyio 的 CancelScope，而 CancelScope **只能在其创建的
# 那个 task 里退出**。旧写法是在「第一个发起请求的那个 task」里 enter 的，那个请求
# 一结束、task 一销毁，之后任何在别的 task 里发生的 close 都会抛：
#   RuntimeError: Attempted to exit a cancel scope that isn't the current tasks's
#   current cancel scope
# 实测就出现在 /ask/stream 的收尾（日志里记成「未捕获异常 POST /ask/stream」），
# 且那之后会话已处于半坏状态。
# 现在：enter / exit 全部锁在同一个长驻 task 内，跨 task 只传「工具名 + 参数」。
_session_task: asyncio.Task | None = None
_session_inbox: asyncio.Queue | None = None
_session_ready: asyncio.Future | None = None
_start_lock = asyncio.Lock()


@contextlib.contextmanager
def as_user(user_id: int | str | None):
    """把当前用户标进上下文 —— 供 `_ensure_base()` 判「这个用户允不允许用音乐」。

    为什么非得走 contextvar：MCP 会话是**全局单例**，建立时不带用户参数，
    只能从这里取当前用户。
    ⚠️ 因此**凡可能触发建会话的入口都必须包一次**（`play` / `control` / `status`，
    以及图里的 `_run_music` 整段）。漏包的后果**不是报错**，而是静默退回部署默认
    （`.env` 没开就一律判「未启用」）—— 用户在设置页开了开关也照样不能用，
    表现出来就像「必须重启服务」。排查这类症状时，先确认这一层有没有设。
    """
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
    """设置自定义播放器路径。改完要**重建常驻会话**才会用新路径（env 是启动时注入的）。"""
    value = (path or "").strip()
    await _put(user_id, EXE_KEY, value or None)
    # 会话是常驻单例且 env 已固化，只能重建；下次调用会自动重新建起来
    await aclose()


async def _get(user_id: str, key: str, default=None):
    from wuwa_rag.core.settings import get_setting

    return await get_setting(user_id, key, default)


async def _put(user_id: int | str, key: str, value) -> None:
    from wuwa_rag.core.settings import delete_setting, set_setting

    if value is None:
        await delete_setting(str(user_id), key)
    else:
        await set_setting(str(user_id), key, value)


# 「未启用」的提示语只定义一次：调用方原样透给用户，别各写一份。
_OFF_HINT = "音乐功能未启用（可在设置页「音乐播放」里打开）"


def _env_ready() -> tuple[bool, str]:
    """**环境**是否具备 —— 与「开关」无关：server 文件在不在、mcp 依赖装没装。

    为什么必须和开关拆成两件事：两者失败的原因完全不同 ——
    开关是用户自己能去设置页开的，环境是部署问题。混在一个函数里返回，
    调用方就分不清「不可用」到底是哪一种，只能整条否定掉。
    （这正是「设置页开了开关、却仍被『未启用』拦住」的根源。）
    """
    if not _SERVER.is_file():
        return False, f"找不到 MCP server：{_SERVER}"
    try:
        import mcp  # noqa: F401
    except ImportError:
        return False, "缺 mcp 依赖（uv sync --extra music）"
    return True, ""


def available() -> tuple[bool, str]:
    """同步可用性判定：**只看部署默认**（.env 的 MUSIC_ENABLED）与环境。

    ⚠️ 这里**故意不接 user_id**：个人设置在 `user_settings` 表里，是**异步**查询，
    同步函数里查不了。曾经的写法是「传了 user_id 就跳过开关检查」，那是个**真 bug** ——
    结果是所有带 user_id 的调用（`now_playing(user.id)` 等）都认为「可用」并真的去
    拉起 MCP 子进程，在某些环境下直接挂死。带个人设置的判定一律走 `available_async`。
    """
    if not get_settings().MUSIC_ENABLED:
        return False, _OFF_HINT
    return _env_ready()


async def available_async(user_id: int | str | None = None) -> tuple[bool, str]:
    """真正可用性判定（含个人开关）：**开关**归 `is_enabled`，**环境**归 `_env_ready`。

    ⚠️ 这里踩过两个坑，别再写回去：
      ① 旧版是「`is_enabled` 为真就直接 `return (True, ...)`」——
         于是环境不合格（缺 mcp 依赖 / 找不到 server）也被当成可用；
      ② 真正拉起子进程的 `_ensure_base()` 与音乐分支的 `_run_music()` 却另外调了
         **同步** `available()` —— 那条只看 .env，完全无视个人开关。
    两条叠一起的后果：用户在设置页打开了开关，系统依旧回「音乐功能未启用」，
    看起来就像「必须改 .env 再重启服务」。**开关必须即时生效，这是本函数的职责。**
    """
    if not await is_enabled(user_id):
        return False, _OFF_HINT
    return _env_ready()


def _fail_pending(inbox: asyncio.Queue, exc: BaseException) -> None:
    """宿主挂了：把排队中的调用全部叫醒，别让它们干等到超时。"""
    while True:
        try:
            item = inbox.get_nowait()
        except asyncio.QueueEmpty:
            return
        if item is not None and not item[0].cancelled():
            item[0].set_exception(exc)


async def _session_main(exe: str, ready: asyncio.Future, inbox: asyncio.Queue) -> None:
    """宿主 task：**在这个 task 里**建立会话、循环处理调用、并在同一个 task 里收尾。

    进去就不再返回，直到收到 `None`（由 `aclose()` 发出）—— 这样 enter/exit 永远同
    task，见文件上方关于 CancelScope 的说明。会话因此是**常驻**的：建一次，之后
    所有点歌/控制都复用它，不必每次冷启动。
    """
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:
        ready.set_exception(MusicUnavailable(f"缺 mcp 依赖（uv sync --extra music）：{exc}"))
        return
    env = dict(os.environ)
    # 个人设置 > 环境变量：留空**不是**「不许用」，而是「不覆盖」——
    # 子进程默认就继承本进程 env，QQMUSIC_EXE 会照常生效。
    # （曾经在这里 pop 掉它是反的：那会让用户设了环境变量却完全不生效。）
    if exe:
        env["QQMUSIC_EXE"] = exe
    params = StdioServerParameters(command=sys.executable,
                                   args=["-X", "utf8", str(_SERVER)],
                                   env=env)
    try:
        async with AsyncExitStack() as stack:
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            log.info("音乐 MCP server 已就绪（常驻）：%s", _SERVER.name)
            ready.set_result(None)
            while True:
                item = await inbox.get()
                if item is None:
                    break      # 用 break 而非 return：让上面的 async with 仍在本 task 里收尾
                fut, tool, kwargs = item
                if fut.cancelled():
                    continue
                try:
                    fut.set_result(await session.call_tool(tool, kwargs))
                except BaseException as exc:   # noqa: BLE001 —— 交给调用方按人话处理
                    fut.set_exception(exc)
    except BaseException as exc:               # noqa: BLE001
        if not ready.done():
            ready.set_exception(exc)
        _fail_pending(inbox, exc)


async def _drop_session() -> None:
    """把起失败的宿主 task 清干净（取消并等它退出），让下次调用重新尝试。"""
    global _session_task, _session_inbox, _session_ready
    task = _session_task
    _session_task, _session_inbox, _session_ready = None, None, None
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(BaseException):
            await task


async def _ensure_base() -> asyncio.Queue:
    """确保宿主 task 在跑，返回它的收件箱（已跑则直接复用）。

    ⚠️ 判定必须走 **available_async**：同步 `available()` 只看 .env 的部署开关，
    会把「用户在设置页打开了开关」误判成未启用 —— 表现就是「开了也没用，像要重启服务」。
    （会话本身与用户无关，这里只是拿当前请求的用户判断「允不允许用」。）
    """
    global _session_task, _session_inbox, _session_ready
    async with _start_lock:
        if _session_task is not None and not _session_task.done():
            return _session_inbox          # type: ignore[return-value]
        ok, why = await available_async(_USER_ID.get())
        if not ok:
            raise MusicUnavailable(why)
        exe = await get_exe(_USER_ID.get())
        loop = asyncio.get_running_loop()
        ready: asyncio.Future = loop.create_future()
        inbox: asyncio.Queue = asyncio.Queue()
        _session_ready, _session_inbox = ready, inbox
        _session_task = asyncio.create_task(_session_main(exe, ready, inbox))
        try:
            # shield：超时只放弃**等待**，不去取消 ready ——
            # 否则宿主稍后 set_result 会撞 InvalidStateError
            await asyncio.wait_for(asyncio.shield(ready), timeout=_START_TIMEOUT)
        except TimeoutError as exc:
            await _drop_session()
            raise MusicUnavailable(
                f"启动 QQ音乐 MCP server 超时（{_START_TIMEOUT:.0f}s）") from exc
        except MusicUnavailable:
            await _drop_session()
            raise
        except BaseException as exc:           # noqa: BLE001
            await _drop_session()
            raise MusicUnavailable(f"启动 QQ音乐 MCP server 失败：{exc}") from exc
        return inbox


async def aclose() -> None:
    """收尾（进程退出 / 测试 / 换播放器路径时用）。

    只负责**通知与等待**：会话的关闭必须在宿主 task 里发生（见文件上方 CancelScope
    的说明），所以这里只给它发一个 `None`，再等它自己退出。
    """
    global _session_task, _session_inbox, _session_ready
    task, inbox = _session_task, _session_inbox
    _session_task, _session_inbox, _session_ready = None, None, None
    if task is None or inbox is None:
        return
    if not task.done():
        with contextlib.suppress(Exception):
            await inbox.put(None)
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=8)
    except BaseException:                      # noqa: BLE001 —— 收尾失败不该影响调用方
        task.cancel()
        with contextlib.suppress(BaseException):
            await task


async def call(tool: str, **kwargs: Any) -> str:
    """调一个音乐工具，返回给用户看的一句话。失败一律变成可读原因，不抛。"""
    try:
        inbox = await _ensure_base()
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await inbox.put((fut, tool, kwargs))
        try:
            result = await asyncio.wait_for(fut, timeout=_CALL_TIMEOUT)
        except TimeoutError:
            fut.cancel()
            return f"音乐工具 {tool} 超时（{_CALL_TIMEOUT:.0f}s）"
    except MusicUnavailable as exc:
        return f"音乐功能不可用：{exc}"
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


async def status(user_id: int | str | None = None) -> str:
    """当前播放状态（给人看的一句话）。`user_id` 用于按该用户的开关判可用性。"""
    with as_user(user_id):
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

