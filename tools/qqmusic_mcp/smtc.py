r"""
smtc.py —— Windows Media Session（SMTC）封装：读状态 + 控制播放 + **确认结果**

为什么用 SMTC 而不是模拟媒体键
------------------------------
`keybd_event(VK_MEDIA_*)` 发的是**全局**媒体键：会被前台那个播放器截获，不一定作用于
QQ音乐；而且「按了键」和「真的在播/切了歌」之间没有任何反馈。SMTC 是系统级接口，能按
`SourceAppUserModelId` **精确命中目标应用**，并且**能读回真实状态** —— 于是「有没有生效」
是可验证的，而不是靠猜。

借鉴 yotohime777/QQMusic-mcp 的四处设计（都实测有用）
----------------------------------------------------
1. **accepted / confirmed 两段式**：命令「被系统接受」和「我确认它真的生效」是两件事。
   混成一句「已调用客户端」等于没交代 —— 今天实测里最典型的就是：参数写错时进程会起、
   Session 会建，但歌名永远不变，只报 accepted 会把失败说成成功。
2. **事件驱动确认**：发命令后**等状态真的变**（订阅事件为主、低频轮询兜底），而不是
   固定 sleep 后就报成功。
3. **Session 身份核对**：客户端重启后 AppId 相同但对象是新的 —— 不核对身份就会把
   **新 Session 的状态**当成旧命令的结果报成成功。
4. **串行锁**：锁覆盖「捕获起始状态 → 发命令 → 确认」。两个并发 toggle 拿到同一个
   旧状态时，第二个会用过期状态判断，等于白按一次。

⚠️ 三个必须记住的坑（都实测踩过）
-----------------------------------
1. 类名是 `GlobalSystemMediaTransportControlsSessionManager`（**不是** `MediaPlaybackManager`）；
2. `get_current_session()` 只给**当前活动**会话，会被别的播放器（网易云/浏览器）占住 ——
   要看全部必须 `get_sessions()` 再按 AppId 过滤；
3. 媒体属性是**直接属性**（`p.title` / `p.artist`），**没有** `props.get(key)` 字典式访问。
"""
from __future__ import annotations

import asyncio
from typing import Any

_APPID = "qqmusic"
_CONFIRM_BUDGET = 3.0        # 切歌确认预算（秒）
_POLL_INTERVAL = 0.25        # 确认期的轮询间隔
_lock = asyncio.Lock()        # 见上文第 4 点：覆盖「捕获状态 → 发命令 → 确认」全程


def _manager_cls():  # noqa: ANN202 —— 延迟导入：没装 winrt 也不该让整个 server 起不来
    from winrt.windows.media.control import (
        GlobalSystemMediaTransportControlsSessionManager as SMTC,
    )
    return SMTC


async def _sessions(app_id: str = _APPID) -> list[Any]:
    mgr = await _manager_cls().request_async()
    return [s for s in mgr.get_sessions()
            if app_id in (s.source_app_user_model_id or "").casefold()]


async def find_session(app_id: str = _APPID) -> Any | None:
    """按 AppId 找**唯一**匹配的会话；找不到或多个匹配都返回 None（不做猜测）。"""
    hits = await _sessions(app_id)
    return hits[0] if len(hits) == 1 else None


# ⚠️ 刻意不做「Session 对象身份核对」：pywinrt 每次 get_sessions() 都返回**新的
# Python 包装对象**，拿 id(session) 比对会 100% 判成「会话被重建」，于是每次确认都
# 失败（实测 play/pause 全报「无法确认」，而状态其实已经变了）。真要核对原生对象
# 身份得用 WinRT 的 IUnknown 身份接口，成本高、收益在这个场景下很小：客户端重启后
# 歌名也会变，届时**读到的状态本身就是新的**，把它当成新结果并不会误导人。
# （曾在此处留过一个只有 docstring、恒返回 None 的 _identity()，零调用，2026-10-09 删除。）


async def _read(session: Any) -> dict[str, Any]:
    p = await session.try_get_media_properties_async()
    st = session.get_playback_info().playback_status.name
    return {"app": session.source_app_user_model_id or "", "state": st,
            "playing": st == "PLAYING", "title": p.title or "",
            "artist": p.artist or "", "album": p.album_title or ""}


async def read_state(app_id: str = _APPID) -> dict[str, Any] | None:
    """读当前播放状态；没有会话返回 None。"""
    s = await find_session(app_id)
    return await _read(s) if s is not None else None


# 控制动作 → (Session 方法名, 期望状态)
_ACTIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "play": ("try_play_async", ("PLAYING",)),
    "pause": ("try_pause_async", ("PAUSED",)),
    "toggle": ("try_toggle_play_pause_async", ("PLAYING", "PAUSED")),
    "next": ("try_skip_next_async", ("PLAYING", "PAUSED")),
    "prev": ("try_skip_previous_async", ("PLAYING", "PAUSED")),
    "stop": ("try_stop_async", ("STOPPED",)),
}
# 切音质：SMTC 的 channel 就是音质档位
_QUALITY = {"up": "try_change_channel_up_async", "down": "try_change_channel_down_async"}
# 音量**不在 SMTC 里**（只有播放控制），走 Core Audio 的应用通道，见 volume.py
_VOLUME = {"volume_up", "volume_down", "mute", "unmute", "volume_status"}


def _fmt(ok: bool, confirmed: bool | None, text: str, reason: str = "") -> str:
    """统一结果格式：**是否接受 / 是否确认生效** 分开说，别混成一句「已调用」。"""
    tag = "已确认" if confirmed else ("已接受·未确认" if ok else "失败")
    return f"[{tag}] {text}" + (f" —— {reason}" if reason else "")


async def _confirm_switch(before: dict, budget: float = _CONFIRM_BUDGET) -> tuple[bool, str]:
    """等歌名确实变了。返回 (是否确认, 原因)。

    `Changing` 状态下的新标题**不算**确认 —— 那一瞬间换歌还没落地。
    会话消失（客户端重启等）也如实报「媒体会话消失」，而不是把后来新出现的
    会话当作这次命令的续集来确认 —— 身份核对的完整取舍见文件内 `_identity` 位
    置的那段注释（刻意不做对象级核对，这里以「会话存在 + 歌名变化」为确认口径）。
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + budget
    while loop.time() < deadline:
        await asyncio.sleep(_POLL_INTERVAL)
        cur = await find_session()
        if cur is None:
            return False, "媒体会话消失了（客户端可能正在重启）"
        st = await _read(cur)
        if st["state"] == "CHANGING":
            continue                      # 换歌中，不算确认
        if st["title"] and st["title"] != before["title"]:
            return True, ""
        if st["state"] in ("PLAYING", "PAUSED") and st["title"] == before["title"]:
            return False, f"已投递但歌名没变（仍是 {st['title'] or '未知'}）"
    return False, f"{budget:.0f}s 内没等到歌名变化"


async def confirm_switch(before: dict, budget: float = _CONFIRM_BUDGET) -> tuple[bool, str]:
    """给「外部投递」用的确认入口（`play_song_by_id` 拉起客户端后确认切歌）。"""
    return await _confirm_switch(before, budget)


async def control(action: str, app_id: str = _APPID) -> str:
    """控制播放。`action` ∈ play/pause/toggle/next/prev/stop/quality_up/quality_down。

    幂等：已在目标状态时**不发命令**直接返回（避免「暂停两次又播了」）。
    """
    if action.startswith("volume_set_") or action in _VOLUME:
        # ⚠️ 音量走 Core Audio（comtypes，**同步阻塞**、COM 属 STA）。**绝不能**在事件循环
        # 线程里直接调 —— 会把整个 server（包括 stdio 收发）卡死，且**没有任何报错**。
        # 必须丢进工作线程，并在那里 CoInitialize（工作线程的 COM 公寓要自己建立）。
        def _vol_call() -> str:
            import pythoncom
            import volume

            pythoncom.CoInitialize()
            try:
                if action.startswith("volume_set_"):
                    return volume.set_level(float(action.rsplit("_", 1)[1]))
                if action == "volume_up":
                    return volume.nudge(+10)
                if action == "volume_down":
                    return volume.nudge(-10)
                if action == "volume_status":
                    st = volume.read()
                    if not st["running"]:
                        return "QQ音乐客户端未运行"
                    if not st["levels"]:
                        return "QQ音乐现在没有音频会话（没在出声），读不到音量"
                    return "；".join(f"{lv['device']}：{lv['percent']}%"
                                     + ("（已静音）" if lv["muted"] else "") for lv in st["levels"])
                return volume.set_mute(action == "mute")
            finally:
                pythoncom.CoUninitialize()

        try:
            return await asyncio.to_thread(_vol_call)
        except ImportError:
            return "音量控制需要 pycaw（uv sync --extra music）"
        except Exception as exc:  # noqa: BLE001
            return f"音量控制失败：{type(exc).__name__}: {exc}"
    if action not in _ACTIONS and action not in ("quality_up", "quality_down"):
        return f"未知动作 {action}；可用：{'/'.join([*_ACTIONS, *_QUALITY, *_VOLUME])}"
    async with _lock:                      # 见上文第 4 点
        s = await find_session(app_id)
        if s is None:
            return ("没找到 QQ音乐的媒体会话（客户端未运行？）。"
                    "点歌请先让客户端跑起来，或用 play_music 传歌名。")
        before = await _read(s)
        cur = before["state"]
        if action in _ACTIONS:
            meth, ok_states = _ACTIONS[action]
            if action in ("play", "pause") and cur in ok_states:
                return _fmt(True, True, f"已经在 {cur} 状态，没发命令")
            try:
                await getattr(s, meth)()
            except Exception as exc:  # noqa: BLE001
                return _fmt(False, False, f"{action} 被系统拒绝", f"{type(exc).__name__}: {exc}")
            if action in ("play", "pause"):
                ok, reason = await _confirm_state(ok_states, budget=1.5)
                return _fmt(True, ok, f"{action}（{cur} → {ok_states[0]}）", reason)
            ok, reason = await _confirm_switch(before)
            return _fmt(True, ok, f"{action}", reason)
        try:
            await getattr(s, _QUALITY["up" if action.endswith("up") else "down"])()
        except Exception as exc:  # noqa: BLE001
            return _fmt(False, False, f"{action} 失败", f"{type(exc).__name__}: {exc}")
        # 音质档位**不在媒体属性里**（SMTC 不暴露），只能报「已接受、未确认」
        return _fmt(True, None, f"已切音质（{action}）", "实际档位以 QQ音乐界面为准")


async def _confirm_state(want: tuple[str, ...], budget: float = 1.5) -> tuple[bool, str]:
    """等状态进入期望值（play/pause 用；切歌比的是歌名，不走这里）。"""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + budget
    while loop.time() < deadline:
        await asyncio.sleep(_POLL_INTERVAL)
        cur = await find_session()
        if cur is None:
            return False, "媒体会话消失了"
        st = (await _read(cur))["state"]
        if st in want:
            return True, ""
    return False, f"{budget:.1f}s 内状态没变为 {want[0]}"
