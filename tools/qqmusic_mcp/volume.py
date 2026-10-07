r"""
volume.py —— QQ音乐**应用级**音量（Core Audio 的 ISimpleAudioVolume）

为什么不用系统音量 / 媒体键
--------------------------
- `keybd_event(VK_VOLUME_UP)` 调的是**系统主音量**：所有应用一起变，还会被前台那个
  播放器截获，发出去也读不回有没有生效。
- SMTC **没有音量接口**（只有播放控制）。
- Core Audio 的 `ISimpleAudioVolume` 才是对口的：它管**某个音频会话**（即某个应用）
  在某个输出设备上的音量，读写都能立刻读回验证。

⚠️ 三个必须记住的坑（都是实测踩的）
-----------------------------------
1. **import 路径**：接口在 `pycaw.api.audiopolicy` / `pycaw.constants` / `pycaw.utils`，
   **不是** `pycaw.pycaw`（顶层模块里没有 IAudioSessionControl2 等）。
2. **要枚举设备再取会话**：`AudioUtilities.GetAllDevices(...)` → 逐个
   `device.AudioSessionManager.GetSessionEnumerator()`。只用顶层
   `GetAudioSessionManager()` 拿到的会话列表里没有可用的音量接口。
3. **音量用 `GetMasterVolume()/SetMasterVolume()`**（0~1 标量）、属性名是
   `session.SimpleAudioVolume`。写成 `ISimpleAudioVolume` / `GetMasterVolumeLevelScalar`
   会 AttributeError —— 而这些异常若被 `except: pass` 兜住，症状是「一个会话都枚举不到」，
   极具误导性（我为此排查了三步）。

另外三条来自 yotohime777/QQMusic-mcp 的实测约束：
- 枚举**所有活动输出设备**（应用可能在非默认设备上出声）；
- 按进程 **exe 完整路径**（+ PID 创建时间）匹配，写入前复核 —— 只按进程名会打到别的进程；
- `SetMasterVolume` 里的 "Master" 指**该音频会话**，不是设备主音量；
  本模块从不调设备级 `IAudioEndpointVolume` 的写方法。

安全姿态：**默认只读**。调音量是有可感知副作用的操作，只在显式要求时写。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("qqmusic.volume")


def _api():
    """pycaw 新版把接口拆到了子模块；顶层 pycaw.pycaw 里没有。"""
    from pycaw.api.audiopolicy import IAudioSessionControl2
    from pycaw.constants import DEVICE_STATE, EDataFlow
    from pycaw.utils import AudioSession, AudioUtilities

    return IAudioSessionControl2, DEVICE_STATE, EDataFlow, AudioSession, AudioUtilities


def _qq_exe() -> str | None:
    """运行中的 QQMusic.exe 完整路径（小写，比较用）。"""
    import psutil

    for p in psutil.process_iter(["name", "exe"]):
        try:
            if (p.info["name"] or "").casefold() == "qqmusic.exe" and p.info["exe"]:
                return str(Path(p.info["exe"]).resolve()).casefold()
        except Exception:  # noqa: BLE001 —— psutil 的 NoSuchProcess/AccessDenied
            continue
    return None


def _all_sessions() -> list[tuple[str, Any, Any]]:
    """所有活动输出设备上的会话 → [(设备名, AudioSession, SimpleAudioVolume)]。"""
    clsid, DEVICE_STATE, EDataFlow, AudioSession, AudioUtilities = _api()
    out = []
    try:
        devices = AudioUtilities.GetAllDevices(EDataFlow.eRender.value, DEVICE_STATE.ACTIVE.value)
    except Exception as exc:  # noqa: BLE001 —— 没有音频设备时直接放弃
        log.debug("枚举输出设备失败: %s", exc)
        return out
    for dev in devices:
        try:
            enum = dev.AudioSessionManager.GetSessionEnumerator()
            for i in range(enum.GetCount()):
                ctl = enum.GetSession(i)
                if ctl is None:
                    continue
                sess = AudioSession(ctl.QueryInterface(clsid))
                out.append((dev.FriendlyName, sess, sess.SimpleAudioVolume))
        except Exception as exc:  # noqa: BLE001 —— 个别设备/会话失效是常态
            log.debug("跳过设备 %s: %s", dev.FriendlyName, exc)
    return out


def _mine() -> list[tuple[str, Any, Any, Any]]:
    """只留 QQ音乐自己的会话，并带上进程对象（写入前要复核身份）。"""
    expected = _qq_exe()
    if expected is None:
        return []
    hits = []
    for devname, sess, vol in _all_sessions():
        proc = sess.Process
        if proc is None:
            continue
        try:
            exe = str(Path(proc.exe()).resolve()).casefold()
        except Exception:  # noqa: BLE001
            continue
        if exe == expected:
            hits.append((devname, sess, vol, proc))
    return hits


def read() -> dict[str, Any]:
    """只读：QQ音乐在每个活动输出设备上的当前音量（%）与静音状态。"""
    if _qq_exe() is None:
        return {"running": False, "levels": []}
    levels = []
    for devname, _s, vol, _p in _mine():
        try:
            levels.append({"device": devname,
                           "percent": round(vol.GetMasterVolume() * 100),
                           "muted": bool(vol.GetMute())})
        except Exception as exc:  # noqa: BLE001
            log.debug("读音量失败(%s): %s", devname, exc)
    return {"running": True, "levels": levels}


def set_level(percent: float) -> str:
    """把 QQ音乐的**应用音量**设为 percent（0~100）。只影响该应用，不动系统音量。"""
    if _qq_exe() is None:
        return "QQ音乐客户端未运行"
    targets = _mine()
    if not targets:
        return "找不到 QQ音乐的音频会话（它现在可能没在出声）"
    target = max(0.0, min(float(percent), 100.0)) / 100.0
    for devname, _s, vol, proc in targets:
        # 写入前复核进程身份：客户端可能刚重启，打到新进程上就错了
        try:
            if str(Path(proc.exe()).resolve()).casefold() != _qq_exe():
                return f"写入前 QQ音乐进程已更换（{devname}），已放弃以免打到别的进程"
            vol.SetMasterVolume(target, None)
        except Exception as exc:  # noqa: BLE001
            return f"设置音量失败（{devname}）：{exc}"
    got = "、".join(f"{lv['device']} {lv['percent']}%" for lv in read()["levels"]) or "无读回"
    return f"QQ音乐音量已设为 {round(target * 100)}%（读回 {got}）—— 只改这个应用，不动系统音量"


def nudge(delta_percent: float) -> str:
    """相对调整（+10 / -10）。"""
    lv = read()["levels"]
    if not lv:
        return "读不到 QQ音乐当前音量（它现在可能没在出声）"
    base = sum(x["percent"] for x in lv) / len(lv)
    return set_level(base + delta_percent)


def set_mute(muted: bool) -> str:
    """静音 / 取消静音（应用级）。"""
    if _qq_exe() is None:
        return "QQ音乐客户端未运行"
    targets = _mine()
    if not targets:
        return "找不到 QQ音乐的音频会话"
    for devname, _s, vol, _p in targets:
        try:
            vol.SetMute(1 if muted else 0, None)
        except Exception as exc:  # noqa: BLE001
            return f"设置静音失败（{devname}）：{exc}"
    return f"已{'静音' if muted else '取消静音'}（应用级，不影响别的播放器）"
