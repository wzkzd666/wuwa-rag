r"""
qqmusic_local_mcp.py —— 自包含的 QQ音乐 MCP server（Windows）

零第三方 QQ音乐包依赖，只用 mcp + httpx。
运行：python -X utf8 tools/qqmusic_mcp/qqmusic_local_mcp.py

搜索链路（2026-10-07 实测打通，全程无需 sign 签名）
----------------------------------------------------
1) 歌名 → mid
   GET https://c.y.qq.com/splcloud/fcgi-bin/smartbox_new.fcg
   ⚠️ 响应是 **JSONP**（`callback({...})`），必须先剥壳再 json.loads。
   ⚠️ 数据在 `data.song.itemlist` —— 键名是 itemlist，**不是** list。
   只给 mid（十六进制），不给 songid。
2) mid → 十进制 songid
   GET https://c.y.qq.com/v8/fcg-bin/fcg_play_single_song.fcg?songmid=<mid>&noplaysong=1
   `data` 是 list，取 data[0]；songid 在 **`data[0]['id']`**（int，**没有 songid 这个键**）。
3) 播放：QQMusic.exe /playbysongid=<songid>

已废弃（不要再用，都是实测过的）
--------------------------------
· c.y.qq.com/soso/fcgi-bin/client_search_cp —— 能通、code=0，但 song.list **恒空**
· u.y.qq.com/cgi-bin/musicu.fcg —— 需 sign（code=500001）
· c.y.qq.com/splcloud/fcgi-bin/fcg_v2_search_cp —— 404

QQMusic.exe 定位优先级
---------------------
1. 环境变量 `QQMUSIC_EXE`（由项目侧在拉起本进程时注入，见 tools/qqmusic_mcp/README.md）
2. 同目录下的 `qqmusic.ini` 里的 `exe=` 行（方便用户直接改文件）
3. 注册表 App Paths + 常见安装目录 + 便携版形态（D:\qq_music\QQMusic\QQMusic<版号>\QQMusic.exe）
"""
from __future__ import annotations

import asyncio
import configparser
import ctypes
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx

# mcp 1.x 是 `mcp.server.fastmcp.FastMCP`；2.x 起改名 `mcp.server.mcpserver.MCPServer`
# 且旧模块**直接不存在**（不是 DeprecationWarning）。两条都兼容，免得换环境就跑不起来。
try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _MCP
except ModuleNotFoundError:  # pragma: no cover —— mcp 1.x
    from mcp.server.fastmcp import FastMCP as _MCP  # type: ignore[no-redef]

sys.path.insert(0, str(Path(__file__).parent))
import smtc  # noqa: E402  —— 同目录模块，按路径导入

mcp = _MCP("qqmusic-local")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Referer": "https://y.qq.com/",
}
SEARCH_URL = "https://c.y.qq.com/splcloud/fcgi-bin/smartbox_new.fcg"
SONGINFO_URL = "https://c.y.qq.com/v8/fcg-bin/fcg_play_single_song.fcg"
INI_PATH = Path(__file__).with_name("qqmusic.ini")
log = logging.getLogger("qqmusic")


# ---------- 定位 QQMusic.exe ----------
def find_qqmusic() -> str | None:
    """按 环境变量 → 配置文件 → 注册表/常见路径 的顺序找 QQMusic.exe。"""
    # ① 环境变量（项目侧注入；用户也可自己 setx）
    override = os.getenv("QQMUSIC_EXE", "").strip().strip('"')
    if override and os.path.isfile(override):
        return override
    # ② 同目录配置文件：exe = D:\...\QQMusic.exe
    if INI_PATH.is_file():
        try:
            cp = configparser.ConfigParser()
            cp.read(INI_PATH, encoding="utf-8")
            exe = cp.get("qqmusic", "exe", fallback="").strip().strip('"')
            if exe and os.path.isfile(exe):
                return exe
        except Exception:  # noqa: BLE001 —— 配置文件读不了就当没配，不该让 server 起不来
            pass
    # ③ 注册表 + 常见安装目录 + 便携版形态
    roots: list[str] = [os.environ.get("ProgramFiles", ""),
                        os.environ.get("ProgramFiles(x86)", "")]
    for d in "CDEFG":
        roots += [f"{d}:\\", f"{d}:\\Program Files", f"{d}:\\Program Files (x86)"]
    cands: list[Path] = []
    for r in roots:
        if not r:
            continue
        # 正规安装：Program Files\Tencent\QQMusic\QQMusic.exe
        cands.append(Path(r) / "Tencent" / "QQMusic" / "QQMusic.exe")
    # 便携版：D:\qq_music\QQMusic\QQMusic<版号>\QQMusic.exe（用户的实际形态）
    for d in "CDEFG":
        portable = Path(f"{d}:\\qq_music\\QQMusic")
        if portable.is_dir():
            cands += sorted(p / "QQMusic.exe" for p in portable.glob("QQMusic*") if p.is_dir())
    try:  # 注册表最权威，装在非标准位置也能找到
        import winreg

        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                k = winreg.OpenKey(hive, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\QQMusic.exe")
                v, _ = winreg.QueryValueEx(k, None)
                if v:
                    cands.insert(0, Path(v))
            except OSError:
                pass
    except Exception:  # noqa: BLE001 —— 非 Windows / 无权限时跳过
        pass
    for p in cands:
        if p.is_file():
            return str(p)
    return None


# ---------- Windows 媒体按键 ----------
VK_MEDIA_NEXT, VK_MEDIA_PREV, VK_MEDIA_STOP, VK_MEDIA_PLAY_PAUSE = 0xB0, 0xB1, 0xB2, 0xB3
VK_VOLUME_MUTE, VK_VOLUME_DOWN, VK_VOLUME_UP = 0xAD, 0xAE, 0xAF
KEYEVENTF_KEYUP = 0x0002
_MAX_VOLUME_STEPS = 20          # 上限保护：keybd_event 是串行循环，steps 给大了会长时间卡住


def tap(vk: int, times: int = 1) -> None:
    times = max(1, min(int(times), _MAX_VOLUME_STEPS))
    for _ in range(times):
        ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
        ctypes.windll.user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)


# ---------- 搜索 ----------
def _unwrap_jsonp(text: str) -> dict[str, Any]:
    """QQ 的这些接口返回 JSONP（`callback({...})`），直接 json.loads 必然抛异常。"""
    s = (text or "").strip()
    if s.startswith("callback("):
        s = s[len("callback("):s.rfind(")")]
    elif s.startswith("jQuery"):
        s = s[s.find("(") + 1:s.rfind(")")]
    return json.loads(s)


async def _search_mid(keyword: str, limit: int = 5) -> list[dict]:
    """第 1 步：歌名 → mid 列表。命中项含 name / singer / mid。"""
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as cli:
        r = await cli.get(SEARCH_URL,
                          params={"key": keyword, "format": "json", "utf8": "1"},
                          headers=HEADERS)
        r.raise_for_status()
        data = _unwrap_jsonp(r.text)
    # ⚠️ 键名是 itemlist，不是 list
    items = (data.get("data", {}).get("song", {}) or {}).get("itemlist") or []
    return [{"name": it.get("name", ""), "singer": it.get("singer", ""),
             "mid": it.get("mid", "")} for it in items[:limit] if it.get("mid")]


async def _mid_to_songid(mid: str) -> dict[str, Any] | None:
    """第 2 步：mid → 十进制 songid。songid 在 data[0]['id']（没有 songid 这个键）。"""
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as cli:
        r = await cli.get(SONGINFO_URL,
                          params={"songmid": mid, "format": "json", "noplaysong": 1},
                          headers=HEADERS)
        r.raise_for_status()
        data = _unwrap_jsonp(r.text)
    arr = data.get("data") or []
    if not isinstance(arr, list) or not arr:
        return None
    it = arr[0]
    sid = it.get("id")
    if not isinstance(sid, int):
        return None
    return {"songid": sid,
            "name": it.get("name", ""),
            "singer": "/".join(s.get("name", "") for s in (it.get("singer") or [])),
            "album": (it.get("album") or {}).get("name", "")}


async def search_song(keyword: str, limit: int = 5) -> list[dict]:
    """搜索并补全 songid（每条都实打实取过 songid，不编造）。"""
    out: list[dict] = []
    for item in await _search_mid(keyword, limit):
        full = await _mid_to_songid(item["mid"])
        if full:
            out.append({**item, **full})
    return out


def _client_running() -> bool:
    """QQMusic.exe 是否已在运行。**必须先有客户端**：`/playbysongid` 是投递给已运行
    实例的启动器参数，客户端没起来时投了也没人接。"""
    try:
        import psutil

        for p in psutil.process_iter(["name"]):
            if (p.info["name"] or "").casefold() == "qqmusic.exe":
                return True
    except Exception as exc:  # noqa: BLE001 —— psutil 缺失/权限不足时退回按需启动
        log.warning("psutil 进程检查失败（改为直接启动）: %s", exc)
    return False


def _launch(songid: int | None = None) -> str:
    """投递播放命令。

    ⚠️ **参数格式是这个功能的命门**（实测踩过）：必须传**两个独立参数** ——
        [exe, '/playbysongid', 'cmd_count==1&&id_0=<songid>&songtype_0==0']
    写成单个 `/playbysongid=<id>`（等号形式）客户端**完全不解析**：进程会起、会话会建，
    但歌名永远不变（SMTC 读回 16~40 秒均为原曲），非常容易误判成「等待不够久」。
    参考实现：github.com/yotohime777/QQMusic-mcp（qqmusic/client.py）。
    """
    exe = find_qqmusic()
    if not exe:
        return ("错误: 找不到 QQMusic.exe —— 它**只从环境变量 QQMUSIC_EXE 读路径**"
                "（或设置页「音乐播放 → 播放器路径」）。请设好后重试。")

    args = [exe]
    if songid is not None:
        if not str(songid).isascii() or not str(songid).isdecimal():
            return f"错误: songid 必须是纯数字（收到 {songid!r}）"
        args += ["/playbysongid", f"cmd_count==1&&id_0=={int(songid)}&&songtype_0==0"]
    _spawn_detached(exe, args[1:])
    return f"已调用 QQ音乐客户端: {args}"


async def dispatch(songid: int) -> str:
    """投递点歌并**确认是否真的切歌**，结果分两段说。

    借鉴 yotohime777 的 accepted/confirmed：命令「发出去了」和「确认生效了」是两件事。
    混成一句「已调用客户端」会把失败说成成功 —— 参数写错时进程照样起、Session 照样建，
    只是歌名永远不变，正是今天实测踩的坑。
    """
    # 冷启动：客户端没在跑就先空跑拉起来，等它真正就绪（**会话出现**才算就绪，
    # 进程出现不算 —— 那时它还没加载完，投递会被吞，实测过一次）再投递。
    # 这样用户说一次就够了，不必「再播一次」。
    if not _client_running():
        exe = find_qqmusic()
        if not exe:
            return "[失败] 找不到 QQMusic.exe（设 QQMUSIC_EXE 或填 qqmusic.ini 的 exe=）"
        _spawn_detached(exe, [])
        if not await _wait_ready(_COLD_START_WAIT):
            return (f"[失败] 已启动 QQ音乐客户端，但 {_COLD_START_WAIT:.0f} 秒内还没就绪"
                    "（首次启动可能要手动登录/初始化）。请再播一次。")
    before = None
    try:
        before = await smtc.read_state()
    except Exception as exc:  # noqa: BLE001 —— 读不到状态只是没法确认，不该挡住投递
        log.debug("投递前读状态失败（将无法确认切歌）: %s", exc)
    out = _launch(songid)
    if out.startswith("错误") or "还没就绪" in out:
        return f"[失败] {out}"
    if before is None:
        return f"[已接受·未确认] {out}（读不到播放状态，无法确认是否切歌）"
    ok, reason = await smtc.confirm_switch(before)
    return f"[已确认] {out}" if ok else f"[已接受·未确认] {out} —— {reason}"


# 冷启动等待预算：官方客户端冷启到能接受启动参数，实测 3~8 秒；给 25 秒余量
_COLD_START_WAIT = 25.0


async def _wait_ready(budget: float) -> bool:
    """等客户端就绪：Media Session 出现即视为就绪。

    为什么以「会话出现」为标准：`_client_running()` 在进程一出现就为真，但此时它还没
    加载完，直接投递会被吞（实测过一次：进程在、参数也给了，歌名 40 秒不变）。
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + budget
    while loop.time() < deadline:
        try:
            if await smtc.read_state() is not None:
                return True
        except Exception:  # noqa: BLE001 —— 会话还没注册时读会抛，继续等
            pass
        await asyncio.sleep(0.5)
    return False


def _spawn_detached(exe: str, args: list[str]) -> None:
    """拉起客户端。stdin/stdout/stderr 全部丢弃 + CREATE_NO_WINDOW：
    ① 不加窗口控制每次弹黑窗；② 继承的管道会让调用方一直等它的输出。
    """
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    devnull = subprocess.DEVNULL
    subprocess.Popen([exe, *args], shell=False, stdin=devnull, stdout=devnull,
                     stderr=devnull, creationflags=flags)


# ---------- 工具 ----------
@mcp.tool()
async def play_music(keyword: str) -> str:
    """搜索关键词并立即用 QQ音乐 PC 客户端播放最匹配的一首。
    用户说「放歌/播放xxx/来一首」时优先用这个，不要拆成搜索+播放两步。"""
    try:
        songs = await search_song(keyword, limit=1)
    except Exception as e:  # noqa: BLE001 —— 面向用户的工具，失败要说人话
        return f"搜索失败（QQ 接口可能又改版了）: {e}"
    if not songs:
        return f"没搜到与「{keyword}」相关的歌曲"
    s = songs[0]
    return (f"{await dispatch(int(s['songid']))}\n"
            f"正在播放: {s['name']} - {s['singer']}（songid={s['songid']}）")


@mcp.tool()
async def search_music(keyword: str, limit: int = 5) -> str:
    """只搜索不播放，返回歌曲列表（含真实 songid，可直接给 play_song_by_id）。"""
    try:
        songs = await search_song(keyword, max(1, min(limit, 10)))
    except Exception as e:  # noqa: BLE001
        return f"搜索失败: {e}"
    if not songs:
        return f"没搜到「{keyword}」"
    return "\n".join(
        f"[{i + 1}] {s['name']} - {s['singer']} | 专辑:{s['album']} | songid={s['songid']}"
        for i, s in enumerate(songs)
    )


@mcp.tool()
async def play_song_by_id(songid: int) -> str:
    """用十进制 songid 播放指定歌曲。songid 必须来自 search_music 的返回值，禁止编造。"""
    return await dispatch(int(songid))


@mcp.tool()
async def player_control(action: str) -> str:
    """控制播放，一个工具覆盖全部动作（减少来回调用）。

    action 取值：play 播放 / pause 暂停 / toggle 播放暂停切换 / next 下一首 /
    prev 上一首 / stop 停止 / quality_up 音质调高 / quality_down 音质调低 /
    volume_up 音量+10 / volume_down 音量-10 / mute 静音 / unmute 取消静音 /
    volume_status 当前音量。

    音量走 Core Audio 的**应用通道**（只改 QQ音乐，不动系统音量、也不影响别的播放器）。

    走 Windows 媒体会话（SMTC）**精确命中 QQ音乐**，不用模拟媒体键 ——
    全局媒体键会被前台那个播放器截经常，且发出去也读不回「到底有没有生效」。
    """
    try:
        return await smtc.control(action.strip().lower())
    except Exception as e:  # noqa: BLE001 —— winrt 没装/系统不支持时给可读原因
        return f"播放控制失败（{type(e).__name__}: {e}）。播放控制需要 winrt 包：uv sync --extra music"


@mcp.tool()
async def player_status() -> str:
    """自检 + 当前播放状态：exe 找没找到、客户端在不在、现在在放什么。

    「进程起来了」不等于「在播」—— 这里读的是系统媒体会话的真实状态。
    """
    exe = find_qqmusic()
    lines = [f"QQMusic.exe: {exe or '未找到（设 QQMUSIC_EXE 或填 qqmusic.ini 的 exe=）'}"]
    lines.append(f"客户端进程: {'运行中' if _client_running() else '未运行'}")
    try:
        st = await smtc.read_state()
    except Exception as e:  # noqa: BLE001
        st = None
        lines.append(f"媒体会话读取失败（{type(e).__name__}）")
    if st:
        lines.append(f"正在播放: {st['title']} - {st['artist']}（{st['state']}）")
    else:
        lines.append("媒体会话: 无（客户端没运行，或已打开但尚未开始播放）")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
