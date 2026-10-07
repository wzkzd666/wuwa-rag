"""音乐播放状态的 HTTP 接口（给前端播放条用）。

可见性：任何登录用户都能读/控自己机器上的播放器 —— 它是**本机**状态，不是账号数据，
所以不存在「越权看别人」的问题。功能未启用时一律返回 available=false，前端据此隐藏条子。
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from wuwa_rag.api import auth as authn
from wuwa_rag.config import get_settings
from wuwa_rag.services import music

router = APIRouter()

# 播放状态读取的硬超时（秒）。前端 5 秒轮询一次，卡住就会持续占用事件循环。
_STATUS_TIMEOUT = 3.0


class MusicControlIn(BaseModel):
    """动作名直接沿用 MCP 工具的 action，取值见 tools/qqmusic_mcp 的 player_control。"""
    action: str = Field(..., description="play/pause/toggle/next/prev/volume_up/volume_down/mute/unmute")


class MusicSettingIn(BaseModel):
    enabled: bool | None = Field(None, description="是否启用音乐播放；null = 恢复部署默认")
    exe: str | None = Field(None, description="自定义 QQMusic.exe 路径；空串/null = 让 server 自动探测")
    reset: bool = Field(False, description="true = 全部恢复默认（.env）")


@router.get("/music/setting")
async def api_music_setting(user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """当前用户的音乐开关（设置页用）。`source` 说明现在是谁决定的。"""
    from wuwa_rag.core.settings import get_setting

    personal = await get_setting(str(user.id), music.SETTING_KEY, None)
    return {"enabled": await music.is_enabled(user.id),
            "personal": personal,
            "deployment": get_settings().MUSIC_ENABLED,
            "source": "deployment" if get_settings().MUSIC_ENABLED else
                      ("personal" if personal is not None else "default"),
            "exe": await music.get_exe(user.id),
            "exe_source": "personal" if personal is not None and
                          await music.get_exe(user.id) != get_settings().MUSIC_EXE else "default"}


@router.put("/music/setting")
async def api_music_setting_put(body: MusicSettingIn,
                                user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """开关音乐播放 / 自定义播放器路径。**立即生效，不用重启服务**。

    改路径会重建 MCP 会话（子进程的 env 是启动时注入的），所以首次点歌会多花一两秒。
    """
    if body.reset:
        await music.set_enabled(user.id, None)
        await music.set_exe(user.id, None)
    else:
        if body.enabled is not None:
            await music.set_enabled(user.id, body.enabled)
        if body.exe is not None:
            await music.set_exe(user.id, body.exe)
    return {"enabled": await music.is_enabled(user.id), "exe": await music.get_exe(user.id)}


@router.get("/music/status")
async def api_music_status(user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """当前播放状态（播放条每 5 秒轮询）。

    ⚠️ 这里有**硬超时**兜底：底层要读系统媒体会话（SMTC），在某些环境下可能挂住
    （实测过：`-c` 内联调用 0.5s 正常、同样代码写成文件脚本必挂到超时，原因未定位）。
    播放条 5 秒一次轮询，一旦卡住就是**持续占用一个事件循环**，所以宁可让它超时返回
    「不可用」、前端隐藏播放条，也不能拖住整个 API。
    """
    try:
        return await asyncio.wait_for(music.now_playing(user.id), timeout=_STATUS_TIMEOUT)
    except (TimeoutError, Exception) as exc:  # noqa: BLE001
        return {"available": False, "playing": False, "paused": False, "title": "",
                "artist": "", "volume": None, "muted": False, "app": "",
                "reason": f"读取超时或失败：{type(exc).__name__}"}


@router.post("/music/control")
async def api_music_control(body: MusicControlIn,
                            user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    return {"result": await music.control(body.action, user.id)}


def install(app) -> None:  # noqa: ANN001
    app.include_router(router, prefix="", tags=["music"])
