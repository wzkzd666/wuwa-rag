"""用户自己的偏好设置（设置页开关）。

与 `services.profile`（user_facts）的区别：那条是**系统从对话里推断**的（会过期、会
软删、同类覆盖）；这里是**用户显式设的**（主题、音乐开关…），不推断、不软删，只有用户
改写才变。

为什么存 PG 而不是浏览器 localStorage：音乐开关决定**服务端**要不要去调本机播放器，
服务端得知道当前设置 —— 开关写在前端、后端不知道，是「开了却没生效」最难查的一类问题。
"""
from __future__ import annotations

from typing import Any

from wuwa_rag.core.db import get_cursor
from wuwa_rag.ww_logger import get_logger

log = get_logger("settings")


async def get_setting(user_id: str, key: str, default: Any = None) -> Any:
    async with get_cursor() as cur:
        await cur.execute("SELECT value FROM user_settings WHERE user_id = %s AND key = %s",
                          (str(user_id), key))
        row = await cur.fetchone()
    return row[0] if row else default


async def set_setting(user_id: str, key: str, value: Any) -> None:
    async with get_cursor() as cur:
        await cur.execute(
            "INSERT INTO user_settings (user_id, key, value) VALUES (%s, %s, %s)"
            " ON CONFLICT (user_id, key) DO UPDATE SET value = EXCLUDED.value,"
            " updated_at = now()",
            (str(user_id), key, _dumps(value)),
        )


async def delete_setting(user_id: str, key: str) -> None:
    """删掉设置 → 回到部署默认值（.env）。设置页的「恢复默认」用。"""
    async with get_cursor() as cur:
        await cur.execute("DELETE FROM user_settings WHERE user_id = %s AND key = %s",
                          (str(user_id), key))


def _dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)
