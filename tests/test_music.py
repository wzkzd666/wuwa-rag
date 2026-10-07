"""音乐指令识别（纯规则，零 LLM）。

为什么单测它：判错的后果是「用户问游戏、assistant 去放歌」。所以**负样本**
（游戏问句、闲聊、时间问句）比正样本更重要。
"""
from __future__ import annotations

import asyncio
import importlib.util

import pytest

from wuwa_rag.dialog.nlu import music_action, needs_clarification


@pytest.mark.parametrize("q,action,keyword", [
    ("放首周杰伦的晴天", "play", "周杰伦的晴天"),
    ("播放稻香", "play", "稻香"),          # 动词按长度降序，否则会切出「放稻香」
    ("来一首告白气球", "play", "告白气球"),
    ("放个稻香", "play", "稻香"),
    # 口语前缀：「我想 / 我要 / 想 / 麻烦 / 能不能」—— 缺了它整句落空。
    # 实测「我想听周杰伦的小夜曲」原本判不出来，用户感受是「说了没反应」。
    ("我想听周杰伦的小夜曲", "play", "周杰伦的小夜曲"),
    ("我要听稻香", "play", "稻香"),
    ("想听青花瓷", "play", "青花瓷"),
    ("麻烦放首稻香", "play", "稻香"),
    ("能不能播放青花瓷", "play", "青花瓷"),
    ("听一下晴天", "play", "晴天"),        # 「一下」必须当量词吃掉，否则会混进歌名
    ("听听小夜曲", "play", "小夜曲"),       # 重复动词也要被前缀吃掉
    ("暂停", "pause", ""),
    ("停一下", "pause", ""),
    ("继续播放", "play", ""),
    ("下一首", "next", ""),
    ("换一首", "next", ""),
    ("上一首", "prev", ""),
    ("停止播放", "stop", ""),
    ("关闭音乐", "stop", ""),     # 最自然的说法；词表原先漏了它 → 表现为「说了没反应」
    ("关掉音乐", "stop", ""),
    ("把音乐关掉", "stop", ""),    # 倒装语序也要兜
    ("关闭音乐功能", "stop", ""),
    ("不要放了", "stop", ""),
    # 音量类：方向两支必须判得出来（曾经整个类目都没有 → 「调小音乐」毫无反应）
    ("调小音乐", "volume_down", ""),
    ("把音乐调小", "volume_down", ""),
    ("调小声音", "volume_down", ""),
    ("声音小一点", "volume_down", ""),
    ("小点声", "volume_down", ""),
    ("调大音乐", "volume_up", ""),
    ("音量调大", "volume_up", ""),
    ("声音大一点", "volume_up", ""),
    ("大声点", "volume_up", ""),
    ("静音", "mute", ""),
    ("取消静音", "unmute", ""),
    ("现在在放什么", "status", ""),
])
def test_音乐指令能识别(q: str, action: str, keyword: str) -> None:
    got = music_action(q)
    assert got is not None, q
    assert got == (action, keyword), f"{q!r} -> {got}"


@pytest.mark.parametrize("q", [
    "卡卡的声骸怎么配",     # 游戏问句含「卡」，不能误判
    "心的连招怎么打",
    "守岸人是谁",
    "今天几号",
    "放首歌",              # 没有具体歌名 → 不瞎搜
    "放点音乐",
    "帮我放首歌",
    "",
])
def test_非音乐指令不误判(q: str) -> None:
    assert music_action(q) is None, q


async def test_音乐服务未启用时如实说不可用(monkeypatch) -> None:
    """开关关闭时必须返回可读原因，不能抛异常打断问答。

    「开关状态」用打桩而不是真查 `user_settings`：其余单测都不连库，这一条也不该连
    （真连会因连不上/锁等待把整个测试套件挂住 —— 实测踩过）。
    """
    from wuwa_rag.services import music

    async def _off(_user_id=None):
        return False

    monkeypatch.setattr(music, "is_enabled", _off)
    out = await music.play("晴天", "any-user")
    assert "不可用" in out
    assert "Traceback" not in out


async def test_音乐开关_个人设置与部署默认任一为开即生效(monkeypatch) -> None:
    """开启条件是「或」：这是**本机能力**不是账号数据，机器装了播放器就能放。"""
    from wuwa_rag.config import get_settings
    from wuwa_rag.services import music

    s = get_settings()
    original = s.MUSIC_ENABLED
    try:
        s.MUSIC_ENABLED = True
        assert await music.is_enabled("u") is True     # 部署默认开 -> 无视个人设置
        s.MUSIC_ENABLED = False

        async def _personal(v):
            return v

        # ⚠️ 要打桩 `core.settings.get_setting`（is_enabled 里**延迟导入**它），
        # 打桩 music._get 没用 —— 那样会真连库，在 pytest 的事件循环下**永久等待**
        # （实测把整个测试文件挂住）。
        from wuwa_rag.core import settings as settings_mod

        monkeypatch.setattr(settings_mod, "get_setting", lambda *a: _personal(True))
        assert await music.is_enabled("u") is True
        monkeypatch.setattr(settings_mod, "get_setting", lambda *a: _personal(False))
        assert await music.is_enabled("u") is False
    finally:
        s.MUSIC_ENABLED = original


async def test_设置页开了开关_即使部署默认没开也必须判定可用(monkeypatch) -> None:
    """守「开关即时生效」—— 这条曾经是坏的，而且坏得很隐蔽。

    旧实现有两处叠加出错：
      · `available_async` 在 `is_enabled` 为真时**无条件** `return (True, ...)`；
      · 真正拉起 MCP 子进程的 `_ensure_session()`、以及 music 分支的 `_run_music()`
        调的却是**同步** `available()` —— 那条只看 .env 的部署开关。
    于是用户在设置页打开开关后，系统依然回「音乐功能未启用（可在设置页打开）」，
    表现出来的就是「开了没用，得像改 .env 那样重启服务」。

    这里只测判定层（不真起子进程）。判据分两半：
      · 环境齐 → 必须可用；
      · 环境不齐 → 给出的原因必须是**环境**问题，绝不能是「未启用」——
        那会把用户指向完全错误的排查方向。
    """
    from wuwa_rag.config import get_settings
    from wuwa_rag.core import settings as settings_mod
    from wuwa_rag.services import music

    s = get_settings()
    original = s.MUSIC_ENABLED
    try:
        s.MUSIC_ENABLED = False                      # 部署默认关（.env 里没开）

        async def _on(*_a, **_k):
            return True                              # 用户在设置页打开了

        monkeypatch.setattr(settings_mod, "get_setting", _on)
        assert await music.is_enabled("u") is True
        ok, why = await music.available_async("u")
        if ok:
            assert why == ""
        else:
            assert "未启用" not in why, f"个人开关已开，不该再报未启用：{why}"

        # 对照：个人设置也没开 → 不可用，且提示要指向设置页（这条能引导用户自己解决）
        async def _off(*_a, **_k):
            return False

        monkeypatch.setattr(settings_mod, "get_setting", _off)
        ok2, why2 = await music.available_async("u")
        assert ok2 is False
        assert "未启用" in why2
    finally:
        s.MUSIC_ENABLED = original


@pytest.mark.skipif(
    importlib.util.find_spec("mcp") is None,
    reason="需要 mcp 依赖（uv sync --extra music）",
)
async def test_常驻会话_可跨任务复用且能干净收尾(monkeypatch) -> None:
    """守「会话的 enter/exit 必须在同一个 task」。

    ⚠️ 这条会真起一次 MCP 子进程（本地、秒级）—— 是本文件唯一这样的用例，但它**必须
    存在**：这个 bug 只在「跨 task」这一种调用形态下才暴露，纯打桩根本测不出来。

    旧实现在「第一个发起请求的那个 task」里 enter、把会话存成模块级全局；那个请求 task
    一销毁，之后任何在别的 task 里 close 都会抛
    `RuntimeError: Attempted to exit a cancel scope that isn't the current tasks's
    current cancel scope` —— 实测就出现在 /ask/stream 的收尾（日志里记成未捕获异常）。
    现在会话由**长驻 task**持有，enter/exit 永远同 task。
    """
    from wuwa_rag.core import settings as settings_mod
    from wuwa_rag.services import music

    async def _get(_uid, key, default=None):
        return True if key == "music_enabled" else default   # 只开开关，不碰播放器路径

    monkeypatch.setattr(settings_mod, "get_setting", _get)

    async def one() -> str:
        with music.as_user(1):
            return await music.call("player_status")          # 只读状态，不出声

    try:
        a = await one()                                   # 在 pytest 的 task 里建会话
        b = await asyncio.create_task(one())              # 换一个 task 复用同一会话
        c = await asyncio.create_task(one())
        assert all(isinstance(x, str) for x in (a, b, c))
        await music.aclose()                              # 跨 task 收尾：旧实现必抛
    finally:
        await music.aclose()


# ---------- 多轮回指（「刚刚的任务你再试试看」）----------

_HIST_MUSIC = [
    {"role": "user", "content": "播放周杰伦的青花瓷"},
    {"role": "assistant", "content": "好的，已经在放了。"},
]


def test_回指_沿用上一条音乐指令() -> None:
    """用户实测：多轮里说「刚刚的任务你再试试看」时音乐完全失效。

    根因是这类句子既没有音乐动词、也没有歌名 —— 判据全落空，
    而「刚刚的任务」只有在**历史**里才解析得出来。
    """
    for q in ("刚刚的任务你再试试看", "再放一次", "那个再来一遍", "还是刚才那首"):
        got = music_action(q, _HIST_MUSIC)
        assert got == ("play", "周杰伦的青花瓷"), (q, got)


def test_回指_没有音乐历史时不猜() -> None:
    """历史里没有音乐指令时，回指句必须落回普通问答 —— 宁可漏判，不可乱判。"""
    hist = [{"role": "user", "content": "卡卡的声骸怎么配"},
            {"role": "assistant", "content": "……"}]
    assert music_action("刚刚的任务你再试试看", hist) is None
    # 游戏问句即使带回指词也不该命中
    assert music_action("卡卡的声骸怎么配", _HIST_MUSIC) is None
    # 没有历史时同样不判
    assert music_action("再放一次") is None


# ---------- 「没理解」→ 主动反问（clarify）----------

@pytest.mark.parametrize("q", [
    "那个", "再试", "再来", "再放一次", "嗯", "额", "刚刚的任务你再试试看",
])
def test_没理解_纯指代与回指会要求澄清(q: str) -> None:
    """判定落空时应当**反问**，而不是硬答（实测硬答会「顺着上一话题编」）。"""
    assert needs_clarification(q) is True


@pytest.mark.parametrize("q", [
    "讲个故事", "你好", "谢谢",     # 短，但语义完整 —— 不该反问
    "卡卡的声骸怎么配", "守岸人是谁", "播放周杰伦的青花瓷",
])
def test_没理解_语义完整就不澄清(q: str) -> None:
    """**误判比漏判更烦人**：中文里大量完整意图只有 4 个字（「讲个故事」）。

    曾经还加过「≤6 字且无槽位角色 → 澄清」，实测立刻把「讲个故事/你好/谢谢」
    打成反问，已删除那条判据 —— 把「短」等同于「不明」是错的。
    """
    assert needs_clarification(q) is False


def test_没理解_有槽位或角色的短句不澄清() -> None:
    """有锚点就有可答内容，哪怕很短（「心」→ 问的是心，不需要反问）。"""
    assert needs_clarification("那个", ["声骸"], None) is False
    assert needs_clarification("再试", None, ["心"]) is False
