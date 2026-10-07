"""音乐指令识别（纯规则，零 LLM）。

为什么单测它：判错的后果是「用户问游戏、assistant 去放歌」。所以**负样本**
（游戏问句、闲聊、时间问句）比正样本更重要。
"""
from __future__ import annotations

import pytest

from wuwa_rag.dialog.nlu import music_action, needs_clarification


@pytest.mark.parametrize("q,action,keyword", [
    ("放首周杰伦的晴天", "play", "周杰伦的晴天"),
    ("播放稻香", "play", "稻香"),          # 动词按长度降序，否则会切出「放稻香」
    ("来一首告白气球", "play", "告白气球"),
    ("放个稻香", "play", "稻香"),
    ("暂停", "pause", ""),
    ("停一下", "pause", ""),
    ("继续播放", "play", ""),
    ("下一首", "next", ""),
    ("换一首", "next", ""),
    ("上一首", "prev", ""),
    ("停止播放", "stop", ""),
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
