"""QQ音乐 MCP server 的**纯逻辑**离线测试（不联网、不启动客户端）。

只测「改错了会静默出错」的地方：JSONP 剥壳、字段解析、exe 定位优先级、参数夹紧。
网络与播放端到端是**手动验证**（会真出声），不在单测里。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "tools" / "qqmusic_mcp" / "qqmusic_local_mcp.py"


def _load():
    spec = importlib.util.spec_from_file_location("qqm", _PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------- JSONP 剥壳 ----------
@pytest.mark.parametrize("raw,expect", [
    ('callback({"a":1})', {"a": 1}),
    ('  callback({"songid":9})  ', {"songid": 9}),
    ('{"a":2}', {"a": 2}),                       # 已经是纯 JSON 也能吃
])
def test_剥壳(raw: str, expect: dict) -> None:
    """QQ 这些接口回 `callback({...})`。不剥就 json.loads 必抛 —— 搜索会一直失败。"""
    assert _load()._unwrap_jsonp(raw) == expect


# ---------- 字段解析 ----------
def test_搜索结果取_itemlist而不是_list() -> None:
    """实测：`data.song` 的键是 **itemlist**。写成 list 会永远取到空 → 搜不到歌。"""
    m = _load()
    payload = {"data": {"song": {"itemlist": [
        {"name": "晴天", "singer": "周杰伦", "mid": "0039MnYb0qxYhV"}]}}}
    assert payload["data"]["song"]["itemlist"][0]["mid"] == "0039MnYb0qxYhV"
    # 顺手钉住「键名写错就空」这件事：换成 list 取不到
    assert payload["data"]["song"].get("list") is None
    assert m is not None


def test_songid在id字段而不是songid字段() -> None:
    """实测：`data[0]['id']` 才是十进制 songid，**没有** `songid` 这个键。"""
    item = {"id": 97773, "name": "晴天",
            "singer": [{"name": "周杰伦"}], "album": {"name": "叶惠美"}}
    assert item["id"] == 97773
    assert "songid" not in item


# ---------- exe 定位 ----------
def test_环境变量优先于其他来源(tmp_path, monkeypatch) -> None:
    """项目侧会注入 QQMUSIC_EXE；它必须压过配置文件，否则用户改了设置不生效。"""
    m = _load()
    exe = tmp_path / "QQMusic.exe"
    exe.write_text("x", encoding="utf-8")
    monkeypatch.setenv("QQMUSIC_EXE", str(exe))
    assert m.find_qqmusic() == str(exe)


def test_环境变量指向不存在的文件时继续往下找(monkeypatch) -> None:
    """路径失效（用户卸载/换盘）不能直接报错，得当没配继续探测。"""
    m = _load()
    monkeypatch.setenv("QQMUSIC_EXE", r"D:/不存在的路径/QQMusic.exe")
    # 探测结果因机器而异，只要求「不抛异常且返回 str|None」
    assert m.find_qqmusic() is None or isinstance(m.find_qqmusic(), str)


# ---------- 媒体键死代码不得复活 ----------
def test_媒体键模拟已删除() -> None:
    """2026-10-09 删除 tap()/_MAX_VOLUME_STEPS/VK_* 全家（零调用者）。

    原测试钉的是「keybd_event 串行循环 steps 夹紧 20」——那是媒体键模拟的护栏；
    媒体键本身是**全局**按键（会被前台播放器截获、发出去读不回结果），控制一律走
    SMTC 精确命中（见 smtc.py），这套按键代码属于误导性死代码，删。
    本测试改为守卫：确认它们不会以「看着还能用」的形态悄悄回来。
    """
    m = _load()
    for gone in ("tap", "_MAX_VOLUME_STEPS", "KEYEVENTF_KEYUP", "VK_MEDIA_NEXT"):
        assert not hasattr(m, gone), f"{gone} 应已随媒体键模拟一并删除"


def test_工具集已精简() -> None:
    """按键模拟那 4 个（toggle/next/prev/stop）已被 SMTC 的 player_control 取代。

    合并成一个 `player_control(action)` 是刻意的：工具越多，模型选错 / 多轮调用的
    概率越高。**音量工具刻意没做** —— SMTC 没有音量接口，只能靠全局媒体键模拟，
    而那个会被前台其它播放器截获，与其给一个「看着能用其实不保证」的按钮，
    不如不留。系统音量用户自己调。
    """
    m = _load()
    for name in ("play_music", "search_music", "play_song_by_id",
                 "player_control", "player_status"):
        assert callable(getattr(m, name)), name
    for gone in ("toggle_play_pause", "next_track", "prev_track", "stop_music", "set_volume"):
        assert not hasattr(m, gone), f"{gone} 应已被 player_control 取代"


# ---------- 播放参数格式（这个功能的命门）----------

def test_播放命令是两个独立参数而不是等号形式() -> None:
    """回归：写成 [exe, '/playbysongid=<id>'] 时客户端**完全不解析**。

    症状极具误导性 —— 进程会起、Media Session 会建，但歌名永远不变，
    看起来像「等不够久」，实际是参数根本没被读（实测读回 16~40 秒均为原曲）。
    正确形式：两个独立参数 + cmd_count 表达式。参考 yotohime777/QQMusic-mcp。

    用 ast 取**函数体里的字符串常量**，避开 docstring —— 说明文字里会故意写出
    等号形式来讲这个坑，按行 grep 会把说明当成实现。
    """
    import ast
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(_load()._launch))
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef))
    body = fn.body[1:]                       # 跳过 docstring
    lits = [n.value for stmt in body for n in ast.walk(stmt)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert any(v == "/playbysongid" for v in lits), "必须把 /playbysongid 作为独立参数"
    assert any("cmd_count==1&&id_0==" in v for v in lits), "必须用 cmd_count 表达式传 id"
    assert not any("/playbysongid=" in v for v in lits), "不能出现等号形式（客户端不解析）"


def test_非数字songid被拒绝() -> None:
    """songid 必须是纯十进制：不能传 songmid / URL / 歌名（会被拼成垃圾命令）。"""
    m = _load()
    assert m._launch("0039MnYb0qxYhV").startswith("错误: songid 必须是纯数字")
    assert m._launch("../../evil").startswith("错误: songid 必须是纯数字")


# ---------- 同曲短路（重放正在播的歌不得被说成「卡住了」） ----------

def test_同曲判定_形态变体() -> None:
    m = _load()
    assert m._is_same_song("无间渡", "无间渡", "胡彦斌的无间渡")
    assert m._is_same_song("无间渡 (Live)", "无间渡", "无间渡")      # SMTC 带后缀
    assert m._is_same_song("无间渡", "无间渡", "播放无间渡")          # 标题在关键词里
    assert not m._is_same_song("稻香", "无间渡", "无间渡")
    assert not m._is_same_song("", "无间渡", "无间渡")               # 没在播


def test_点正在播的歌_不再投递直接确认() -> None:
    """2026-10-10 实测：无间渡本来就在放，再点一次 → confirm_switch 见歌名没变
    → 回「已接受·未确认」→ 模型转述成「是不是卡住了」。同曲必须短路。"""
    import asyncio
    m = _load()
    calls = {"dispatch": 0}

    async def fake_search(keyword, limit=5):
        return [{"name": "无间渡", "singer": "胡彦斌", "songid": 123}]

    async def fake_dispatch(songid):
        calls["dispatch"] += 1
        return "不该被调用"

    async def fake_read(app_id=None):
        return {"title": "无间渡", "artist": "胡彦斌", "playing": True, "state": "PLAYING"}

    m.search_song = fake_search
    m.dispatch = fake_dispatch
    m.smtc.read_state = fake_read
    out = asyncio.run(m.play_music("胡彦斌的无间渡"))
    assert calls["dispatch"] == 0, "同曲不该重新投递"
    assert "正在播放" in out and "无间渡" in out
    assert "没变" not in out and "未确认" not in out


# ---------- stop = 真停（QQ音乐不实现 SMTC StopRequested，2026-10-10 实测） ----------

def test_stop_媒体命令被忽略时关进程(monkeypatch) -> None:
    """pause≠stop：用户要彻底停。Stop 被播放器忽略 → 兜底关客户端进程，且如实报。"""
    import asyncio
    m = _load()
    monkeypatch.setattr(m.smtc, "control", lambda a: _coro("[已接受·未确认] stop —— 播放器没停"))
    monkeypatch.setattr(m, "_client_running", lambda: True)
    monkeypatch.setattr(m, "_close_client", lambda: _coro("已关闭客户端进程"))
    out = asyncio.run(m.player_control("stop"))
    assert out.startswith("[已确认]") and "关闭客户端进程" in out and "冷启动" in out


def test_stop_客户端没跑直接确认(monkeypatch) -> None:
    import asyncio
    m = _load()
    monkeypatch.setattr(m.smtc, "control",
                        lambda a: _coro("没找到 QQ音乐的媒体会话（客户端未运行？）"))
    monkeypatch.setattr(m, "_client_running", lambda: False)
    out = asyncio.run(m.player_control("stop"))
    assert out.startswith("[已确认]") and "没在跑" in out


def test_同曲但暂停中_恢复播放而非重投(monkeypatch) -> None:
    """暂停中再点同一首：重投会**从头播**（唱到一半突然重来）→ 必须走 SMTC play 恢复。"""
    import asyncio
    m = _load()

    async def fake_search(keyword, limit=5):
        return [{"name": "无间渡", "singer": "胡彦斌", "songid": 1}]

    async def fake_dispatch(songid):
        raise AssertionError("暂停中重播同一首不该走投递")

    async def fake_read(app_id=None):
        return {"title": "无间渡", "artist": "胡彦斌", "playing": False, "state": "PAUSED"}

    monkeypatch.setattr(m, "search_song", fake_search)
    monkeypatch.setattr(m, "dispatch", fake_dispatch)
    monkeypatch.setattr(m.smtc, "read_state", fake_read)
    monkeypatch.setattr(m.smtc, "control", lambda a: _coro("[已确认] play（PAUSED → PLAYING）"))
    out = asyncio.run(m.play_music("无间渡"))
    assert "恢复播放" in out


def _coro(value):
    import asyncio
    fut: asyncio.Future = asyncio.get_event_loop_policy().new_event_loop().create_future()
    fut.set_result(value)
    async def one():
        return value
    return one()
