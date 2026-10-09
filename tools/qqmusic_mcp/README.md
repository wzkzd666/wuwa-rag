# QQ音乐 MCP server

用 QQ音乐 PC 客户端播放歌曲的 MCP 工具集。**零第三方 QQ音乐包依赖**，只用 `mcp` + `httpx`。

## 快速开始

1. 装 QQ音乐 PC 客户端（便携版也行）
2. 让 server 找得到 `QQMusic.exe`（三选一，见下）
3. 挂到你的 MCP 客户端，或由项目拉起（`QQMUSIC_ENABLED=true`）

```bash
# 独立运行（stdio）
python -X utf8 tools/qqmusic_mcp/qqmusic_local_mcp.py
```

## 配置 `QQMusic.exe` 位置（优先级从高到低）

| 方式 | 做法 |
|---|---|
| ① 环境变量 | `setx QQMUSIC_EXE "D:/qq_music/QQMusic/QQMusic2261.14.23.51/QQMusic.exe"` |
| ② 配置文件 | 编辑同目录 `qqmusic.ini` 的 `[qqmusic] exe=` |
| ③ 自动探测 | 注册表 App Paths / `Program Files\Tencent\QQMusic\` / `X:/qq_music/QQMusic/QQMusic*版号*/` |

不确定是否配好时，调用 `player_status` 工具自检，它会告诉你三种配置方式各自怎么填。

## 工具

| 工具 | 用途 |
|---|---|
| `play_music(keyword)` | 搜索并播放最匹配的一首（**首选**，省去两步） |
| `search_music(keyword, limit)` | 只搜索，返回带真实 songid 的列表 |
| `play_song_by_id(songid)` | 按 songid 播放（songid 必须来自 `search_music`，禁止编造） |
| `player_control(action)` | 播放控制，一个工具覆盖全部：`play`/`pause`/`toggle`/`next`/`prev`/`stop`/`quality_up`/`quality_down`/`volume_up`/`volume_down`/`mute`/`unmute`/`volume_status` |
| `player_status()` | 自检 + **现在在放什么**（读系统媒体会话的真实状态） |

**为什么控制合并成一个工具**：工具越多，模型选错、多轮调用的概率越高。
`player_control` 内部走 SMTC 精确命中 QQ音乐，并在动作后**读回状态确认**，
而不是「发出去就当成功」。

**音量为什么不是媒体键**：SMTC **没有**音量接口，全局媒体键（`keybd_event`）调的是
**系统主音量**（所有应用一起变、还会被前台播放器截获）。这里走 Core Audio 的
**应用通道**（`ISimpleAudioVolume`，见 `volume.py`）—— 只改 QQ音乐，不动系统音量、
不影响别的播放器，每步都读回验证。实现借鉴 yotohime777 的实测约束：枚举**所有活动输出
设备**、按进程 **exe 完整路径**匹配、写入前复核 PID 身份。

**幂等**：已经在目标状态（play 时已 PLAYING、pause 时已 PAUSED）时**不发命令**直接返回，
避免「暂停两次又播了」。

**两段式回报（accepted / confirmed）**：结果分 `[已确认]` / `[已接受·未确认]` / `[失败]`。
「命令发出去了」和「确认它真的生效了」是两件事 —— 混成一句「已调用客户端」会把失败说成
成功（今天实测踩过：参数写错时进程照样起、Session 照样建，只是歌名永远不变）。
以下情况**如实报「未确认」**而不是假装成功：

- 切歌前后**歌名相同**（包括「本来就在放这首歌」——不能据此证明切歌成功）
- 音质档位：SMTC **不暴露**音质信息，无法确认
- 读不到媒体会话 / 3 秒内歌名没变 / 会话中途消失

确认逻辑：投递前先记状态 → 发命令 → 等歌名真的变（`CHANGING` 中间态不算确认）。
play/pause 则等状态进入期望值。锁覆盖「捕获状态 → 发命令 → 确认」全程，避免并发
toggle 用同一个旧状态。

## 搜索链路（2026-10-07 实测，全程无需签名）

1. **歌名 → mid**：`c.y.qq.com/splcloud/fcgi-bin/smartbox_new.fcg`
   ⚠️ 响应是 **JSONP**（`callback({...})`），要先剥壳；数据在 `data.song.itemlist`（**不是** `list`）
2. **mid → songid**：`c.y.qq.com/v8/fcg-bin/fcg_play_single_song.fcg`
   ⚠️ songid 在 `data[0]['id']`（**没有** `songid` 这个键）
3. **播放**（⚠️ 参数格式是命门，见下）：
   `[QQMusic.exe, '/playbysongid', 'cmd_count==1&&id_0==<songid>&&songtype_0==0']`

### ⚠️ `/playbysongid` 必须传**两个独立参数**

```
✅ [exe, '/playbysongid', 'cmd_count==1&&id_0==97773&&songtype_0==0']   # 2 秒切歌
❌ [exe, '/playbysongid=97773']                                        # 永远不生效
```

写成**单个等号形式**时，客户端**完全不解析**：进程会起、Media Session 会建、但歌名
永远不变（实测读回 16~40 秒均为原曲）。这个失败形态极具误导性 —— 看起来像
「等不够久」，实际是参数根本没被读。

其它两个前提：
- **客户端必须已在运行**。`/playbysongid` 是投递给**已运行实例**的启动器参数，
  客户端没起来时投了没人接。本实现会检测进程，没运行就先拉起来并提示稍后重试。
- songid 必须是**纯十进制**（`data[0]['id']`），不能传 songmid / URL / 歌名。

参考实现：<https://github.com/yotohime777/QQMusic-mcp>（`qqmusic/client.py`）。

### 用 Media Session 确认播放是否真的发生

进程和参数都正确**不等于**播了。`winrt-Windows.Media.Control` 能读回真实状态：

```python
from winrt.windows.media.control import GlobalSystemMediaTransportControlsSessionManager as SMTC
m = await SMTC.request_async()
for s in m.get_sessions():                  # 注意是 get_sessions()，不是 get_current_session()
    if 'qqmusic' in (s.source_app_user_model_id or '').lower():
        p = await s.try_get_media_properties_async()
        print(p.title, p.artist, s.get_playback_info().playback_status.name)
```

属性是**直接属性**（`p.title`），没有 `props.get(key)` 那种字典式访问；
`get_current_session()` 会被别的播放器（网易云/浏览器）占住，必须枚举 `get_sessions()`。
SMTC 还能 `try_play_async` / `try_pause_async` / `try_skip_next_async` / `try_stop_async`
—— 实测对 QQ 音乐有效，可当「播放控制」用（但**没有 setTrack**，指定不了曲目）。

### 已废弃的接口（别再试）

- `c.y.qq.com/soso/fcgi-bin/client_search_cp` —— 能通、`code=0`，但结果**恒空**
- `u.y.qq.com/cgi-bin/musicu.fcg` —— 需 sign 签名（`code=500001`）
- `c.y.qq.com/splcloud/fcgi-bin/fcg_v2_search_cp` —— 404

QQ 音乐接口变动频繁。搜索返回空时，先怀疑接口，改这里而不是改上层调用。

## 兼容性

- **mcp 1.x / 2.x 都支持**：2.x 起 `FastMCP` 改名 `MCPServer`，本文件两条都兼容
- 仅 Windows：依赖 SMTC（`winrt`）与 Core Audio（`pycaw`/COM）。
  早期版本的媒体键模拟（`keybd_event`）已删除——它是**全局**按键，会被前台播放器截获、
  不保证作用于 QQ 音乐，且发出去读不回结果；控制一律走 SMTC 精确命中（见 smtc.py）
