import { useCallback, useEffect, useRef, useState } from 'react'
import {
  Loader2, Music4, Pause, Play, SkipBack, SkipForward, SlidersHorizontal, Volume2, VolumeX,
} from 'lucide-react'

import { useStore } from '../store/useStore'
import * as api from '../lib/api'
import type { MusicState } from '../types'

/**
 * 音乐播放条：显示当前在放什么，并给出播放/暂停/上下首/音量控制。
 *
 * 为什么不轮询：靠 SSE 推播放状态需要后端常驻一个媒体监听，而 SMTC 本身也没有
 * 「状态变化」推送（只有 session 被替换时的会话事件）。5 秒轮询足够，且没在放时
 * 直接停（available=false 就不再问了）。
 */
export default function MusicBar() {
  const apiBase = useStore((s) => s.settings.apiBase)
  const toast = useStore((s) => s.toast)
  const [st, setSt] = useState<MusicState | null>(null)
  const [busy, setBusy] = useState('')
  // 拖动滑块时先本地跟手，松手才发请求 —— 否则每像素一次接口
  const [vol, setVol] = useState<number | null>(null)
  const volRef = useRef<number | null>(null)
  const [volOpen, setVolOpen] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const s = await api.musicStatus(apiBase)
      setSt(s)
      return s
    } catch {
      return null
    }
  }, [apiBase])

  useEffect(() => {
    let alive = true
    let timer = 0
    const tick = async () => {
      if (!alive) return
      const s = await refresh()
      if (!alive) return
      // ⚠️ **绝不能**因为「现在没在放」就 return 掉轮询（旧写法如此，是个真 bug）：
      // 播放状态是**会自己变**的 —— 用户随时可能去点歌、或手动打开 QQ音乐，
      // 一旦停了轮询，播放条就再也不会出现了（表现出来就是「明明在放歌，顶栏却空的」）。
      // 没在放时只是把间隔放宽，省掉大部分无效请求。
      timer = window.setTimeout(tick, s?.available ? 5000 : 15000)
    }
    timer = window.setTimeout(tick, 0)
    return () => {
      alive = false
      window.clearTimeout(timer)
    }
  }, [refresh])

  const ctl = async (action: string) => {
    setBusy(action)
    try {
      const r = await api.musicControl(action, apiBase)
      if (r.result && action.startsWith('volume') === false) toast('ok', r.result)
      await refresh()
    } catch (e) {
      toast('err', e instanceof Error ? e.message : String(e))
    } finally {
      setBusy('')
    }
  }

  // 音量**按需查**：后端刻意不在 5 秒轮询里读音量（Core Audio 的同步 COM 调用会拖住
  // 请求），所以点开音量区时先查一次，之后拖动滑块靠本地 state 跟手。
  const openVol = async () => {
    const next = !volOpen
    setVolOpen(next)
    if (!next) return
    try {
      const r = await api.musicControl('volume_status', apiBase)
      const m = /(\d+)%/.exec(r.result || '')
      if (m) setVol(Number(m[1]))
    } catch {
      /* 读不到就保持原样，不打扰用户 */
    }
  }

  // 只在「有会话且确实在放什么」时出现
  const show = !!st && st.available && !!st.title
  if (!show) return null

  return (
    <div className="music-bar">
      <Music4 size={15} className="music-bar-icon" />
      <span className="music-bar-title" title={`${st!.title} - ${st!.artist}`}>
        {st!.title}
      </span>
      <span className="music-bar-artist">{st!.artist}</span>
      <div className="music-bar-ctl">
        <button className="msg-action" title="上一首" disabled={!!busy}
                onClick={() => void ctl('prev')}>
          <SkipBack size={14} />
        </button>
        <button className="msg-action" title={st!.playing ? '暂停' : '播放'} disabled={!!busy}
                onClick={() => void ctl(st!.playing ? 'pause' : 'play')}>
          {busy === 'play' || busy === 'pause'
            ? <Loader2 size={14} className="spin" />
            : st!.playing ? <Pause size={14} /> : <Play size={14} />}
        </button>
        <button className="msg-action" title="下一首" disabled={!!busy}
                onClick={() => void ctl('next')}>
          <SkipForward size={14} />
        </button>
        <span className="music-bar-vol">
          <button className="msg-action" title="静音 / 取消静音" disabled={!!busy}
                  onClick={() => void ctl(st!.muted ? 'unmute' : 'mute')}>
            {st!.muted ? <VolumeX size={14} /> : <Volume2 size={14} />}
          </button>
          <button className="msg-action" title="音量（仅调整 QQ音乐，不影响系统音量）" disabled={!!busy}
                  onClick={() => void openVol()}>
            <SlidersHorizontal size={14} />
          </button>
          {volOpen && (
          <input
            type="range"
            min={0}
            max={100}
            value={vol ?? st!.volume ?? 100}
            disabled={!!busy}
            title={`QQ音乐音量 ${vol ?? st!.volume ?? '—'}%（仅作用于 QQ音乐）`}
            onChange={(e) => {
              const v = Number(e.target.value)
              setVol(v)
              volRef.current = v
            }}
            onMouseUp={() => {
              if (volRef.current != null) void ctl('volume_set_' + volRef.current)
              volRef.current = null
            }}
            onKeyUp={() => {
              if (volRef.current != null) void ctl('volume_set_' + volRef.current)
              volRef.current = null
            }}
          />
          )}
        </span>
      </div>
    </div>
  )
}
