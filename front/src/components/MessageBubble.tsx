import { memo, useEffect, useRef, useState } from 'react'
import { Bot, User, AlertTriangle, RefreshCw, Copy, Check, Target, Layers, Users, FileText, Scissors, ChevronDown, Volume2, Square } from 'lucide-react'
import type { Message } from '../types'
import { renderMarkdown } from '../lib/markdown'
import * as api from '../lib/api'
import { useStore } from '../store/useStore'
import './MessageBubble.css'

const INTENT_LABEL: Record<string, string> = {
  fact: '事实查询',
  semantic: '语义问答',
  hybrid: '混合检索',
  chitchat: '闲聊',
}

function CopyBtn({ text }: { text: string }) {
  const [ok, setOk] = useState(false)
  return (
    <button
      className="msg-action"
      title="复制"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text)
          setOk(true)
          setTimeout(() => setOk(false), 1500)
        } catch {
          /* 忽略 */
        }
      }}
    >
      {ok ? <Check size={13} /> : <Copy size={13} />}
    </button>
  )
}

/** TTS 播放按钮：调后端 /tts 拿音频 URL 再播放。
 *
 * 三态：idle（播放图标）→ loading（转圈）→ playing（方块=停止）。
 * ⚠️ 后端 TTS 默认关闭，此时返回 **200 + ok=false + 可读 reason**，
 *    所以不能只看 HTTP 状态，必须检查 ok 字段并把 reason 弹 toast。
 * ⚠️ 卸载/重播时必须 pause + 解绑 src，否则连续点几个气泡会叠音。
 */
function TtsBtn({ text, emotion }: { text: string; emotion?: string }) {
  const [state, setState] = useState<'idle' | 'loading' | 'playing'>('idle')
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const toast = useStore((s) => s.toast)
  const apiBase = useStore((s) => s.settings.apiBase)

  // 卸载时停止播放，避免离开页面后音频还在响
  useEffect(() => () => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current = null
    }
  }, [])

  const stop = () => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current = null
    }
    setState('idle')
  }

  const play = async () => {
    setState('loading')
    try {
      const out = await api.tts(text, emotion || '', apiBase)
      if (!out.ok || !out.url) {
        setState('idle')
        toast('err', out.error || '语音合成失败')
        return
      }
      const audio = new Audio(out.url)
      audioRef.current = audio
      audio.onended = () => { audioRef.current = null; setState('idle') }
      // 播放失败（浏览器策略/URL 过期）也要回到 idle，否则按钮卡在 playing
      audio.onerror = () => {
        audioRef.current = null
        setState('idle')
        toast('err', '音频播放失败，链接可能已过期（有效期 24 小时）')
      }
      await audio.play()
      setState('playing')
    } catch (err) {
      setState('idle')
      toast('err', err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <button
      className="msg-action"
      title={state === 'playing' ? '停止播放' : '朗读这条回答'}
      disabled={state === 'loading'}
      onClick={() => (state === 'playing' ? stop() : void play())}
    >
      {state === 'loading' ? <RefreshCw size={13} className="spin" />
        : state === 'playing' ? <Square size={12} />
        : <Volume2 size={13} />}
    </button>
  )
}

interface Props {
  msg: Message
  onRegenerate?: (id: string) => void
  canRegenerate?: boolean
}

/**
 * 引用来源折叠面板：默认只显示「N 条引用」，点开列出召回文档的面包屑
 * （`角色 › 模块 › 组件`，后端 doc_sources 去重保序返回）。
 * 无 sources 时退化为纯展示标签（兼容旧会话记录）。
 */
function SourcePanel({ count, sources }: { count: number; sources?: string[] }) {
  const [open, setOpen] = useState(false)
  const has = !!sources && sources.length > 0
  if (!has) {
    return (
      <span className="tag">
        <FileText size={11} />
        {count} 条引用
      </span>
    )
  }
  return (
    <div className={`tag tag-src${open ? ' open' : ''}`}>
      <button
        type="button"
        className="src-toggle"
        onClick={() => setOpen((v) => !v)}
        title="查看引用来源"
      >
        <FileText size={11} />
        {count} 条引用
        <ChevronDown size={11} className="src-caret" />
      </button>
      {open && (
        <ul className="src-list">
          {sources.map((s) => (
            <li key={s}>{s}</li>
          ))}
        </ul>
      )}
    </div>
  )
}

function MessageBubble({ msg, onRegenerate, canRegenerate }: Props) {
  const isUser = msg.role === 'user'
  const html = !isUser && msg.content ? renderMarkdown(msg.content) : ''
  // 个性化头像：设置里上传后，气泡头像用图；空则回落默认图标
  const avatarAssistant = useStore((s) => s.settings.avatarAssistant)
  const avatarUser = useStore((s) => s.settings.avatarUser)
  // 语音开关：关闭时不渲染播放按钮（后端 TTS 默认也是关的，两边口径一致）
  const ttsEnabled = useStore((s) => s.settings.ttsEnabled)

  return (
    <div className={`msg-row ${isUser ? 'msg-user' : 'msg-bot'} fade-up`}>
      <div className={`avatar ${isUser ? 'avatar-user' : 'avatar-bot'}`}>
        {isUser ? (
          avatarUser ? <img src={avatarUser} alt="我" /> : <User size={16} />
        ) : (
          avatarAssistant ? <img src={avatarAssistant} alt="AI" /> : <Bot size={16} />
        )}
      </div>

      <div className="msg-main">
        <div className={`bubble ${isUser ? 'bubble-user' : 'bubble-bot'}`}>
          {isUser ? (
            <span className="msg-plain">{msg.content}</span>
          ) : msg.status === 'retrieving' && !msg.content ? (
            <span className="retrieving">
              <span className="dots">
                <i />
                <i />
                <i />
              </span>
              {msg.stageLabel ? `${msg.stageLabel}…` : '正在检索知识库…'}
            </span>
          ) : msg.status === 'error' ? (
            <div className="msg-error">
              <AlertTriangle size={15} />
              <div>
                <b>请求出错</b>
                <p>{msg.error || '未知错误'}</p>
                {msg.content && <pre className="msg-error-partial">{msg.content}</pre>}
              </div>
            </div>
          ) : (
            <>
              <div className="markdown" dangerouslySetInnerHTML={{ __html: html }} />
              {msg.streaming && <span className="caret" />}
            </>
          )}
        </div>

        {/* 元数据 chips */}
        {!isUser && msg.meta && msg.status !== 'error' && (
          <div className="meta-row">
            {msg.meta.intent && (
              <span className="tag tag-violet">
                <Target size={11} />
                {INTENT_LABEL[msg.meta.intent] || msg.meta.intent}
              </span>
            )}
            {msg.meta.characters && msg.meta.characters.length > 0 && (
              <span className="tag tag-accent">
                <Users size={11} />
                {msg.meta.characters.join('、')}
              </span>
            )}
            {msg.meta.slots && msg.meta.slots.length > 0 && (
              <span className="tag tag-pink">
                <Layers size={11} />
                {msg.meta.slots.join('·')}
              </span>
            )}
            {typeof msg.meta.docs === 'number' && msg.meta.docs > 0 && (
              <SourcePanel count={msg.meta.docs} sources={msg.meta.sources} />
            )}
            {msg.meta.truncated && (
              <span className="tag tag-warn" title="模型出现重复输出，已自动截断">
                <Scissors size={11} />
                已截断重复内容
              </span>
            )}
          </div>
        )}

        {/* 操作栏 */}
        {!isUser && !msg.streaming && msg.content && msg.status !== 'error' && (
          <div className="msg-actions">
            <CopyBtn text={msg.content} />
            {ttsEnabled && <TtsBtn text={msg.content} emotion={msg.meta?.emotion} />}
            {canRegenerate && onRegenerate && (
              <button className="msg-action" title="重新生成" onClick={() => onRegenerate(msg.id)}>
                <RefreshCw size={13} />
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default memo(MessageBubble)
