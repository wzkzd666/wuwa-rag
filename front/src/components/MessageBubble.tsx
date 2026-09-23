import { memo, useState } from 'react'
import { Bot, User, AlertTriangle, RefreshCw, Copy, Check, Target, Layers, Users, FileText, Scissors, ChevronDown } from 'lucide-react'
import type { Message } from '../types'
import { renderMarkdown } from '../lib/markdown'
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
