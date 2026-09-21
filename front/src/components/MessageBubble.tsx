import { memo, useState } from 'react'
import { Bot, User, AlertTriangle, RefreshCw, Copy, Check, Target, Layers, Users, FileText, Scissors } from 'lucide-react'
import type { Message } from '../types'
import { renderMarkdown } from '../lib/markdown'
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

function MessageBubble({ msg, onRegenerate, canRegenerate }: Props) {
  const isUser = msg.role === 'user'
  const html = !isUser && msg.content ? renderMarkdown(msg.content) : ''

  return (
    <div className={`msg-row ${isUser ? 'msg-user' : 'msg-bot'} fade-up`}>
      <div className={`avatar ${isUser ? 'avatar-user' : 'avatar-bot'}`}>
        {isUser ? <User size={16} /> : <Bot size={16} />}
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
              <span className="tag">
                <FileText size={11} />
                {msg.meta.docs} 条引用
              </span>
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
