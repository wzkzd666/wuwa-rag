import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Plus, MessageSquare, Trash2, Pencil, Check, X, Sparkles, Library, Swords, Gem, Coins } from 'lucide-react'
import { useStore, ensureActiveConversation } from '../store/useStore'
import MessageBubble from '../components/MessageBubble'
import Composer from '../components/Composer'
import './ChatPage.css'

const EXAMPLES = [
  { icon: Gem, text: '卡卡罗毕业配装用什么声骸？', tag: '配装' },
  { icon: Coins, text: '忌炎六阶突破要多少贝币？', tag: '突破' },
  { icon: Swords, text: '长离的共鸣链效果是什么？', tag: '共鸣链' },
  { icon: Sparkles, text: '今汐怎么玩？', tag: '攻略' },
]

/** 左侧会话列表 */
function SessionList() {
  const conversations = useStore((s) => s.conversations)
  const activeId = useStore((s) => s.activeId)
  const select = useStore((s) => s.selectConversation)
  const create = useStore((s) => s.newConversation)
  const remove = useStore((s) => s.deleteConversation)
  const rename = useStore((s) => s.renameConversation)
  const busyId = useStore((s) => s.busyId)
  const navigate = useNavigate()

  const [editing, setEditing] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  // 两步删除确认：首次点击标记，再次点击才删
  const [confirmDel, setConfirmDel] = useState<string | null>(null)
  // 收录入口仅管理员可见（后端 /ingest 也是 admin 守卫）
  const isAdmin = useStore((s) => s.auth?.role === 'admin')

  const startEdit = (id: string, title: string) => {
    setEditing(id)
    setDraft(title)
  }
  const commit = (id: string) => {
    rename(id, draft)
    setEditing(null)
  }

  useEffect(() => {
    if (!confirmDel) return
    const t = setTimeout(() => setConfirmDel(null), 3000)
    return () => clearTimeout(t)
  }, [confirmDel])

  return (
    <div className="session-list">
      <button className="btn btn-primary new-chat" onClick={() => create()}>
        <Plus size={15} /> 新建对话
      </button>
      <div className="session-scroll">
        {conversations.length === 0 && (
          <div className="session-empty">暂无会话</div>
        )}
        {conversations.map((c) => (
          <div
            key={c.id}
            className={`session-item ${c.id === activeId ? 'session-active' : ''}`}
            onClick={() => !busyId && select(c.id)}
          >
            <MessageSquare size={14} className="session-icon" />
            {editing === c.id ? (
              <span className="session-edit" onClick={(e) => e.stopPropagation()}>
                <input
                  autoFocus
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') commit(c.id)
                    if (e.key === 'Escape') setEditing(null)
                  }}
                />
                <button onClick={() => commit(c.id)} title="保存">
                  <Check size={13} />
                </button>
                <button onClick={() => setEditing(null)} title="取消">
                  <X size={13} />
                </button>
              </span>
            ) : (
              <>
                <span className="session-title">{c.title}</span>
                <span className="session-ops" onClick={(e) => e.stopPropagation()}>
                  <button title="重命名" onClick={() => startEdit(c.id, c.title)}>
                    <Pencil size={12} />
                  </button>
                  <button
                    title={confirmDel === c.id ? '再点一次确认删除' : '删除'}
                    className={confirmDel === c.id ? 'danger-confirm' : ''}
                    onClick={() => {
                      if (confirmDel === c.id) {
                        remove(c.id)
                        setConfirmDel(null)
                      } else {
                        setConfirmDel(c.id)
                      }
                    }}
                  >
                    <Trash2 size={12} />
                  </button>
                </span>
              </>
            )}
          </div>
        ))}
      </div>
      {isAdmin && (
        <button className="btn btn-ghost goto-knowledge" onClick={() => navigate('/knowledge')}>
          <Library size={14} /> 收录新角色
        </button>
      )}
    </div>
  )
}

export default function ChatPage() {
  const activeId = useStore((s) => s.activeId)
  const conversations = useStore((s) => s.conversations)
  const busyId = useStore((s) => s.busyId)
  const send = useStore((s) => s.send)
  const stop = useStore((s) => s.stop)
  const regenerate = useStore((s) => s.regenerate)
  const health = useStore((s) => s.health)
  const avatarAssistant = useStore((s) => s.settings.avatarAssistant)
  const navigate = useNavigate()

  const conv = conversations.find((c) => c.id === activeId)
  const messages = conv?.messages ?? []
  const busy = busyId === activeId

  const scrollRef = useRef<HTMLDivElement>(null)

  // 新消息自动滚到底部
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
  }, [messages.length, messages[messages.length - 1]?.content])

  // 监听侧栏「收录新角色」跳转
  useEffect(() => {
    const handler = (e: Event) => navigate((e as CustomEvent).detail)
    window.addEventListener('wuwa-nav', handler)
    return () => window.removeEventListener('wuwa-nav', handler)
  }, [navigate])

  // 保证有活跃会话
  useEffect(() => {
    if (!activeId) ensureActiveConversation()
  }, [activeId])

  const isEmpty = messages.length === 0

  return (
    <div className="chat-page">
      <SessionList />

      <div className="chat-main">
        <div className="chat-scroll" ref={scrollRef}>
          {isEmpty ? (
            <div className="chat-welcome">
              <div className="welcome-logo">
                {avatarAssistant ? <img src={avatarAssistant} alt="爱弥斯" /> : <Sparkles size={26} />}
              </div>
              <h1>
                你好，我是<b className="grad-text">爱弥斯</b>
              </h1>
              <p className="welcome-sub">
                鸣潮角色知识助手 · 声骸配装 / 突破材料 / 共鸣链 / 角色攻略，问我吧
              </p>
              {health === 'down' && (
                <div className="welcome-warn">
                  后端服务未连接（127.0.0.1:8000）。请先启动 FastAPI 与 Celery worker，或在「设置」中检查 API 地址。
                </div>
              )}
              <div className="example-grid">
                {EXAMPLES.map(({ icon: Icon, text, tag }) => (
                  <button key={text} className="example-card" onClick={() => send(text)}>
                    <span className="example-icon">
                      <Icon size={16} />
                    </span>
                    <span className="example-text">{text}</span>
                    <span className="tag">{tag}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="msg-list">
              {messages.map((m, i) => (
                <MessageBubble
                  key={m.id}
                  msg={m}
                  canRegenerate={m.role === 'assistant' && i === messages.length - 1 && !busy}
                  onRegenerate={regenerate}
                />
              ))}
            </div>
          )}
        </div>

        <Composer
          onSend={send}
          onStop={stop}
          busy={busy}
          placeholder={health === 'down' ? '后端离线，仍可输入（将报错）…' : undefined}
        />
      </div>
    </div>
  )
}
