import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { History, Search, Download, Trash2, MessageSquare, ArrowRight, Inbox } from 'lucide-react'
import { useStore } from '../store/useStore'
import './HistoryPage.css'

export default function HistoryPage() {
  const conversations = useStore((s) => s.conversations)
  const select = useStore((s) => s.selectConversation)
  const remove = useStore((s) => s.deleteConversation)
  const toast = useStore((s) => s.toast)
  const navigate = useNavigate()

  const [kw, setKw] = useState('')
  const [confirmDel, setConfirmDel] = useState<string | null>(null)

  const filtered = useMemo(() => {
    const k = kw.trim().toLowerCase()
    const list = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt)
    if (!k) return list
    return list.filter(
      (c) =>
        c.title.toLowerCase().includes(k) ||
        c.messages.some((m) => m.content.toLowerCase().includes(k)),
    )
  }, [conversations, kw])

  const exportOne = (id: string) => {
    const c = conversations.find((x) => x.id === id)
    if (!c) return
    const blob = new Blob([JSON.stringify(c, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${c.title.replace(/[\\/:*?"<>|]/g, '_')}.json`
    a.click()
    URL.revokeObjectURL(url)
    toast('ok', '已导出会话 JSON')
  }

  const exportAll = () => {
    if (conversations.length === 0) return
    const blob = new Blob([JSON.stringify(conversations, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `潮声智库-全部会话-${new Date().toISOString().slice(0, 10)}.json`
    a.click()
    URL.revokeObjectURL(url)
    toast('ok', `已导出 ${conversations.length} 个会话`)
  }

  const openChat = (id: string) => {
    select(id)
    navigate('/')
  }

  return (
    <div className="page history-page">
      <div className="page-head">
        <div>
          <h2 className="page-title">
            <History size={19} className="grad-text" /> 历史会话
          </h2>
          <p className="page-desc">共 {conversations.length} 个会话，本地保存于浏览器。可搜索、继续对话或导出为 JSON。</p>
        </div>
        <button className="btn btn-ghost head-btn" onClick={exportAll} disabled={conversations.length === 0}>
          <Download size={15} /> 导出全部
        </button>
      </div>

      <div className="hist-search">
        <Search size={15} />
        <input
          value={kw}
          placeholder="搜索标题或消息内容…"
          onChange={(e) => setKw(e.target.value)}
        />
      </div>

      {filtered.length === 0 ? (
        <div className="empty-state">
          <Inbox size={30} />
          <span>{conversations.length === 0 ? '还没有任何会话，去问答页开始吧' : '没有匹配的会话'}</span>
        </div>
      ) : (
        <div className="hist-list">
          {filtered.map((c) => {
            const last = c.messages[c.messages.length - 1]
            const preview = last
              ? (last.role === 'user' ? '你：' : '爱弥斯：') + last.content.replace(/\s+/g, ' ').slice(0, 60)
              : '空会话'
            return (
              <div key={c.id} className="card hist-item">
                <div className="hist-main" onClick={() => openChat(c.id)}>
                  <div className="hist-title">
                    <MessageSquare size={14} />
                    <b>{c.title}</b>
                    <span className="tag">{c.messages.length} 条</span>
                  </div>
                  <div className="hist-preview">{preview || '（无内容）'}</div>
                  <div className="hist-time">
                    {new Date(c.updatedAt).toLocaleString('zh-CN')} · thread {c.threadId}
                  </div>
                </div>
                <div className="hist-ops">
                  <button className="btn btn-icon" title="继续对话" onClick={() => openChat(c.id)}>
                    <ArrowRight size={16} />
                  </button>
                  <button className="btn btn-icon" title="导出 JSON" onClick={() => exportOne(c.id)}>
                    <Download size={15} />
                  </button>
                  <button
                    className={`btn btn-icon btn-danger ${confirmDel === c.id ? 'armed' : ''}`}
                    title={confirmDel === c.id ? '再点一次确认删除' : '删除'}
                    onClick={() => {
                      if (confirmDel === c.id) {
                        remove(c.id)
                        setConfirmDel(null)
                        toast('info', '会话已删除')
                      } else {
                        setConfirmDel(c.id)
                        setTimeout(() => setConfirmDel((v) => (v === c.id ? null : v)), 3000)
                      }
                    }}
                  >
                    <Trash2 size={15} />
                  </button>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
