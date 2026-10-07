import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { History, Search, Download, Trash2, MessageSquare, ArrowRight, Inbox } from 'lucide-react'
import { useStore, previewOf } from '../store/useStore'
import * as api from '../lib/api'
import type { ConversationMeta } from '../types'
import './HistoryPage.css'

export default function HistoryPage() {
  const convs = useStore((s) => s.convs)
  const select = useStore((s) => s.selectConversation)
  const remove = useStore((s) => s.deleteConversation)
  const toast = useStore((s) => s.toast)
  const apiBase = useStore((s) => s.settings.apiBase)
  const navigate = useNavigate()

  const [kw, setKw] = useState('')
  /** 服务端搜索结果；null = 没在搜索，显示全量列表 */
  const [hits, setHits] = useState<ConversationMeta[] | null>(null)
  const [confirmDel, setConfirmDel] = useState<string | null>(null)

  // 搜索走服务端：转录在服务端存着，只过滤手头这份列表就只能搜到标题和最后一条预览。
  // 300ms 防抖，避免每敲一个字打一次请求。
  useEffect(() => {
    const k = kw.trim()
    if (!k) {
      setHits(null)
      return
    }
    const t = setTimeout(async () => {
      try {
        setHits((await api.listConversations(k, apiBase)).conversations)
      } catch {
        setHits(null)   // 搜索失败就退回本地过滤，不打断使用
      }
    }, 300)
    return () => clearTimeout(t)
  }, [kw, apiBase])

  const filtered = useMemo(() => {
    const list = hits ?? [...convs].sort((a, b) => Date.parse(b.updated_at) - Date.parse(a.updated_at))
    const k = kw.trim().toLowerCase()
    if (!k || hits) return list
    return list.filter((c) => c.title.toLowerCase().includes(k) || c.last_content.toLowerCase().includes(k))
  }, [convs, hits, kw])

  // 导出要带全文，而列表里只有摘要 —— 单独拉一次详情
  const fetchDetail = (id: string) => api.getConversation(id, apiBase)

  const download = (name: string, data: unknown) => {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = name
    a.click()
    URL.revokeObjectURL(url)
  }

  const exportOne = async (id: string) => {
    try {
      const det = await fetchDetail(id)
      download(`${det.title.replace(/[\\/:*?"<>|]/g, '_')}.json`, det)
      toast('ok', '已导出会话')
    } catch (err) {
      toast('err', '导出失败：' + (err instanceof Error ? err.message : String(err)))
    }
  }

  const exportAll = async () => {
    if (convs.length === 0) return
    try {
      const all = await Promise.all(convs.map((c) => fetchDetail(c.thread_id)))
      download(`潮声智库-全部会话-${new Date().toISOString().slice(0, 10)}.json`, all)
      toast('ok', `已导出 ${all.length} 个会话`)
    } catch (err) {
      toast('err', '导出失败：' + (err instanceof Error ? err.message : String(err)))
    }
  }

  const openChat = (id: string) => {
    void select(id)
    navigate('/')
  }

  return (
    <div className="page history-page">
      <div className="page-head">
        <div>
          <h2 className="page-title">
            <History size={19} className="grad-text" /> 历史会话
          </h2>
          <p className="page-desc">
            共 {convs.length} 个会话，仅你本人可见。支持检索、续接对话与导出。
          </p>
        </div>
        <button className="btn btn-ghost head-btn" onClick={() => void exportAll()} disabled={convs.length === 0}>
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
          <span>{convs.length === 0 ? '还没有任何会话，去问答页开始吧' : '没有匹配的会话'}</span>
        </div>
      ) : (
        <div className="hist-list">
          {filtered.map((c) => (
            <div key={c.thread_id} className="card hist-item">
              <div className="hist-main" onClick={() => openChat(c.thread_id)}>
                <div className="hist-title">
                  <MessageSquare size={14} />
                  <b>{c.title}</b>
                  <span className="tag">{c.message_count} 条</span>
                </div>
                <div className="hist-preview">{previewOf(c)}</div>
                <div className="hist-time">
                  {new Date(c.updated_at).toLocaleString('zh-CN')} · thread {c.thread_id}
                </div>
              </div>
              <div className="hist-ops">
                <button className="btn btn-icon" title="继续对话" onClick={() => openChat(c.thread_id)}>
                  <ArrowRight size={16} />
                </button>
                <button className="btn btn-icon" title="导出会话" onClick={() => void exportOne(c.thread_id)}>
                  <Download size={15} />
                </button>
                <button
                  className={`btn btn-icon btn-danger ${confirmDel === c.thread_id ? 'armed' : ''}`}
                  title={confirmDel === c.thread_id ? '再点一次确认删除' : '删除'}
                  onClick={() => {
                    if (confirmDel === c.thread_id) {
                      void remove(c.thread_id)
                      setConfirmDel(null)
                      toast('info', '会话已删除')
                    } else {
                      setConfirmDel(c.thread_id)
                      setTimeout(() => setConfirmDel((v) => (v === c.thread_id ? null : v)), 3000)
                    }
                  }}
                >
                  <Trash2 size={15} />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
