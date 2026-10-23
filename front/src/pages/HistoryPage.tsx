import { fmtTime } from '../lib/format'
import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { History, Search, Download, Trash2, MessageSquare, ArrowRight, Inbox, Loader2 } from 'lucide-react'
import { useStore, previewOf } from '../store/useStore'
import AdminUsersCard from '../components/AdminUsersCard'
import ConfirmButton from '../components/ConfirmButton'
import * as api from '../lib/api'
import type { ConversationMeta } from '../types'
import './HistoryPage.css'

export default function HistoryPage() {
  const convs = useStore((s) => s.convs)
  // 加载期与空态必须分开（这个字段本来就是为了这个用途加的，别让它再次只写不读）
  const loadingConvs = useStore((s) => s.loadingConvs)
  const select = useStore((s) => s.selectConversation)
  const remove = useStore((s) => s.deleteConversation)
  const toast = useStore((s) => s.toast)
  const apiBase = useStore((s) => s.settings.apiBase)
  // 与 KnowledgePage 同一道门禁：藏卡片只是体验，真拦人是后端 require_admin
  const isAdmin = useStore((s) => s.auth?.role === 'admin')
  const navigate = useNavigate()

  const [kw, setKw] = useState('')
  /** 服务端搜索结果；null = 没在搜索，显示全量列表 */
  const [hits, setHits] = useState<ConversationMeta[] | null>(null)

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

      {loadingConvs && convs.length === 0 ? (
        <div className="empty-state">
          <Loader2 size={30} className="spin" />
          <span>正在读取会话列表…</span>
        </div>
      ) : filtered.length === 0 ? (
        <div className="empty-state">
          <Inbox size={30} />
          <span>{convs.length === 0 ? '还没有任何会话，去问答页开始吧' : '没有匹配的会话'}</span>
        </div>
      ) : (
        <div className="hist-list">
          {filtered.map((c) => (
            <div key={c.thread_id} className="card hist-item">
              {/* 点开会话是这页的主动作，原来却是个只挂了 onClick 的 div ——
                  键盘 Tab 不进去、读屏念不出「这是个可点的行」。改成真的 <button>，
                  类名保持不变（外观由 HistoryPage.css 里紧挨着的那条重置规则兜平）。 */}
              <button type="button" className="hist-main" onClick={() => openChat(c.thread_id)}>
                <div className="hist-title">
                  <MessageSquare size={14} />
                  <b>{c.title}</b>
                  <span className="tag">{c.message_count} 条</span>
                </div>
                <div className="hist-preview">{previewOf(c)}</div>
                <div className="hist-time">
                  {fmtTime(c.updated_at)} · thread {c.thread_id}
                </div>
              </button>
              <div className="hist-ops">
                <button className="btn btn-icon" title="继续对话" onClick={() => openChat(c.thread_id)}>
                  <ArrowRight size={16} />
                </button>
                <button className="btn btn-icon" title="导出会话" onClick={() => void exportOne(c.thread_id)}>
                  <Download size={15} />
                </button>
                <ConfirmButton
                  className="btn btn-icon btn-danger"
                  icon={Trash2}
                  iconSize={15}
                  title="删除"
                  confirmLabel="再点一次确认删除"
                  onConfirm={async () => {
                    // 成功了才说「已删除」：remove 失败时会自己弹「删除失败：…」，
                    // 无条件报成功就会出现同一屏里「失败」与「已删除」并列
                    // （与 store.clearAll 同一条判据，别再各写一遍）。
                    if (await remove(c.thread_id)) toast('info', '会话已删除')
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* 管理员用户清单挂在本页底部：这里已是「翻查沉淀数据」的页面，
          开新路由要动 Layout 的导航，代价大于收益（详见组件文件头注释）。 */}
      {isAdmin && <AdminUsersCard />}
    </div>
  )
}
