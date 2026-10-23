import { fmtTime } from '../lib/format'
import { useCallback, useEffect, useState } from 'react'
import { AlertTriangle, ChevronLeft, ChevronRight, Loader2, RefreshCw, ShieldCheck, Users } from 'lucide-react'

import { useStore } from '../store/useStore'
import { adminUsers } from '../lib/api'
import type { AdminUserRow } from '../types'

/**
 * 管理员用户清单（GET /admin/users）：账号、角色、注册时间、活跃画像条数。
 *
 * 挂载在历史会话页底部 —— 那是管理员已有的「翻查沉淀数据」的页面，开新路由
 * 需要动 Layout.tsx 的导航（不在本次改动范围）。
 *
 * ⚠️ 类名沿用 KnowledgePage 的卡片/表格词汇（kb-records / kb-table / kb-pager，
 * 全局 CSS 随打包生效，UsagePanel 已是先例），故意**不新增任何样式**。
 * 「画像条数」是服务端算出来的只读列，界面上不提供任何可手改的字段。
 */

/** 人数随部署增长；属主规则：长列表必须可折叠或分页。这里选分页（与已收录角色同口径）。 */
const PAGE_SIZE = 15

export default function AdminUsersCard() {
  const apiBase = useStore((s) => s.settings.apiBase)
  const [rows, setRows] = useState<AdminUserRow[]>([])
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState('')
  const [page, setPage] = useState(1)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const out = await adminUsers(apiBase)
      setRows(out.users)
      setErr('')
    } catch (e) {
      // 401 → UnauthorizedError（消息即「登录已过期」）；403 的 detail 是后端写好的中文，
      // 都会原样落进这一行警示条 —— 不必再按状态码猜文案。
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [apiBase])

  useEffect(() => {
    void load()
  }, [load])

  const totalPages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE))
  // 页码只做越界收敛（与 KnowledgePage 同做法），不写额外 effect
  const safePage = Math.min(page, totalPages)
  const pageRows = rows.slice((safePage - 1) * PAGE_SIZE, safePage * PAGE_SIZE)

  return (
    <section className="card kb-records">
      <div className="kb-owned-head">
        <h3>
          <Users size={15} /> 用户清单{rows.length > 0 && <span className="kb-count">{rows.length}</span>}
        </h3>
        <div className="kb-owned-tools">
          <button className="btn btn-ghost btn-sm" onClick={() => void load()} disabled={loading}>
            {loading ? <Loader2 size={13} className="spin" /> : <RefreshCw size={13} />} 刷新
          </button>
        </div>
      </div>

      {err && (
        <div className="kb-warn">
          <AlertTriangle size={15} /> 读取用户清单失败：{err}
        </div>
      )}

      {!err && rows.length === 0 && !loading ? (
        <div className="empty-state">
          <Users size={26} />
          <span>还没有任何注册用户</span>
        </div>
      ) : (
        !err && (
          <table className="kb-table">
            <thead>
              <tr>
                <th>用户</th>
                <th>角色</th>
                <th>注册时间</th>
                <th className="kb-act-col">画像条数</th>
              </tr>
            </thead>
            <tbody>
              {pageRows.map((r) => (
                <tr key={r.username}>
                  <td className="kb-char">{r.username}</td>
                  <td>
                    {r.role === 'admin' ? (
                      <span className="tag tag-violet">
                        <ShieldCheck size={11} /> 管理员
                      </span>
                    ) : (
                      <span className="tag">游客</span>
                    )}
                  </td>
                  <td className="kb-time">{fmtTime(r.created_at)}</td>
                  <td className="kb-mono">{r.facts}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )
      )}

      {totalPages > 1 && (
        <div className="kb-pager">
          <span className="kb-pager-info">
            第 {safePage} / {totalPages} 页 · 共 {rows.length} 个用户
          </span>
          <div className="kb-pager-btns">
            <button className="btn btn-ghost btn-sm" onClick={() => setPage(safePage - 1)} disabled={safePage <= 1}>
              <ChevronLeft size={13} /> 上一页
            </button>
            <button className="btn btn-ghost btn-sm" onClick={() => setPage(safePage + 1)} disabled={safePage >= totalPages}>
              下一页 <ChevronRight size={13} />
            </button>
          </div>
        </div>
      )}
    </section>
  )
}
