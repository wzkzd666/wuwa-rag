import { fmtTime } from '../lib/format'
import { useCallback, useEffect, useState } from 'react'
import { ChevronLeft, ChevronRight, Loader2, RefreshCw, ShieldCheck, Trash2, Users } from 'lucide-react'

import { useStore } from '../store/useStore'
import ConfirmButton from './ConfirmButton'
import LoadFail from './LoadFail'
import { adminDeleteUser, adminUsers } from '../lib/api'
import type { AdminUserRow } from '../types'

/**
 * 管理员用户清单（GET /admin/users）：账号、角色、注册时间、活跃画像条数，
 * 以及**移除账号**（连带回收他名下的会话/画像/评价/用量/偏好/加密凭据）。
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
  const me = useStore((s) => s.auth?.username ?? '')
  const toast = useStore((s) => s.toast)
  const [rows, setRows] = useState<AdminUserRow[]>([])
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState('')
  const [page, setPage] = useState(1)
  /** 正在移除的那个用户名：只灰那一行的按钮，不把整张表锁住 */
  const [removing, setRemoving] = useState('')

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

  const removeUser = async (r: AdminUserRow) => {
    setRemoving(r.username)
    try {
      const out = await adminDeleteUser(r.id, apiBase)
      const p = out.purged
      toast('ok', `已移除 ${out.removed}：会话 ${p.sessions} · 画像 ${p.facts} · 评价 ${p.feedback} · 用量 ${p.usage}`)
      await load()
    } catch (e) {
      // 后端的 detail 本身就是原因（「不能删除自己这个账号」/「这是最后一个管理员」/
      // 非管理员的 403 文案），原样回显；前端不另写一套判断（同一语义两处判据必漂移）
      toast('err', e instanceof Error ? e.message : String(e))
    } finally {
      setRemoving('')
    }
  }

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
        <LoadFail what="用户清单" error={err} onRetry={() => void load()}
                  note="这不代表一个用户都没有" />
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
                <th className="kb-act-col">操作</th>
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
                  <td>
                    {/* 破坏性操作走全站唯一的两步确认实现；`title` 写清回收范围（不可恢复），
                        待确认态也不会被抑制 —— 那一刻正是最需要读到后果的瞬间。 */}
                    <ConfirmButton
                      className="btn btn-ghost btn-sm"
                      icon={Trash2}
                      iconSize={13}
                      label="移除"
                      confirmLabel="再点一次永久移除"
                      busy={removing === r.username}
                      disabled={removing !== '' || r.username === me}
                      title={r.username === me
                        ? '不能删除自己这个账号'
                        : `移除 ${r.username}：账号与令牌、会话与转录、模型侧记忆、画像事实、评价、用量、偏好、云端凭据密文，全部不可恢复`}
                      onConfirm={() => void removeUser(r)}
                    />
                  </td>
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
