import { useCallback, useEffect, useMemo, useState } from 'react'
import { AlertTriangle, Loader2, RefreshCw } from 'lucide-react'

import UsageCharts from './UsageCharts'
import { useStore } from '../store/useStore'
import { usageSummary } from '../lib/api'
import type { UsageSummary } from '../types'

const n = (v: number | null | undefined) => (v ?? 0).toLocaleString('zh-CN')

/**
 * token 用量（独立一页）。答案反馈在另一页（`FeedbackPanel`）—— 两件事的
 * 读者与操作频率都不同，混在一页会互相淹没。
 *
 * 管理员可「看全员」或「只看某个人」；**筛选项由后端 `user` 参数强制执行**，
 * 非管理员传了也只会拿到自己（后端强制，前端不藏选项）。
 */
export default function UsagePanel() {
  const apiBase = useStore((s) => s.settings.apiBase)
  const [days, setDays] = useState(7)
  const [sum, setSum] = useState<UsageSummary | null>(null)
  const [who, setWho] = useState('')          // '' = 全员（仅管理员可用）
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState('')

  const load = useCallback(async () => {
    setLoading(true)
    try {
      setSum(await usageSummary(days, who || undefined, apiBase))
      setErr('')
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [days, who, apiBase])

  useEffect(() => {
    void load()
  }, [load])

  const isAdmin = useStore((s) => s.auth?.role === 'admin')
  const people = useMemo(() => {
    const out: { id: string; name: string }[] = []
    const seen = new Set<string>()
    for (const r of sum?.rows ?? []) {
      const id = r.user_id ?? ''
      if (!id || seen.has(id)) continue
      seen.add(id)
      out.push({ id, name: r.username || id })
    }
    return out
  }, [sum])

  return (
    <div className="usage-panel">
      <div className="usage-bar">
        <div className="set-row-ctl">
          {[1, 7, 30].map((d) => (
            <button
              key={d}
              className={`btn btn-sm ${days === d ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => setDays(d)}
            >
              {d === 1 ? '今天' : `近 ${d} 天`}
            </button>
          ))}
        </div>
        <div className="usage-filters">
          {isAdmin && (
            <select
              className="input usage-select"
              value={who}
              onChange={(e) => setWho(e.target.value)}
              title="按用户查看用量"
            >
              <option value="">全部用户（综合）</option>
              {people.map((p) => (
                <option key={p.id} value={p.id}>{p.name}</option>
              ))}
            </select>
          )}
          <button className="btn btn-ghost btn-sm" onClick={() => void load()} disabled={loading}>
            {loading ? <Loader2 size={13} className="spin" /> : <RefreshCw size={13} />} 刷新
          </button>
        </div>
      </div>

      {err && (
        <div className="kb-warn">
          <AlertTriangle size={15} /> 读取失败：{err}
        </div>
      )}

      {sum && <UsageCharts daily={sum.daily} />}

      {sum && (
        <table className="kb-table">
          <thead>
            <tr>
              <th>{isAdmin && who ? '用户（已筛选）' : '用户'}</th>
              <th>本地（输入 / 输出）</th>
              <th>云端（输入 / 输出）</th>
              <th>调用</th>
            </tr>
          </thead>
          <tbody>
            {sum.rows.length === 0 && (
              <tr>
                <td colSpan={4} className="usage-empty">近 {days} 天没有用量记录</td>
              </tr>
            )}
            {sum.rows.map((r) => (
              <tr key={r.user_id ?? 'none'}>
                <td>{r.username || '（未登录）'}</td>
                <td>{n(r.local_prompt)} / {n(r.local_completion)}</td>
                <td>{n(r.cloud_prompt)} / {n(r.cloud_completion)}</td>
                <td>{n(r.calls)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

    </div>
  )
}
