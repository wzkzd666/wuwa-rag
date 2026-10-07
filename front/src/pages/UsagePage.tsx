import { BarChart3 } from 'lucide-react'

import UsagePanel from '../components/UsagePanel'
import { useStore } from '../store/useStore'
import './UsagePage.css'

/**
 * token 用量与答案反馈（单独一页）。
 *
 * 可见性完全由**后端**决定：普通用户请求 `/usage/summary` 只会拿到自己那一行
 * （`scope='self'`），管理员拿 `scope=all`。这一页不做任何过滤 ——
 * 越权必须后端拦，前端藏按钮不算防护。
 */
export default function UsagePage() {
  const isAdmin = useStore((s) => s.auth?.role === 'admin')
  const username = useStore((s) => s.auth?.username ?? '')

  return (
    <div className="page usage-page">
      <div className="page-head">
        <div>
          <h2 className="page-title">
            <BarChart3 size={19} className="grad-text" /> 用量
          </h2>
          <p className="page-desc">
            {isAdmin ? '全员 token 用量与答案满意度' : '你的 token 用量与你提交的评价'}
            <span className="usage-who">{username}</span>
          </p>
        </div>
      </div>

      <section className="card usage-card">
        <UsagePanel />
      </section>
    </div>
  )
}
