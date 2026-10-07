import { useState } from 'react'
import { MessageSquare } from 'lucide-react'

import FeedbackPanel from '../components/FeedbackPanel'
import { useStore } from '../store/useStore'
import './FeedbackPage.css'

/**
 * 答案反馈（独立一页）。
 *
 * 与「用量」分开：两件事的读者与查看频率不同 —— 用量是每天扫一眼的趋势，
 * 反馈是**逐条读内容**的，混在一页会互相淹没。
 */
export default function FeedbackPage() {
  const isAdmin = useStore((s) => s.auth?.role === 'admin')
  const username = useStore((s) => s.auth?.username ?? '')
  const [days, setDays] = useState(7)

  return (
    <div className="page feedback-page">
      <div className="page-head">
        <div>
          <h2 className="page-title">
            <MessageSquare size={19} className="grad-text" /> 答案反馈
          </h2>
          <p className="page-desc">
            {isAdmin ? '全部用户的答案评价' : '你提交的评价'}
            <span className="usage-who">{username}</span>
          </p>
        </div>
      </div>

      <section className="card usage-card">
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
        </div>
        <FeedbackPanel days={days} />
      </section>
    </div>
  )
}
