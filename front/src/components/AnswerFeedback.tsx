import { useState } from 'react'
import { Check, ThumbsDown, ThumbsUp, X } from 'lucide-react'

import { useStore } from '../store/useStore'
import { feedbackSubmit } from '../lib/api'
import type { Message } from '../types'

/**
 * 单条回答的满意度反馈。
 *
 * 交互：点 👍 直接提交；点 👎 展开可选的补充说明（不满意往往说不清哪里不对，
 * 给了文字框才有分析价值）。
 *
 * ⚠️ 提交前**明确告知**管理员会查看这次问答 —— 反馈里带着问题与回答原文，
 * 这是知情而不是偷偷收集。反馈内容会随截图/导出离开本机，所以文案写得很直白。
 */
export default function AnswerFeedback({ msg, question }: { msg: Message; question: string }) {
  const apiBase = useStore((s) => s.settings.apiBase)
  const threadId = useStore((s) => s.activeThreadId)
  const toast = useStore((s) => s.toast)
  const [rated, setRated] = useState<0 | 1 | -1>(0)
  const [open, setOpen] = useState(false)
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)

  const send = async (rating: 1 | -1, comment: string) => {
    setBusy(true)
    try {
      await feedbackSubmit(
        {
          thread_id: threadId || 'unknown',
          target_id: msg.id,
          rating,
          comment,
          question: question.slice(0, 500),
          answer: (msg.content || '').slice(0, 2000),
        },
        apiBase,
      )
      setRated(rating)
      setOpen(false)
      setText('')
      toast('ok', rating === 1 ? '已记录，谢谢' : '已记录，管理员会看到这条差评')
    } catch (err) {
      toast('err', err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="fb-wrap">
      <button
        className={`msg-action fb-btn ${rated === 1 ? 'fb-on' : ''}`}
        title="这条回答有帮助"
        disabled={busy}
        onClick={() => void send(1, '')}
      >
        {rated === 1 ? <Check size={13} /> : <ThumbsUp size={13} />}
      </button>
      <button
        className={`msg-action fb-btn ${rated === -1 ? 'fb-bad' : ''}`}
        title="这条回答不准确 / 未解决问题（可补充说明）"
        disabled={busy}
        onClick={() => setOpen((v) => !v)}
      >
        {rated === -1 ? <Check size={13} /> : <ThumbsDown size={13} />}
      </button>

      {open && (
        <div className="fb-panel">
          <div className="fb-panel-head">
            <b>哪里不对？</b>
            <button className="msg-action" title="关闭" onClick={() => setOpen(false)}>
              <X size={12} />
            </button>
          </div>
          <textarea
            className="fb-textarea"
            value={text}
            placeholder="例如：答的是守岸人，我问的是心；COST3 说错了……（可留空）"
            onChange={(e) => setText(e.target.value)}
            rows={3}
          />
          <div className="fb-panel-foot">
            <span className="fb-notice">
              提交后**管理员可查看这次问答原文**用于改进（不含你的账号密码等信息）。
            </span>
            <button
              className="btn btn-primary btn-sm"
              disabled={busy}
              onClick={() => void send(-1, text.trim())}
            >
              提交
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
