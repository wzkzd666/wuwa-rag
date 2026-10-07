import { useCallback, useEffect, useState } from 'react'
import { ChevronDown, Loader2, RefreshCw, ThumbsDown, ThumbsUp, Trash2 } from 'lucide-react'

import { useStore } from '../store/useStore'
import { feedbackDelete, feedbackList } from '../lib/api'
import type { FeedbackList } from '../types'
import './UsagePanel.css'

/**
 * 答案反馈（独立一页）。
 *
 * 可见性由后端决定：普通用户只拿到自己的反馈，管理员拿到全员；每条带 `can_delete`
 * （管理员任意 / 普通用户仅自己的），前端只照办。
 */
export default function FeedbackPanel({ days }: { days: number }) {
  const apiBase = useStore((s) => s.settings.apiBase)
  const toast = useStore((s) => s.toast)
  const [fb, setFb] = useState<FeedbackList | null>(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState('')
  const [openFa, setOpenFa] = useState(true)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      setFb(await feedbackList(days, apiBase))
      setErr('')
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [days, apiBase])

  useEffect(() => {
    void load()
  }, [load])

  const remove = async (id: number) => {
    try {
      await feedbackDelete(id, apiBase)
      setFb((prev) => (prev ? { ...prev, items: prev.items.filter((i) => i.id !== id) } : prev))
      void load()
      toast('ok', '已删除该条反馈')
    } catch (e) {
      toast('err', e instanceof Error ? e.message : String(e))
    }
  }

  const st = fb?.summary
  const downRate = st?.down_ratio

  return (
    <div className="usage-panel">
      <div className="usage-bar">
        <div className="fb-fold-stats">
          <span className="tag tag-ok"><ThumbsUp size={11} /> {st?.up ?? 0}</span>
          <span className="tag tag-err"><ThumbsDown size={11} /> {st?.down ?? 0}</span>
          <span className="fb-fold-rate">
            差评率 {downRate === null || downRate === undefined ? '—' : `${(downRate * 100).toFixed(0)}%`}
          </span>
        </div>
        <button className="btn btn-ghost btn-sm" onClick={() => void load()} disabled={loading}>
          {loading ? <Loader2 size={13} className="spin" /> : <RefreshCw size={13} />} 刷新
        </button>
      </div>

      {err && <div className="kb-warn">{err}</div>}

      <div className={`fb-fold ${openFa ? 'fb-fold-open' : ''}`}>
        <button className="fb-fold-head" onClick={() => setOpenFa((v) => !v)}>
          <ChevronDown size={15} className="fb-fold-caret" />
          <span className="fb-fold-title">全部反馈（{fb?.items.length ?? 0}）</span>
        </button>
        {openFa && (
          <div className="fb-fold-body">
            {fb && fb.items.length === 0 && <div className="usage-empty">近 {days} 天没有反馈</div>}
            {fb?.items.map((it) => (
              <div key={it.id} className={`fb-item ${it.rating === -1 ? 'fb-item-bad' : ''}`}>
                <div className="fb-item-head">
                  <b>{it.username}</b>
                  <span className="fb-item-time">
                    {new Date(it.created_at || '').toLocaleString('zh-CN')}
                  </span>
                  {it.rating === -1
                    ? <span className="tag tag-err">不满意</span>
                    : <span className="tag tag-ok">满意</span>}
                  {it.can_delete && (
                    <button className="msg-action fb-del" title="删除该条反馈"
                            onClick={() => void remove(it.id)}>
                      <Trash2 size={12} />
                    </button>
                  )}
                </div>
                {it.question && <p className="fb-item-q">问：{it.question}</p>}
                {it.answer && <p className="fb-item-a">答：{it.answer}</p>}
                {it.comment && <p className="fb-item-c">补充：{it.comment}</p>}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
