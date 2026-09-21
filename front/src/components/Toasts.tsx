import { CheckCircle2, XCircle, Info, X } from 'lucide-react'
import { useStore } from '../store/useStore'
import './Toasts.css'

export default function Toasts() {
  const toasts = useStore((s) => s.toasts)
  const dismiss = useStore((s) => s.dismissToast)

  if (toasts.length === 0) return null

  return (
    <div className="toast-wrap">
      {toasts.map((t) => (
        <div key={t.id} className={`toast toast-${t.kind}`}>
          {t.kind === 'ok' && <CheckCircle2 size={15} color="var(--ok)" />}
          {t.kind === 'err' && <XCircle size={15} color="var(--err)" />}
          {t.kind === 'info' && <Info size={15} color="var(--accent-2)" />}
          <span>{t.text}</span>
          <button className="toast-x" onClick={() => dismiss(t.id)} aria-label="关闭">
            <X size={13} />
          </button>
        </div>
      ))}
    </div>
  )
}
