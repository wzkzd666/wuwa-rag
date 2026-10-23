import { CheckCircle2, XCircle, Info, X } from 'lucide-react'
import { useStore } from '../store/useStore'
import './Toasts.css'

export default function Toasts() {
  const toasts = useStore((s) => s.toasts)
  const dismiss = useStore((s) => s.dismissToast)

  // 容器**始终挂着**（原来 toasts 为空就 return null）：aria-live 区域必须
  // 先存在于无障碍树里，读屏才会在内容插入时播报；和文字同一帧挂载的区域
  // 经常被直接跳过 —— 那样加了 role 也等于没加。空容器是 0×0 的 fixed 盒，
  // 不拦截点击、也没有任何视觉痕迹。
  return (
    <div className="toast-wrap" role="status" aria-live="polite">
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
