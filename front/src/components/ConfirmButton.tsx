import { useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { Loader2 } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import './ConfirmButton.css'

/**
 * 破坏性操作的「两步确认」按钮 —— 全站唯一实现。
 *
 * 为什么要收口成一个组件：同一个动作原本有四种写法（会话行的 `.danger-confirm` 换色、
 * 历史行的 `.armed`、设置页的 `.btn-danger.armed`，以及知识库的两处 `window.confirm`）。
 * 前三份 CSS 互不相让、靠一串 `!important` 压制（成因见 ConfirmButton.css），
 * 第四份直接跳出设计系统 —— 原生弹窗不吃主题色板、文案是操作系统的 chrome、
 * 在手机上还是全屏系统对话框。
 *
 * 交互手感沿用设置页那一份（实测出来的，不要改）：点一次进入待确认 →
 * 到点没再点就自动解除 → 第二击才真正执行。
 */

/** 待确认态自动解除的毫秒数：再长就开始考验耐心，再短等于没给反悔窗口 */
const AUTO_DISARM_MS = 4000

export interface ConfirmButtonProps {
  /** 常态文案；纯图标按钮可以不传（此时 `title` 就是它的无障碍名称） */
  label?: ReactNode
  /** 待确认文案：这套交互唯一的提示，必须写清「再点一次」会失去什么 */
  confirmLabel: ReactNode
  /** 第二击执行的动作；调用方自己负责失败提示（各页现都在 handler 里 try/catch） */
  onConfirm: () => unknown
  icon?: LucideIcon
  /** 图标尺寸，常态图标与进行中的转圈共用一个数，免得两处各写各的 */
  iconSize?: number
  /** 动作进行中：转圈并禁点，防连点重复提交 */
  busy?: boolean
  disabled?: boolean
  /** 宿主按钮既有的类名（`.btn` / `.btn-danger` / 各页自己的尺寸类），外观由它们决定 */
  className?: string
  /**
   * 悬浮说明：常态是「这个按钮干什么」，待确认态是「再点一次会失去什么」——
   * 知识库两处删除把后果（清文档/索引/图谱、只删账本不动流水线）只写在这里。
   * ⚠️ 待确认时**不能**抑制它：那一刻正是用户最需要读到后果的瞬间，
   * 而按钮脸上的文案只有「再点一次确认删除」，不含后果。
   */
  title?: string
}

export default function ConfirmButton({
  label,
  confirmLabel,
  onConfirm,
  icon: Icon,
  iconSize = 14,
  busy = false,
  disabled = false,
  className = '',
  title,
}: ConfirmButtonProps) {
  const [armed, setArmed] = useState(false)
  const timer = useRef<number | null>(null)

  const cancelDisarm = () => {
    if (timer.current !== null) {
      window.clearTimeout(timer.current)
      timer.current = null
    }
  }

  // 卸载必须清定时器：列表行删完组件就消失了，老写法里那个 setTimeout 会在行
  // 已经不在了之后回调（设置页原来是把整页 state 置 false，所以没暴露这个问题）。
  useEffect(() => cancelDisarm, [])

  // 外部把按钮置灰时（同一行的别的操作在跑），待确认态不能留着：
  // 恢复可点的那一瞬间用户看到的还是「再点一次」，等于默认放行了一次破坏性操作。
  useEffect(() => {
    if (disabled || busy) {
      cancelDisarm()
      setArmed(false)
    }
  }, [disabled, busy])

  const locked = disabled || busy

  const click = async () => {
    if (locked) return
    if (!armed) {
      setArmed(true)
      cancelDisarm()
      timer.current = window.setTimeout(() => setArmed(false), AUTO_DISARM_MS)
      return
    }
    cancelDisarm()
    setArmed(false)
    await onConfirm()
  }

  return (
    <button
      type="button"
      className={`confirm-btn ${armed ? 'confirm-armed' : ''} ${className}`}
      onClick={() => void click()}
      disabled={locked}
      aria-busy={busy || undefined}
      title={title}
    >
      {busy ? <Loader2 size={iconSize} className="spin" /> : Icon ? <Icon size={iconSize} /> : null}
      {armed ? confirmLabel : label}
    </button>
  )
}
