import { type ReactNode } from 'react'
import { ChevronDown } from 'lucide-react'

/**
 * 页面级可折叠区块。
 *
 * 设置项越堆越长（模型/语音/账号三块都是「配一次就再也不看」的），全展开会把
 * 频率高的「外观 / 背景」挤到屏幕外，所以默认只留一行标题 + 一句说明。
 *
 * ⚠️ 标题**整行可点**，且收起后行尾的 `aside` 状态标签必须留着——「语音合成」
 * 收起来却看不到「未就绪」，用户就得逐个点开去找，比不折叠更糟。
 * 折叠状态由调用方持有（见 SettingsPage 的 open/toggle），组件本身无状态。
 */
export function Fold({ title, icon, aside, hint, open, onToggle, children }: {
  title: string
  icon: ReactNode
  aside?: ReactNode
  hint?: string
  open: boolean
  onToggle: () => void
  children: ReactNode
}) {
  return (
    <section className={`card set-card${open ? '' : ' set-folded'}`}>
      <div className="set-head-row">
        <button type="button" className="set-h set-h-btn" onClick={onToggle} aria-expanded={open}>
          {icon} {title}
          <ChevronDown size={15} className={`set-caret${open ? ' set-caret-open' : ''}`} />
        </button>
        {aside}
      </div>
      {!open && hint && <p className="set-fold-hint">{hint}</p>}
      {open && children}
    </section>
  )
}

/** 卡片内部的次级折叠：比 Fold 轻一档，只在卡片里划一小块可收起的区域。
 *  用途是「同一张卡里次要的一堆字段」——例如云端模型的凭据表单、
 *  语音合成的密钥/音色/语气指令，默认收起，避免一屏全是输入框。 */
export function InlineFold({ title, open, onToggle, children }: {
  title: string
  open: boolean
  onToggle: () => void
  children: ReactNode
}) {
  return (
    <div className="set-inline-fold">
      <button type="button" className="set-inline-btn" onClick={onToggle} aria-expanded={open}>
        <ChevronDown size={14} className={`set-caret${open ? ' set-caret-open' : ''}`} />
        {title}
      </button>
      {open && <div className="set-inline-body">{children}</div>}
    </div>
  )
}
