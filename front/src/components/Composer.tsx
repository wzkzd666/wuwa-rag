import { useRef, useState, KeyboardEvent, useEffect } from 'react'
import { ArrowUp, Square } from 'lucide-react'
import './Composer.css'

interface Props {
  onSend: (text: string) => void
  onStop: () => void
  busy: boolean
  disabled?: boolean
  placeholder?: string
}

/** 自动伸缩的多行输入框：Enter 发送，Shift+Enter 换行 */
export default function Composer({ onSend, onStop, busy, disabled, placeholder }: Props) {
  const [text, setText] = useState('')
  const ref = useRef<HTMLTextAreaElement>(null)

  // 自适应高度（最高 200px）
  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 200) + 'px'
  }, [text])

  // busy 解除后重新聚焦
  useEffect(() => {
    if (!busy) ref.current?.focus()
  }, [busy])

  const submit = () => {
    const t = text.trim()
    if (!t || busy || disabled) return
    onSend(t)
    setText('')
  }

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="composer">
      <div className="composer-box glass">
        <textarea
          ref={ref}
          className="composer-input"
          rows={1}
          value={text}
          placeholder={placeholder || '问问鸣潮角色的配装、突破材料、共鸣链…（Enter 发送，Shift+Enter 换行）'}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          disabled={disabled}
        />
        {busy ? (
          <button className="send-btn stop-btn" onClick={onStop} title="停止生成">
            <Square size={14} fill="currentColor" />
          </button>
        ) : (
          <button
            className="send-btn"
            onClick={submit}
            disabled={!text.trim() || disabled}
            title="发送"
          >
            <ArrowUp size={17} strokeWidth={2.4} />
          </button>
        )}
      </div>
      <div className="composer-hint">
        答案由微调模型「爱弥斯」生成，内容来自本地知识库，仅供参考
      </div>
    </div>
  )
}
