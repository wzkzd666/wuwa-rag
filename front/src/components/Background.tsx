import type { CSSProperties } from 'react'
import { useStore } from '../store/useStore'

/**
 * 应用背景层。
 *
 * 结构（都在 z-index:0，.layout 是 z-index:1，所以永远压在内容之下）：
 *   .app-bg      背景本体：预设渐变 / 自定义图片
 *   .app-bg-veil 自定义图片时的遮罩，保证正文可读
 *
 * 预设为 'default' 时不画任何东西，交给 global.css 里 body::before 的光晕，
 * 这样「默认」和改造前完全一致。
 */
export default function Background() {
  const bgPreset = useStore((s) => s.settings.bgPreset)
  const bgImage = useStore((s) => s.settings.bgImage)
  const bgDim = useStore((s) => s.settings.bgDim)
  const bgBlur = useStore((s) => s.settings.bgBlur)

  const custom = bgPreset === 'custom' && !!bgImage

  const style: CSSProperties = {}
  if (custom) {
    style.backgroundImage = `url("${bgImage}")`
    if (bgBlur > 0) {
      style.filter = `blur(${bgBlur}px)`
      // 放大一点，盖住模糊后边缘的透明过渡
      style.transform = 'scale(1.06)'
    }
  }

  return (
    <>
      <div className="app-bg" data-preset={bgPreset} style={style} aria-hidden />
      {custom && <div className="app-bg-veil" style={{ opacity: bgDim }} aria-hidden />}
    </>
  )
}
