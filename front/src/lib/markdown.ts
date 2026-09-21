import { marked } from 'marked'
import DOMPurify from 'dompurify'

marked.setOptions({
  gfm: true,
  breaks: true,
})

/**
 * 转义代码区之外的波浪号，避免 GFM 删除线误伤。
 *
 * 背景：爱弥斯人设回复爱用 ~ / ~~ 作语气破折号（「来~玩呀~」「很累~~真的」），
 * 而 marked 的 GFM 把成对波浪号（甚至单个 ~ 配另一个 ~）当删除线定界符，
 * 导致半句话被划上删除线。本域内从不需要删除线语法，故一律禁用。
 *
 * 实测（marked 5.x）：修复前 `来~玩呀~` -> `来<del>玩呀</del>`；修复后正常显示波浪号。
 * 代码块与行内代码内的 ~ 必须保留（如 `x~~y`、正则 /a~~b/），故跳过这些区域。
 */
function escapeTildesOutsideCode(src: string): string {
  const CODE_FENCE = /^(```|~~~)/
  const INLINE_CODE = /(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/g
  // CommonMark 反斜杠转义按单个标点计，所以 ~~ 要转成 \~\~
  const esc = (run: string) => run.split('').map((c) => '\\' + c).join('')
  const TILDE = /~+/g
  const out: string[] = []
  let inFence = false
  for (const line of src.split('\n')) {
    const isFenceLine = CODE_FENCE.test(line)
    if (!inFence && !isFenceLine) {
      let cursor = 0
      for (const m of line.matchAll(INLINE_CODE)) {
        const idx = m.index ?? 0
        out.push(line.slice(cursor, idx).replace(TILDE, esc))
        out.push(m[0]) // 行内代码原样保留
        cursor = idx + m[0].length
      }
      out.push(line.slice(cursor).replace(TILDE, esc))
    } else {
      out.push(line) // 代码块原样保留
    }
    if (isFenceLine) inFence = !inFence
    out.push('\n')
  }
  return out.join('').replace(/\n$/, '')
}

/**
 * 把 Markdown 文本渲染成安全的 HTML 字符串。
 * 用 DOMPurify 净化，防止答案里混入的脚本造成 XSS。
 */
export function renderMarkdown(src: string): string {
  const safe = escapeTildesOutsideCode(src ?? '')
  const raw = marked.parse(safe, { async: false }) as string
  return DOMPurify.sanitize(raw, {
    ADD_ATTR: ['target', 'rel'],
  })
}
