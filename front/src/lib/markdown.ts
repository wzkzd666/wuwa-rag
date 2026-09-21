import { marked } from 'marked'
import DOMPurify from 'dompurify'

marked.setOptions({
  gfm: true,
  breaks: true,
})

/**
 * 把 Markdown 文本渲染成安全的 HTML 字符串。
 * 用 DOMPurify 净化，防止答案里混入的脚本造成 XSS。
 */
export function renderMarkdown(src: string): string {
  const raw = marked.parse(src ?? '', { async: false }) as string
  return DOMPurify.sanitize(raw, {
    ADD_ATTR: ['target', 'rel'],
  })
}
