/**
 * 时间/字节的展示格式化 —— 唯一一处实现。
 *
 * 原来 `fmtTime` 在 `KnowledgePage.tsx` 与 `AdminUsersCard.tsx` 里**逐字重复两份**，
 * 而 `FeedbackPanel.tsx` 与 `HistoryPage.tsx` 干脆裸写 `new Date(x).toLocaleString()`。
 * 裸写的那两处坏在同一个地方：`created_at`/`updated_at` 为空串或 null 时，
 * `new Date('')` 是 Invalid Date，`toLocaleString()` 会把 **"Invalid Date"** 直接印到界面上。
 * 两份重复 + 两处裸写 = 四种口径，改一处必漏一处（本项目的老毛病）。
 */

/** 时间戳缺失或异常一律成「—」，不把 Invalid Date 印到界面上 */
export function fmtTime(iso: string | null | undefined): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '—' : d.toLocaleString('zh-CN', { hour12: false })
}

/** 只有日期没有时间（画像事实的落库日，用户不关心几点） */
export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return ''
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleDateString('zh-CN')
}

/** 文件大小：KB 一位小数，没有就「—」（原始资料没上传成功时 size 是 null） */
export function fmtSize(n: number | null | undefined): string {
  if (!n) return '—'
  return n >= 1024 ? `${(n / 1024).toFixed(1)} KB` : `${n} B`
}
