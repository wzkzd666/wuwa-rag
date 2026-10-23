/**
 * `lib/format` 的三条判据。
 * 立此存照的原因：`fmtTime` 曾有两份逐字重复、另两处裸写 `new Date(x).toLocaleString()`，
 * 裸写那两处遇到 null/空串会把 "Invalid Date" 直接印在界面上。
 */
import { describe, expect, it } from 'vitest'

import { fmtDate, fmtSize, fmtTime } from './format'

describe('时间与字节的展示格式化', () => {
  it('缺失与异常一律破折号，绝不出现 Invalid Date', () => {
    for (const bad of [null, undefined, '', '不是时间', '2026-13-45T99:99:99']) {
      expect(fmtTime(bad)).toBe('—')
    }
  })

  it('正常时间戳按 24 小时制渲染', () => {
    const out = fmtTime('2026-10-23T05:00:00.000Z')
    expect(out).not.toBe('—')
    expect(out).not.toMatch(/Invalid/)
    expect(out).not.toMatch(/(上午|下午)/)      // hour12: false
  })

  it('日期档空值给空串（它出现在表格单元里，不是「没有数据」那种语义）', () => {
    expect(fmtDate(null)).toBe('')
    expect(fmtDate('')).toBe('')
    expect(fmtDate('2026-10-23T05:00:00Z')).toMatch(/2026/)
  })

  it('字节：KB 一位小数，0/null 走破折号', () => {
    expect(fmtSize(null)).toBe('—')
    expect(fmtSize(0)).toBe('—')
    expect(fmtSize(512)).toBe('512 B')
    expect(fmtSize(2048)).toBe('2.0 KB')
  })
})
