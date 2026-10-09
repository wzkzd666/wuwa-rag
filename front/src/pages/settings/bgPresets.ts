import type { BgPreset } from '../../types'

/** 背景预设（key 对齐 types.ts 的 BgPreset 与 global.css 的 data-preset） */
export const BG_PRESETS: { key: BgPreset; label: string; css: string }[] = [
  { key: 'default', label: '默认', css: 'linear-gradient(135deg, #0b1020, #17213a)' },
  { key: 'aurora', label: '极光', css: 'linear-gradient(160deg, #0a1024, #16224a 42%, #2c1f52)' },
  { key: 'dusk', label: '暮紫', css: 'linear-gradient(160deg, #1a1030, #3a1d4e 46%, #6b2d55)' },
  { key: 'cyber', label: '深青', css: 'linear-gradient(160deg, #04121c, #0b2b3a 46%, #123d4a)' },
  { key: 'plain', label: '纯色', css: '#070b16' },
  { key: 'custom', label: '自定义图片', css: '' },
]
