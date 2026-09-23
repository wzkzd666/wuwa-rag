/**
 * 图片选择与压缩工具（头像 / 自定义背景用）。
 *
 * 为什么必须压缩：设置存进 localStorage（zustand persist）。原图动辄几 MB，
 * 直接塞进去会触发 QuotaExceededError，而且每次读写都要 base64 一遍，很卡。
 * 所以统一走 canvas 缩放 + webp 编码（保留透明通道，体积远小于 png）。
 */

export interface EncodeOptions {
  /** 长边上限（square=true 时是边长），单位 px */
  max: number
  /** 编码质量 0~1 */
  quality?: number
  /** true = 先居中裁成正方形（头像用） */
  square?: boolean
}

/** 打开系统文件选择框。用户取消时 resolve(null)。 */
export function pickImageFile(): Promise<File | null> {
  return new Promise((resolve) => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = 'image/*'
    input.style.position = 'fixed'
    input.style.left = '-10000px'

    let settled = false
    const done = (file: File | null) => {
      if (settled) return
      settled = true
      window.removeEventListener('focus', onFocus)
      input.remove()
      resolve(file)
    }
    // 取消选择时不会触发 change：靠窗口重新获得焦点兜底（延迟避开 change 竞态）
    const onFocus = () => window.setTimeout(() => done(input.files?.[0] ?? null), 400)

    input.addEventListener('change', () => done(input.files?.[0] ?? null))
    window.addEventListener('focus', onFocus, { once: true })
    document.body.appendChild(input)
    input.click()
  })
}

/** 浏览器是否支持 webp 编码（现代 Chrome/Edge/Firefox 都支持；不支持则回落 png） */
function pickMimeType(quality: number): { type: string; q?: number } {
  try {
    const c = document.createElement('canvas')
    c.width = 1
    c.height = 1
    if (c.toDataURL('image/webp').startsWith('data:image/webp')) {
      return { type: 'image/webp', q: quality }
    }
  } catch {
    /* 忽略，走 png */
  }
  return { type: 'image/png' }
}

type Drawable = ImageBitmap | HTMLImageElement

async function loadDrawable(file: File): Promise<Drawable> {
  if (typeof createImageBitmap === 'function') {
    try {
      return await createImageBitmap(file)
    } catch {
      /* 某些格式（如 svg）createImageBitmap 会失败，回落到 <img> */
    }
  }
  const url = URL.createObjectURL(file)
  try {
    const img = new Image()
    img.decoding = 'async'
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve()
      img.onerror = () => reject(new Error('图片解码失败'))
      img.src = url
    })
    return img
  } finally {
    URL.revokeObjectURL(url)
  }
}

function sizeOf(d: Drawable): { w: number; h: number } {
  return d instanceof HTMLImageElement
    ? { w: d.naturalWidth || d.width, h: d.naturalHeight || d.height }
    : { w: d.width, h: d.height }
}

/** 把文件读成压缩后的 dataURL */
export async function fileToDataUrl(file: File, opts: EncodeOptions): Promise<string> {
  const { max, quality = 0.9, square = false } = opts
  if (!file.type.startsWith('image/')) throw new Error('不是图片文件')

  const drawable = await loadDrawable(file)
  const { w: sw0, h: sh0 } = sizeOf(drawable)
  if (!sw0 || !sh0) throw new Error('图片尺寸读取失败')

  // 裁切源区域（square 时取中心正方形）
  let sx = 0
  let sy = 0
  let sw = sw0
  let sh = sh0
  if (square) {
    const side = Math.min(sw0, sh0)
    sx = (sw0 - side) / 2
    sy = (sh0 - side) / 2
    sw = side
    sh = side
  }

  const scale = Math.min(1, max / Math.max(sw, sh))
  const w = Math.max(1, Math.round(sw * scale))
  const h = Math.max(1, Math.round(sh * scale))

  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('当前浏览器不支持 canvas')
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(drawable as CanvasImageSource, sx, sy, sw, sh, 0, 0, w, h)
  if (!(drawable instanceof HTMLImageElement)) drawable.close()

  const { type, q } = pickMimeType(quality)
  return canvas.toDataURL(type, q)
}

/** 估算 dataURL 的字节数（base64 ≈ 原文 × 4/3） */
export function dataUrlBytes(dataUrl: string): number {
  const i = dataUrl.indexOf(',')
  if (i < 0) return 0
  return Math.round(((dataUrl.length - i - 1) * 3) / 4)
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`
  return `${(n / 1024 / 1024).toFixed(1)} MB`
}
