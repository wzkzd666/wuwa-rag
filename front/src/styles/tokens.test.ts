/**
 * 设计令牌的可读性契约 —— 唯一一处「配色会不会把人洗掉」的自动门禁。
 *
 * 立项原因（2026-10-23 实测）：浅色主题当初是**事后补**的，只覆盖了 `--text-*` 三档，
 * 语义色（`--accent/--ok/--warn/--err/--accent-2`）在浅色块里根本没有覆盖值。
 * 这些饱和色在深色底上有 7~13:1，同一份色值放到白底上只有 1.29~2.98:1 ——
 * 全站 36 处 `color:` 引用它们，于是浅色主题里错误按钮、「已收录」标签、markdown 链接、
 * 点赞/点踩选中态全部洗成一片淡彩。而这类问题**人眼扫一遍很容易漏**（默认是深色主题），
 * 单测又完全没覆盖 CSS，所以把它变成断言。
 *
 * ⚠️ 这里只核「文字角色」的令牌，不核所有颜色：`--grad-ink`（渐变面上的墨色）
 * 在白底上算出来是 16:1、在深色面板底上只有 1.2:1，若按「对所有底色都要过」一刀切，
 * 它必被误判。逐底色核没有意义，逐**用途**核才有。
 */
import { readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

const SRC = fileURLToPath(new URL('..', import.meta.url)) // front/src
const GLOBAL_CSS = readFileSync(join(SRC, 'styles', 'global.css'), 'utf-8')

/** 递归收集 src 下所有 CSS（只收 .css，测试自身不算） */
function cssFiles(dir: string): string[] {
  const out: string[] = []
  for (const name of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, name.name)
    if (name.isDirectory()) out.push(...cssFiles(p))
    else if (name.name.endsWith('.css')) out.push(p)
  }
  return out
}

/** 取 `选择器 { … }` 的块体，按花括号配对（先剥注释，注释里的括号不算） */
function blockOf(css: string, selector: string): string {
  const bare = css.replace(/\/\*[\s\S]*?\*\//g, '')
  const at = bare.indexOf(selector)
  if (at < 0) throw new Error(`找不到选择器 ${selector}`)
  const open = bare.indexOf('{', at)
  let depth = 0
  for (let i = open; i < bare.length; i++) {
    if (bare[i] === '{') depth++
    else if (bare[i] === '}') {
      if (--depth === 0) return bare.slice(open + 1, i)
    }
  }
  throw new Error(`${selector} 的花括号没闭合`)
}

/** 块内的 `--x: value`（不含嵌套块里的：这里取的块都是扁平的令牌声明） */
function tokens(block: string): Record<string, string> {
  const out: Record<string, string> = {}
  for (const m of block.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)) {
    out[m[1]] = m[2].trim()
  }
  return out
}

const DARK = tokens(blockOf(GLOBAL_CSS, ':root'))
const LIGHT = { ...DARK, ...tokens(blockOf(GLOBAL_CSS, "[data-theme='light']")) }

/** 解一层 var() 别名：`--accent-text: var(--accent)` 要拿到真色值 */
function resolve(map: Record<string, string>, name: string, seen = 0): string {
  const raw = map[name]
  if (raw === undefined) throw new Error(`令牌 ${name} 未定义`)
  if (seen > 3) throw new Error(`令牌 ${name} 循环引用`)
  const m = /^var\((--[\w-]+)\)$/.exec(raw)
  return m ? resolve(map, m[1], seen + 1) : raw
}

type Rgba = { r: number; g: number; b: number; a: number }

/** 吃 `#rrggbb` / `rgb(r g b / a)` / `rgba(r, g, b, a)` 三种写法（令牌里三种都在用） */
function toRgba(raw: string): Rgba {
  const hex = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(raw.trim())
  if (hex) {
    const h = hex[1].length === 3 ? hex[1].split('').map((c) => c + c).join('') : hex[1]
    return {
      r: parseInt(h.slice(0, 2), 16), g: parseInt(h.slice(2, 4), 16), b: parseInt(h.slice(4, 6), 16), a: 1,
    }
  }
  const nums = raw.replace(/^[^(]*\(/, '').replace(/\)$/, '').split(/[,/\s]+/).filter(Boolean).map(Number)
  if (nums.length < 3) throw new Error(`认不出的色值：${raw}`)
  return { r: nums[0], g: nums[1], b: nums[2], a: nums.length > 3 ? nums[3] : 1 }
}

const asHex = (c: Rgba) =>
  '#' + [c.r, c.g, c.b].map((v) => Math.round(v).toString(16).padStart(2, '0')).join('')

/** fg 以自身 alpha 叠在 bg 上（半透明芯片/渐变层的合成口径，与浏览器实测一致） */
function composite(fg: Rgba, bg: Rgba): Rgba {
  return {
    r: fg.r * fg.a + bg.r * (1 - fg.a),
    g: fg.g * fg.a + bg.g * (1 - fg.a),
    b: fg.b * fg.a + bg.b * (1 - fg.a),
    a: 1,
  }
}

function luminance(c: Rgba): number {
  const f = (v: number) => {
    const s = v / 255
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4
  }
  return 0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b)
}

function contrast(a: Rgba, b: Rgba): number {
  const la = luminance(a)
  const lb = luminance(b)
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05)
}

/**
 * 一条 `background:` 值里的**半透明色层**（芯片/玻璃底），按声明顺序。
 * 逐层解析而不是写死色值：换品牌色时这条断言要跟着走，不然守的是旧颜色。
 * 支持三种写法混用：`rgba(...)` 字面量、`var(--grad-soft)`、`var(--accent-2-tint-14)`
 * （后者展开成 `rgb(var(--accent-2-rgb) / 0.14)`，三元组与 alpha 分在两处写）。
 */
function layersOf(map: Record<string, string>, bgValue: string): Rgba[] {
  const layers: Rgba[] = []
  const expand = (value: string, depth = 0): string => {
    if (depth > 3) return value
    return value.replace(/var\((--[\w-]+)\)/g, (_all, name: string) => expand(map[name] ?? '', depth + 1))
  }
  const flat = expand(bgValue)
  for (const lit of flat.match(/rgba?\([^()]*(?:\([^()]*\)[^()]*)*\)/g) ?? []) {
    const m = /rgb[a]?\(([^)]*)\)/.exec(lit)?.[1] ?? ''
    const rgbRef = /var\((--[\w-]+-rgb)\)/.exec(m)
    if (rgbRef) {
      const [r, g, b] = resolve(map, rgbRef[1]).trim().split(/\s+/).map(Number)
      const a = /\/\s*([\d.]+)\s*$/.exec(m.trim())
      layers.push({ r, g, b, a: a ? Number(a[1]) : 1 })
    } else {
      layers.push(toRgba(lit))
    }
  }
  return layers
}

/** 文字角色令牌：饱和语义色只能做装饰（圆点/边框/滑块），读字必须走这一档 */
const TEXT_TOKENS = [
  '--text-primary', '--text-secondary', '--text-muted',
  '--accent-text', '--accent-2-text', '--accent-3-text',
  '--ok-text', '--warn-text', '--err-text',
]

const SURFACES = ['--bg-base', '--bg-deep', '--bg-surface', '--bg-elevated', '--bg-hover']

/** 规则是否只在某个主题下生效（`[data-theme='light'] xxx`） */
function themeScope(sel: string): 'light' | 'dark' | null {
  const m = /^\[data-theme='(light|dark)'\]/.exec(sel)
  return m ? (m[1] as 'light' | 'dark') : null
}

/** 去掉主题前缀后的本体选择器；`*::selection` 与 `::selection` 是同一个目标 */
function tailOf(sel: string): string {
  return sel
    .replace(/^\[data-theme='(?:light|dark)'\]\s*/, '')
    .replace(/^\*+(?=::)/, '')
    .trim()
}

describe('配色令牌必须真的能读', () => {
  it.each(TEXT_TOKENS)('%s 在两个主题的每块底上都 ≥4.5:1', (name) => {
    for (const [theme, map] of [['深色', DARK], ['浅色', LIGHT]] as const) {
      const fg = toRgba(resolve(map, name))
      for (const bgName of SURFACES) {
        const bg = toRgba(resolve(map, bgName))
        const ratio = contrast(fg, bg)
        expect(
          ratio >= 4.5,
          `${theme} ${name} / ${bgName} ${resolve(map, bgName)}: ${ratio.toFixed(2)}`,
        ).toBe(true)
      }
    }
  })

  it('同一条规则里的「文字色 × 半透明底」组合也要过 AA', () => {
    /**
     * 只核**真实出现过的配对**（一条规则里同时写了 color 和半透明 background），
     * 而不是「所有文字令牌 × 所有芯片」——后者会把 `--err-text` 拿去和 `.nav-active`
     * 的品牌渐变底组合，那种配对代码里根本没有，报的是假案。
     *
     * 立项案例：`.kb-count` 的芯片底是 `--grad-soft`（两层 0.16 alpha 品牌色叠在白底上
     * → rgb(211,232,248)，比任何一块平底都深），`--accent-text` 取 #0f766e 时五块平底
     * 全过（最差的 4.76），落到这个芯片上却只有 4.34:1 —— 浏览器实测才抓到，
     * 所以把它固化成静态断言（芯片底要按 alpha 合成，不能拿令牌原色算）。
     */
    const failures: string[] = []
    for (const file of cssFiles(SRC)) {
      const css = readFileSync(file, 'utf-8').replace(/\/\*[\s\S]*?\*\//g, '')
      const rules = [...css.matchAll(/([^{}]+)\{([^{}]*)\}/g)].map((m) => ({
        sel: m[1].trim().split('\n').pop()!.trim(),
        body: m[2],
      }))
      // 被主题作用域规则覆盖过的选择器：它的基础规则在该主题下**不生效**，
      // 拿基础规则的色值去算就是假案（`.markdown code` 的 `#b9c2ff` 只在深色用）。
      const overridden = { light: new Set<string>(), dark: new Set<string>() }
      for (const r of rules) {
        const scope = themeScope(r.sel)
        if (!scope) continue
        if (/color\s*:/.test(r.body) || /background\s*:/.test(r.body)) overridden[scope].add(tailOf(r.sel))
      }
      for (const r of rules) {
        const scope = themeScope(r.sel)
        const fgRaw = /(?:^|[;.\s])color\s*:\s*([^;]+)/.exec(r.body)?.[1]?.trim()
        const bgRaw = /background(?:-image)?\s*:\s*([^;]+)/.exec(r.body)?.[1]?.trim()
        if (!fgRaw || !bgRaw) continue
        if (!/rgba?\(|var\(--grad-soft\)|var\(--[\w-]+-tint-\d+\)/.test(bgRaw)) continue
        for (const [theme, map] of [['深色', DARK], ['浅色', LIGHT]] as const) {
          const key = theme === '浅色' ? 'light' : 'dark'
          if (scope && scope !== key) continue
          if (!scope && overridden[key].has(tailOf(r.sel))) continue
          const fgText = /^var\((--[\w-]+)\)$/.test(fgRaw) ? resolve(map, fgRaw.slice(4, -1)) : fgRaw
          const fg = toRgba(fgText)
          if (fg.a === 0) continue            // 渐变文字（background-clip:text）这里算不了
          for (const bgName of SURFACES) {
            const chip = layersOf(map, bgRaw).reduce(
              (acc, l) => composite(l, acc),
              toRgba(resolve(map, bgName)),
            )
            const ratio = contrast(fg, chip)
            if (ratio < 4.5) {
              failures.push(`${file.split(/[\\/]/).pop()} ${r.sel} | ${theme} ${bgName} ${asHex(toRgba(resolve(map, bgName)))}→${asHex(chip)}: ${ratio.toFixed(2)}`)
            }
          }
        }
      }
    }
    expect(failures.join('\n')).toBe('')
  })

  it('语义饱和色不得再被直接当文字色（必须走 -text 档）', () => {
    // 负向前瞻：`--accent` 不能吃掉 `--accent-2`；后顾排除 accent-color / border-color，
    // 那两类是装饰用途，留饱和色是对的。
    const re = /(?<![\w-])color\s*:\s*var\(--(?:accent|accent-2|accent-3|ok|warn|err)(?![\w-])\)/
    const hits = cssFiles(SRC)
      .flatMap((f) => (readFileSync(f, 'utf-8').match(new RegExp(re, 'g')) ?? []).map((m) => `${f}: ${m}`))
    expect(hits.join('\n')).toBe('')
  })

  it('浅色主题必须真的覆盖六个文字角色令牌（漏一个就又回到洗掉的状态）', () => {
    const lightBlock = blockOf(GLOBAL_CSS, "[data-theme='light']")
    const overridden = Object.keys(tokens(lightBlock))
    for (const name of ['--accent-text', '--accent-2-text', '--accent-3-text', '--ok-text', '--warn-text', '--err-text']) {
      expect(overridden, `浅色主题没覆盖 ${name}`).toContain(name)
    }
  })
})
