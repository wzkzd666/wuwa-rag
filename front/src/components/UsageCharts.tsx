import { useMemo } from 'react'
import './UsagePanel.css'



export interface DailyPoint {
  d: string
  local_prompt: number | null
  local_completion: number | null
  cloud_prompt: number | null
  cloud_completion: number | null
  calls: number
}

/** 大数缩写：12345 → 12.3k。坐标轴上写「12345」会把图挤扁。 */
function short(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`
  return String(Math.round(n))
}

const COLOR_LOCAL = 'var(--accent-2)'
const COLOR_CLOUD = 'var(--warn)'
const AXIS = 'var(--line-strong)'
const GRID = 'var(--line)'

/** 每天的最大值，用来定 y 轴上界；全为 0 时给个 1，避免除零把柱子画成 NaN。 */
function peakOf(series: number[][]): number {
  const max = Math.max(0, ...series.flat())
  return max > 0 ? max : 1
}

/** 7 天 token 用量：上柱状（每日消耗，local/cloud 堆叠）、下折线（累计趋势）。 */
export default function UsageCharts({ daily }: { daily: DailyPoint[] }) {
  const data = useMemo(() => daily ?? [], [daily])

  const totals = useMemo(() => {
    let acc = 0
    return data.map((p) => {
      acc += (p.local_prompt ?? 0) + (p.local_completion ?? 0)
        + (p.cloud_prompt ?? 0) + (p.cloud_completion ?? 0)
      return acc
    })
  }, [data])

  if (data.length === 0) {
    return <div className="chart-empty">暂无用量数据</div>
  }

  const local = data.map((p) => (p.local_prompt ?? 0) + (p.local_completion ?? 0))
  const cloud = data.map((p) => (p.cloud_prompt ?? 0) + (p.cloud_completion ?? 0))
  const barPeak = peakOf([local, cloud])
  const linePeak = peakOf([totals])
  const allZero = local.every((v) => v === 0) && cloud.every((v) => v === 0)

  return (
    <div className="charts">
      <div className="chart-card">
        <div className="chart-head">
          <b>每日 token 消耗</b>
          <span className="chart-legend">
            <i style={{ background: COLOR_LOCAL }} />本地
            <i style={{ background: COLOR_CLOUD }} />云端
          </span>
        </div>
        <BarChart data={data} local={local} cloud={cloud} peak={barPeak} allZero={allZero} />
      </div>

      <div className="chart-card">
        <div className="chart-head">
          <b>累计 token 趋势</b>
        </div>
        <LineChart data={data} totals={totals} peak={linePeak} allZero={allZero} />
      </div>
    </div>
  )
}

const W = 640
const H = 168
const PAD = { top: 12, right: 8, bottom: 22, left: 38 }

function BarChart({ data, local, cloud, peak, allZero }: {
  data: DailyPoint[]; local: number[]; cloud: number[]; peak: number; allZero: boolean
}) {
  const iw = W - PAD.left - PAD.right
  const ih = H - PAD.top - PAD.bottom
  const slot = iw / data.length
  const bw = Math.min(26, slot * 0.5)
  const y = (v: number) => PAD.top + ih - (v / peak) * ih

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg" role="img" aria-label="每日 token 消耗柱状图">
      {[0, 0.5, 1].map((f) => (
        <g key={f}>
          <line x1={PAD.left} x2={W - PAD.right} y1={y(peak * f)} y2={y(peak * f)}
                stroke={GRID} strokeWidth={1} />
          <text x={PAD.left - 6} y={y(peak * f) + 3} className="chart-tick" textAnchor="end">
            {short(peak * f)}
          </text>
        </g>
      ))}
      <line x1={PAD.left} x2={W - PAD.right} y1={PAD.top + ih} y2={PAD.top + ih} stroke={AXIS} />
      {data.map((p, i) => {
        const cx = PAD.left + slot * i + slot / 2
        const lh = (local[i] / peak) * ih
        const ch = (cloud[i] / peak) * ih
        return (
          <g key={p.d}>
            {/* 本地在下、云端在上，堆叠；一侧为 0 时那一段高度就是 0，不画 */}
            <rect x={cx - bw / 2} y={PAD.top + ih - lh} width={bw} height={lh}
                  rx={3} fill={COLOR_LOCAL} />
            {ch > 0 && (
              <rect x={cx - bw / 2} y={PAD.top + ih - lh - ch} width={bw} height={ch}
                    rx={3} fill={COLOR_CLOUD} />
            )}
            <text x={cx} y={PAD.top + ih + 14} className="chart-tick" textAnchor="middle">
              {p.d.slice(5)}
            </text>
            <title>
              {`${p.d}\n本地 ${local[i].toLocaleString('zh-CN')}｜云端 ${cloud[i].toLocaleString('zh-CN')}｜${p.calls} 次调用`}
            </title>
          </g>
        )
      })}
      {allZero && <text x={W / 2} y={H / 2} className="chart-empty-txt" textAnchor="middle">近 7 天没有用量</text>}
    </svg>
  )
}

function LineChart({ data, totals, peak, allZero }: {
  data: DailyPoint[]; totals: number[]; peak: number; allZero: boolean
}) {
  const iw = W - PAD.left - PAD.right
  const ih = H - PAD.top - PAD.bottom
  const step = data.length > 1 ? iw / (data.length - 1) : 0
  const x = (i: number) => PAD.left + step * i
  const y = (v: number) => PAD.top + ih - (v / peak) * ih
  const path = totals.map((v, i) => `${i === 0 ? 'M' : 'L'}${x(i)},${y(v)}`).join(' ')
  const area = `${path} L${x(totals.length - 1)},${PAD.top + ih} L${x(0)},${PAD.top + ih} Z`

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg" role="img" aria-label="累计 token 趋势折线图">
      <defs>
        <linearGradient id="usageArea" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="var(--accent-2)" stopOpacity="0.28" />
          <stop offset="100%" stopColor="var(--accent-2)" stopOpacity="0.02" />
        </linearGradient>
      </defs>
      {[0, 0.5, 1].map((f) => (
        <g key={f}>
          <line x1={PAD.left} x2={W - PAD.right} y1={y(peak * f)} y2={y(peak * f)}
                stroke={GRID} strokeWidth={1} />
          <text x={PAD.left - 6} y={y(peak * f) + 3} className="chart-tick" textAnchor="end">
            {short(peak * f)}
          </text>
        </g>
      ))}
      {!allZero && (
        <>
          <path d={area} fill="url(#usageArea)" />
          <path d={path} fill="none" stroke={COLOR_LOCAL} strokeWidth={2}
                strokeLinejoin="round" strokeLinecap="round" />
          {totals.map((v, i) => (
            <circle key={i} cx={x(i)} cy={y(v)} r={3} fill="var(--bg-elevated)"
                    stroke={COLOR_LOCAL} strokeWidth={2}>
              <title>{`${data[i].d}｜累计 ${v.toLocaleString('zh-CN')}`}</title>
            </circle>
          ))}
        </>
      )}
      <line x1={PAD.left} x2={W - PAD.right} y1={PAD.top + ih} y2={PAD.top + ih} stroke={AXIS} />
      {data.map((p, i) => (
        <text key={p.d} x={x(i)} y={PAD.top + ih + 14} className="chart-tick" textAnchor="middle">
          {p.d.slice(5)}
        </text>
      ))}
      {allZero && <text x={W / 2} y={H / 2} className="chart-empty-txt" textAnchor="middle">近 7 天没有用量</text>}
    </svg>
  )
}
