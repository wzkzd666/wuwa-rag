import { feedbackMine } from './api'

/**
 * 「这条回答我之前评过几分」的批量回填。
 *
 * 为什么要有这一层：原来每条气泡挂载时各发一次 `GET /feedback/mine?target_id=…`，
 * 打开一个 30 条的会话就是 30 个请求 —— 而 `feedbackMine` 本来就收 id 数组。
 * 逐个请求既挤占按用户的兜底限额（200/min，见 `api/ratelimit.py`），
 * 也把首屏的网络瀑布拖成一串。
 *
 * 没有选择「把 rated 从 ChatPage 往下传」：那要动三处渲染的 props 与状态归属，
 * 而回填本质是**每个气泡自己的显示增益**，不该把调度权交给列表。
 * 这里做的是把同一帧内的登记**合并成一个请求**，结果按 id 分发给各自订阅者。
 *
 * 四条不变量（改这个文件前先看）：
 * - 只认服务端 id：历史里的消息 id 是 DB 行号（纯数字），而本轮刚生成的气泡先是随机串，
 *   收到 `done.message_id` 之后才被换成行号（见 store 里那三处 adoption）——
 *   所以这里的数字判定既挡住了「还没落库就问」，也挡住了异常/中断留下的 uid 气泡；
 * - 失败静默且不落缓存：回填是增益不是依赖，绝不把网络错误演变成气泡上的红条；
 *   但下次挂载必须能重问，否则永远回填不上；
 * - 成功要落缓存：切走再切回同一会话时组件会重新挂载、state 归零，
 *   没有缓存就会把已经确认过的「评过 👍」显示成没评；
 * - 单批有上限：一屏几百条时不把 URL 撑爆，超出的留下一窗口继续发。
 */

export type Rating = 0 | 1 | -1

/** 合并窗口：够长到覆盖一次 React 提交，又短到用户感觉不出延迟 */
const BATCH_WINDOW_MS = 60
const MAX_IDS_PER_REQUEST = 100

/** 服务端真值缓存（含 0 = 确实没评过）：id → rating */
const cached = new Map<string, Rating>()
/** 正在等的回调：id → 订阅者（同一 id 可能同时挂在两处） */
const waiters = new Map<string, Set<(r: Rating) => void>>()
/** 按 apiBase 分组排队：设置页改过地址后，不能把新地址混进旧地址那一批 */
const queued = new Map<string, Set<string>>()
let timer: ReturnType<typeof setTimeout> | null = null

const NOOP = () => { /* 不订阅就别的东西 */ }

function settle(id: string, rating: Rating) {
  const set = waiters.get(id)
  waiters.delete(id)
  if (!set) return
  for (const cb of set) cb(rating)
}

async function send(apiBase: string, ids: string[]) {
  try {
    const out = await feedbackMine(ids, apiBase)
    const ratings = out.ratings ?? {}
    for (const id of ids) {
      const r = ratings[id]
      const rating: Rating = r === 1 || r === -1 ? r : 0
      cached.set(id, rating)
      settle(id, rating)
    }
  } catch {
    // 不写 cached：下次挂载会重新登记、重新问一遍（失败不该变成永久失明）
    for (const id of ids) waiters.delete(id)
  }
}

function runBatch() {
  timer = null
  for (const apiBase of [...queued.keys()]) {
    const bucket = queued.get(apiBase)
    if (!bucket?.size) {
      queued.delete(apiBase)
      continue
    }
    const take = [...bucket].slice(0, MAX_IDS_PER_REQUEST)
    for (const id of take) bucket.delete(id)
    if (!bucket.size) queued.delete(apiBase)
    void send(apiBase, take)
  }
  // 还有没发完的（超出单批上限）：下一个窗口接着发，不能丢
  if (queued.size && timer === null) timer = setTimeout(runBatch, BATCH_WINDOW_MS)
}

/**
 * 把「刚提交的评价」写进缓存。
 * 必须调用：缓存里可能已经有这条的 0（挂载时问过、那时还没评）。
 * 不更新的话，用户点完 👍 切走再切回——组件重新挂载读到缓存的 0，
 * 按钮就打回「没评过」的样子（而服务端明明记着 👍）。
 */
export function rememberRating(id: string, rating: Rating) {
  cached.set(id, rating)
}

/**
 * 登记一条待回填的评价，立刻返回取消函数。
 * 取消只丢回调，已排队的请求照发 —— 同一会话里通常还有别的气泡等同一个 id。
 */
export function hydrateRating(id: string, apiBase: string, apply: (r: Rating) => void): () => void {
  if (!/^\d+$/.test(id)) return NOOP
  const known = cached.get(id)
  if (known !== undefined) {
    apply(known)
    return NOOP
  }
  const set = waiters.get(id) ?? new Set<(r: Rating) => void>()
  set.add(apply)
  waiters.set(id, set)
  const bucket = queued.get(apiBase) ?? new Set<string>()
  bucket.add(id)
  queued.set(apiBase, bucket)
  if (timer === null) timer = setTimeout(runBatch, BATCH_WINDOW_MS)
  return () => { set.delete(apply) }
}
