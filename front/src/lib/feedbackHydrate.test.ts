/**
 * 反馈回填的合并行为。
 *
 * 这里守的是「一次挂载一串请求」这个老毛病：每条气泡各发一次 `GET /feedback/mine`，
 * 30 条会话就是 30 个请求（详见 lib/feedbackHydrate.ts 的文件头）。
 * 每条用例都用 `vi.resetModules()` 拿一份干净的模块级缓存 —— 缓存本身也是被测对象，
 * 共享一份模块会让用例互相污染。
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'

const mine = vi.fn()
vi.mock('./api', () => ({
  feedbackMine: (ids: string[], base?: string) => mine(ids, base),
}))

type Mod = typeof import('./feedbackHydrate')

async function freshModule(): Promise<Mod> {
  vi.resetModules()
  return import('./feedbackHydrate')
}

beforeEach(() => {
  vi.useFakeTimers()
  mine.mockReset()
  mine.mockResolvedValue({ ratings: {} })
})

describe('同帧登记合并成一个请求', () => {
  it('三条气泡只发一次，且三个 id 都在参数里', async () => {
    const m = await freshModule()
    const seen: Record<string, number> = {}
    for (const id of ['11', '12', '13']) m.hydrateRating(id, '/api', (r) => { seen[id] = r })
    expect(mine).not.toHaveBeenCalled()      // 窗口内不该提前发
    await vi.advanceTimersByTimeAsync(100)
    await vi.waitFor(() => expect(mine).toHaveBeenCalledTimes(1))
    expect(mine.mock.calls[0][0].sort()).toEqual(['11', '12', '13'])
  })

  it('非服务端 id（本轮新生成的随机串）一个请求都不发', async () => {
    const m = await freshModule()
    m.hydrateRating('u-abc123', '/api', () => { })
    await vi.advanceTimersByTimeAsync(1000)
    expect(mine).not.toHaveBeenCalled()
  })

  it('不同 apiBase 分开发，不混进同一批', async () => {
    const m = await freshModule()
    m.hydrateRating('1', 'http://a/api', () => { })
    m.hydrateRating('2', 'http://b/api', () => { })
    await vi.advanceTimersByTimeAsync(100)
    await vi.waitFor(() => expect(mine).toHaveBeenCalledTimes(2))
    const bases = mine.mock.calls.map((c) => c[1]).sort()
    expect(bases).toEqual(['http://a/api', 'http://b/api'])
  })
})

describe('缓存必须与用户看到的一致', () => {
  it('成功回填后落缓存：同一 id 再登记不发请求、直接给服务端真值', async () => {
    mine.mockResolvedValueOnce({ ratings: { '7': 1 } })
    const m = await freshModule()
    const first: number[] = []
    m.hydrateRating('7', '/api', (r) => first.push(r))
    await vi.advanceTimersByTimeAsync(100)
    await vi.waitFor(() => expect(first).toEqual([1]))

    const second: number[] = []
    m.hydrateRating('7', '/api', (r) => second.push(r))
    await vi.advanceTimersByTimeAsync(1000)
    expect(second).toEqual([1])
    expect(mine).toHaveBeenCalledTimes(1)     // 没有第二次请求
  })

  it('提交后 rememberRating 覆盖旧缓存：切走再切回不会显示成「没评过」', async () => {
    mine.mockResolvedValueOnce({ ratings: {} })   // 挂载时确实没评
    const m = await freshModule()
    const before: number[] = []
    m.hydrateRating('9', '/api', (r) => before.push(r))
    await vi.advanceTimersByTimeAsync(100)
    await vi.waitFor(() => expect(before).toEqual([0]))

    m.rememberRating('9', -1)                 // 用户点了 👎
    const after: number[] = []
    m.hydrateRating('9', '/api', (r) => after.push(r))
    expect(after).toEqual([-1])
    expect(mine).toHaveBeenCalledTimes(1)
  })

  it('失败静默且不落缓存：下次登记会重新问（否则永远回填不上）', async () => {
    mine.mockRejectedValueOnce(new Error('离线'))
    const m = await freshModule()
    const got: number[] = []
    m.hydrateRating('5', '/api', (r) => got.push(r))
    await vi.advanceTimersByTimeAsync(100)
    await vi.waitFor(() => expect(mine).toHaveBeenCalledTimes(1))
    expect(got).toEqual([])                    // 失败不打扰用户

    mine.mockResolvedValueOnce({ ratings: { '5': 1 } })
    m.hydrateRating('5', '/api', (r) => got.push(r))
    await vi.advanceTimersByTimeAsync(100)
    await vi.waitFor(() => expect(got).toEqual([1]))
    expect(mine).toHaveBeenCalledTimes(2)
  })

  it('取消只丢回调：同一 id 的另一个订阅者照样收到结果', async () => {
    mine.mockResolvedValueOnce({ ratings: { '3': 1 } })
    const m = await freshModule()
    const kept: number[] = []
    const gone: number[] = []
    const cancel = m.hydrateRating('3', '/api', (r) => gone.push(r))
    m.hydrateRating('3', '/api', (r) => kept.push(r))
    cancel()
    await vi.advanceTimersByTimeAsync(100)
    await vi.waitFor(() => expect(kept).toEqual([1]))
    expect(gone).toEqual([])
  })
})
