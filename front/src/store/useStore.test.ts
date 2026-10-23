/**
 * store 层的行为基线（vitest，纯 node 环境，不起浏览器）。
 *
 * 为什么前端也要有测试：后端 436 条离线用例当提交门禁，前端**一条都没有**，
 * 于是这一批缺陷全是「看得出来但没人能证明」的形态——
 *   · 流被干净关闭却没发 done → 气泡永久停在「帮家人找资料…」且没有重试入口；
 *   · 同一段逻辑在 send 与 regenerate 各抄一份 → 修了一边漏一边（真实发生过）；
 *   · loadingConvs/loadingMsgs 只写不读 → 加载期拿空状态冒充「没有数据」；
 *   · clearAll 吞掉异常却照常报成功 → 用户同时看到「失败」与「已清空」。
 * 这四条本文件各有一条用例，且都是**能红**的（把代码改回旧写法即失败）。
 *
 * 约定：不发真实网络请求（整个 `../lib/api` 被替身顶掉），也不依赖后端在跑。
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'

// 必须在 import store 之前 hoist：vitest 会把 vi.mock 提升到模块顶部
vi.mock('../lib/api', () => {
  class UnauthorizedError extends Error {}
  return {
    UnauthorizedError,
    setAuthToken: vi.fn(),
    health: vi.fn(async () => ({ ok: true })),
    listConversations: vi.fn(async () => ({ conversations: [] })),
    getConversation: vi.fn(async () => ({ messages: [] })),
    deleteConversation: vi.fn(async () => ({ ok: true })),
    renameConversation: vi.fn(async () => ({ ok: true })),
    clearConversations: vi.fn(async () => ({ ok: true })),
    ingest: vi.fn(async () => ({ chain_id: 'x' })),
    ask: vi.fn(async () => ({
      answer: 'A', thread_id: 't', intent: '', slots: [], characters: [], docs: 0,
    })),
    askStream: vi.fn(),
    regenerateStream: vi.fn(),
  }
})

import * as api from '../lib/api'
import { useStore } from './useStore'
import type { StreamEvent } from '../types'

/**
 * 造一个「按序吐完事件然后正常结束」的假 SSE。
 * `done` 不发就等价于**干净断流**（反代超时、后端进程被杀的典型形态）。
 * 参数按真实签名写死，避免 `mockImplementation` 因类型退化而被 tsc 拒掉。
 */
const fakeAsk = (events: StreamEvent[]) =>
  (_q: string, _t: string | null, onEvent: (e: StreamEvent) => void) => {
    for (const e of events) onEvent(e)
    return { controller: new AbortController(), done: Promise.resolve() }
  }

const fakeRegen = (events: StreamEvent[]) =>
  (_t: string, onEvent: (e: StreamEvent) => void) => {
    for (const e of events) onEvent(e)
    return { controller: new AbortController(), done: Promise.resolve() }
  }

const DONE = (over: Partial<Extract<StreamEvent, { done: true }>> = {}): StreamEvent =>
  ({
    done: true,
    answer: '这是完整答案',
    intent: 'chitchat',
    slots: [],
    characters: [],
    docs: 0,
    sources: ['守岸人 › 角色攻略 › 配队'],
    truncated: false,
    emotion: '',
    thread_id: 'thread-1',
    ...over,
  }) as Extract<StreamEvent, { done: true }>

const lastMsg = () => {
  const msgs = useStore.getState().messages
  return msgs[msgs.length - 1]
}

beforeEach(() => {
  vi.clearAllMocks()
  useStore.setState({
    auth: { username: 'tester', role: 'guest', token: 'tk' },
    messages: [],
    convs: [],
    activeThreadId: '',
    busy: false,
    loadingConvs: false,
    convsErr: '',
    loadingMsgs: false,
    settings: { ...useStore.getState().settings, stream: true, apiBase: '' },
  })
})

describe('流式收尾：没有 done 就是没答完', () => {
  it('send：干净断流必须落 error 态，而不是永久停在检索中', async () => {
    vi.mocked(api.askStream).mockImplementation(
      fakeAsk([{ stage: 'vector', label: '帮家人找资料' }]),
    )

    await useStore.getState().send('随便问一句')

    const m = lastMsg()
    expect(m.streaming).toBe(false)
    expect(m.status).toBe('error')       // 旧写法停在这里，且没有重试入口
    expect(m.error).toContain('中断')
  })

  it('send：收到 done 则正常收尾，并把 web_reason 带进 meta', async () => {
    vi.mocked(api.askStream).mockImplementation(
      fakeAsk([DONE({ web_reason: '联网搜索没成功（API Key 无效或已过期）' })]),
    )
    // 真实服务端在新建会话后就查得到它。不这么替会撞上 loadConversations 里
    // 「当前会话已不存在 → 退回草稿态并清空 messages」那条**正确**逻辑，
    // 测出来的是替身不像真，而不是代码有问题。
    vi.mocked(api.listConversations).mockResolvedValueOnce({
      conversations: [{ thread_id: 'thread-1', title: '随便问一句', updated_at: '', message_count: 2 }],
    } as never)

    await useStore.getState().send('随便问一句')

    const m = lastMsg()
    expect(m.status).toBe('done')
    expect(m.content).toBe('这是完整答案')
    // meta 键表只有一份实现：后端新增字段必须这里也读得到（否则引用面板静默缺一类来源）
    expect(m.meta?.sources).toEqual(['守岸人 › 角色攻略 › 配队'])
    expect(m.meta?.web_reason).toContain('API Key')
  })

  it('send：后端已下发 error 事件时，保留后端文案不被泛化文案覆盖', async () => {
    vi.mocked(api.askStream).mockImplementation(
      fakeAsk([{ error: '模型服务暂时不可用', detail: 'connection refused' } as StreamEvent]),
    )

    await useStore.getState().send('随便问一句')

    const m = lastMsg()
    expect(m.status).toBe('error')
    expect(m.error).toContain('模型服务暂时不可用')
    expect(m.error).not.toContain('连接提前结束')
  })

  it('regenerate：同一条断流判据（这条路径以前漏了，正是「修一边漏一边」）', async () => {
    useStore.setState({
      activeThreadId: 'thread-1',
      messages: [
        { id: 'u1', role: 'user', content: '守岸人配队', createdAt: 1 },
        { id: 'a1', role: 'assistant', content: '旧答案', createdAt: 2 },
      ],
    })
    vi.mocked(api.regenerateStream).mockImplementation(
      fakeRegen([{ stage: 'vector', label: '帮家人找资料' }]),
    )

    await useStore.getState().regenerate('a1')

    const m = lastMsg()
    expect(m.streaming).toBe(false)
    expect(m.status).toBe('error')
  })
  it('done 事件的每一个字段都必须落到 meta（键表是对外契约，不是实现细节）', async () => {
    // 为什么单独钉这一条：`web_reason` 差点成为下一个漏网的——后端加字段时只需改
    // `metaOf` 一处，但**没有任何测试**能发现「加了一个字段、meta 里没有它」。
    // 期望键表写死在这里就是故意的：它就是前后端之间那份契约的副本。
    vi.mocked(api.askStream).mockImplementation(
      fakeAsk([DONE({
        intent: 'hybrid', slots: ['队友'], characters: ['守岸人'], docs: 3,
        sources: ['守岸人 › 角色攻略 › 配队', '联网 › 攻略 · https://a.example/x'],
        truncated: true, emotion: '[excited]',
        web_reason: '联网搜索没成功（API Key 无效或已过期）',
      })]),
    )
    vi.mocked(api.listConversations).mockResolvedValueOnce({
      conversations: [{ thread_id: 'thread-1', title: 't', updated_at: '', message_count: 2 }],
    } as never)

    await useStore.getState().send('守岸人配队')

    const meta = lastMsg().meta ?? {}
    expect(Object.keys(meta).sort()).toEqual([
      "characters", "docs", "emotion", "intent", "slots", "sources", "truncated", "web_reason",
    ])
  })
})

describe('加载态与清空结果必须可被消费', () => {
  it('loadConversations 期间 loadingConvs 为真、结束后为假', async () => {
    let released!: () => void
    const gate = new Promise<void>((r) => (released = r))
    vi.mocked(api.listConversations).mockImplementation(async () => {
      await gate
      return { conversations: [] }
    })

    const p = useStore.getState().loadConversations()
    // UI 靠它区分「正在加载」与「你真的没有会话」；这个字段曾只写不读
    expect(useStore.getState().loadingConvs).toBe(true)
    released()
    await p
    expect(useStore.getState().loadingConvs).toBe(false)
  })

  it('loadConversations 读失败：convsErr 记下原因，列表保留上一次的', async () => {
    // 这条守的是「读不到 ≠ 没有」：store 里必须有失败原因这个出口，
    // 否则会话栏/历史页/设置页三处都会把请求挂了读成「暂无会话」。
    useStore.setState({ convs: [{ thread_id: 'old', title: '上一次读到的', updated_at: '' } as never] })
    vi.mocked(api.listConversations).mockRejectedValueOnce(new Error('502 Bad Gateway'))

    await useStore.getState().loadConversations()

    expect(useStore.getState().convsErr).toContain('502')
    expect(useStore.getState().convs).toHaveLength(1)   // 保持上一次的可用状态比清空好
  })

  it('loadConversations 读成功：必须把 convsErr 清零', async () => {
    // 只写不清的话，一次偶发失败会永久挂在界面上（重试成功后仍显示「读取失败」）
    useStore.setState({ convsErr: '上一次的失败' })
    vi.mocked(api.listConversations).mockResolvedValueOnce({ conversations: [] } as never)

    await useStore.getState().loadConversations()

    expect(useStore.getState().convsErr).toBe('')
  })

  it('clearAll：后端失败要如实返回 false，且不许清本地列表', async () => {
    useStore.setState({ convs: [{ thread_id: 't1', title: 'A', updated_at: '' } as never] })
    vi.mocked(api.clearConversations).mockRejectedValueOnce(new Error('boom'))

    const ok = await useStore.getState().clearAll()

    expect(ok).toBe(false)
    expect(useStore.getState().convs).toHaveLength(1)   // 没清掉就不许装作清掉了
  })

  it('clearAll：成功时返回 true 并清空', async () => {
    useStore.setState({ convs: [{ thread_id: 't1', title: 'A', updated_at: '' } as never] })

    const ok = await useStore.getState().clearAll()

    expect(ok).toBe(true)
    expect(useStore.getState().convs).toHaveLength(0)
  })
})

/** 让 loadConversations 认为「当前会话还在」——否则它会退回草稿态并清空 messages（真实且正确的行为） */
function seedConversation() {
  vi.mocked(api.listConversations).mockImplementation(async () => ({
    conversations: [{ thread_id: 'thread-1', title: 't', updated_at: '', message_count: 2 }] as never,
  }))
}

describe('回答落库后的真 id 必须回到气泡上', () => {
  /**
   * 本轮新生成的气泡在前端只有本地随机 uid，而满意度反馈按 `msg.id` 提交。
   * 不把 `done.message_id`（= messages 表的行号）换上去，就会出现两件事：
   *   · 刷新后回填不到那条评价（服务端存着的 target_id 是一个再也不会出现的 uid）；
   *   * 同一条回答可以被评两次（一次 uid、一次数字 id）。
   */
  it('send：done 带 message_id 时气泡 id 换成它', async () => {
    seedConversation()
    vi.mocked(api.askStream).mockImplementation(fakeAsk([{ token: '答' }, DONE({ message_id: 162 })]))

    await useStore.getState().send('守岸人配队')

    expect(lastMsg().id).toBe('162')
  })

  it('send：没有 message_id（早退/空答案）时保留原 id，不许变成 undefined', async () => {
    seedConversation()
    const events = [{ token: '答' }, DONE()]
    vi.mocked(api.askStream).mockImplementation(fakeAsk(events))

    await useStore.getState().send('守岸人配队')

    expect(lastMsg().id).toBeTruthy()
    expect(lastMsg().id).not.toBe('undefined')
  })

  it('非流式 /ask：meta 键集合与流式一致，且同样采纳 message_id', async () => {
    // 关掉 stream 走 /ask 这条出口：它原先手打七个键、漏了 web_reason（同一语义两处判据的必然结局）
    seedConversation()
    useStore.setState({ settings: { ...useStore.getState().settings, stream: false } })
    vi.mocked(api.ask).mockResolvedValue({
      answer: 'A', thread_id: 'thread-1', intent: 'fact', slots: ['队友'], characters: ['守岸人'],
      docs: 2, sources: ['守岸人 › 配队'], truncated: false, emotion: '',
      web_reason: '联网搜索没成功（API Key 无效或已过期）', message_id: 177,
    })

    await useStore.getState().send('守岸人配队')

    const m = lastMsg()
    expect(m.id).toBe('177')
    expect(Object.keys(m.meta ?? {}).sort()).toEqual(
      ['characters', 'docs', 'emotion', 'intent', 'slots', 'sources', 'truncated', 'web_reason'],
    )
    expect(m.meta?.web_reason).toContain('API Key')
  })

  it('regenerate：同样换成新的真 id', async () => {
    seedConversation()
    // regenerate 要先在本地找到「最后一句提问」才动手（找不到就直接 return）
    useStore.setState({
      activeThreadId: 'thread-1',
      messages: [{ id: 'q1', role: 'user', content: '守岸人配队', createdAt: 1 } as never],
    })
    vi.mocked(api.regenerateStream).mockImplementation(
      fakeRegen([{ token: '新答案' }, DONE({ message_id: 205 })]),
    )

    await useStore.getState().regenerate('任意一条消息 id')

    expect(lastMsg().id).toBe('205')
  })
})
