import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { Conversation, Message, Settings, AskMeta, IngestRecord, StreamEvent, AuthInfo } from '../types'
import * as api from '../lib/api'

/** 生成短 id */
function uid(): string {
  return Math.random().toString(36).slice(2, 10)
}

function nowConv(): Conversation {
  const t = Date.now()
  return {
    id: uid(),
    threadId: uid(),
    title: '新对话',
    messages: [],
    createdAt: t,
    updatedAt: t,
  }
}

/** 从首条用户消息里提取一个标题 */
function titleFrom(text: string): string {
  const clean = text.replace(/\s+/g, ' ').trim()
  if (!clean) return '新对话'
  return clean.length > 22 ? clean.slice(0, 22) + '…' : clean
}

const DEFAULT_SETTINGS: Settings = {
  apiBase: '',
  theme: 'dark',
  stream: true,
  fontSize: 14,
  sidebarCollapsed: false,
  avatarAssistant: '',
  avatarUser: '',
  bgPreset: 'default',
  bgImage: '',
  bgDim: 0.45,
  bgBlur: 0,
}

interface Toast {
  id: string
  kind: 'ok' | 'err' | 'info'
  text: string
}

interface Store {
  conversations: Conversation[]
  activeId: string
  settings: Settings
  /** 登录态（token/用户名/角色）；null = 未登录。persist 到 localStorage */
  auth: AuthInfo | null
  /** 正在流式接收的会话 id（用于禁用输入、显示停止按钮） */
  busyId: string | null
  toasts: Toast[]
  ingests: IngestRecord[]
  /** 后端连通性：unknown | ok | down */
  health: 'unknown' | 'ok' | 'down'

  // 鉴权
  setAuth: (a: AuthInfo) => void
  clearAuth: () => void

  // 会话
  newConversation: () => string
  selectConversation: (id: string) => void
  renameConversation: (id: string, title: string) => void
  deleteConversation: (id: string) => void
  clearAll: () => void

  // 问答
  send: (text: string) => Promise<void>
  stop: () => void
  regenerate: (messageId: string) => Promise<void>

  // ingest
  ingestCharacter: (character: string) => Promise<void>

  // 设置 / toast / health
  setSettings: (patch: Partial<Settings>) => void
  toast: (kind: Toast['kind'], text: string) => void
  dismissToast: (id: string) => void
  checkHealth: () => Promise<void>
}

/** 当前正在进行的一次流式请求的中断器 */
let activeAbort: AbortController | null = null

export const useStore = create<Store>()(
  persist(
    (set, get) => {
      // —— 内部工具：在指定会话追加/更新消息 ——
      const patchConv = (id: string, fn: (c: Conversation) => Conversation) => {
        set((s) => ({
          conversations: s.conversations.map((c) => (c.id === id ? fn(c) : c)),
        }))
      }
      const updateMsg = (convId: string, msgId: string, patch: Partial<Message>) => {
        patchConv(convId, (c) => ({
          ...c,
          updatedAt: Date.now(),
          messages: c.messages.map((m) => (m.id === msgId ? { ...m, ...patch } : m)),
        }))
      }
      const pushMsg = (convId: string, msg: Message) => {
        patchConv(convId, (c) => ({ ...c, updatedAt: Date.now(), messages: [...c.messages, msg] }))
      }

      return {
        conversations: [],
        activeId: '',
        settings: DEFAULT_SETTINGS,
        auth: null,
        busyId: null,
        toasts: [],
        ingests: [],
        health: 'unknown',

        setAuth: (a) => {
          api.setAuthToken(a.token)
          set({ auth: a })
        },

        clearAuth: () => {
          api.setAuthToken('')
          set({ auth: null })
        },

        newConversation: () => {
          const c = nowConv()
          set((s) => ({ conversations: [c, ...s.conversations], activeId: c.id }))
          return c.id
        },

        selectConversation: (id) => set({ activeId: id }),

        renameConversation: (id, title) =>
          patchConv(id, (c) => ({ ...c, title: title.trim() || c.title })),

        deleteConversation: (id) =>
          set((s) => {
            const conversations = s.conversations.filter((c) => c.id !== id)
            let activeId = s.activeId
            if (activeId === id) {
              activeId = conversations[0]?.id ?? ''
            }
            return { conversations, activeId }
          }),

        clearAll: () =>
          set((s) => {
            const fresh = nowConv()
            return {
              conversations: [fresh],
              activeId: fresh.id,
              ingests: [],
              settings: { ...s.settings },
            }
          }),

        send: async (text) => {
          const q = text.trim()
          if (!q) return
          const s = get()
          if (s.busyId) return

          // 确保有活跃会话
          let convId = s.activeId
          if (!convId || !s.conversations.find((c) => c.id === convId)) {
            convId = get().newConversation()
          }
          const conv = get().conversations.find((c) => c.id === convId)!

          // 若是首条消息，用它当标题
          if (conv.messages.length === 0) {
            patchConv(convId, (c) => ({ ...c, title: titleFrom(q) }))
          }

          const userMsg: Message = { id: uid(), role: 'user', content: q, createdAt: Date.now() }
          const botMsg: Message = {
            id: uid(),
            role: 'assistant',
            content: '',
            createdAt: Date.now(),
            streaming: true,
            status: 'retrieving',
          }
          pushMsg(convId, userMsg)
          pushMsg(convId, botMsg)
          set({ busyId: convId })

          const { settings } = get()
          const base = settings.apiBase
          const threadId = conv.threadId

          try {
            if (settings.stream) {
              const { controller, done } = api.askStream(
                q,
                threadId,
                (evt: StreamEvent) => {
                  if ('stage' in evt) {
                    // 细粒度阶段：抽取/检索/生成。检索实测约 19s，这段必须让用户看到进展
                    updateMsg(convId, botMsg.id, { status: 'retrieving', stageLabel: evt.label })
                  } else if ('status' in evt && evt.status === 'retrieving') {
                    updateMsg(convId, botMsg.id, { status: 'retrieving' })
                  } else if ('token' in evt) {
                    // 首个 token 到达即切换为生成态
                    patchConv(convId, (c) => ({
                      ...c,
                      messages: c.messages.map((m) =>
                        m.id === botMsg.id ? { ...m, content: m.content + evt.token, status: undefined } : m,
                      ),
                    }))
                  } else if ('error' in evt) {
                    // 后端 SSE 流中途异常（响应头已发出，错误以事件下发）：
                    // 已吐出的 token 保留在气泡里，标记 error 态供重生成
                    updateMsg(convId, botMsg.id, {
                      streaming: false,
                      status: 'error',
                      stageLabel: undefined,
                      error: evt.error + (evt.detail ? `（${evt.detail}）` : ''),
                    })
                    get().toast('err', evt.error)
                  } else if ('done' in evt && evt.done) {
                    const meta: AskMeta = {
                      intent: evt.intent,
                      slots: evt.slots,
                      characters: evt.characters,
                      docs: evt.docs,
                      sources: evt.sources,
                      truncated: evt.truncated,
                    }
                    // done.answer 是后端给出的权威全文：
                    // 触发复读兜底时它是「截断后」的内容，必须覆盖掉已流式渲染的复读文本，
                    // 否则前端仍显示重复内容，后端兜底等于白做。
                    // 仅在 answer 为空（异常兜底）时保留已渲染内容，避免把气泡清空。
                    const streamed = get()
                      .conversations.find((c) => c.id === convId)!
                      .messages.find((m) => m.id === botMsg.id)!.content
                    updateMsg(convId, botMsg.id, {
                      streaming: false,
                      status: 'done',
                      stageLabel: undefined,
                      meta,
                      content: evt.answer || streamed,
                    })
                  }
                },
                base,
              )
              activeAbort = controller
              await done
              // 收尾：确保非流式态
              updateMsg(convId, botMsg.id, { streaming: false })
            } else {
              const out = await api.ask(q, threadId, base)
              updateMsg(convId, botMsg.id, {
                content: out.answer,
                streaming: false,
                status: 'done',
                meta: {
                  intent: out.intent,
                  slots: out.slots,
                  characters: out.characters,
                  docs: out.docs,
                  sources: out.sources,
                  truncated: out.truncated,
                },
              })
            }
          } catch (err) {
            const msg = err instanceof Error ? err.message : String(err)
            if (msg.includes('abort')) {
              updateMsg(convId, botMsg.id, { streaming: false, status: 'done' })
            } else if (err instanceof api.UnauthorizedError) {
              // token 过期/被撤：清登录态，App 会自动切到登录页
              updateMsg(convId, botMsg.id, { streaming: false, status: 'error', error: msg })
              get().clearAuth()
              get().toast('err', msg)
            } else {
              updateMsg(convId, botMsg.id, {
                streaming: false,
                status: 'error',
                error: msg,
                content: get().conversations.find((c) => c.id === convId)!.messages.find((m) => m.id === botMsg.id)!.content,
              })
              get().toast('err', '请求失败：' + msg)
              set({ health: 'down' })
            }
          } finally {
            activeAbort = null
            set({ busyId: null })
          }
        },

        stop: () => {
          if (activeAbort) {
            activeAbort.abort()
            activeAbort = null
          }
          const s = get()
          if (s.busyId) {
            const convId = s.busyId
            patchConv(convId, (c) => ({
              ...c,
              messages: c.messages.map((m) => (m.streaming ? { ...m, streaming: false, status: 'done' } : m)),
            }))
            set({ busyId: null })
          }
        },

        regenerate: async (messageId) => {
          const s = get()
          if (s.busyId) return
          const conv = s.conversations.find((c) => c.messages.some((m) => m.id === messageId))
          if (!conv) return
          // 找到该 assistant 消息之前最近的 user 消息
          const idx = conv.messages.findIndex((m) => m.id === messageId)
          let userText = ''
          for (let i = idx - 1; i >= 0; i--) {
            if (conv.messages[i].role === 'user') {
              userText = conv.messages[i].content
              break
            }
          }
          if (!userText) return
          // 删除该 assistant 消息及其之后所有消息，重发
          patchConv(conv.id, (c) => ({ ...c, messages: c.messages.slice(0, idx) }))
          set({ activeId: conv.id })
          // 复用 send：先移除刚保留的最后一条 user 消息，避免重复
          patchConv(conv.id, (c) => ({ ...c, messages: c.messages.slice(0, -1) }))
          await get().send(userText)
        },

        ingestCharacter: async (character) => {
          const name = character.trim()
          if (!name) return
          const { settings } = get()
          const rec: IngestRecord = {
            id: uid(),
            character: name,
            chainId: '',
            state: 'PENDING',
            createdAt: Date.now(),
            ok: false,
          }
          set((s) => ({ ingests: [rec, ...s.ingests] }))
          try {
            const out = await api.ingest(name, settings.apiBase)
            set((s) => ({
              ingests: s.ingests.map((r) =>
                r.id === rec.id ? { ...r, chainId: out.chain_id, state: out.state, ok: true } : r,
              ),
            }))
            get().toast('ok', `已提交「${name}」入库流水线`)
          } catch (err) {
            const msg = err instanceof Error ? err.message : String(err)
            set((s) => ({
              ingests: s.ingests.map((r) =>
                r.id === rec.id ? { ...r, state: 'FAILED', ok: false, error: msg } : r,
              ),
            }))
            get().toast('err', '入库失败：' + msg)
            set({ health: 'down' })
          }
        },

        setSettings: (patch) => set((s) => ({ settings: { ...s.settings, ...patch } })),

        toast: (kind, text) => {
          const id = uid()
          set((s) => ({ toasts: [...s.toasts, { id, kind, text }] }))
          setTimeout(() => get().dismissToast(id), 3600)
        },

        dismissToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),

        checkHealth: async () => {
          try {
            await api.health(get().settings.apiBase)
            set({ health: 'ok' })
          } catch {
            set({ health: 'down' })
          }
        },
      }
    },
    {
      name: 'wuwa-rag-front',
      version: 1,
      // persist 的默认合并是**浅合并**：老存档里的 settings 是个完整对象，会整体顶掉
      // DEFAULT_SETTINGS，新增的个性化字段（头像 / 背景 / 侧栏折叠）全成 undefined。
      // 这里显式深合一层 settings，保证旧存档升级后新字段有默认值。
      merge: (persisted, current) => {
        const p = (persisted ?? {}) as Partial<Store>
        return {
          ...current,
          ...p,
          settings: { ...DEFAULT_SETTINGS, ...(p.settings ?? {}) },
        }
      },
      partialize: (s) => ({
        conversations: s.conversations.map((c) => ({
          ...c,
          // 不持久化流式中间态
          messages: c.messages.map((m) => ({ ...m, streaming: false })),
        })),
        activeId: s.activeId,
        settings: s.settings,
        auth: s.auth,
        ingests: s.ingests.slice(0, 50),
      }),
    },
  ),
)

/** 保证至少有一个会话；返回其 id */
export function ensureActiveConversation(): string {
  const s = useStore.getState()
  if (s.activeId && s.conversations.some((c) => c.id === s.activeId)) {
    return s.activeId
  }
  return s.newConversation()
}
