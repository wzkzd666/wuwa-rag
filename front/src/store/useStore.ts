import { create } from 'zustand'
import type {
  AuthInfo, ConversationMeta, IngestRecord, Message, Settings, StreamEvent, StoredMessage,
} from '../types'
import * as api from '../lib/api'

/** 生成短 id（流式期间的临时消息用；服务端消息用它的数字 id 转字符串） */
function uid(): string {
  return Math.random().toString(36).slice(2, 10)
}

/**
 * 本地持久化分**两格**——这两类数据的归属根本不同，塞在一起就是上一版的问题：
 *
 *   `wuwa-rag-front:auth`        —— 设备级。谁在这台机器上登录过（单值）。
 *   `wuwa-rag-front:u:<用户名>`   —— 用户级。**界面偏好**：主题/字号/侧栏折叠/
 *                                    头像/背景/TTS 开关。
 *
 * **会话与消息不在这里**：服务端 PG 才是真源（后端 conversations.py，按 user_id 隔离）。
 * 以前是一个键 `wuwa-rag-front` 装下所有东西，换个账号登录就会看到上一个人的会话列表、
 * 消息、头像和背景 —— 那正是这次要修的问题。
 */
const LS_AUTH = 'wuwa-rag-front:auth'
const LS_SETTINGS = (username: string) => `wuwa-rag-front:u:${username}`
/** 旧版单键存档（zustand persist 的结构是 `{state, version}`），只用于一次性迁移读取。 */
const LS_LEGACY = 'wuwa-rag-front'

function readJSON<T>(key: string): T | null {
  try {
    const raw = localStorage.getItem(key)
    return raw ? (JSON.parse(raw) as T) : null
  } catch {
    return null
  }
}

function writeJSON(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch {
    // 隐私模式 / 超配额：最坏是「下次打开不记得设置」，不该让界面报错
  }
}

function dropKey(key: string): void {
  try {
    localStorage.removeItem(key)
  } catch { /* 同上 */ }
}

/** 会话摘要 → 预览文案（历史页与侧栏共用） */
export function previewOf(c: ConversationMeta): string {
  if (!c.last_content) return '空会话'
  const head = c.last_role === 'user' ? '你：' : '爱弥斯：'
  return head + c.last_content.replace(/\s+/g, ' ').slice(0, 60)
}

/** 服务端消息 → 前端消息 */
function toMessage(m: StoredMessage): Message {
  return {
    id: String(m.id),
    role: m.role,
    content: m.content,
    createdAt: Date.parse(m.created_at) || Date.now(),
    status: 'done',
    meta: m.meta,
  }
}

const DEFAULT_SETTINGS: Settings = {
  apiBase: '',
  theme: 'dark',
  stream: true,
  fontSize: 14,
  // 会话栏默认**展开**：它是「切回上一轮对话」的主要入口，收起只留「新建」会让人
  // 以为历史丢了。窗口窄时用户自己收或用下面的断点自动收。
  sessionListCollapsed: false,
  // 会话栏默认**展开**：它是「切回上一轮对话」的主要入口，收起只留「新建」会让人
  sidebarCollapsed: false,
  avatarAssistant: '',
  avatarUser: '',
  bgPreset: 'default',
  bgImage: '',
  bgDim: 0.45,
  bgBlur: 0,
  // 面板磨砂：默认半透明（0.72）+ 12px 模糊。1.0 = 实心白框（改版前的观感），
  // 调低更透、背景图更有存在感，但表格文字对比度会下降。
  panelAlpha: 0.72,
  panelBlur: 12,
  // 语音朗读开关：这是**本机显示偏好**（关掉只是不渲染朗读按钮）。
  // 后端 TTS 总开关已于 2026-09-30 开启，且密钥改由用户在设置页自持，
  // 所以前端默认一并打开，开箱即可看到朗读按钮；凭据没配好时设置页的
  // 「语音合成」卡会显示「未就绪」并给出具体原因，点击朗读按钮也会提示。
  ttsEnabled: true,
}

/**
 * 旧存档一次性迁移：把老键里的 auth / settings 拆进新两格。
 *
 * 老的 `conversations` **直接丢弃**，不迁移：它没有用户归属，无法判断属于谁，
 * 硬塞给下一个登录的人就是继续泄漏。会话正文本来也只是缓存，服务端没有对应数据。
 */
function migrateLegacy(): AuthInfo | null {
  const legacy = readJSON<{ state?: { auth?: AuthInfo | null; settings?: Partial<Settings> } }>(LS_LEGACY)
  const auth = legacy?.state?.auth ?? null
  if (!auth) return null
  const saved = legacy?.state?.settings
  if (saved) writeJSON(LS_SETTINGS(auth.username), { ...DEFAULT_SETTINGS, ...saved })
  writeJSON(LS_AUTH, auth)
  return auth
}

interface Toast {
  id: string
  kind: 'ok' | 'err' | 'info'
  text: string
}

interface Store {
  /** 登录态（token/用户名/角色）；null = 未登录。设备级持久化 */
  auth: AuthInfo | null
  /** 界面偏好。**按用户**持久化（不同账号互不影响） */
  settings: Settings

  // —— 会话：服务端为真源，本地只做当前视图的缓存 ——
  /** 会话列表（摘要，不含正文），新的在前 */
  convs: ConversationMeta[]
  /** 当前会话的 thread_id；'' = 草稿态（还没和任何服务端会话绑定） */
  activeThreadId: string
  /** 当前会话的消息（服务端拉取的 + 流式中的临时消息） */
  messages: Message[]
  loadingConvs: boolean
  loadingMsgs: boolean
  /** 是否正在流式接收（一个时刻只会有一个会话在跑） */
  busy: boolean

  toasts: Toast[]
  ingests: IngestRecord[]
  /** 后端连通性：unknown | ok | down */
  health: 'unknown' | 'ok' | 'down'

  // 鉴权
  setAuth: (a: AuthInfo) => void
  clearAuth: () => void

  // 会话
  loadConversations: () => Promise<void>
  newConversation: () => void
  selectConversation: (threadId: string) => Promise<void>
  renameConversation: (threadId: string, title: string) => Promise<void>
  deleteConversation: (threadId: string) => Promise<void>
  clearAll: () => Promise<void>

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

/** 启动时读档：先看设备级登录态，再按它取该用户的界面偏好 */
const bootAuth = readJSON<AuthInfo>(LS_AUTH) ?? migrateLegacy()
const bootSettings: Settings = {
  ...DEFAULT_SETTINGS,
  ...(bootAuth ? readJSON<Partial<Settings>>(LS_SETTINGS(bootAuth.username)) ?? {} : {}),
}

export const useStore = create<Store>()((set, get) => {
  /** 把某条消息就地打个补丁（只改当前会话的最后一条流式消息） */
  const patchMsg = (msgId: string, patch: Partial<Message>) =>
    set((s) => ({ messages: s.messages.map((m) => (m.id === msgId ? { ...m, ...patch } : m)) }))

  /** 401 / 网络异常的统一收尾：登录过期就踢回登录页 */
  const fail = (err: unknown, prefix: string) => {
    const msg = err instanceof Error ? err.message : String(err)
    if (err instanceof api.UnauthorizedError) {
      get().clearAuth()
      get().toast('err', msg)
      return
    }
    get().toast('err', prefix + msg)
  }

  return {
    auth: bootAuth,
    settings: bootSettings,

    convs: [],
    activeThreadId: '',
    messages: [],
    loadingConvs: false,
    loadingMsgs: false,
    busy: false,

    toasts: [],
    ingests: [],
    health: 'unknown',

    setAuth: (a) => {
      api.setAuthToken(a.token)
      writeJSON(LS_AUTH, a)
      // 换用户 = 换一整套界面偏好与空会话视图。必须先清空再拉列表，
      // 否则上一秒还挂着上一个账号的会话，会闪一下别人的历史。
      const saved = readJSON<Partial<Settings>>(LS_SETTINGS(a.username)) ?? {}
      set({
        auth: a,
        settings: { ...DEFAULT_SETTINGS, ...saved },
        convs: [],
        activeThreadId: '',
        messages: [],
        busy: false,
      })
      void get().loadConversations()
    },

    clearAuth: () => {
      api.setAuthToken('')
      dropKey(LS_AUTH)
      set({
        auth: null,
        convs: [],
        activeThreadId: '',
        messages: [],
        busy: false,
        ingests: [],
      })
    },

    loadConversations: async () => {
      if (!get().auth) {
        set({ convs: [] })
        return
      }
      set({ loadingConvs: true })
      try {
        const { conversations } = await api.listConversations('', get().settings.apiBase)
        set({ convs: conversations })
        // 当前会话在服务端已不存在（别处删了）→ 退回草稿态，别停在一个空壳会话上
        const cur = get().activeThreadId
        if (cur && !conversations.some((c) => c.thread_id === cur)) {
          set({ activeThreadId: '', messages: [] })
        }
      } catch (err) {
        if (err instanceof api.UnauthorizedError) get().clearAuth()
        // 其它错误不动列表：保持上一次的可用状态比清空好
      } finally {
        set({ loadingConvs: false })
      }
    },

    newConversation: () => {
      // 纯本地草稿：服务端会话在**第一句问答**时才创建（后端 /ask 里 ensure_conversation），
      // 这样点「新建对话」又没说话不会留下一堆空会话。
      set({ activeThreadId: '', messages: [] })
    },

    selectConversation: async (threadId) => {
      set({ activeThreadId: threadId, messages: [], loadingMsgs: true })
      try {
        const det = await api.getConversation(threadId, get().settings.apiBase)
        // 防竞态：用户快速连点多个会话时，只认最后一次选中的那个
        if (get().activeThreadId !== threadId) return
        set({ messages: det.messages.map(toMessage) })
      } catch (err) {
        if (get().activeThreadId !== threadId) return
        set({ activeThreadId: '', messages: [] })
        fail(err, '打开会话失败：')
      } finally {
        if (get().activeThreadId === threadId) set({ loadingMsgs: false })
      }
    },

    renameConversation: async (threadId, title) => {
      const t = title.trim()
      if (!t) return
      const prev = get().convs
      // 乐观更新：重命名要立刻看到反馈，失败再回滚
      set({ convs: prev.map((c) => (c.thread_id === threadId ? { ...c, title: t } : c)) })
      try {
        await api.renameConversation(threadId, t, get().settings.apiBase)
      } catch (err) {
        set({ convs: prev })
        fail(err, '重命名失败：')
      }
    },

    deleteConversation: async (threadId) => {
      try {
        await api.deleteConversation(threadId, get().settings.apiBase)
      } catch (err) {
        fail(err, '删除失败：')
        return
      }
      set((s) => {
        const convs = s.convs.filter((c) => c.thread_id !== threadId)
        if (s.activeThreadId !== threadId) return { convs }
        return { convs, activeThreadId: '', messages: [] }
      })
    },

    clearAll: async () => {
      try {
        await api.clearConversations(get().settings.apiBase)
      } catch (err) {
        fail(err, '清空失败：')
        return
      }
      // 只清会话与入库记录；界面偏好是另一个维度，不动
      set({ convs: [], activeThreadId: '', messages: [], ingests: [] })
    },

    send: async (text) => {
      const q = text.trim()
      if (!q) return
      const s = get()
      if (s.busy || !s.auth) return

      const userMsg: Message = { id: uid(), role: 'user', content: q, createdAt: Date.now() }
      const botMsg: Message = {
        id: uid(),
        role: 'assistant',
        content: '',
        createdAt: Date.now(),
        streaming: true,
        status: 'retrieving',
      }
      set({ messages: [...s.messages, userMsg, botMsg], busy: true })

      // 草稿态传 null：由服务端建会话并把 id 回传（见 StreamEvent.thread_id）
      const threadId = s.activeThreadId || null
      const base = s.settings.apiBase
      const onDone = (evt: Extract<StreamEvent, { done: true }>) => {
        const meta = {
          intent: evt.intent,
          slots: evt.slots,
          characters: evt.characters,
          docs: evt.docs,
          sources: evt.sources,
          truncated: evt.truncated,
          emotion: evt.emotion,
        }
        // done.answer 是后端给出的权威全文：触发复读兜底时它是「截断后」的内容，
        // 必须覆盖掉已流式渲染的复读文本，否则前端仍显示重复内容，后端兜底等于白做。
        // 仅在 answer 为空（异常兜底）时保留已渲染内容，避免把气泡清空。
        set((st) => ({
          activeThreadId: evt.thread_id || st.activeThreadId,
          messages: st.messages.map((m) =>
            m.id === botMsg.id
              ? {
                  ...m,
                  streaming: false,
                  status: 'done' as const,
                  stageLabel: undefined,
                  meta,
                  content: evt.answer || m.content,
                }
              : m,
          ),
        }))
      }

      try {
        if (s.settings.stream) {
          const { controller, done } = api.askStream(
            q,
            threadId,
            (evt: StreamEvent) => {
              if ('stage' in evt) {
                // 细粒度阶段：抽取/检索/生成。检索实测约 19s，这段必须让用户看到进展
                patchMsg(botMsg.id, { status: 'retrieving', stageLabel: evt.label })
              } else if ('status' in evt && evt.status === 'retrieving') {
                patchMsg(botMsg.id, { status: 'retrieving' })
              } else if ('token' in evt) {
                // 首个 token 到达即切换为生成态
                set((st) => ({
                  messages: st.messages.map((m) =>
                    m.id === botMsg.id
                      ? { ...m, content: m.content + evt.token, status: undefined }
                      : m,
                  ),
                }))
              } else if ('error' in evt) {
                // 后端 SSE 流中途异常（响应头已发出，错误以事件下发）：
                // 已吐出的 token 保留在气泡里，标记 error 态供重生成
                patchMsg(botMsg.id, {
                  streaming: false,
                  status: 'error',
                  stageLabel: undefined,
                  error: evt.error + (evt.detail ? `（${evt.detail}）` : ''),
                })
                get().toast('err', evt.error)
              } else if ('done' in evt && evt.done) {
                onDone(evt)
              }
            },
            base,
          )
          activeAbort = controller
          await done
          patchMsg(botMsg.id, { streaming: false })
        } else {
          const out = await api.ask(q, threadId, base)
          set((st) => ({
            activeThreadId: out.thread_id || st.activeThreadId,
            messages: st.messages.map((m) =>
              m.id === botMsg.id
                ? {
                    ...m,
                    content: out.answer,
                    streaming: false,
                    status: 'done' as const,
                    meta: {
                      intent: out.intent,
                      slots: out.slots,
                      characters: out.characters,
                      docs: out.docs,
                      sources: out.sources,
                      truncated: out.truncated,
                      emotion: out.emotion,
                    },
                  }
                : m,
            ),
          }))
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err)
        if (msg.includes('abort')) {
          patchMsg(botMsg.id, { streaming: false, status: 'done' })
        } else if (err instanceof api.UnauthorizedError) {
          // token 过期/被撤：清登录态，App 会自动切到登录页
          patchMsg(botMsg.id, { streaming: false, status: 'error', error: msg })
          get().clearAuth()
          get().toast('err', msg)
        } else {
          patchMsg(botMsg.id, { streaming: false, status: 'error', error: msg })
          get().toast('err', '请求失败：' + msg)
          set({ health: 'down' })
        }
      } finally {
        activeAbort = null
        set({ busy: false })
        // 拉一次列表：拿服务端定的标题 / 预览 / 时间；草稿态提问时也顺带把
        // 新会话塞进列表（activeThreadId 已在 onDone 里认领，不会被重置）
        await get().loadConversations()
      }
    },

    stop: () => {
      if (activeAbort) {
        activeAbort.abort()
        activeAbort = null
      }
      if (get().busy) {
        set((s) => ({
          busy: false,
          messages: s.messages.map((m) =>
            m.streaming ? { ...m, streaming: false, status: 'done' as const } : m,
          ),
        }))
      }
    },

    regenerate: async (messageId) => {
      // messageId 只表示「用户点的是哪一条」，服务端一律重答最后一句提问；
      // 界面上只在最后一条助手消息上给按钮，两者天然一致，所以这里不用它。
      void messageId
      const s = get()
      if (s.busy || !s.activeThreadId) return

      // 本地先砍到「最后一句提问」为止，让旧回答立刻消失（服务端会同步删转录）
      let idx = -1
      for (let i = s.messages.length - 1; i >= 0; i--) {
        if (s.messages[i].role === 'user') {
          idx = i
          break
        }
      }
      if (idx < 0) return

      const botMsg: Message = {
        id: uid(),
        role: 'assistant',
        content: '',
        createdAt: Date.now(),
        streaming: true,
        status: 'retrieving',
      }
      set({ messages: [...s.messages.slice(0, idx + 1), botMsg], busy: true })

      try {
        const { controller, done } = api.regenerateStream(
          s.activeThreadId,
          (evt: StreamEvent) => {
            if ('stage' in evt) {
              patchMsg(botMsg.id, { status: 'retrieving', stageLabel: evt.label })
            } else if ('status' in evt && evt.status === 'retrieving') {
              patchMsg(botMsg.id, { status: 'retrieving' })
            } else if ('token' in evt) {
              set((st) => ({
                messages: st.messages.map((m) =>
                  m.id === botMsg.id
                    ? { ...m, content: m.content + evt.token, status: undefined }
                    : m,
                ),
              }))
            } else if ('error' in evt) {
              patchMsg(botMsg.id, {
                streaming: false,
                status: 'error',
                stageLabel: undefined,
                error: evt.error + (evt.detail ? `（${evt.detail}）` : ''),
              })
              get().toast('err', evt.error)
            } else if ('done' in evt && evt.done) {
              patchMsg(botMsg.id, {
                streaming: false,
                status: 'done',
                stageLabel: undefined,
                content: evt.answer || get().messages.find((m) => m.id === botMsg.id)?.content || '',
                meta: {
                  intent: evt.intent,
                  slots: evt.slots,
                  characters: evt.characters,
                  docs: evt.docs,
                  sources: evt.sources,
                  truncated: evt.truncated,
                  emotion: evt.emotion,
                },
              })
            }
          },
          s.settings.apiBase,
        )
        activeAbort = controller
        await done
        patchMsg(botMsg.id, { streaming: false })
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err)
        if (msg.includes('abort')) {
          patchMsg(botMsg.id, { streaming: false, status: 'done' })
        } else {
          patchMsg(botMsg.id, { streaming: false, status: 'error', error: msg })
          fail(err, '重新生成失败：')
        }
      } finally {
        activeAbort = null
        set({ busy: false })
      }
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

    setSettings: (patch) => {
      const next = { ...get().settings, ...patch }
      // 界面偏好按**登录用户**落盘：换个账号进来是另一套观感，不互相串
      const username = get().auth?.username
      if (username) writeJSON(LS_SETTINGS(username), next)
      set({ settings: next })
    },

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
})
