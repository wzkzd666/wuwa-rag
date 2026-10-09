import type {
  AskOut, ConversationDetail, ConversationMeta, FeedbackList, IngestOut, IngestRecordRow,
  IngestStatus, KnowledgeOut, LlmConfig,
  LlmConfigIn, LlmTestOut, ProviderPreset, StreamEvent, TtsConfigIn, TtsConfigOut, TtsOut, TtsStatus,
  UsageSummary, UserFact,
  MusicState,
} from '../types'

/**
 * API 层。默认走 Vite 代理前缀 /api（开发期转发到 127.0.0.1:8000）。
 * 生产或用户自定义时，可在设置里填写完整后端地址，此时直接用绝对地址。
 *
 * 鉴权（2026-09-22）：登录后 setAuthToken 存 token，所有请求自动带
 * `Authorization: Bearer <token>`；401 统一抛 UnauthorizedError，由调用方踢回登录页。
 */

function resolveBase(customBase?: string): string {
  if (customBase && customBase.trim()) {
    return customBase.trim().replace(/\/+$/, '')
  }
  return '/api'
}

let authToken = ''

/** 登录态变化时由 store 调用；token 为空 = 未登录 */
export function setAuthToken(token: string): void {
  authToken = token
}

/** 401 专用错误：store / 页面据此清登录态、跳登录页 */
export class UnauthorizedError extends Error {
  constructor() {
    super('登录已过期，请重新登录')
    this.name = 'UnauthorizedError'
  }
}

function authHeaders(): Record<string, string> {
  return authToken ? { Authorization: `Bearer ${authToken}` } : {}
}

async function request<T>(path: string, init?: RequestInit, base?: string): Promise<T> {
  const res = await fetch(`${resolveBase(base)}${path}`, {
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    ...init,
  })
  if (res.status === 401) throw new UnauthorizedError()
  if (!res.ok) {
    // 后端 {detail: ...} 的业务错误（注册重名 / 密码错误等）直接透出给用户
    let msg = `HTTP ${res.status} ${res.statusText}`
    try {
      const j = await res.json()
      if (j?.detail) msg = String(j.detail)
    } catch { /* 保留默认消息 */ }
    throw new Error(msg)
  }
  return (await res.json()) as T
}

// ---------- 鉴权 ----------

export interface AuthOut {
  token: string
  username: string
  role: 'admin' | 'guest'
  /** true = 该账号仍是部署者配置的初始口令，前端应引导尽快改密 */
  must_change_password?: boolean
}

export function register(username: string, password: string, base?: string): Promise<AuthOut> {
  return request('/auth/register', { method: 'POST', body: JSON.stringify({ username, password }) }, base)
}

export function login(username: string, password: string, base?: string): Promise<AuthOut> {
  return request('/auth/login', { method: 'POST', body: JSON.stringify({ username, password }) }, base)
}

/** 校验 token 是否仍有效（启动时用）；无效抛 UnauthorizedError */
export function me(base?: string): Promise<{ username: string; role: string }> {
  return request('/auth/me', { method: 'GET' }, base)
}

export function logout(base?: string): Promise<{ ok: boolean }> {
  return request('/auth/logout', { method: 'POST' }, base)
}

// ---------- 用户画像 ----------

export function getProfile(base?: string): Promise<{ username: string; facts: UserFact[] }> {
  return request('/profile', { method: 'GET' }, base)
}

export function deleteFact(factId: number, base?: string): Promise<{ ok: boolean }> {
  return request(`/profile/fact/${factId}`, { method: 'DELETE' }, base)
}

/** GET /health */
export function health(base?: string): Promise<{ ok: boolean }> {
  return request('/health', { method: 'GET' }, base)
}

/** POST /ask（非流式，一次性返回完整答案） */
export function ask(question: string, threadId: string | null, base?: string): Promise<AskOut> {
  return request('/ask', { method: 'POST', body: JSON.stringify({ question, thread_id: threadId }) }, base)
}

/** POST /ingest */
export function ingest(character: string, base?: string): Promise<IngestOut> {
  return request('/ingest', { method: 'POST', body: JSON.stringify({ character }) }, base)
}

/** GET /ingest/status?character=xxx —— 按角色查五步入库进度 */
export function ingestStatus(character: string, base?: string): Promise<IngestStatus> {
  return request(`/ingest/status?character=${encodeURIComponent(character)}`, { method: 'GET' }, base)
}

/** POST /ingest/control —— 暂停 / 继续 / 取消一条正在跑的入库链（按角色，不按 chain_id） */
export function ingestControl(
  character: string,
  action: 'pause' | 'resume' | 'cancel',
  base?: string,
): Promise<{ character: string; action: string; paused: boolean; cancelled: boolean }> {
  return request(
    '/ingest/control',
    { method: 'POST', body: JSON.stringify({ character, action }) },
    base,
  )
}

/** GET /ingest/records —— 服务端提交账本（含提交人，刷新/换设备都在） */
export function ingestRecords(
  limit: number,
  base?: string,
): Promise<{ items: IngestRecordRow[]; total: number }> {
  return request(`/ingest/records?limit=${limit}`, { method: 'GET' }, base)
}

/** DELETE /ingest/records/{id} —— 删掉一条提交记录（只删账本，不碰后台流水线） */
export function ingestRecordDelete(
  id: number,
  base?: string,
): Promise<{ id: number; character: string }> {
  return request(`/ingest/records/${id}`, { method: 'DELETE' }, base)
}

// ---------- 知识库视图 ----------

/** GET /knowledge/characters —— 知识库里实际拥有的角色（含来源、块数、更新时间） */
export function knowledgeCharacters(base?: string): Promise<KnowledgeOut> {
  return request('/knowledge/characters', { method: 'GET' }, base)
}

/** POST /knowledge/refresh —— 先清旧知识再重跑五步链（管理员） */
export function knowledgeRefresh(character: string, base?: string): Promise<IngestOut> {
  return request('/knowledge/refresh', { method: 'POST', body: JSON.stringify({ character }) }, base)
}

/** DELETE /knowledge/characters/{name} —— 删除该角色知识库（管理员，后台异步执行） */
export function knowledgeDelete(
  character: string,
  base?: string,
): Promise<{ character: string; task_id: string }> {
  return request(
    `/knowledge/characters/${encodeURIComponent(character)}`,
    { method: 'DELETE' },
    base,
  )
}

// ---------- 用户自定义云端模型（2026-09-29）----------
// 密钥体系：加密密钥由**用户自持的加密口令**派生，只活在服务端进程内存里。
// 进程重启 = 上锁 → 需重新 unlock；未解锁期间云端模型自动回落本地默认 agent。

/** PUT /llm/config 返回：保存后的概要。⚠️ 不含 unlocked 等完整状态字段，
 *  需要完整状态请重新 GET /llm/config（保存后界面就是这么刷新的）。 */
export interface LlmSaveOut {
  configured: boolean
  enabled: boolean
  provider: string
  base_url: string
  model: string
  key_hint: string
  unlocked?: boolean
}

/** POST /llm/unlock —— 解锁本次会话的密钥。两条通道任选：
 *  password（登录密码，自动解锁通道）或 passphrase（加密口令，兜底通道）。
 *  正常登录时后端已自动解锁；只有进程重启后仍持旧 token、或当初只用口令建密钥时才需手调。 */
export function unlockLlm(cred: { password?: string; passphrase?: string }, base?: string):
  Promise<{ ok: boolean; unlocked: boolean }> {
  return request('/llm/unlock', { method: 'POST', body: JSON.stringify(cred) }, base)
}

/** POST /llm/lock —— 丢弃内存里的派生密钥，之后回落本地默认 agent */
export function lockLlm(base?: string): Promise<{ ok: boolean; unlocked: boolean }> {
  return request('/llm/lock', { method: 'POST' }, base)
}

/** POST /llm/passphrase —— 更换加密口令（必须提供原口令，服务端不持有主密钥） */
export function changeLlmPassphrase(oldPp: string, newPp: string, base?: string):
  Promise<{ ok: boolean; unlocked: boolean }> {
  return request('/llm/passphrase', { method: 'POST', body: JSON.stringify({ old: oldPp, new: newPp }) }, base)
}

/** POST /auth/password —— 改登录密码。后端会先用旧密码重绑云端密钥再更新哈希，
 *  所以改完密码已存的 API Key 依然能自动解开。 */
export function changePassword(oldPw: string, newPw: string, base?: string):
  Promise<{ ok: boolean }> {
  return request('/auth/password', { method: 'POST', body: JSON.stringify({ old: oldPw, new: newPw }) }, base)
}

/** GET /llm/providers —— provider 预设列表（选完自动带出 base_url） */
export function llmProviders(base?: string): Promise<{ providers: ProviderPreset[] }> {
  return request('/llm/providers', { method: 'GET' }, base)
}

/** GET /llm/config —— 读自己的云端配置。后端只返回掩码，任何情况都拿不到明文 key */
export function getLlmConfig(base?: string): Promise<LlmConfig> {
  return request('/llm/config', { method: 'GET' }, base)
}

/** PUT /llm/config —— 保存配置（api_key 留空表示保留已存的 key）。
 *  passphrase：会话尚未解锁时必填（首次建立口令 / 之后解锁）。 */
export function saveLlmConfig(cfg: LlmConfigIn, base?: string): Promise<LlmSaveOut> {
  return request('/llm/config', { method: 'PUT', body: JSON.stringify(cfg) }, base)
}

/** POST /llm/config/test —— 连通性测试 + 拉可选模型列表（模型自选）。
 *  ⚠️ 后端不接受 query 传 key（防明文进日志/代理/浏览器历史），
 *  key 只走 body，或留空让后端用已保存的 key 测。 */
export function testLlmConfig(cfg: LlmConfigIn, base?: string): Promise<LlmTestOut> {
  return request('/llm/config/test', { method: 'POST', body: JSON.stringify(cfg) }, base)
}

/** POST /llm/enabled —— 只切换「本地默认 / 云端自定义」，凭据原样保留。
 *  这是「切回本地」唯一能落库的入口：不走保存表单（凭据不需要重填），
 *  也不需要口令/解锁。返回启用后的真实状态，`configured=false` 表示没配置过。 */
export function setLlmEnabled(
  enabled: boolean, base?: string,
): Promise<{ ok: boolean; enabled: boolean; configured: boolean; unlocked: boolean }> {
  return request('/llm/enabled', { method: 'POST', body: JSON.stringify({ enabled }) }, base)
}

/** DELETE /llm/config —— 删除配置（含密文），之后回落本地默认 agent */
export function deleteLlmConfig(base?: string): Promise<{ ok: boolean; deleted: boolean }> {
  return request('/llm/config', { method: 'DELETE' }, base)
}

// ---------- TTS 语音合成（Qwen-Audio-3.1-TTS-Flash，密钥由用户自持）----------

/** GET /tts/status —— 当前用户能否朗读、不可用原因、生效模型/音色与默认值 */
export function ttsStatus(base?: string): Promise<TtsStatus> {
  return request('/tts/status', { method: 'GET' }, base)
}

/** POST /tts —— 合成语音，返回 24h 有效的音频 URL。
 *  ⚠️ 服务未配置时后端返回 **200 + ok=false**（配置缺失不是服务故障），
 *  所以这里不能只看 HTTP 状态码，必须检查 ok 字段并把 error 提示给用户。 */
export function tts(text: string, emotion: string, base?: string): Promise<TtsOut> {
  return request('/tts', { method: 'POST', body: JSON.stringify({ text, emotion }) }, base)
}

/** GET /tts/config —— 读自己的语音凭据。后端只返回掩码，任何情况都拿不到明文 key */
export function getTtsConfig(base?: string): Promise<TtsConfigOut> {
  return request('/tts/config', { method: 'GET' }, base)
}

/** PUT /tts/config —— 保存语音凭据（api_key 加密落库）。
 *  api_key 留空表示保留已存的 key；未解锁时 password / passphrase 至少给一个。 */
export function saveTtsConfig(cfg: TtsConfigIn, base?: string): Promise<TtsConfigOut> {
  return request('/tts/config', { method: 'PUT', body: JSON.stringify(cfg) }, base)
}

/** DELETE /tts/config —— 删除自己的语音凭据（含密文），之后回落全局兜底或不可用 */
export function deleteTtsConfig(base?: string): Promise<{ ok: boolean; deleted: boolean }> {
  return request('/tts/config', { method: 'DELETE' }, base)
}

// ---------- 会话历史（服务端存，按登录用户隔离）----------
// 会话与消息的服务端真源见后端 conversations.py；前端不再把会话存 localStorage
// （那正是「换个账号登录就看到上一个人的历史」的根因）。

/** GET /conversations —— 当前用户的会话列表（不含正文，只带最后一条预览）。
 *  q 非空时按标题**或任意一条消息正文**模糊搜索（转录在服务端，只能在那搜）。 */
export function listConversations(q = '', base?: string): Promise<{ conversations: ConversationMeta[] }> {
  const qs = q.trim() ? `?q=${encodeURIComponent(q.trim())}` : ''
  return request(`/conversations${qs}`, { method: 'GET' }, base)
}

/** GET /conversations/{tid} —— 单个会话 + 全部消息。别人的会话一律 404 */
export function getConversation(threadId: string, base?: string): Promise<ConversationDetail> {
  return request(`/conversations/${encodeURIComponent(threadId)}`, { method: 'GET' }, base)
}

/** PATCH /conversations/{tid} —— 重命名 */
export function renameConversation(threadId: string, title: string, base?: string):
  Promise<{ ok: boolean }> {
  return request(
    `/conversations/${encodeURIComponent(threadId)}`,
    { method: 'PATCH', body: JSON.stringify({ title }) },
    base,
  )
}

/** DELETE /conversations/{tid} —— 删除会话（连带 LangGraph 里该 thread 的记忆） */
export function deleteConversation(threadId: string, base?: string): Promise<{ ok: boolean }> {
  return request(`/conversations/${encodeURIComponent(threadId)}`, { method: 'DELETE' }, base)
}

/** DELETE /conversations —— 清空当前用户全部会话 */
export function clearConversations(base?: string): Promise<{ ok: boolean; deleted: number }> {
  return request('/conversations', { method: 'DELETE' }, base)
}

/**
 * POST /conversations/{tid}/regenerate —— 重新生成最后一条回答（SSE）。
 *
 * 注意：这里**不传**要重答哪一条。服务端一律重答最后一句用户提问，而界面上
 * 只在最后一条助手消息上给出「重新生成」按钮，两者天然一致。
 * 服务端会先删掉「最后一句问 + 它的答」的转录、并把模型侧记忆回放成删完之后的窗口，
 * 否则模型记得自己刚被删掉的那个回答，重新生成大概率吐出同一段话。
 */
export function regenerateStream(
  threadId: string,
  onEvent: (evt: StreamEvent) => void,
  base?: string,
): { controller: AbortController; done: Promise<void> } {
  return sseStream(`/conversations/${encodeURIComponent(threadId)}/regenerate`, null, onEvent, base)
}

/**
 * 把一次 SSE 请求包成「逐事件回调 + 可中断」。
 *
 * 用 fetch + ReadableStream 手动解析（原生 EventSource 不支持 POST 与自定义 body），
 * 后端逐行发 `data: {json}\n\n`。
 */
function sseStream(
  path: string,
  body: unknown,
  onEvent: (evt: StreamEvent) => void,
  base?: string,
): { controller: AbortController; done: Promise<void> } {
  const controller = new AbortController()

  const done = (async () => {
    const res = await fetch(`${resolveBase(base)}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream', ...authHeaders() },
      body: JSON.stringify(body),
      signal: controller.signal,
    })
    if (res.status === 401) throw new UnauthorizedError()
    if (!res.ok || !res.body) {
      throw new Error(`HTTP ${res.status} ${res.statusText}`)
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''

    // SSE 以空行（\n\n）分隔事件；按事件块解析 data: 行
    while (true) {
      const { value, done: streamDone } = await reader.read()
      if (streamDone) break
      buffer += decoder.decode(value, { stream: true })

      let sep: number
      while ((sep = buffer.indexOf('\n\n')) !== -1) {
        const rawEvent = buffer.slice(0, sep)
        buffer = buffer.slice(sep + 2)

        const dataLines = rawEvent
          .split('\n')
          .filter((l) => l.startsWith('data:'))
          .map((l) => l.slice(5).replace(/^ /, ''))

        if (dataLines.length === 0) continue
        const payload = dataLines.join('\n').trim()
        if (!payload) continue

        try {
          onEvent(JSON.parse(payload) as StreamEvent)
        } catch {
          // 忽略无法解析的分片（例如心跳注释行）
        }
      }
    }
  })()

  return { controller, done }
}

/**
 * POST /ask/stream —— 流式问答。
 *
 * threadId 传 null = 「这是个还没有会话的新问题」：服务端会建好会话，并把 id 放在
 * done 事件里回传（见 StreamEvent 的 thread_id），前端据此认领。
 */
export function askStream(
  question: string,
  threadId: string | null,
  onEvent: (evt: StreamEvent) => void,
  base?: string,
): { controller: AbortController; done: Promise<void> } {
  return sseStream('/ask/stream', { question, thread_id: threadId }, onEvent, base)
}

/** GET /usage/summary —— 近 N 天 token 用量。普通用户只会拿到自己那一行（后端强制） */
export function usageSummary(days: number, user?: string, base?: string): Promise<UsageSummary> {
  const q = user ? `&user=${encodeURIComponent(user)}` : ''
  return request(`/usage/summary?days=${days}${q}`, { method: 'GET' }, base)
}

/** POST /feedback —— 对一条回答点赞/点踩（可带文字），重复提交视为改主意 */
export function feedbackSubmit(
  body: {
    thread_id: string
    target_id: string
    rating: 1 | -1
    comment?: string
    question?: string
    answer?: string
    provider?: string
    model?: string
  },
  base?: string,
): Promise<{ id: number; rating: number }> {
  return request('/feedback', { method: 'POST', body: JSON.stringify(body) }, base)
}

/** GET /feedback —— 反馈列表（普通用户只看自己的，管理员看全员） */
export function feedbackList(days: number, base?: string): Promise<FeedbackList> {
  return request(`/feedback?days=${days}`, { method: 'GET' }, base)
}

/** GET /feedback/mine —— 我点过哪些回答（避免重复弹窗） */
export function feedbackMine(
  targetIds: string[],
  base?: string,
): Promise<{ ratings: Record<string, number> }> {
  const q = targetIds.slice(0, 200).join(',')
  return request(`/feedback/mine?target_ids=${encodeURIComponent(q)}`, { method: 'GET' }, base)
}

/** DELETE /feedback/{id} —— 删掉一条反馈（管理员任意 / 普通用户仅自己的） */
export function feedbackDelete(
  id: number,
  base?: string,
): Promise<{ id: number; character: string | null }> {
  return request(`/feedback/${id}`, { method: 'DELETE' }, base)
}

/** GET /music/status —— 当前播放状态（播放条用；未启用/未运行时 available=false） */
export function musicStatus(base?: string): Promise<MusicState> {
  return request('/music/status', { method: 'GET' }, base)
}

/** POST /music/control —— 控制本机播放器。action 见 tools/qqmusic_mcp 的 player_control */
export function musicControl(action: string, base?: string): Promise<{ result: string }> {
  return request('/music/control', { method: 'POST', body: JSON.stringify({ action }) }, base)
}

/** 音乐设置（设置页用） */
export interface MusicSetting {
  /** 现在是否生效（个人设置 或 .env 任一为开） */
  enabled: boolean
  /** 个人设置里的值（null = 没设过，走默认） */
  personal: boolean | null
  /** .env 里的部署级默认 */
  deployment: boolean
  /** 当前生效的是谁：personal / deployment / default */
  source: 'personal' | 'deployment' | 'default'
  exe: string
}

export function musicSetting(base?: string): Promise<MusicSetting> {
  return request('/music/setting', { method: 'GET' }, base)
}

export function musicSettingPut(
  body: { enabled?: boolean | null; exe?: string | null; reset?: boolean },
  base?: string,
): Promise<{ enabled: boolean; exe: string }> {
  return request('/music/setting', { method: 'PUT', body: JSON.stringify(body) }, base)
}
