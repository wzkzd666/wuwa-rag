import type {
  AskOut, IngestOut, IngestStatus, LlmConfig, LlmConfigIn, LlmTestOut,
  ProviderPreset, StreamEvent, TtsOut, TtsStatus, UserFact,
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

// ---------- 用户自定义云端模型（2026-09-29）----------

/** PUT /llm/config 返回：保存后的概要。⚠️ 不含 crypto_available 等字段，
 *  需要完整状态请重新 GET /llm/config（保存后界面就是这么刷新的）。 */
export interface LlmSaveOut {
  configured: boolean
  enabled: boolean
  provider: string
  base_url: string
  model: string
  key_hint: string
}

/** GET /llm/providers —— provider 预设列表（选完自动带出 base_url） */
export function llmProviders(base?: string): Promise<{ providers: ProviderPreset[] }> {
  return request('/llm/providers', { method: 'GET' }, base)
}

/** GET /llm/config —— 读自己的云端配置。后端只返回掩码，任何情况都拿不到明文 key */
export function getLlmConfig(base?: string): Promise<LlmConfig> {
  return request('/llm/config', { method: 'GET' }, base)
}

/** PUT /llm/config —— 保存配置（api_key 留空表示保留已存的 key） */
export function saveLlmConfig(cfg: LlmConfigIn, base?: string): Promise<LlmSaveOut> {
  return request('/llm/config', { method: 'PUT', body: JSON.stringify(cfg) }, base)
}

/** POST /llm/config/test —— 连通性测试 + 拉可选模型列表（模型自选）。
 *  ⚠️ 后端不接受 query 传 key（防明文进日志/代理/浏览器历史），
 *  key 只走 body，或留空让后端用已保存的 key 测。 */
export function testLlmConfig(cfg: LlmConfigIn, base?: string): Promise<LlmTestOut> {
  return request('/llm/config/test', { method: 'POST', body: JSON.stringify(cfg) }, base)
}

/** DELETE /llm/config —— 删除配置（含密文），之后回落本地默认 agent */
export function deleteLlmConfig(base?: string): Promise<{ ok: boolean; deleted: boolean }> {
  return request('/llm/config', { method: 'DELETE' }, base)
}

// ---------- TTS 语音合成（2026-09-29，后端默认关闭）----------

/** GET /tts/status —— 三重开关是否全满足；未就绪时 reason 给可读原因 */
export function ttsStatus(base?: string): Promise<TtsStatus> {
  return request('/tts/status', { method: 'GET' }, base)
}

/** POST /tts —— 合成语音，返回 24h 有效的音频 URL。
 *  ⚠️ 未开启/配置不全时后端返回 **200 + ok=false**（预留未开不是服务故障），
 *  所以这里不能只看 HTTP 状态码，必须检查 ok 字段。 */
export function tts(text: string, emotion: string, base?: string): Promise<TtsOut> {
  return request('/tts', { method: 'POST', body: JSON.stringify({ text, emotion }) }, base)
}

/**
 * POST /ask/stream —— 服务端发送事件（SSE）。
 * 后端逐行发 `data: {json}\n\n`。这里用 fetch + ReadableStream 手动解析，
 * 因为原生 EventSource 不支持 POST 与自定义 body。
 *
 * onEvent 每收到一个事件回调一次；返回一个可用于中断的 AbortController。
 */
export function askStream(
  question: string,
  threadId: string | null,
  onEvent: (evt: StreamEvent) => void,
  base?: string,
): { controller: AbortController; done: Promise<void> } {
  const controller = new AbortController()

  const done = (async () => {
    const res = await fetch(`${resolveBase(base)}/ask/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream', ...authHeaders() },
      body: JSON.stringify({ question, thread_id: threadId }),
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
