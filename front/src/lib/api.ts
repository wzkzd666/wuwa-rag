import type { AskOut, IngestOut, IngestStatus, StreamEvent, UserFact } from '../types'

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
