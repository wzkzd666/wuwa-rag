import type { AskOut, IngestOut, IngestStatus, StreamEvent } from '../types'

/**
 * API 层。默认走 Vite 代理前缀 /api（开发期转发到 127.0.0.1:8000）。
 * 生产或用户自定义时，可在设置里填写完整后端地址，此时直接用绝对地址。
 */

function resolveBase(customBase?: string): string {
  if (customBase && customBase.trim()) {
    return customBase.trim().replace(/\/+$/, '')
  }
  return '/api'
}

async function request<T>(path: string, init?: RequestInit, base?: string): Promise<T> {
  const res = await fetch(`${resolveBase(base)}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} ${res.statusText}`)
  }
  return (await res.json()) as T
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
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify({ question, thread_id: threadId }),
      signal: controller.signal,
    })
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
