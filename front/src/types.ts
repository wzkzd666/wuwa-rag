// 后端接口的数据类型定义（对齐 FastAPI 的 AskIn/AskOut/IngestIn 与 SSE 事件）

/** 单条消息 */
export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  createdAt: number
  /** 仅 assistant：流式进行中 */
  streaming?: boolean
  /** 仅 assistant：后端返回的检索状态标记 */
  status?: 'retrieving' | 'done' | 'error'
  /** 仅 assistant：当前阶段的用户可读文案（抽取/检索/生成），流式期间更新 */
  stageLabel?: string
  /** 仅 assistant：问答元数据 */
  meta?: AskMeta
  /** 仅 assistant：出错信息 */
  error?: string
}

/** /ask 与 SSE done 事件里的元数据 */
export interface AskMeta {
  intent?: string
  slots?: string[]
  characters?: string[]
  docs?: number
  /** 后端检测到模型复读并截断了答案 */
  truncated?: boolean
}

/** 一个会话 */
export interface Conversation {
  id: string
  /** 与后端对应的 thread_id */
  threadId: string
  title: string
  messages: Message[]
  createdAt: number
  updatedAt: number
}

/** /ask 非流式返回体 */
export interface AskOut {
  answer: string
  thread_id: string
  intent: string
  slots: string[]
  characters: string[]
  docs: number
  /** 后端复读兜底触发、答案被截断过 */
  truncated?: boolean
}

/** /ingest 返回体 */
export interface IngestOut {
  character: string
  chain_id: string
  state: string
}

/** SSE 流事件（后端 ask_stream 逐条 yield 的 JSON） */
export type StreamEvent =
  | { token: string }
  | { status: 'retrieving' }
  /** 细粒度阶段：抽取/检索/生成，用于在无输出期间告知用户在做什么 */
  | { stage: string; label: string }
  /** 后端流中途异常：连接不会裸断了，错误以事件下发 */
  | { error: string; detail?: string }
  | {
      done: true
      answer: string
      intent: string
      slots: string[]
      characters: string[]
      docs: number
      /** 后端复读兜底触发：answer 是截断后的权威全文，前端须用它覆盖已流式渲染的内容 */
      truncated?: boolean
    }

/** 一次 ingest 提交记录 */
export interface IngestRecord {
  id: string
  character: string
  chainId: string
  state: string
  createdAt: number
  ok: boolean
  error?: string
}

/** GET /ingest/status 返回：五步流水线实时进度 */
export interface IngestStep {
  key: string
  label: string
  status: 'pending' | 'running' | 'success' | 'failed'
  error: string | null
}

export interface IngestStatus {
  character: string
  status: 'pending' | 'running' | 'success' | 'failed'
  found: boolean
  steps: IngestStep[]
  updated_at: number | null
}

export interface Settings {
  apiBase: string
  theme: 'dark' | 'light'
  stream: boolean
  fontSize: number
}
