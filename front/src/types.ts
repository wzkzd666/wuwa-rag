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
  /** 引用来源面包屑（角色 › 模块 › 组件），已去重保序 */
  sources?: string[]
  /** 后端检测到模型复读并截断了答案 */
  truncated?: boolean
  /** 情绪标签（TTS 用）；后端 TTS 未开启时为空串 */
  emotion?: string
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
  /** 引用来源面包屑（角色 › 模块 › 组件），已去重保序 */
  sources?: string[]
  /** 后端复读兜底触发、答案被截断过 */
  truncated?: boolean
  /** 情绪标签（TTS 用）；后端 TTS 未开启时为空串 */
  emotion?: string
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
      /** 引用来源面包屑（角色 › 模块 › 组件），已去重保序 */
      sources?: string[]
      /** 后端复读兜底触发：answer 是截断后的权威全文，前端须用它覆盖已流式渲染的内容 */
      truncated?: boolean
      /** 情绪标签（TTS 用）；后端 TTS 未开启时为空串 */
      emotion?: string
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

// ---------- 鉴权 + 用户画像（2026-09-22） ----------

/** 登录态：token 存 localStorage，请求带 Authorization: Bearer */
export interface AuthInfo {
  token: string
  username: string
  role: 'admin' | 'guest'
}

/** 一条用户画像事实（user_facts 表，后端 /profile 返回） */
export interface UserFact {
  id: number
  fact: string
  confidence: number | null
  source: string | null
  created_at: string
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

/** 聊天背景预设；'custom' 表示使用用户上传的图片（bgImage） */
export type BgPreset = 'default' | 'aurora' | 'dusk' | 'cyber' | 'plain' | 'custom'

export interface Settings {
  apiBase: string
  theme: 'dark' | 'light'
  stream: boolean
  fontSize: number

  // ---------- 个性化（2026-09-22 新增） ----------
  /** 桌面端把侧边栏折叠成图标窄栏 */
  sidebarCollapsed: boolean
  /** 助手头像 dataURL；空 = 用默认图标。统一用于侧栏 logo / 欢迎页 logo / 机器人气泡头像 */
  avatarAssistant: string
  /** 我的头像 dataURL；空 = 用默认图标 */
  avatarUser: string
  /** 背景预设；'custom' 时使用 bgImage */
  bgPreset: BgPreset
  /** 自定义背景图 dataURL（仅 bgPreset === 'custom' 时生效） */
  bgImage: string
  /** 自定义背景的遮罩浓度 0~0.85，越高文字越清楚 */
  bgDim: number
  /** 自定义背景的模糊半径 0~16px */
  bgBlur: number

  // ---------- 语音（2026-09-29 新增）----------
  /** 前端语音开关：关闭则不显示播放按钮、不调 /tts（后端还有一道 TTS_ENABLED 总开关） */
  ttsEnabled: boolean
}

// ---------- 用户自定义云端模型（2026-09-29）----------

/** provider 预设（GET /llm/providers）：选完自动带出 base_url */
export interface ProviderPreset {
  key: string
  label: string
  base_url: string
}

/** GET /llm/config：⚠️ 只返回掩码，后端任何情况都不回显明文 key */
export interface LlmConfig {
  configured: boolean
  enabled: boolean
  provider: string
  base_url: string
  model: string
  /** 形如 `sk-****abcd`，仅供用户认出是哪把 key */
  key_hint: string
  updated_at?: string | null
  /** 后端未配 SECRET_KEY 时为 false：此时保存会被拒绝，界面需提示 */
  crypto_available: boolean
  /** 是否由该云端模型兼任情绪判定（false = 走本地 qwen3:8b） */
  emotion_enabled: boolean
  /** 本地默认 agent（回落时用），如 aemeath */
  default_provider?: string
  default_model?: string
}

/** PUT /llm/config 入参：api_key 留空表示保留原 key */
export interface LlmConfigIn {
  base_url: string
  model: string
  api_key: string
  provider: string
  enabled: boolean
  /** true = 用该模型兼任情绪判定（默认 false：走本地 qwen3:8b） */
  emotion_enabled: boolean
}

/** POST /llm/config/test：连通性测试 + 可选模型列表（模型自选） */
export interface LlmTestOut {
  ok: boolean
  error: string
  models: string[]
}

// ---------- TTS 语音合成（2026-09-29）----------

/** GET /tts/status */
export interface TtsStatus {
  enabled: boolean
  /** 三重开关全满足才为 true（未开启 / 缺 key / 缺 WorkspaceId 都为 false） */
  ready: boolean
  reason: string
  model: string
  voice: string
  /** 后端支持的情绪枚举，界面提示用 */
  emotions: string[]
}

/** POST /tts：未开启时 ok=false + 可读 error（200 而非 503，属预留未开而非故障） */
export interface TtsOut {
  ok: boolean
  url: string
  error: string
  emotion?: string
  model?: string
  voice?: string
  elapsed_ms?: number
}
