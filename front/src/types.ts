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
  /** 情绪标签（TTS 用）；语音服务不可用（未配置密钥/业务空间）时为空串 */
  emotion?: string
}

/**
 * 一个会话的**摘要**（GET /conversations 返回）。
 *
 * 会话历史自 2026-09-30 起存在服务端 PG（见后端 conversations.py），按登录用户隔离，
 * 所以前端不再保存会话正文，只保存这份摘要 + 当前打开的那个会话的消息。
 * `thread_id` 同时也是 LangGraph 的多轮记忆键——服务端会加 `u<user_id>:` 前缀再落库，
 * 前端拿到/回传的永远是这里的短 id。
 */
export interface ConversationMeta {
  thread_id: string
  title: string
  message_count: number
  created_at: string
  updated_at: string
  /** 最后一条消息的角色（'' 表示空会话），供列表做预览文案 */
  last_role: string
  last_content: string
}

/** GET /conversations/{tid} 里的一条消息（服务端 id 是数字，本地流式期间用 uid 字符串） */
export interface StoredMessage {
  id: number
  role: 'user' | 'assistant'
  content: string
  meta: AskMeta
  created_at: string
}

/** GET /conversations/{tid} 返回体 */
export interface ConversationDetail {
  thread_id: string
  title: string
  created_at: string
  messages: StoredMessage[]
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
  /** 情绪标签（TTS 用）；语音服务不可用（未配置密钥/业务空间）时为空串 */
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
      /** 情绪标签（TTS 用）；语音服务不可用（未配置密钥/业务空间）时为空串 */
      emotion?: string
      /**
       * 本轮所属的会话 id。
       *
       * 会话现在由**服务端**创建：草稿态（还没绑定会话）提问时前端传 thread_id=null，
       * 后端建好会话并把 id 放在 done 事件里回传，前端靠它认领——所以这个字段必须有，
       * 否则「新建对话」后的第一句问答，前端永远不知道自己属于哪个会话。
       */
      thread_id?: string
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

/** GET /knowledge/characters 的单条：知识库里**实际拥有**的一个角色 */
export interface KnowledgeCharacter {
  character: string
  /** 来源标识（当前恒为 kurobbs，即鸣潮 WIKI） */
  source: string
  title: string | null
  /** RustFS 原文对象指针 —— 用来核对「这份知识是从哪份原文来的」 */
  raw_uri: string | null
  raw_size: number | null
  created_at: string | null
  updated_at: string | null
  /** 实际块数；为 0 说明入了库但索引没跑成，是有用的健康信号 */
  chunks: number
  /** 是否属于内置种子名册。false = 靠自动爬取发现并入库的新角色 */
  seeded: boolean
}

/** GET /knowledge/characters 返回体 */
export interface KnowledgeOut {
  items: KnowledgeCharacter[]
  total: number
  /**
   * 种子名册里**尚未入库**的角色。
   *
   * 由后端下发而不是前端硬编码：候选名册只允许有一个来源，否则加新角色时
   * 前后端两份清单必然不同步（这正是改造前那份 `ROSTER` 常量的毛病）。
   */
  seeded_only: string[]
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
  /** `cancelled` 是本项目加的第四种终态：单看 steps 分不出「失败」与「被取消」 */
  status: 'pending' | 'running' | 'success' | 'failed' | 'cancelled'
  found: boolean
  /** worker 侧 Redis 控制旗标：暂停期间那一步保持 pending，靠这两个布尔告诉前端 */
  paused: boolean
  cancelled: boolean
  steps: IngestStep[]
  updated_at: number | null
}

/**
 * GET /ingest/records 的一条：**服务端**的抓取/入库提交账本（表 crawl_runs）。
 * 与 `IngestRecord` 的区别：那个是浏览器会话内的临时列表（驱动轮询与置顶条），
 * 这个刷新/换设备都在，并且带**提交人**。
 */
export interface IngestRecordRow {
  id: number
  character: string
  submitted_by: string
  submitted_by_name: string
  chain_id: string | null
  /** 库里那行的状态（running / success / failed） */
  state: string
  error: string | null
  created_at: string | null
  finished_at: string | null
  /** 叠加 Redis 实时进度算出的活状态：unknown / running / success / failed */
  status: 'unknown' | 'running' | 'success' | 'failed'
  /**
   * 当前登录者能否暂停/取消/删除这条：admin 全部可以；普通用户只对**自己提交的**为 true。
   * 由服务端判定并逐行下发 —— 前端拿不到 user.id，让它自己猜必然会出现「按钮能点但 403」。
   */
  can_control: boolean
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
  /** 面板（卡片 / 页面标题块）底色不透明度 0.3~1，越低越透 */
  panelAlpha: number
  /** 面板磨砂模糊半径 0~24px */
  panelBlur: number

  // ---------- 语音朗读（Qwen-Audio-3.1-TTS-Flash）----------
  /** 前端语音开关：关闭则不显示朗读按钮、不调 /tts（后端另有一道 TTS_ENABLED 总开关） */
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
  /** 后端加密库（cryptography）是否可用；false 时保存会被拒绝，界面需提示 */
  crypto_ready: boolean
  /** 本会话是否已解锁。false = 云端模型不生效，自动回落本地默认 agent */
  unlocked: boolean
  /** 是否已绑定登录密码通道：true = 下次登录自动解密，无需任何额外输入 */
  auto_unlock: boolean
  /** 是否已绑定加密口令通道（独立于登录密码的兜底通道） */
  pp_bound: boolean
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
  /** 登录密码：未解锁时用它解锁；已解锁时传它会补建「登录自动解锁」通道 */
  password: string
  /** 加密口令（兜底通道）：首次保存时可一并设置，服务端不保存 */
  passphrase: string
}

/** POST /llm/config/test：连通性测试 + 可选模型列表（模型自选） */
export interface LlmTestOut {
  ok: boolean
  error: string
  models: string[]
}

// ---------- TTS 语音合成（2026-09-30：密钥改为用户自持）----------

/** 生效来源：user=用户自持凭据 / global=部署者 .env 兜底 / none=都没配 */
export type TtsSource = "user" | "global" | "none"

/** GET /tts/status */
export interface TtsStatus {
  enabled: boolean
  /** 当前用户此刻能否合成（凭据齐全 + 已解锁 + 未被停用） */
  ready: boolean
  /** 未就绪时的可读原因（直接展示给用户，含「缺什么 / 该做什么」） */
  reason: string
  /** 生效来源 */
  source: TtsSource
  /** 当前**实际生效**的模型 id */
  model: string
  /** 模型中文展示名，如 Qwen-Audio-3.1-TTS-Flash */
  model_label: string
  /** 当前**实际生效**的音色 id（内部值，如 longanlingxi_v3.1） */
  voice: string
  /** 音色中文展示名，如「龙安灵希 · 可爱甜美（社交陪伴）」 */
  voice_label: string
  /** 后端支持的情绪枚举，界面提示用 */
  emotions: string[]
  /** 该用户是否已保存过自己的凭据 */
  configured: boolean
  /** 本会话是否已解锁（进程重启后需重新登录/输入口令） */
  unlocked: boolean
  /** 可选音色（3.1 音色与模型强绑定，填错版本会 400） */
  voices: { id: string; label: string }[]
  /** 代码默认值：输入框用 placeholder 提示「留空则用默认」 */
  defaults: {
    model: string
    voice: string
    voice_label: string
    instruction: string
  }
}

/** GET /tts/config：用户自持凭据（**只回掩码**，任何情况都不回显明文 key） */
export interface TtsConfigOut {
  configured: boolean
  /** 掩码 key，如 sk-****wxyz */
  key_hint: string
  workspace_id: string
  model: string
  voice: string
  instruction: string
  crypto_ready: boolean
  unlocked: boolean
  auto_unlock: boolean
  pp_bound: boolean
}

/** PUT /tts/config 入参：api_key 留空表示保留原 key。
 *  其余字段是**整体替换**语义，留空即回落代码默认值。 */
export interface TtsConfigIn {
  api_key: string
  workspace_id: string
  model: string
  voice: string
  /** 指令控制（音色性格/语速基调），留空则用默认值 */
  instruction: string
  /** 登录密码：未解锁时用它解锁；已解锁时传它会补建「登录自动解锁」通道 */
  password: string
  /** 加密口令（兜底通道）：首次保存时可一并设置，服务端不保存 */
  passphrase: string
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

/** GET /usage/summary 的一行：某个用户近 N 天的用量（local / cloud 分开） */
export interface UsageRow {
  user_id: string | null
  username: string | null
  local_prompt: number | null
  local_completion: number | null
  cloud_prompt: number | null
  cloud_completion: number | null
  calls: number
  last_at: string | null
}

/** GET /usage/summary 里的按天序列（画图用；缺失的日期后端已补零） */
export interface DailyUsage {
  d: string
  local_prompt: number | null
  local_completion: number | null
  cloud_prompt: number | null
  cloud_completion: number | null
  calls: number
}

export interface UsageSummary {
  days: number
  rows: UsageRow[]
  daily: DailyUsage[]
  /** all = 管理员看全员；self = 普通用户只看自己（后端强制） */
  scope: 'all' | 'self'
  feedback?: { up: number; down: number; total: number; down_ratio: number | null }
}

/** GET /feedback 的一条 */
export interface FeedbackItem {
  id: number
  user_id: string
  username: string
  thread_id: string
  target_id: string
  rating: 1 | -1
  comment: string | null
  question: string | null
  answer: string | null
  provider: string | null
  model: string | null
  created_at: string | null
  /** 管理员可删任意；普通用户仅自己的（后端逐条判定） */
  can_delete: boolean
}

export interface FeedbackList {
  days: number
  items: FeedbackItem[]
  summary: { up: number; down: number; total: number; down_ratio: number | null }
}
