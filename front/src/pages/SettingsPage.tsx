import { useEffect, useState } from 'react'
import {
  Settings as SettingsIcon, Sun, Moon, Zap, Type, Server, Trash2, Activity, RefreshCw,
  Camera, Image as ImageIcon, Palette, RotateCcw, User as UserIcon, Sparkles, Eraser,
  Cpu, Volume2, AlertTriangle, Check, Smile,
} from 'lucide-react'
import { useStore } from '../store/useStore'
import * as api from '../lib/api'
import { pickImageFile, fileToDataUrl, formatBytes, dataUrlBytes } from '../lib/image'
import type { BgPreset, LlmConfig, ProviderPreset, UserFact } from '../types'
import './SettingsPage.css'

/** 背景预设（key 对齐 types.ts 的 BgPreset 与 global.css 的 data-preset） */
const BG_PRESETS: { key: BgPreset; label: string; css: string }[] = [
  { key: 'default', label: '默认', css: 'linear-gradient(135deg, #0b1020, #17213a)' },
  { key: 'aurora', label: '极光', css: 'linear-gradient(160deg, #0a1024, #16224a 42%, #2c1f52)' },
  { key: 'dusk', label: '暮紫', css: 'linear-gradient(160deg, #1a1030, #3a1d4e 46%, #6b2d55)' },
  { key: 'cyber', label: '深青', css: 'linear-gradient(160deg, #04121c, #0b2b3a 46%, #123d4a)' },
  { key: 'plain', label: '纯色', css: '#070b16' },
  { key: 'custom', label: '自定义图片', css: '' },
]

/**
 * 模型服务：① 默认用项目自带 agent（本地 aemeath）；② 可切换到用户自己的云端大模型。
 *
 * 隐私：api_key 输入后只在「保存」时 POST 给后端加密落库，读回永远是掩码（key_hint），
 * 前端不留存明文、不写 localStorage。已配置时输入框留空 = 保留原 key。
 * ⚠️ crypto_available=false（服务端没配 SECRET_KEY）时禁止保存，明确提示——
 *    否则用户以为存好了，实际后端会拒绝。
 */
function LlmSection() {
  const toast = useStore((s) => s.toast)
  const apiBase = useStore((s) => s.settings.apiBase)
  const [providers, setProviders] = useState<ProviderPreset[]>([])
  const [cfg, setCfg] = useState<LlmConfig | null>(null)
  const [provider, setProvider] = useState('openai')
  const [baseUrl, setBaseUrl] = useState('')
  const [model, setModel] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [models, setModels] = useState<string[]>([])   // 测试后拉回的可选模型
  const [testing, setTesting] = useState(false)
  const [saving, setSaving] = useState(false)
  const [useCloud, setUseCloud] = useState(false)
  // 情绪判定模型：默认本地 qwen3:8b；打开则交由上面的云端自定义模型兼任
  const [emotionViaCloud, setEmotionViaCloud] = useState(false)

  const load = async () => {
    try {
      const [p, c] = await Promise.all([api.llmProviders(apiBase), api.getLlmConfig(apiBase)])
      setProviders(p.providers)
      setCfg(c)
      if (c.configured) {
        setProvider(c.provider || 'openai')
        setBaseUrl(c.base_url)
        setModel(c.model)
        setUseCloud(c.enabled)
        setEmotionViaCloud(!!c.emotion_enabled)
      }
    } catch {
      /* 未登录/后端未起：静默，不打扰 */
    }
  }
  useEffect(() => { void load() }, [])   // eslint-disable-line react-hooks/exhaustive-deps

  const pickProvider = (key: string) => {
    setProvider(key)
    const hit = providers.find((x) => x.key === key)
    if (hit && hit.base_url) setBaseUrl(hit.base_url)   // 选预设自动带出地址
  }

  const doTest = async () => {
    if (!baseUrl.trim()) { toast('err', '请先填 API 地址'); return }
    setTesting(true)
    try {
      const out = await api.testLlmConfig(
        { base_url: baseUrl.trim(), model: model.trim(), api_key: apiKey,
          provider, enabled: true, emotion_enabled: emotionViaCloud },
        apiBase,
      )
      if (out.ok) {
        setModels(out.models)
        toast('ok', `连接成功，拉到 ${out.models.length} 个可选模型`)
      } else {
        toast('err', out.error || '连接失败')
      }
    } catch (err) {
      toast('err', err instanceof Error ? err.message : String(err))
    } finally {
      setTesting(false)
    }
  }

  const doSave = async () => {
    if (cfg && !cfg.crypto_available) { toast('err', '服务端未配置 SECRET_KEY，无法安全保存密钥'); return }
    if (!baseUrl.trim() || !model.trim()) { toast('err', 'API 地址与模型都不能为空'); return }
    if (!apiKey.trim() && !(cfg?.configured)) { toast('err', '请填写 API Key'); return }
    setSaving(true)
    try {
      await api.saveLlmConfig(
        { base_url: baseUrl.trim(), model: model.trim(), api_key: apiKey.trim(),
          provider, enabled: useCloud, emotion_enabled: emotionViaCloud },
        apiBase,
      )
      setApiKey('')             // 保存后清空明文输入
      toast('ok', useCloud ? '已保存并启用云端模型' : '已保存（当前仍用本地默认模型）')
      await load()
    } catch (err) {
      toast('err', err instanceof Error ? err.message : String(err))
    } finally {
      setSaving(false)
    }
  }

  const doDelete = async () => {
    try {
      await api.deleteLlmConfig(apiBase)
      setApiKey(''); setModels([]); setUseCloud(false); setEmotionViaCloud(false)
      toast('info', '已删除云端配置，回到本地默认模型')
      await load()
    } catch (err) {
      toast('err', err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <section className="card set-card">
      <h3 className="set-h"><Cpu size={15} /> 模型服务</h3>
      <div className="set-row">
        <div className="set-row-main">
          <label>作答模型</label>
          <p>
            默认用项目自带的本地 agent（{cfg?.default_model || 'aemeath'}，人设内置）。
            也可切换到你自己的云端大模型 API —— 配置只属于你，密钥加密存储、绝不回显。
          </p>
        </div>
        <div className="set-row-ctl">
          <div className="seg">
            <button className={`seg-btn ${!useCloud ? 'seg-on' : ''}`} onClick={() => setUseCloud(false)}>
              <Server size={13} /> 本地默认
            </button>
            <button className={`seg-btn ${useCloud ? 'seg-on' : ''}`} onClick={() => setUseCloud(true)}>
              <Cpu size={13} /> 云端自定义
            </button>
          </div>
        </div>
      </div>

      {useCloud && (
        <>
          {cfg && !cfg.crypto_available && (
            <div className="set-warn">
              <AlertTriangle size={14} /> 服务端未配置 <code>SECRET_KEY</code>，云模型功能已关闭（不会明文存密钥）。请在后端 .env 设置后再用。
            </div>
          )}
          <div className="set-row">
            <div className="set-row-main">
              <label>服务商</label>
              <p>选一个预设自动带出 API 地址，也可选「自定义」手填任意 OpenAI 兼容服务。</p>
            </div>
            <div className="set-row-ctl">
              <select className="input api-input" value={provider} onChange={(e) => pickProvider(e.target.value)}>
                {providers.map((p) => <option key={p.key} value={p.key}>{p.label}</option>)}
              </select>
            </div>
          </div>
          <div className="set-row">
            <div className="set-row-main">
              <label>API 地址</label>
              <p>到 <code>/v1</code> 为止，例如 <code>https://api.deepseek.com/v1</code>。</p>
            </div>
            <div className="set-row-ctl">
              <input className="input api-input" value={baseUrl} placeholder="https://…/v1"
                     onChange={(e) => setBaseUrl(e.target.value)} />
            </div>
          </div>
          <div className="set-row">
            <div className="set-row-main">
              <label>API Key</label>
              <p>
                {cfg?.configured
                  ? <>已保存（<code>{cfg.key_hint}</code>）。留空 = 保留原 Key 不改。</>
                  : '仅在保存时加密上传，前端不留存、不回显明文。'}
              </p>
            </div>
            <div className="set-row-ctl">
              <input className="input api-input" type="password" value={apiKey}
                     placeholder={cfg?.configured ? '留空保留原 Key' : 'sk-…'}
                     onChange={(e) => setApiKey(e.target.value)} autoComplete="off" />
            </div>
          </div>
          <div className="set-row">
            <div className="set-row-main">
              <label>模型</label>
              <p>先点「测试连接」拉取可选模型，再从下拉选；也可直接手填模型 id。</p>
            </div>
            <div className="set-row-ctl model-ctl">
              <input className="input api-input" list="llm-models" value={model}
                     placeholder="如 deepseek-chat" onChange={(e) => setModel(e.target.value)} />
              <datalist id="llm-models">
                {models.map((m) => <option key={m} value={m} />)}
              </datalist>
              <button className="btn btn-ghost" onClick={doTest} disabled={testing}>
                {testing ? <RefreshCw size={14} className="spin" /> : <Zap size={14} />} 测试连接
              </button>
            </div>
          </div>
          <div className="set-row">
            <div className="set-row-main">
              <label><Smile size={13} /> 情绪判定</label>
              <p>
                朗读前要判定这段话的语气。默认用本地 qwen3:8b（不额外消耗你的额度）；
                打开则由上面的云端模型一并负责。
              </p>
            </div>
            <div className="set-row-ctl">
              <button
                className={`switch ${emotionViaCloud ? 'switch-on' : ''}`}
                onClick={() => setEmotionViaCloud(!emotionViaCloud)}
                role="switch"
                aria-checked={emotionViaCloud}
              >
                <span className="switch-knob" />
              </button>
            </div>
          </div>
          <div className="set-row">
            <div className="set-row-main">
              <label>&nbsp;</label>
              <p>保存后立即生效；切回「本地默认」或删除配置即回落自带 agent。</p>
            </div>
            <div className="set-row-ctl status-ctl">
              <button className="btn btn-primary" onClick={doSave} disabled={saving || (cfg ? !cfg.crypto_available : false)}>
                {saving ? <RefreshCw size={14} className="spin" /> : <Check size={14} />} 保存
              </button>
              {cfg?.configured && (
                <button className="btn btn-danger" onClick={doDelete}>
                  <Trash2 size={14} /> 删除配置
                </button>
              )}
            </div>
          </div>
        </>
      )}
    </section>
  )
}

export default function SettingsPage() {
  const settings = useStore((s) => s.settings)
  const setSettings = useStore((s) => s.setSettings)
  const health = useStore((s) => s.health)
  const checkHealth = useStore((s) => s.checkHealth)
  const clearAll = useStore((s) => s.clearAll)
  const conversations = useStore((s) => s.conversations)
  const ingests = useStore((s) => s.ingests)
  const toast = useStore((s) => s.toast)

  const [testing, setTesting] = useState(false)
  const [armed, setArmed] = useState(false)
  // 语音合成后端可用性：null = 未拉到（后端未起/未登录），界面不显示状态提示
  const [ttsReady, setTtsReady] = useState<boolean | null>(null)
  const [ttsVoice, setTtsVoice] = useState('')
  const [ttsReason, setTtsReason] = useState('')
  // 我的画像（user_facts 表，后端 /profile）
  const auth = useStore((s) => s.auth)
  const apiBase = useStore((s) => s.settings.apiBase)
  const [facts, setFacts] = useState<UserFact[] | null>(null)

  // 拉一次后端语音可用性（三重开关：TTS_ENABLED / key / WorkspaceId）
  useEffect(() => {
    let alive = true
    void (async () => {
      try {
        const s = await api.ttsStatus(apiBase)
        if (!alive) return
        setTtsReady(s.ready)
        setTtsVoice(s.voice)
        setTtsReason(s.reason)
      } catch {
        if (alive) setTtsReady(null)   // 未登录或后端未起：静默，不打扰
      }
    })()
    return () => { alive = false }
  }, [apiBase])

  const loadFacts = async () => {
    try {
      const out = await api.getProfile(apiBase)
      setFacts(out.facts)
    } catch {
      setFacts([])
    }
  }

  /** 选图并压缩成 dataURL（头像 256px 方图 / 背景长边 1600px），写进对应设置项 */
  const uploadImage = async (
    key: 'avatarAssistant' | 'avatarUser' | 'bgImage',
    opts: { max: number; square: boolean; quality: number },
    okText: string,
  ) => {
    const file = await pickImageFile()
    if (!file) return
    try {
      const dataUrl = await fileToDataUrl(file, { max: opts.max, square: opts.square, quality: opts.quality })
      if (key === 'bgImage') {
        // 背景图选中即切到「自定义图片」预设，所见即所得
        setSettings({ bgImage: dataUrl, bgPreset: 'custom' })
      } else {
        setSettings(key === 'avatarAssistant' ? { avatarAssistant: dataUrl } : { avatarUser: dataUrl })
      }
      toast('ok', `${okText}（${formatBytes(dataUrlBytes(dataUrl))}）`)
    } catch (err) {
      toast('err', '图片处理失败：' + (err instanceof Error ? err.message : String(err)))
    }
  }

  const testConnection = async () => {
    setTesting(true)
    await checkHealth()
    setTesting(false)
    const ok = useStore.getState().health === 'ok'
    toast(ok ? 'ok' : 'err', ok ? '后端连接正常' : '无法连接后端，请检查服务与地址')
  }

  const totalMessages = conversations.reduce((n, c) => n + c.messages.length, 0)

  return (
    <div className="page settings-page">
      <div className="page-head">
        <div>
          <h2 className="page-title">
            <SettingsIcon size={19} className="grad-text" /> 设置
          </h2>
          <p className="page-desc">连接后端、外观与数据管理。设置保存在浏览器本地。</p>
        </div>
      </div>

      {/* 后端连接 */}
      <section className="card set-card">
        <h3 className="set-h">
          <Server size={15} /> 后端连接
        </h3>
        <div className="set-row">
          <div className="set-row-main">
            <label>API 地址</label>
            <p>留空则走开发代理 /api（转发到 127.0.0.1:8000）。跨机访问时填完整地址，如 http://192.168.1.10:8000，此时后端需允许 CORS。</p>
          </div>
          <div className="set-row-ctl">
            <input
              className="input api-input"
              value={settings.apiBase}
              placeholder="默认：/api（开发代理）"
              onChange={(e) => setSettings({ apiBase: e.target.value })}
            />
          </div>
        </div>
        <div className="set-row">
          <div className="set-row-main">
            <label>连接状态</label>
            <p>每 30 秒自动探测 GET /health。</p>
          </div>
          <div className="set-row-ctl status-ctl">
            <span className={`tag ${health === 'ok' ? 'tag-ok' : health === 'down' ? 'tag-err' : 'tag-warn'}`}>
              <Activity size={12} />
              {health === 'ok' ? '在线' : health === 'down' ? '离线' : '检测中'}
            </span>
            <button className="btn btn-ghost" onClick={testConnection} disabled={testing}>
              {testing ? <RefreshCw size={14} className="spin" /> : <Zap size={14} />} 测试连接
            </button>
          </div>
        </div>
        <div className="set-row">
          <div className="set-row-main">
            <label>流式输出</label>
            <p>开启走 SSE 逐字吐答案（/ask/stream）；关闭则一次性返回（/ask）。</p>
          </div>
          <div className="set-row-ctl">
            <button
              className={`switch ${settings.stream ? 'switch-on' : ''}`}
              onClick={() => setSettings({ stream: !settings.stream })}
              role="switch"
              aria-checked={settings.stream}
            >
              <span className="switch-knob" />
            </button>
          </div>
        </div>
        <div className="set-row">
          <div className="set-row-main">
            <label><Volume2 size={13} /> 语音朗读</label>
            <p>
              开启后每条回答下方出现朗读按钮，用爱弥斯的语气把答案念出来（带情绪标签）。
              {ttsReady === null ? '' : ttsReady
                ? ` 当前音色：${ttsVoice || '默认'}。`
                : ` 后端未就绪：${ttsReason}`}
            </p>
          </div>
          <div className="set-row-ctl">
            <button
              className={`switch ${settings.ttsEnabled ? 'switch-on' : ''}`}
              onClick={() => setSettings({ ttsEnabled: !settings.ttsEnabled })}
              role="switch"
              aria-checked={settings.ttsEnabled}
              title={ttsReady === false ? '后端 TTS 未开启，按钮点了会提示' : undefined}
            >
              <span className="switch-knob" />
            </button>
          </div>
        </div>
      </section>

      {/* 模型服务：默认本地 agent / 用户自定义云端 API（2026-09-29） */}
      <LlmSection />

      {/* 外观 */}
      <section className="card set-card">
        <h3 className="set-h">
          <Sun size={15} /> 外观
        </h3>
        <div className="set-row">
          <div className="set-row-main">
            <label>主题</label>
            <p>深空暗色（默认）或浅色模式。</p>
          </div>
          <div className="set-row-ctl">
            <div className="seg">
              <button
                className={`seg-btn ${settings.theme === 'dark' ? 'seg-on' : ''}`}
                onClick={() => setSettings({ theme: 'dark' })}
              >
                <Moon size={13} /> 暗色
              </button>
              <button
                className={`seg-btn ${settings.theme === 'light' ? 'seg-on' : ''}`}
                onClick={() => setSettings({ theme: 'light' })}
              >
                <Sun size={13} /> 浅色
              </button>
            </div>
          </div>
        </div>
        <div className="set-row">
          <div className="set-row-main">
            <label>基础字号</label>
            <p>当前 {settings.fontSize}px，影响全局文字大小。</p>
          </div>
          <div className="set-row-ctl font-ctl">
            <input
              type="range"
              min={12}
              max={18}
              step={1}
              value={settings.fontSize}
              onChange={(e) => setSettings({ fontSize: Number(e.target.value) })}
            />
            <span className="font-val">
              <Type size={13} /> {settings.fontSize}px
            </span>
          </div>
        </div>
      </section>

      {/* 个性化：头像 + 背景 */}
      <section className="card set-card">
        <h3 className="set-h">
          <Palette size={15} /> 个性化
        </h3>

        {/* 助手头像 */}
        <div className="set-row">
          <div className="set-row-main">
            <label>助手头像</label>
            <p>用于侧栏 logo、欢迎页与 AI 气泡头像。点左侧栏的 logo 也能直接换。</p>
          </div>
          <div className="set-row-ctl">
            <span className={`avatar-prev ${settings.avatarAssistant ? '' : 'avatar-prev-empty'}`}>
              {settings.avatarAssistant ? (
                <img src={settings.avatarAssistant} alt="助手头像" />
              ) : (
                <Camera size={18} />
              )}
            </span>
            <button
              className="btn btn-ghost"
              onClick={() => uploadImage('avatarAssistant', { max: 256, square: true, quality: 0.92 }, '助手头像已更新')}
            >
              <Camera size={14} /> 更换
            </button>
            {settings.avatarAssistant && (
              <button
                className="btn btn-ghost"
                title="恢复默认图标"
                onClick={() => {
                  setSettings({ avatarAssistant: '' })
                  toast('info', '助手头像已恢复默认')
                }}
              >
                <RotateCcw size={14} />
              </button>
            )}
          </div>
        </div>

        {/* 我的头像 */}
        <div className="set-row">
          <div className="set-row-main">
            <label>我的头像</label>
            <p>用于你发送消息的气泡头像。</p>
          </div>
          <div className="set-row-ctl">
            <span className={`avatar-prev avatar-prev-user ${settings.avatarUser ? '' : 'avatar-prev-empty'}`}>
              {settings.avatarUser ? (
                <img src={settings.avatarUser} alt="我的头像" />
              ) : (
                <UserIcon size={18} />
              )}
            </span>
            <button
              className="btn btn-ghost"
              onClick={() => uploadImage('avatarUser', { max: 256, square: true, quality: 0.92 }, '头像已更新')}
            >
              <Camera size={14} /> 更换
            </button>
            {settings.avatarUser && (
              <button
                className="btn btn-ghost"
                title="恢复默认图标"
                onClick={() => {
                  setSettings({ avatarUser: '' })
                  toast('info', '头像已恢复默认')
                }}
              >
                <RotateCcw size={14} />
              </button>
            )}
          </div>
        </div>

        {/* 聊天背景 */}
        <div className="set-row bg-row">
          <div className="set-row-main">
            <label>聊天背景</label>
            <p>预设渐变或自定义图片。自定义图片存在浏览器本地（自动压缩到长边 1600px）。</p>
          </div>
          <div className="set-row-ctl bg-swatches">
            {BG_PRESETS.map(({ key, label, css }) => (
              <button
                key={key}
                className={`bg-swatch ${settings.bgPreset === key ? 'bg-swatch-on' : ''}`}
                title={label}
                onClick={() => setSettings({ bgPreset: key })}
              >
                <span
                  className="bg-swatch-fill"
                  style={key === 'custom'
                    ? (settings.bgImage ? { backgroundImage: `url("${settings.bgImage}")` } : undefined)
                    : { backgroundImage: css }}
                >
                  {key === 'custom' && !settings.bgImage && <ImageIcon size={13} />}
                </span>
                <span className="bg-swatch-label">{label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* 自定义图片的专属控制：上传 / 遮罩 / 模糊 */}
        {settings.bgPreset === 'custom' && (
          <>
            <div className="set-row">
              <div className="set-row-main">
                <label>背景图片</label>
                <p>{settings.bgImage ? `当前图片 ${formatBytes(dataUrlBytes(settings.bgImage))}。` : '还没选图片。'}</p>
              </div>
              <div className="set-row-ctl">
                <button
                  className="btn btn-ghost"
                  onClick={() => uploadImage('bgImage', { max: 1600, square: false, quality: 0.82 }, '背景已更新')}
                >
                  <ImageIcon size={14} /> {settings.bgImage ? '更换图片' : '上传图片'}
                </button>
                {settings.bgImage && (
                  <button
                    className="btn btn-ghost"
                    title='清除图片（回到「默认」背景）'
                    onClick={() => {
                      setSettings({ bgImage: '', bgPreset: 'default' })
                      toast('info', '已恢复默认背景')
                    }}
                  >
                    <Trash2 size={14} />
                  </button>
                )}
              </div>
            </div>
            <div className="set-row">
              <div className="set-row-main">
                <label>遮罩浓度</label>
                <p>在图片上叠一层暗色（浅色主题为亮色），越高文字越清楚。当前 {(settings.bgDim * 100).toFixed(0)}%。</p>
              </div>
              <div className="set-row-ctl font-ctl">
                <input
                  type="range"
                  min={0}
                  max={0.85}
                  step={0.05}
                  value={settings.bgDim}
                  onChange={(e) => setSettings({ bgDim: Number(e.target.value) })}
                />
                <span className="font-val">{(settings.bgDim * 100).toFixed(0)}%</span>
              </div>
            </div>
            <div className="set-row">
              <div className="set-row-main">
                <label>背景模糊</label>
                <p>模糊半径，0 为不模糊。当前 {settings.bgBlur}px。</p>
              </div>
              <div className="set-row-ctl font-ctl">
                <input
                  type="range"
                  min={0}
                  max={16}
                  step={1}
                  value={settings.bgBlur}
                  onChange={(e) => setSettings({ bgBlur: Number(e.target.value) })}
                />
                <span className="font-val">{settings.bgBlur}px</span>
              </div>
            </div>
          </>
        )}
      </section>

      {/* 我的画像（user_facts 表） */}
      <section className="card set-card" ref={(el) => {
        // 挂载后拉一次画像；简单起见不搞轮询——设置页本来就不是常驻页面
        if (el && facts === null) loadFacts()
      }}>
        <h3 className="set-h">
          <Sparkles size={15} /> 我的画像
        </h3>
        <p className="profile-desc">
          系统会在你提问后自动记住稳定偏好（主玩角色、熟悉程度等），下次回答时贴合你的情况。
          {auth?.role === 'admin' ? '管理员也拥有收录知识库的权限。' : ''}
        </p>
        {facts === null ? (
          <div className="profile-loading">读取中…</div>
        ) : facts.length === 0 ? (
          <div className="profile-loading">
            还没有画像。多聊几轮（比如「我主玩守岸人，是萌新」），这里就会长出来~
          </div>
        ) : (
          <ul className="profile-list">
            {facts.map((f) => (
              <li key={f.id} className="profile-item">
                <span className="profile-text">{f.fact}</span>
                <span className="profile-meta">
                  {f.created_at ? new Date(f.created_at).toLocaleDateString('zh-CN') : ''}
                </span>
                <button
                  className="profile-del"
                  title="不再记住这条"
                  onClick={async () => {
                    try {
                      await api.deleteFact(f.id, apiBase)
                      setFacts((prev) => (prev ?? []).filter((x) => x.id !== f.id))
                      toast('info', '已删除该画像')
                    } catch (err) {
                      toast('err', err instanceof Error ? err.message : String(err))
                    }
                  }}
                >
                  <Eraser size={12} />
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      {/* 数据 */}
      <section className="card set-card">
        <h3 className="set-h">
          <Trash2 size={15} /> 数据管理
        </h3>
        <div className="set-stats">
          <div className="stat">
            <b>{conversations.length}</b>
            <span>会话</span>
          </div>
          <div className="stat">
            <b>{totalMessages}</b>
            <span>消息</span>
          </div>
          <div className="stat">
            <b>{ingests.length}</b>
            <span>入库记录</span>
          </div>
        </div>
        <div className="set-row danger-row">
          <div className="set-row-main">
            <label>清空全部数据</label>
            <p>删除所有会话、消息与入库记录，并重置为一个新对话。此操作不可恢复。</p>
          </div>
          <div className="set-row-ctl">
            <button
              className={`btn btn-danger ${armed ? 'armed' : ''}`}
              onClick={() => {
                if (armed) {
                  clearAll()
                  setArmed(false)
                  toast('info', '已清空全部本地数据')
                } else {
                  setArmed(true)
                  setTimeout(() => setArmed(false), 4000)
                }
              }}
            >
              <Trash2 size={14} /> {armed ? '确认清空' : '清空数据'}
            </button>
          </div>
        </div>
      </section>

      <div className="set-about">
        潮声智库 · 鸣潮角色知识助手前端 v1.0 · 后端 FastAPI + LangGraph 混合检索 RAG
      </div>
    </div>
  )
}
