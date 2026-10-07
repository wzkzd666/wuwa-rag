import { useCallback, useEffect, useState, type ReactNode } from 'react'
import {
  Settings as SettingsIcon, Sun, Moon, Zap, Type, Server, Trash2, Activity, RefreshCw,
  Camera, Image as ImageIcon, Palette, RotateCcw, User as UserIcon, Sparkles, Eraser,
  Cpu, Music4, Volume2, Loader2, AlertTriangle, Check, Smile, KeyRound, Unlock, Lock, Mic, ChevronDown,
} from 'lucide-react'
import { useStore } from '../store/useStore'
import * as api from '../lib/api'
import { pickImageFile, fileToDataUrl, formatBytes, dataUrlBytes } from '../lib/image'
import type {
  BgPreset, LlmConfig, ProviderPreset, TtsConfigOut, TtsStatus, UserFact,
} from '../types'
import type { MusicSetting } from '../lib/api'
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
 * 页面级可折叠区块。
 *
 * 设置项越堆越长（模型/语音/账号三块都是「配一次就再也不看」的），全展开会把
 * 频率高的「外观 / 背景」挤到屏幕外，所以默认只留一行标题 + 一句说明。
 *
 * ⚠️ 标题**整行可点**，且收起后行尾的 `aside` 状态标签必须留着——「语音合成」
 * 收起来却看不到「未就绪」，用户就得逐个点开去找，比不折叠更糟。
 * 折叠状态由调用方持有（见 SettingsPage 的 open/toggle），组件本身无状态。
 */
function Fold({ title, icon, aside, hint, open, onToggle, children }: {
  title: string
  icon: ReactNode
  aside?: ReactNode
  hint?: string
  open: boolean
  onToggle: () => void
  children: ReactNode
}) {
  return (
    <section className={`card set-card${open ? '' : ' set-folded'}`}>
      <div className="set-head-row">
        <button type="button" className="set-h set-h-btn" onClick={onToggle} aria-expanded={open}>
          {icon} {title}
          <ChevronDown size={15} className={`set-caret${open ? ' set-caret-open' : ''}`} />
        </button>
        {aside}
      </div>
      {!open && hint && <p className="set-fold-hint">{hint}</p>}
      {open && children}
    </section>
  )
}

/** 卡片内部的次级折叠：比 Fold 轻一档，只在卡片里划一小块可收起的区域。
 *  用途是「同一张卡里次要的一堆字段」——例如云端模型的凭据表单、
 *  语音合成的密钥/音色/语气指令，默认收起，避免一屏全是输入框。 */
function InlineFold({ title, open, onToggle, children }: {
  title: string
  open: boolean
  onToggle: () => void
  children: ReactNode
}) {
  return (
    <div className="set-inline-fold">
      <button type="button" className="set-inline-btn" onClick={onToggle} aria-expanded={open}>
        <ChevronDown size={14} className={`set-caret${open ? ' set-caret-open' : ''}`} />
        {title}
      </button>
      {open && <div className="set-inline-body">{children}</div>}
    </div>
  )
}

/**
 * 模型服务：① 默认用项目自带 agent（本地 aemeath）；② 可切换到用户自己的云端大模型。
 *
 * 隐私：api_key 输入后只在「保存」时 POST 给后端加密落库，读回永远是掩码（key_hint），
 * 前端不留存明文。已配置时输入框留空 = 保留原 key。
 *
 * 加密体系（双层密钥 DEK/KEK，详见 rag/llmstore.py）：
 *   - **登录密码通道**：登录时后端用它派生 KEK 解出 DEK → 下次登录自动连上，
 *     用户不需要额外输入任何口令；
 *   - **加密口令通道**：独立于登录密码的兜底通道，两者互不影响；
 *   - 服务端**不持有主密钥**（不在 .env、不落库），解出的 DEK 只在进程内存里；
 *     进程重启会清空 —— 重新登录即恢复，或用加密口令兜底解锁。
 * 界面必须讲清楚：凭据只用于当次提交，前端一律不留存（不写 localStorage）。
 *
 * ⚠️ 「本地 / 云端」开关**必须落库**（走 POST /llm/enabled），不能只改本地 state：
 *    它原本只是个组件内的 useState，保存按钮又长在云端表单里，切到本地后表单收起，
 *    这个选择就永远没机会被写进库；刷新页面时 GET /llm/config 读回 enabled=true，
 *    界面又跳回「云端自定义」——用户看到的就是「切回本地后自动跳回云端」。
 *    当时唯一的持久化办法是「删除配置」，而它会把 api_key 密文一并抹掉
 *    （那一行还承载语音凭据，TTS 也一起没了）。现在只翻开关，凭据原样保留。
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
  const [switching, setSwitching] = useState(false)    // 切换本地/云端开关的请求中
  // 云端配置表单的展开态：null = 自动（在云端，或还没配置过时自动展开）
  const [cloudOpen, setCloudOpen] = useState<boolean | null>(null)
  // 情绪判定模型：默认本地 qwen3:8b；打开则交由上面的云端自定义模型兼任
  const [emotionViaCloud, setEmotionViaCloud] = useState(false)
  // 两种凭据都只活在组件内存里，提交成功即清空 —— 前端绝不留存
  const [password, setPassword] = useState('')      // 登录密码（自动解锁通道）
  const [binding, setBinding] = useState(false)
  const [passphrase, setPassphrase] = useState('')  // 加密口令（兜底通道）
  const [unlocking, setUnlocking] = useState(false)
  const [changing, setChanging] = useState(false)
  const [oldPp, setOldPp] = useState('')
  const [newPp, setNewPp] = useState('')

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

  /** 切换作答模型来源：只翻开关，凭据与语音配置原样保留 */
  const switchModel = async (next: boolean) => {
    if (next === useCloud) return
    setUseCloud(next)                 // 乐观更新，开关手感不受网络影响
    if (!next) {
      setSwitching(true)
      try {
        await api.setLlmEnabled(false, apiBase)
        toast('ok', '已切回本地默认模型（云端配置与语音凭据都留着）')
      } catch (err) {
        toast('err', err instanceof Error ? err.message : String(err))
      } finally {
        setSwitching(false)
        await load()                  // 以服务端为准回填，避免乐观更新与真实状态不一致
      }
      return
    }
    // 切到云端：没配置过就别发请求了，直接把表单摊开等用户填
    if (!cfg?.configured) {
      setCloudOpen(true)
      setUseCloud(false)              // 未配置时不允许停留在「云端」假状态
      toast('info', '还没保存过云端配置，填好 API 信息并点「保存」即启用')
      return
    }
    setSwitching(true)
    try {
      const out = await api.setLlmEnabled(true, apiBase)
      toast(out.unlocked ? 'ok' : 'info',
        out.unlocked ? '已切到云端自定义模型'
          : '已启用云端配置，但密钥本进程尚未解锁，重新登录或用加密口令解锁后才会生效')
    } catch (err) {
      toast('err', err instanceof Error ? err.message : String(err))
    } finally {
      setSwitching(false)
      await load()
    }
  }

  const pickProvider = (key: string) => {
    setProvider(key)
    const hit = providers.find((x) => x.key === key)
    if (hit && hit.base_url) setBaseUrl(hit.base_url)   // 选预设自动带出地址
  }

  /** 用登录密码解锁；已解锁但尚未绑定该通道时，随一次「保留原 Key」的保存补建 */
  const doBindPassword = async () => {
    if (!password) { toast('err', '请输入登录密码'); return }
    if (!cfg?.configured) {
      toast('info', '请先在下面填写 API 信息并保存，保存时会带上这个密码完成绑定')
      return
    }
    setBinding(true)
    try {
      if (cfg.unlocked) {
        await api.saveLlmConfig({
          base_url: baseUrl.trim(), model: model.trim(), api_key: '',
          provider, enabled: useCloud, emotion_enabled: emotionViaCloud,
          password, passphrase: '',
        }, apiBase)
        toast('ok', '已绑定登录密码，下次登录自动连上')
      } else {
        await api.unlockLlm({ password }, apiBase)
        toast('ok', '已解锁，云端模型生效')
      }
      setPassword('')
      await load()
    } catch (err) {
      toast('err', err instanceof Error ? err.message : String(err))
    } finally {
      setBinding(false)
    }
  }

  /** 用加密口令解锁（兜底通道，独立于登录密码） */
  const doUnlock = async () => {
    if (!passphrase) { toast('err', '请输入加密口令'); return }
    if (!cfg?.configured) {
      toast('info', '请先在下面填写 API 信息并保存，保存时可一并设置口令')
      return
    }
    setUnlocking(true)
    try {
      await api.unlockLlm({ passphrase }, apiBase)
      setPassphrase('')
      toast('ok', '已解锁，云端模型生效')
      await load()
    } catch (err) {
      toast('err', err instanceof Error ? err.message : String(err))
    } finally {
      setUnlocking(false)
    }
  }

  const doLock = async () => {
    try {
      await api.lockLlm(apiBase)
      toast('info', '已锁定，云端模型回落本地默认')
      await load()
    } catch (err) {
      toast('err', err instanceof Error ? err.message : String(err))
    }
  }

  const doChangePp = async () => {
    if (!oldPp) { toast('err', '请先填写原口令'); return }
    if (newPp.length < 8) { toast('err', '新口令至少 8 位'); return }
    try {
      await api.changeLlmPassphrase(oldPp, newPp, apiBase)
      setOldPp(''); setNewPp(''); setChanging(false)
      toast('ok', '加密口令已更换')
      await load()
    } catch (err) {
      toast('err', err instanceof Error ? err.message : String(err))
    }
  }

  const doTest = async () => {
    if (!baseUrl.trim()) { toast('err', '请先填 API 地址'); return }
    setTesting(true)
    try {
      const out = await api.testLlmConfig(
        { base_url: baseUrl.trim(), model: model.trim(), api_key: apiKey,
          provider, enabled: true, emotion_enabled: emotionViaCloud,
          password, passphrase },
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
    if (cfg && !cfg.crypto_ready) { toast('err', '服务端缺少加密依赖，无法安全保存密钥'); return }
    if (!baseUrl.trim() || !model.trim()) { toast('err', 'API 地址与模型都不能为空'); return }
    if (!apiKey.trim() && !(cfg?.configured)) { toast('err', '请填写 API Key'); return }
    // ⚠️ 判据是「DEK 是否已解锁」，**不是**「云端模型自己是否配过」——DEK 与语音凭据
    // 共用，解锁一次两边都能写入。用 `configured` 判断会在「先配了语音、再配模型」时
    // 要求用户重输口令，而那时口令输入框已经变成「已解锁」徽章 —— 无处可填，直接死路。
    if (cfg && !cfg.unlocked && !password.trim() && !passphrase.trim()) {
      toast('err', '请填写登录密码或加密口令，用于建立 / 解开加密密钥'); return
    }
    setSaving(true)
    try {
      await api.saveLlmConfig(
        { base_url: baseUrl.trim(), model: model.trim(), api_key: apiKey.trim(),
          provider, enabled: useCloud, emotion_enabled: emotionViaCloud,
          password, passphrase },
        apiBase,
      )
      setApiKey('')             // 保存后清空明文输入
      setPassword(''); setPassphrase('')
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
      setPassword(''); setPassphrase('')
      toast('info', '已删除云端配置，回到本地默认模型（语音凭据也一并清空，需重新配置）')
      await load()
    } catch (err) {
      toast('err', err instanceof Error ? err.message : String(err))
    }
  }

  // 「云端配置」表单是否可见：null = 自动（在云端，或还没配置过时展开让用户能填）
  const cloudShown = cloudOpen ?? (useCloud || !cfg?.configured)
  return (
    <>
      <div className="set-row">
        <div className="set-row-main">
          <label>作答模型</label>
          <p>
            默认用项目自带的本地 agent（{cfg?.default_model || 'aemeath'}，人设内置）。
            也可切换到你自己的云端大模型 API —— 配置只属于你，密钥加密存储、绝不回显。
          </p>
          <p className="set-status">
            当前生效：{useCloud
              ? `云端自定义（${model.trim() || cfg?.model || '未选模型'}）`
              : `本地默认（${cfg?.default_model || 'aemeath'}）`}
            {useCloud && cfg && !cfg.unlocked && '；密钥本进程尚未解锁，实际仍回落本地'}
            {'。'}
          </p>
        </div>
        <div className="set-row-ctl">
          <div className="seg">
            <button className={`seg-btn ${!useCloud ? 'seg-on' : ''}`} disabled={switching}
                    onClick={() => void switchModel(false)}>
              <Server size={13} /> 本地默认
            </button>
            <button className={`seg-btn ${useCloud ? 'seg-on' : ''}`} disabled={switching}
                    onClick={() => void switchModel(true)}>
              <Cpu size={13} /> 云端自定义
            </button>
          </div>
        </div>
      </div>

      {/* 凭据表单收在次级折叠里：切到本地后自动收起，但**仍可手动展开编辑**——
          否则想改 Key 就得先切到云端（等于启用一个自己还没配好的模型）。 */}
      <InlineFold
        title={`云端模型配置${cfg?.configured ? `（已保存 ${cfg.key_hint}）` : '（尚未配置）'}`}
        open={cloudShown}
        onToggle={() => setCloudOpen(!cloudShown)}
      >
        {cfg && !cfg.crypto_ready && (
          <div className="set-warn">
            <AlertTriangle size={14} /> 服务端缺少 <code>cryptography</code> 依赖，云模型功能已关闭（不会明文存密钥）。请在后端安装该包后再用。
          </div>
        )}
          {/* ── 通道一：登录密码（自动解锁）──
              服务端登录时用它派生 KEK 解出 DEK，用户无需任何额外输入 */}
          <div className="set-row">
            <div className="set-row-main">
              <label><KeyRound size={13} /> 登录密码（自动解锁）</label>
              <p>
                {cfg?.auto_unlock
                  ? '已绑定：每次登录自动解密你的 Key，无需再输任何口令。'
                  : '填一次登录密码即可绑定 —— 之后每次登录自动连上你的云端模型。'}
              </p>
            </div>
            <div className="set-row-ctl status-ctl">
              {cfg?.auto_unlock && cfg?.unlocked ? (
                <>
                  <span className="tag"><Check size={12} /> 已解锁</span>
                  <button className="btn btn-ghost" onClick={doLock}>
                    <Lock size={14} /> 锁定
                  </button>
                </>
              ) : (
                <>
                  <input className="input api-input" type="password" value={password}
                         placeholder="当前登录密码"
                         onChange={(e) => setPassword(e.target.value)}
                         autoComplete="current-password" />
                  <button className="btn btn-ghost" onClick={doBindPassword} disabled={binding}>
                    {binding ? <RefreshCw size={14} className="spin" /> : <Unlock size={14} />}
                    {cfg?.configured ? '解锁' : '绑定'}
                  </button>
                </>
              )}
            </div>
          </div>

          {/* ── 通道二：加密口令（兜底，独立于登录密码）── */}
          <div className="set-row">
            <div className="set-row-main">
              <label>加密口令（兜底）</label>
              <p>
                独立于登录密码的第二条通道。忘了它也不影响登录自动解锁，
                反过来登录密码改了它照样能解开。
              </p>
            </div>
            <div className="set-row-ctl status-ctl">
              {cfg?.pp_bound ? (
                <>
                  <span className="tag"><Check size={12} /> 已设置</span>
                  <button className="btn btn-ghost" onClick={() => setChanging(!changing)}>
                    改口令
                  </button>
                </>
              ) : (
                <>
                  <input className="input api-input" type="password" value={passphrase}
                         placeholder="设置口令（≥8 位，可留空）"
                         onChange={(e) => setPassphrase(e.target.value)} autoComplete="off" />
                  <button className="btn btn-ghost" onClick={doUnlock} disabled={unlocking}>
                    {unlocking ? <RefreshCw size={14} className="spin" /> : <Unlock size={14} />} 解锁
                  </button>
                </>
              )}
            </div>
          </div>
          {changing && (
            <div className="set-row">
              <div className="set-row-main">
                <label>更换口令</label>
                <p>必须提供原口令。只换口令本身，已存的 API Key 密文不动。</p>
              </div>
              <div className="set-row-ctl status-ctl">
                <input className="input api-input" type="password" value={oldPp}
                       placeholder="原口令" onChange={(e) => setOldPp(e.target.value)}
                       autoComplete="off" />
                <input className="input api-input" type="password" value={newPp}
                       placeholder="新口令（≥8 位）" onChange={(e) => setNewPp(e.target.value)}
                       autoComplete="off" />
                <button className="btn btn-ghost" onClick={doChangePp}>
                  <Check size={14} /> 确认
                </button>
              </div>
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
              <p>
                保存后立即生效；切回「本地默认」只是停用，配置与语音凭据都留着，随时可切回来。
                「删除配置」才会把密钥密文一并抹掉（那一行同时承载语音凭据，删除后朗读也要重配）。
              </p>
            </div>
            <div className="set-row-ctl status-ctl">
              <button className="btn btn-primary" onClick={doSave} disabled={saving || (cfg ? !cfg.crypto_ready : false)}>
                {saving ? <RefreshCw size={14} className="spin" /> : <Check size={14} />} 保存
              </button>
              {cfg?.configured && (
                <button className="btn btn-danger" onClick={doDelete}>
                  <Trash2 size={14} /> 删除配置
                </button>
              )}
            </div>
          </div>
      </InlineFold>
    </>
  )
}

/**
 * 语音合成：阿里云百炼 · Qwen-Audio-3.1-TTS-Flash。
 *
 * 密钥**由用户自持**（开源分发下部署者不替用户垫额度）：每个用户填自己的
 * API Key + 业务空间 ID，加密落库（与云端模型共用同一把 DEK，见 rag/llmstore.py），
 * 因此「登录自动解锁 / 改密码自动重绑 / 加密口令兜底」三条能力直接继承，
 * 用户只需维护一套口令。
 *
 * ⚠️ 官方支持的音色与模型代次**强绑定**：3.1 只认 `_v3.1` 后缀音色，
 * 填成 3.0 的音色会被服务端拒绝（界面用下拉列出可选音色，避免手填踩坑）。
 *
 * 优先级：**用户自持凭据 > 部署者 .env 兜底**。
 */
function TtsSection() {
  const toast = useStore((s) => s.toast)
  const apiBase = useStore((s) => s.settings.apiBase)
  const [st, setSt] = useState<TtsStatus | null>(null)
  const [cfg, setCfg] = useState<TtsConfigOut | null>(null)
  const [apiKey, setApiKey] = useState('')
  const [workspaceId, setWorkspaceId] = useState('')
  const [model, setModel] = useState('')
  const [voice, setVoice] = useState('')
  const [instruction, setInstruction] = useState('')
  // 两种凭据只活在组件内存里，提交成功即清空 —— 前端绝不留存
  const [password, setPassword] = useState('')
  const [passphrase, setPassphrase] = useState('')
  const [saving, setSaving] = useState(false)
  const [unlocking, setUnlocking] = useState(false)
  // 凭据/音色表单的展开态：null = 自动（还没配过就展开，配过就收起）
  const [credOpen, setCredOpen] = useState<boolean | null>(null)

  const load = async () => {
    try {
      const [s, c] = await Promise.all([api.ttsStatus(apiBase), api.getTtsConfig(apiBase)])
      setSt(s)
      setCfg(c)
      if (c.configured) {
        // 已存的值回填；留空的项用 placeholder 提示默认值，不必替用户填死
        setWorkspaceId(c.workspace_id)
        setModel(c.model)
        setVoice(c.voice)
        setInstruction(c.instruction)
      }
    } catch {
      /* 未登录/后端未起：静默，不打扰 */
    }
  }
  useEffect(() => { void load() }, [])   // eslint-disable-line react-hooks/exhaustive-deps

  /** 复用云端模型那套解锁接口（同一把 DEK，解锁一次两边都生效） */
  const doUnlock = async () => {
    if (!password.trim() && !passphrase.trim()) {
      toast('err', '请输入登录密码或加密口令')
      return
    }
    setUnlocking(true)
    try {
      await api.unlockLlm({ password, passphrase }, apiBase)
      setPassword(''); setPassphrase('')
      toast('ok', '已解锁，可正常朗读')
      await load()
    } catch (err) {
      toast('err', err instanceof Error ? err.message : String(err))
    } finally {
      setUnlocking(false)
    }
  }

  const doSave = async () => {
    if (cfg && !cfg.crypto_ready) {
      toast('err', '服务端缺少加密依赖，无法安全保存密钥'); return
    }
    if (!workspaceId.trim()) { toast('err', '请填写业务空间 ID'); return }
    if (!apiKey.trim() && !cfg?.configured) { toast('err', '请填写 API Key'); return }
    // ⚠️ 判据是「DEK 是否已解锁」，**不是**「TTS 自己是否配过」——DEK 由云端模型与语音
    // 共用，登录自动解锁（或先配过云端模型）之后，保存语音凭据根本不需要再给口令。
    // 用 `configured` 判断会把这些人全部挡在门外，而界面上又没有地方填口令（死路）。
    if (cfg && !cfg.unlocked && !password.trim() && !passphrase.trim()) {
      toast('err', '请填写登录密码或加密口令，用于建立 / 解开加密密钥'); return
    }
    // 只有「要新建口令通道」时才卡长度；已绑定过的口令不允许出现 <8 位，故不会误伤解锁
    if (cfg && !cfg.unlocked && !bound && !password.trim()
        && passphrase.trim() && passphrase.trim().length < 8) {
      toast('err', '加密口令至少 8 位（服务端不保存，忘了无法找回）'); return
    }
    setSaving(true)
    try {
      await api.saveTtsConfig({
        api_key: apiKey.trim(), workspace_id: workspaceId.trim(),
        model: model.trim(), voice: voice.trim(), instruction,
        password, passphrase,
      }, apiBase)
      setApiKey(''); setPassword(''); setPassphrase('')   // 保存后清空明文
      toast('ok', '语音配置已保存')
      await load()
    } catch (err) {
      toast('err', err instanceof Error ? err.message : String(err))
    } finally {
      setSaving(false)
    }
  }

  const doDelete = async () => {
    try {
      await api.deleteTtsConfig(apiBase)
      setApiKey(''); setWorkspaceId(''); setModel(''); setVoice(''); setInstruction('')
      setPassword(''); setPassphrase('')
      toast('info', '已删除语音配置')
      await load()
    } catch (err) {
      toast('err', err instanceof Error ? err.message : String(err))
    }
  }

  const voices = st?.voices || []
  const ready = st?.ready === true
  const srcText = st?.source === 'user' ? '使用你自己的凭据'
    : st?.source === 'global' ? '使用部署者预置的凭据' : '尚未配置'
  // 是否已建立过解锁通道（登录密码 / 加密口令任一条）。false = 本次保存将新建 DEK，
  // 此时才需要卡口令长度、才需要提示「填一次以后免输」。
  const bound = !!(cfg?.auto_unlock || cfg?.pp_bound)
  const unlocked = cfg?.unlocked === true
  const unlockHint = cfg?.auto_unlock
    ? '密钥已保存但本进程尚未解锁：重新登录即可自动恢复，也可以在这里立刻解锁。'
    : cfg?.pp_bound
      ? '密钥已保存但本进程尚未解锁：输入当初设置的加密口令解锁（这条通道不随登录自动恢复）。'
      : '第一次保存需要先建立加密密钥：填一次登录密码（推荐 —— 之后每次登录自动解锁），'
        + '或设置一个至少 8 位的加密口令。密钥与云端模型共用，两边只需建立一次。'

  const credShown = credOpen ?? !cfg?.configured
  return (
    <>
      <div className="set-row">
        <div className="set-row-main">
          <label>朗读服务</label>
          <p>
            由你自己的阿里云百炼账号提供语音合成（Qwen-Audio-3.1-TTS-Flash，北京地域）。
            密钥加密入库、只回掩码，与云端模型共用一套解锁口令。
          </p>
          <p>
            当前：{srcText}；音色 {st?.voice_label || st?.voice || '默认'}；
            模型 {st?.model_label || st?.model || '默认'}。
          </p>
          {!ready && st?.reason && <p>不可用原因：{st.reason}。</p>}
        </div>
        <div className="set-row-ctl status-ctl">
          {st !== null && (
            ready
              ? <span className="tag tag-ok"><Check size={12} /> 已就绪</span>
              : <span className="tag tag-warn"><AlertTriangle size={12} /> 未就绪</span>
          )}
        </div>
      </div>

      {cfg && !cfg.crypto_ready && (
        <div className="set-warn">
          <AlertTriangle size={14} /> 服务端缺少 <code>cryptography</code> 依赖，语音配置已关闭（不会明文存密钥）。请在后端安装该包后再用。
        </div>
      )}

      {/* 只要**未解锁**就要露出口令入口 —— 不能只在「已配过」时露：
          首次保存同样需要凭据来建立 DEK，只显示已配置用户的入口会让新人卡死
          （校验要求填口令，界面上却无处可填）。
          `cfg === null`（拉取失败）时也露出：那种情况下只有后端知道真实状态，
          留着入口才能接住后端可能的「请填写口令」，不至于又变成无处可填。
          同一把 DEK，解锁接口复用云端模型那套；已解锁则整行隐藏，保存无需任何口令。 */}
      {!unlocked && (
        <div className="set-row">
          <div className="set-row-main">
            <label><KeyRound size={13} /> {bound ? '解锁密钥' : '加密密钥'}</label>
            <p>{unlockHint}</p>
          </div>
          <div className="set-row-ctl status-ctl">
            <input className="input api-input" type="password" value={password}
                   placeholder={bound && !cfg?.auto_unlock ? '登录密码（未绑定）' : '登录密码'}
                   onChange={(e) => setPassword(e.target.value)}
                   autoComplete="current-password" />
            <input className="input api-input" type="password" value={passphrase}
                   placeholder={bound ? '或加密口令' : '或加密口令（≥8 位）'}
                   onChange={(e) => setPassphrase(e.target.value)}
                   autoComplete="off" />
            {/* 已建立通道才谈得上「解锁」；全新用户是「随保存一起建立」，没有可解锁的东西 */}
            {bound && (
              <button className="btn btn-ghost" onClick={doUnlock} disabled={unlocking}>
                {unlocking ? <RefreshCw size={14} className="spin" /> : <Unlock size={14} />} 解锁
              </button>
            )}
          </div>
        </div>
      )}

      {/* 凭据与音色收进次级折叠：这几项「配一次就不动」，全摊开会把
          「朗读服务」的状态与保存按钮挤得很远。默认折叠策略见 credShown。 */}
      <InlineFold
        title={`凭据与音色${cfg?.configured ? `（已保存 ${cfg.key_hint}）` : '（尚未配置）'}`}
        open={credShown}
        onToggle={() => setCredOpen(!credShown)}
      >
      <div className="set-row">
        <div className="set-row-main">
          <label>API Key</label>
          <p>
            {cfg?.configured
              ? <>已保存（<code>{cfg.key_hint}</code>）。留空 = 保留原 Key 不改。</>
              : '百炼控制台的 API Key，必须是**北京地域**的。仅保存时加密上传，前端不留存、不回显明文。'}
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
          <label>业务空间 ID</label>
          <p>百炼控制台「业务空间」里的一串 id，语音端点域名要用到它。只允许字母、数字与连字符。</p>
        </div>
        <div className="set-row-ctl">
          <input className="input api-input" value={workspaceId}
                 placeholder="如 llm-xxxxxxxx" onChange={(e) => setWorkspaceId(e.target.value)}
                 autoComplete="off" />
        </div>
      </div>

      <div className="set-row">
        <div className="set-row-main">
          <label>音色</label>
          <p>
            音色与模型代次**强绑定**：3.1 只认 <code>_v3.1</code> 后缀的音色。
            留空则用默认（{st?.defaults.voice_label || st?.defaults.voice || '内置默认'}）。
          </p>
        </div>
        <div className="set-row-ctl model-ctl">
          <input className="input api-input" list="tts-voices" value={voice}
                 placeholder={st?.defaults.voice || '留空用默认'}
                 onChange={(e) => setVoice(e.target.value)} autoComplete="off" />
          <datalist id="tts-voices">
            {voices.map((v) => <option key={v.id} value={v.id}>{v.label}</option>)}
          </datalist>
        </div>
      </div>

      <div className="set-row">
        <div className="set-row-main">
          <label>模型</label>
          <p>一般不需要改。留空则用默认（{st?.defaults.model || 'qwen-audio-3.1-tts-flash'}）。</p>
        </div>
        <div className="set-row-ctl">
          <input className="input api-input" value={model}
                 placeholder={st?.defaults.model || '留空用默认'}
                 onChange={(e) => setModel(e.target.value)} autoComplete="off" />
        </div>
      </div>

      <div className="set-row">
        <div className="set-row-main">
          <label>语气指令</label>
          <p>
            自然语言描述音色性格与语速基调（官方「指令控制」，≤100 字符、汉字按 2 字符计）。
            留空则用默认：{st?.defaults.instruction || '内置默认'}。
          </p>
        </div>
        <div className="set-row-ctl">
          <input className="input api-input" value={instruction}
                 placeholder="留空用默认语气" onChange={(e) => setInstruction(e.target.value)}
                 autoComplete="off" />
        </div>
      </div>
      </InlineFold>

      <div className="set-row">
        <div className="set-row-main">
          <label>&nbsp;</label>
          <p>保存后立即生效；删除配置即回落部署者预置的凭据（若已配置），否则朗读不可用。</p>
        </div>
        <div className="set-row-ctl status-ctl">
          <button className="btn btn-primary" onClick={doSave}
                  disabled={saving || (cfg ? !cfg.crypto_ready : false)}>
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
  )
}

/**
 * 账号安全：改登录密码。
 *
 * ⚠️ 顺序敏感：后端会**先用旧密码重绑**云端 API-KEY 的密钥（DEK）再更新密码哈希。
 * 少了这一步，改完密码就再也解不开已存的 Key（只能删配置重填），所以这里必须
 * 让用户输入原密码，不能只填新密码。
 */
function AccountSection() {
  const toast = useStore((s) => s.toast)
  const apiBase = useStore((s) => s.settings.apiBase)
  const [oldPw, setOldPw] = useState('')
  const [newPw, setNewPw] = useState('')
  const [saving, setSaving] = useState(false)

  const doChange = async () => {
    if (!oldPw) { toast('err', '请输入原密码'); return }
    if (newPw.length < 6) { toast('err', '新密码至少 6 位'); return }
    setSaving(true)
    try {
      await api.changePassword(oldPw, newPw, apiBase)
      setOldPw(''); setNewPw('')
      toast('ok', '密码已修改；已存的云端 Key 已同步重绑，下次登录照常自动连上')
    } catch (err) {
      toast('err', err instanceof Error ? err.message : String(err))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="set-row">
      <div className="set-row-main">
        <label>登录密码</label>
        <p>
          改密码时后端会先用旧密码把云端 API-KEY 的密钥重新绑定一遍，
          所以改完之后已存的 Key 依然能自动解开（不用重新填）。
        </p>
      </div>
      <div className="set-row-ctl status-ctl">
        <input className="input api-input" type="password" value={oldPw}
               placeholder="原密码" onChange={(e) => setOldPw(e.target.value)}
               autoComplete="current-password" />
        <input className="input api-input" type="password" value={newPw}
               placeholder="新密码（≥6 位）" onChange={(e) => setNewPw(e.target.value)}
               autoComplete="new-password" />
        <button className="btn btn-ghost" onClick={doChange} disabled={saving}>
          {saving ? <RefreshCw size={14} className="spin" /> : <Check size={14} />} 修改
        </button>
      </div>
    </div>
  )
}

export default function SettingsPage() {
  const settings = useStore((s) => s.settings)
  const setSettings = useStore((s) => s.setSettings)
  const health = useStore((s) => s.health)
  const checkHealth = useStore((s) => s.checkHealth)
  const clearAll = useStore((s) => s.clearAll)
  const convs = useStore((s) => s.convs)
  const ingests = useStore((s) => s.ingests)
  const toast = useStore((s) => s.toast)

  const [testing, setTesting] = useState(false)
  const [armed, setArmed] = useState(false)
  // 各区块的折叠态：默认只展开高频项（后端连接 / 模型服务 / 外观），
  // 其余「配一次就不看」的（语音、账号、个性化、画像、数据）收起，避免一屏堆满。
  const [fold, setFold] = useState<Record<string, boolean>>({
    backend: true, model: true, tts: false, account: false, music: false,
    appearance: true, personal: false, profile: false, data: false,
  })
  const toggle = (k: string) => setFold((f) => ({ ...f, [k]: !f[k] }))

  // ---- 音乐设置（开关 + 播放器路径）----
  const [mset, setMset] = useState<MusicSetting | null>(null)
  const [msetBusy, setMsetBusy] = useState(false)
  const [exeDraft, setExeDraft] = useState<string | null>(null)
  const loadMusicSet = useCallback(async () => {
    try {
      setMset(await api.musicSetting(settings.apiBase))
    } catch {
      /* 读不到就让开关保持禁用，不假装有设置 */
    }
  }, [settings.apiBase])
  useEffect(() => {
    if (fold.music) void loadMusicSet()
  }, [fold.music, loadMusicSet])
  const saveMusic = async (body: { enabled?: boolean | null; exe?: string | null }) => {
    setMsetBusy(true)
    try {
      await api.musicSettingPut(body, settings.apiBase)
      await loadMusicSet()
      setExeDraft(null)
      toast('ok', '音乐设置已保存，立即生效')
    } catch (e) {
      toast('err', e instanceof Error ? e.message : String(e))
    } finally {
      setMsetBusy(false)
    }
  }
  // 我的画像（user_facts 表，后端 /profile）
  const auth = useStore((s) => s.auth)
  const apiBase = useStore((s) => s.settings.apiBase)
  const [facts, setFacts] = useState<UserFact[] | null>(null)

  const loadFacts = async () => {
    try {
      const out = await api.getProfile(apiBase)
      setFacts(out.facts)
    } catch {
      setFacts([])
    }
  }
  // 挂载后拉一次画像（简单起见不轮询——设置页本来就不是常驻页面）。
  // 不用 <section ref> 触发：区块已交给 Fold 渲染，ref 拿不到。
  useEffect(() => { void loadFacts() }, [])   // eslint-disable-line react-hooks/exhaustive-deps

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

  const totalMessages = convs.reduce((n, c) => n + c.message_count, 0)

  return (
    <div className="page settings-page">
      <div className="page-head">
        <div>
          <h2 className="page-title">
            <SettingsIcon size={19} className="grad-text" /> 设置
          </h2>
          <p className="page-desc">配置后端连接、界面外观与数据管理。</p>
        </div>
      </div>

      {/* 后端连接 */}
      <Fold title="后端连接" icon={<Server size={15} />} open={fold.backend}
            onToggle={() => toggle('backend')}
            hint="接口地址、连通状态、流式开关与语音朗读总开关。">
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
              为每条回答提供语音合成：系统先识别回答的语气，再以匹配的情绪朗读出来。
              这是**本机显示偏好**（关掉只是不显示朗读按钮）；凭据与音色在下面
              「语音合成」里配置。
            </p>
          </div>
          <div className="set-row-ctl">
            <button
              className={`switch ${settings.ttsEnabled ? 'switch-on' : ''}`}
              onClick={() => setSettings({ ttsEnabled: !settings.ttsEnabled })}
              role="switch"
              aria-checked={settings.ttsEnabled}
            >
              <span className="switch-knob" />
            </button>
          </div>
        </div>
      </Fold>

      {/* 模型服务：默认本地 agent / 用户自定义云端 API（2026-09-29） */}
      <Fold title="模型服务" icon={<Cpu size={15} />} open={fold.model}
            onToggle={() => toggle('model')}
            hint="本地默认与云端自定义的切换；凭据加密保存，切回本地不会删除云端配置。">
        <LlmSection />
      </Fold>

      {/* 语音合成：用户自持百炼凭据 + 音色（2026-09-30） */}
      <Fold title="语音合成" icon={<Mic size={15} />} open={fold.tts}
            onToggle={() => toggle('tts')}
            hint="朗读用的百炼密钥、业务空间、音色与语气指令（与云端模型共用一套解锁口令）。">
        <TtsSection />
      </Fold>

      {/* 账号安全：改登录密码会自动重绑云端密钥（2026-09-30） */}
      {/* 音乐播放：开关与播放器路径都在这里改，**立即生效不用重启**。
          路径留空的语义是「**读环境变量 QQMUSIC_EXE**」，不是「自动满世界找」——
          路径属机器/部署级，应当由环境变量声明；这里填则覆盖环境变量。 */}
      <Fold title="音乐播放" icon={<Music4 size={15} />} open={fold.music}
            onToggle={() => toggle('music')}
            hint="开启后可直接说「放首周杰伦的晴天」「暂停」「小声点」">
        <div className="music-set">
          <label className="music-set-row">
            <input
              type="checkbox"
              checked={!!mset?.enabled}
              disabled={msetBusy || mset === null}
              onChange={(e) => void saveMusic({ enabled: e.target.checked })}
            />
            <span>
              启用音乐播放
              <em className="music-set-hint">
                {mset?.source === 'deployment'
                  ? '当前由部署配置（.env MUSIC_ENABLED）开启'
                  : mset?.personal === true
                    ? '已由你开启'
                    : '关闭时问「放歌」会得到「音乐功能不可用」的提示'}
                {mset && !mset.exe && mset.enabled && (
                  <em className="music-set-hint music-set-warn">
                    未设置路径：将以环境变量 <code>QQMUSIC_EXE</code> 为准
                    （<code>setx QQMUSIC_EXE "D://path//to//QQMusic.exe"</code> 后需重启本服务）。
                  </em>
                )}
              </em>
            </span>
          </label>
          <div className="music-set-row">
            <span className="music-set-label">播放器路径</span>
            <input
              className="input music-set-exe"
              value={exeDraft ?? mset?.exe ?? ''}
              placeholder="留空 = 读环境变量 QQMUSIC_EXE"
              disabled={msetBusy || mset === null}
              onChange={(e) => setExeDraft(e.target.value)}
              onBlur={() => {
                const v = (exeDraft ?? mset?.exe ?? '').trim()
                if (v !== (mset?.exe ?? '')) void saveMusic({ exe: v })
              }}
            />
            {msetBusy ? <Loader2 size={13} className="spin" /> : (
              <button
                className="btn btn-ghost btn-sm"
                disabled={!mset || (exeDraft ?? mset.exe) === mset.exe}
                onClick={() => void saveMusic({ exe: (exeDraft ?? '').trim() })}
              >
                保存
              </button>
            )}
          </div>
        </div>
      </Fold>

      <Fold title="账号安全" icon={<KeyRound size={15} />} open={fold.account}
            onToggle={() => toggle('account')}
            hint="修改登录密码（会自动重绑云端密钥，已存的 Key 仍能自动解开）。">
        <AccountSection />
      </Fold>

      {/* 外观 */}
      <Fold title="外观" icon={<Sun size={15} />} open={fold.appearance}
            onToggle={() => toggle('appearance')}
            hint="主题与全局基础字号。">
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
      </Fold>

      {/* 个性化：头像 + 背景 */}
      <Fold title="个性化" icon={<Palette size={15} />} open={fold.personal}
            onToggle={() => toggle('personal')}
            hint="助手/我的头像、聊天背景（预设渐变或自定义图片、遮罩与模糊）。">
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
            <p>预设渐变或自定义图片。图片自动压缩至长边 1600px，仅在本机留存，不上传服务端。</p>
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

        {/* 面板材质：卡片 / 页面标题块 / 欢迎卡统一走 --panel-*，对所有背景预设都生效 */}
        <div className="set-row">
          <div className="set-row-main">
            <label>面板不透明度</label>
            <p>
              卡片与标题块的底色浓度。100% 为实心，调低更透、背景更有存在感，但表格文字对比度会下降。
              当前 {(settings.panelAlpha * 100).toFixed(0)}%。
            </p>
          </div>
          <div className="set-row-ctl font-ctl">
            <input
              type="range"
              min={0.3}
              max={1}
              step={0.02}
              value={settings.panelAlpha}
              onChange={(e) => setSettings({ panelAlpha: Number(e.target.value) })}
            />
            <span className="font-val">{(settings.panelAlpha * 100).toFixed(0)}%</span>
          </div>
        </div>
        <div className="set-row">
          <div className="set-row-main">
            <label>面板磨砂</label>
            <p>面板背后的模糊半径，0 为不模糊。当前 {settings.panelBlur}px。</p>
          </div>
          <div className="set-row-ctl font-ctl">
            <input
              type="range"
              min={0}
              max={24}
              step={1}
              value={settings.panelBlur}
              onChange={(e) => setSettings({ panelBlur: Number(e.target.value) })}
            />
            <span className="font-val">{settings.panelBlur}px</span>
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
      </Fold>

      {/* 我的画像（user_facts 表）。
          ⚠️ 拉取时机改动过：原先靠 <section ref> 在挂载时触发 loadFacts，
          改成 Fold 之后 section 由 Fold 渲染、拿不到 ref，所以挪成显式 useEffect
          （见上方 loadFacts 定义处）——否则收起状态下画像永远不加载。 */}
      <Fold title="我的画像" icon={<Sparkles size={15} />} open={fold.profile}
            onToggle={() => toggle('profile')}
            hint="系统自动记住的稳定偏好（主玩角色、熟悉程度等），可逐条删除。">
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
      </Fold>

      {/* 数据 */}
      <Fold title="数据管理" icon={<Trash2 size={15} />} open={fold.data}
            onToggle={() => toggle('data')}
            hint="会话/消息/入库记录统计，以及清空全部会话（不可恢复）。">
        <div className="set-stats">
          <div className="stat">
            <b>{convs.length}</b>
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
            <label>清空全部会话</label>
            <p>删除你账号名下的所有会话与消息（服务端转录 + 模型侧记忆）。
              界面偏好（主题/字号/头像/背景）不受影响。此操作不可恢复。</p>
          </div>
          <div className="set-row-ctl">
            <button
              className={`btn btn-danger ${armed ? 'armed' : ''}`}
              onClick={() => {
                if (armed) {
                  void clearAll()
                  setArmed(false)
                  toast('info', '已清空全部会话')
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
      </Fold>

      <div className="set-about">
        潮声智库 · 鸣潮角色知识助手前端 v1.0 · 后端 FastAPI + LangGraph 混合检索 RAG
      </div>
    </div>
  )
}
