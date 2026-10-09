import { useEffect, useState } from 'react'
import { Zap, Server, Trash2, RefreshCw, Cpu, AlertTriangle, Check, Smile, KeyRound, Unlock, Lock } from 'lucide-react'
import { useStore } from '../../store/useStore'
import * as api from '../../lib/api'
import type { LlmConfig, ProviderPreset } from '../../types'
import { InlineFold } from './Fold'

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
export function LlmSection() {
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
                <p>必须提供原口令。仅更换口令本身，已保存的密钥不受影响。</p>
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
