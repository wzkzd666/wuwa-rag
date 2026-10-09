import { useEffect, useState } from 'react'
import { Trash2, RefreshCw, AlertTriangle, Check, KeyRound, Unlock } from 'lucide-react'
import { useStore } from '../../store/useStore'
import * as api from '../../lib/api'
import type { TtsConfigOut, TtsStatus } from '../../types'
import { InlineFold } from './Fold'

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
export function TtsSection() {
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
