import { useCallback, useEffect, useState } from 'react'
import { Settings as SettingsIcon, Sun, Moon, Zap, Type, Server, Trash2, Activity, RefreshCw, Camera, Image as ImageIcon, Palette, RotateCcw, User as UserIcon, Sparkles, Eraser, Cpu, Music4, Volume2, Loader2, KeyRound, Mic } from 'lucide-react'
import { useStore } from '../store/useStore'
import * as api from '../lib/api'
import { pickImageFile, fileToDataUrl, formatBytes, dataUrlBytes } from '../lib/image'
import type { UserFact } from '../types'
import type { MusicSetting } from '../lib/api'
import './SettingsPage.css'
import { BG_PRESETS } from './settings/bgPresets'
import { Fold } from './settings/Fold'
import { LlmSection } from './settings/LlmSection'
import { TtsSection } from './settings/TtsSection'
import { AccountSection } from './settings/AccountSection'

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

  // ---- 布局类设置（内容区宽度 / 基础字号）：拖拽期间不动布局，松手才提交 ----
  //
  // 为什么不能边拖边改：这两项都会改变页面的几何 —— 前者改主卡片宽度、后者改整页 zoom，
  // 而**滑块自己就长在被改动的那个容器里**。实时应用等于「一边拖、一边把滑块从光标底下
  // 挪走」：滑块被挪动 / 缩放后，同样的鼠标位移映射出的值就变了，拖到一半就脱手，
  // 主观感受就是「拖拽被打断」。所以拖动时只更新草稿 —— 手柄和右侧数值照常跟手 ——
  // 松手（pointerup / 键盘 keyup / 失焦）才写进 store。
  const [draftWidth, setDraftWidth] = useState<number | null>(null)
  const [draftFontSize, setDraftFontSize] = useState<number | null>(null)

  /**
   * 提交布局类设置。先挂上落位标记再写值，让松手那一次是从旧值**平滑**滑到新值，
   * 而不是生硬跳变（过渡规则定义在 global.css，用 :root[data-layout-anim] 限定生效范围，
   * 免得给页面加载、窗口缩放这些几何变化也带上动画）。
   */
  const commitLayout = (key: 'contentWidth' | 'fontSize', value: number | null) => {
    if (value == null) return
    const root = document.documentElement
    root.dataset.layoutAnim = '1'
    setSettings(key === 'contentWidth' ? { contentWidth: value } : { fontSize: value })
    if (key === 'contentWidth') setDraftWidth(null)
    else setDraftFontSize(null)
    window.setTimeout(() => { delete root.dataset.layoutAnim }, 340)
  }

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
            <p>留空则使用默认的本地服务地址。跨机访问时填后端完整地址，如 http://192.168.1.10:8000。</p>
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
            <p>每 30 秒自动检测一次连接状态。</p>
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
            <p>开启后逐字显示回答；关闭则一次性显示完整回答。</p>
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
              value={draftFontSize ?? settings.fontSize}
              onChange={(e) => setDraftFontSize(Number(e.target.value))}
              onPointerUp={(e) => commitLayout('fontSize', Number(e.currentTarget.value))}
              onKeyUp={(e) => commitLayout('fontSize', Number(e.currentTarget.value))}
            />
            <span className="font-val">
              <Type size={13} /> {draftFontSize ?? settings.fontSize}px
            </span>
          </div>
        </div>

        {/* 内容区宽度：标题块与所有卡片共用这一个值（用户此前反馈「宽度不一致」）。 */}
        <div className="set-row">
          <div className="set-row-main">
            <label>内容区宽度</label>
            <p>当前 {draftWidth ?? settings.contentWidth}px，标题块与所有卡片同步。</p>
          </div>
          <div className="set-row-ctl font-ctl">
            <input
              type="range"
              min={640}
              max={1120}
              step={20}
              value={draftWidth ?? settings.contentWidth}
              onChange={(e) => setDraftWidth(Number(e.target.value))}
              onPointerUp={(e) => commitLayout('contentWidth', Number(e.currentTarget.value))}
              onKeyUp={(e) => commitLayout('contentWidth', Number(e.currentTarget.value))}
            />
            <span className="font-val">{draftWidth ?? settings.contentWidth}px</span>
          </div>
        </div>

        {/* 数据区字号：用量表 / 答案反馈这些「一屏要看很多东西」的地方单独调。
            跟基础字号分开放，因为两者诉求相反：聊天区要易读，数据区要紧凑。 */}
        <div className="set-row">
          <div className="set-row-main">
            <label>数据区字号</label>
            <p>当前 {settings.dataFontSize}px，用于用量表与答案反馈。</p>
          </div>
          <div className="set-row-ctl font-ctl">
            <input
              type="range"
              min={10}
              max={16}
              step={1}
              value={settings.dataFontSize}
              onChange={(e) => setSettings({ dataFontSize: Number(e.target.value) })}
            />
            <span className="font-val">
              <Type size={13} /> {settings.dataFontSize}px
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
                  title="不再保留此项"
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
