import { useEffect, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import {
  MessageSquare,
  Library,
  History,
  Settings,
  Waves,
  Menu,
  X,
  PanelLeftClose,
  PanelLeftOpen,
  Camera,
  LogOut,
  ShieldCheck,
  User as UserIcon,
  BarChart3,
} from 'lucide-react'
import { useStore } from '../store/useStore'
import MusicBar from './MusicBar'
import { pickImageFile, fileToDataUrl, formatBytes, dataUrlBytes } from '../lib/image'
import './Layout.css'

const NAV = [
  { to: '/', label: '问答', icon: MessageSquare, end: true, title: '智能问答' },
  { to: '/knowledge', label: '知识库', icon: Library, title: '角色知识库' },
  { to: '/usage', label: '用量', icon: BarChart3, title: 'token 用量' },
  { to: '/feedback', label: '反馈', icon: MessageSquare, title: '答案反馈' },
  { to: '/history', label: '历史', icon: History, title: '历史会话' },
  { to: '/settings', label: '设置', icon: Settings, title: '设置' },
]

/** 与 Layout.css 里抽屉模式的媒体查询保持一致 */
const NARROW_QUERY = '(max-width: 860px)'

/** 视口是否处于「移动端抽屉」档位 */
function useNarrow(): boolean {
  const [narrow, setNarrow] = useState(
    () => typeof window !== 'undefined' && window.matchMedia(NARROW_QUERY).matches,
  )
  useEffect(() => {
    const mq = window.matchMedia(NARROW_QUERY)
    const onChange = () => setNarrow(mq.matches)
    onChange()
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [])
  return narrow
}

export default function Layout() {
  const health = useStore((s) => s.health)
  const collapsed = useStore((s) => s.settings.sidebarCollapsed)
  const avatarAssistant = useStore((s) => s.settings.avatarAssistant)
  const setSettings = useStore((s) => s.setSettings)
  const toast = useStore((s) => s.toast)
  const auth = useStore((s) => s.auth)
  const clearAuth = useStore((s) => s.clearAuth)

  const logout = async () => {
    try {
      await import('../lib/api').then((api) => api.logout())
    } catch {
      /* 后端不可达也照样本地登出 */
    }
    clearAuth()
    toast('info', '已退出登录')
  }

  // 移动端抽屉开合（桌面端不用它，改用持久化的 sidebarCollapsed）
  const [open, setOpen] = useState(false)
  const narrow = useNarrow()

  const loc = useLocation()
  const current = NAV.find((n) => (n.end ? loc.pathname === n.to : loc.pathname.startsWith(n.to)))

  // 落在窄屏时，把桌面折叠态清掉，避免媒体查询切换时留个「半折叠」的怪状态
  useEffect(() => {
    if (!narrow) setOpen(false)
  }, [narrow])

  const toggleNav = () => {
    if (narrow) setOpen((v) => !v)
    else setSettings({ sidebarCollapsed: !collapsed })
  }

  const toggleIcon = narrow
    ? open
      ? <X size={18} />
      : <Menu size={18} />
    : collapsed
      ? <PanelLeftOpen size={18} />
      : <PanelLeftClose size={18} />

  const toggleLabel = narrow
    ? open
      ? '关闭菜单'
      : '打开菜单'
    : collapsed
      ? '展开侧边栏'
      : '折叠侧边栏'

  /** 侧栏 logo 即头像替换入口：点一下直接选图 */
  const changeAssistantAvatar = async () => {
    const file = await pickImageFile()
    if (!file) return
    try {
      const dataUrl = await fileToDataUrl(file, { max: 256, square: true, quality: 0.92 })
      setSettings({ avatarAssistant: dataUrl })
      toast('ok', `助手头像已更新（${formatBytes(dataUrlBytes(dataUrl))}）`)
    } catch (err) {
      toast('err', '头像处理失败：' + (err instanceof Error ? err.message : String(err)))
    }
  }

  return (
    <div className="layout">
      <aside className={`sidebar ${open ? 'sidebar-open' : ''} ${collapsed ? 'rail' : ''}`}>
        <div className="brand">
          <button
            type="button"
            className="brand-logo-btn"
            onClick={changeAssistantAvatar}
            title="更换助手头像"
            aria-label="更换助手头像"
          >
            <span className="brand-logo">
              {avatarAssistant ? <img src={avatarAssistant} alt="助手头像" /> : <Waves size={20} />}
            </span>
            <span className="brand-badge" aria-hidden>
              <Camera size={11} />
            </span>
          </button>
          <div className="brand-text">
            <b className="grad-text">潮声智库</b>
            <span>鸣潮角色知识助手</span>
          </div>
        </div>

        {/* 侧栏**自身**的收纳按钮：落在「潮声智库」那一行的**下方**、导航之上 ——
            贴着被收纳内容的顶端，点之前就知道会收起谁。
            ⚠️ 不能塞进 .brand 与 logo 同行：折叠成图标窄栏（rail）后侧栏只有 64px，
            38px 的 logo 已把那一行占满；独立成行，窄栏下图标才有地方居中。 */}
        <button
          className="sidebar-fold-btn"
          onClick={toggleNav}
          title={toggleLabel}
          aria-label={toggleLabel}
          aria-expanded={narrow ? open : !collapsed}
        >
          {toggleIcon}
          <span>导航</span>
        </button>

        <nav className="nav">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              title={label}
              className={({ isActive }) => `nav-item ${isActive ? 'nav-active' : ''}`}
              onClick={() => setOpen(false)}
            >
              <Icon size={17} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-foot">
          <span className={`health-dot health-${health}`} />
          <span className="health-text">
            {health === 'ok' ? '后端在线' : health === 'down' ? '后端离线' : '检测中…'}
          </span>
        </div>
      </aside>

      {open && <div className="sidebar-mask" onClick={() => setOpen(false)} />}

      <div className="main">
        <header className="topbar">
          {/* 顶栏最左只在**窄屏档**放按钮：那时侧栏是抽屉、平时整块藏在屏幕外，
              开合它的按钮必须在抽屉外面，否则抽屉一关就再也打不开了。
              桌面档这里不放按钮 —— 侧栏常驻可见，它自己的收纳按钮落在侧栏「导航」之上；
              会话列表的收纳开关则归会话栏自己（见 ChatPage 的 .session-toggle）。 */}
          {narrow && (
            <button
              className="btn btn-icon menu-btn"
              onClick={toggleNav}
              title={toggleLabel}
              aria-label={toggleLabel}
              aria-expanded={open}
            >
              {toggleIcon}
            </button>
          )}
          <div className="topbar-title">{current?.title ?? '潮声智库'}</div>
          {/* 音乐播放条：只在真的有歌在放时出现（组件内部按 available/title 决定） */}
          <MusicBar />
          <div className="topbar-right">
            <span className={`health-pill health-${health}`}>
              <span className="health-dot" />
              {health === 'ok' ? '在线' : health === 'down' ? '离线' : '…'}
            </span>
            {auth && (
              <span className="user-pill" title={auth.role === 'admin' ? '管理员' : '游客'}>
                {auth.role === 'admin' ? <ShieldCheck size={13} /> : <UserIcon size={13} />}
                {auth.username}
                <button className="user-logout" title="退出登录" onClick={logout}>
                  <LogOut size={13} />
                </button>
              </span>
            )}
          </div>
        </header>
        <main className="content">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
