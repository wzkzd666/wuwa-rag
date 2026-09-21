import { useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { MessageSquare, Library, History, Settings, Waves, Menu, X } from 'lucide-react'
import { useStore } from '../store/useStore'
import './Layout.css'

const NAV = [
  { to: '/', label: '问答', icon: MessageSquare, end: true, title: '智能问答' },
  { to: '/knowledge', label: '知识库', icon: Library, title: '角色知识库' },
  { to: '/history', label: '历史', icon: History, title: '历史会话' },
  { to: '/settings', label: '设置', icon: Settings, title: '设置' },
]

export default function Layout() {
  const health = useStore((s) => s.health)
  const [open, setOpen] = useState(false)
  const loc = useLocation()
  const current = NAV.find((n) => (n.end ? loc.pathname === n.to : loc.pathname.startsWith(n.to)))

  return (
    <div className="layout">
      <aside className={`sidebar ${open ? 'sidebar-open' : ''}`}>
        <div className="brand">
          <div className="brand-logo">
            <Waves size={20} />
          </div>
          <div className="brand-text">
            <b className="grad-text">潮声智库</b>
            <span>鸣潮角色知识助手</span>
          </div>
        </div>

        <nav className="nav">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
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
          <button className="btn btn-icon menu-btn" onClick={() => setOpen((v) => !v)}>
            {open ? <X size={18} /> : <Menu size={18} />}
          </button>
          <div className="topbar-title">{current?.title ?? '潮声智库'}</div>
          <div className="topbar-right">
            <span className={`health-pill health-${health}`}>
              <span className="health-dot" />
              {health === 'ok' ? '在线' : health === 'down' ? '离线' : '…'}
            </span>
          </div>
        </header>
        <main className="content">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
