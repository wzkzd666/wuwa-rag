import { useState } from 'react'
import { LogIn, UserPlus, Waves } from 'lucide-react'
import { useStore } from '../store/useStore'
import * as api from '../lib/api'
import './AuthPage.css'

/**
 * 登录 / 注册页（未登录时的全屏门禁）。
 * - admin 由后端启动时种子写入：ADMIN_PASSWORD 显式配置，留空则首启随机生成
 *   一次性口令（见 core/authdb.ensure_schema，不再有内置弱口令）；
 * - 游客走注册，注册即登录；
 * - 登录态（token）persist 在 localStorage，刷新不掉线，30 天过期。
 */
export default function AuthPage() {
  const setAuth = useStore((s) => s.setAuth)
  const toast = useStore((s) => s.toast)
  const apiBase = useStore((s) => s.settings.apiBase)

  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [busy, setBusy] = useState(false)

  const submit = async () => {
    const u = username.trim()
    if (!u || !password) {
      toast('err', '请输入用户名和密码')
      return
    }
    if (mode === 'register' && password !== confirm) {
      toast('err', '两次输入的密码不一致')
      return
    }
    setBusy(true)
    try {
      const out =
        mode === 'login'
          ? await api.login(u, password, apiBase)
          : await api.register(u, password, apiBase)
      setAuth({ token: out.token, username: out.username, role: out.role })
      toast('ok', `欢迎${mode === 'register' ? '' : '回来'}，${out.username}~`)
      // 部署者用 ADMIN_PASSWORD 配的初始口令属于配置文件明文，登录后引导改密。
      // 只是提示不强拦：改密入口在设置页（Layout 顶栏 → 设置 → 修改密码）。
      if (out.must_change_password) {
        toast('info', '当前仍是初始口令，建议尽快到设置页修改密码')
      }
    } catch (err) {
      toast('err', err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-card glass fade-up">
        <div className="auth-logo">
          <Waves size={22} />
        </div>
        <h1 className="grad-text">潮声智库</h1>
        <p className="auth-sub">鸣潮角色知识助手 · 请先登录</p>

        <div className="seg auth-seg">
          <button
            className={`seg-btn ${mode === 'login' ? 'seg-on' : ''}`}
            onClick={() => setMode('login')}
          >
            <LogIn size={13} /> 登录
          </button>
          <button
            className={`seg-btn ${mode === 'register' ? 'seg-on' : ''}`}
            onClick={() => setMode('register')}
          >
            <UserPlus size={13} /> 注册
          </button>
        </div>

        <input
          className="input auth-input"
          placeholder="用户名"
          value={username}
          maxLength={24}
          onChange={(e) => setUsername(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && submit()}
        />
        <input
          className="input auth-input"
          type="password"
          placeholder="密码"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && submit()}
        />
        {mode === 'register' && (
          <input
            className="input auth-input"
            type="password"
            placeholder="再输一遍密码"
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && submit()}
          />
        )}

        <button className="btn btn-primary auth-submit" onClick={submit} disabled={busy}>
          {busy ? '请稍候…' : mode === 'login' ? '登录' : '注册并登录'}
        </button>

        <p className="auth-hint">
          <UserPlus size={12} />
          新用户注册后即可问答
        </p>
      </div>
    </div>
  )
}
