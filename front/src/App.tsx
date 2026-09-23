import { useEffect } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Background from './components/Background'
import ChatPage from './pages/ChatPage'
import KnowledgePage from './pages/KnowledgePage'
import HistoryPage from './pages/HistoryPage'
import SettingsPage from './pages/SettingsPage'
import AuthPage from './pages/AuthPage'
import Toasts from './components/Toasts'
import { useStore, ensureActiveConversation } from './store/useStore'
import * as api from './lib/api'

export default function App() {
  const settings = useStore((s) => s.settings)
  const checkHealth = useStore((s) => s.checkHealth)
  const auth = useStore((s) => s.auth)
  const clearAuth = useStore((s) => s.clearAuth)
  const toast = useStore((s) => s.toast)

  // 应用主题与字号
  useEffect(() => {
    document.documentElement.dataset.theme = settings.theme
    document.documentElement.style.fontSize = settings.fontSize + 'px'
  }, [settings.theme, settings.fontSize])

  // 启动时把 persist 恢复的 token 接回 api 层，并校验是否仍有效
  // （30 天过期 / 后端重启清库 → 静默登出，不做多余弹窗）
  useEffect(() => {
    api.setAuthToken(auth?.token ?? '')
    if (!auth) return
    api.me(settings.apiBase).catch((err) => {
      if (err instanceof api.UnauthorizedError) {
        clearAuth()
        toast('info', '登录已过期，请重新登录')
      }
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // 启动时保证有会话 + 探测后端连通性（每 30s 一次）
  useEffect(() => {
    ensureActiveConversation()
    checkHealth()
    const t = setInterval(checkHealth, 30000)
    return () => clearInterval(t)
  }, [checkHealth])

  // 未登录 → 全屏登录/注册页（门禁；后端 /ask /ingest 等也都要求鉴权）
  if (!auth) {
    return (
      <>
        <Background />
        <AuthPage />
        <Toasts />
      </>
    )
  }

  return (
    <>
      <Background />
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<ChatPage />} />
          <Route path="/knowledge" element={<KnowledgePage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
      <Toasts />
    </>
  )
}
