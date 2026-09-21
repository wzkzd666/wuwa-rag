import { useEffect } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import ChatPage from './pages/ChatPage'
import KnowledgePage from './pages/KnowledgePage'
import HistoryPage from './pages/HistoryPage'
import SettingsPage from './pages/SettingsPage'
import Toasts from './components/Toasts'
import { useStore, ensureActiveConversation } from './store/useStore'

export default function App() {
  const settings = useStore((s) => s.settings)
  const checkHealth = useStore((s) => s.checkHealth)

  // 应用主题与字号
  useEffect(() => {
    document.documentElement.dataset.theme = settings.theme
    document.documentElement.style.fontSize = settings.fontSize + 'px'
  }, [settings.theme, settings.fontSize])

  // 启动时保证有会话 + 探测后端连通性（每 30s 一次）
  useEffect(() => {
    ensureActiveConversation()
    checkHealth()
    const t = setInterval(checkHealth, 30000)
    return () => clearInterval(t)
  }, [checkHealth])

  return (
    <>
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
