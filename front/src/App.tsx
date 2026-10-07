import { useEffect, useLayoutEffect } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Background from './components/Background'
import ChatPage from './pages/ChatPage'
import KnowledgePage from './pages/KnowledgePage'
import UsagePage from './pages/UsagePage'
import FeedbackPage from './pages/FeedbackPage'
import HistoryPage from './pages/HistoryPage'
import SettingsPage from './pages/SettingsPage'
import AuthPage from './pages/AuthPage'
import Toasts from './components/Toasts'
import { useStore } from './store/useStore'
import * as api from './lib/api'

// 字号基准：与 global.css 里 body 的 font-size:14px、以及 DEFAULT_SETTINGS.fontSize 对齐。
// 设置页滑块存的是「期望的基准字号」，在这里换算成整体缩放倍率。
const BASE_FONT_SIZE = 14

/**
 * 把外观设置写成 :root 上的 CSS 变量。
 *
 * 为什么独立成组件、而不是写在 App 里：App 一旦订阅 `settings` **整个对象**，
 * 改其中任意一个字段（哪怕只是拖「内容区宽度」滑块）都会重渲染整棵路由树
 * （App → Routes → Layout → 当前页），而这里真正要做的不过是把几个值写进 :root。
 * 拆出来之后，拖滑块时重渲染的只剩这个 `return null` 的组件，页面树完全不动。
 */
function ThemeVars() {
  const theme = useStore((s) => s.settings.theme)
  const fontSize = useStore((s) => s.settings.fontSize)
  const dataFontSize = useStore((s) => s.settings.dataFontSize)
  const contentWidth = useStore((s) => s.settings.contentWidth)
  const panelAlpha = useStore((s) => s.settings.panelAlpha)
  const panelBlur = useStore((s) => s.settings.panelBlur)

  // 应用主题与字号
  //
  // 字号走「整体 zoom 缩放」。不要改回「设置根字号」的老写法：全站 10 个 CSS 文件共 75 处
  // font-size（70 个 px + 5 个 em），没有任何一处用 rem，body 自身就写着 font-size:14px，
  // 所以改 documentElement.style.fontSize 不会级联到任何元素 —— 设置页的字号滑块拖了没反应
  // 就是这个原因。
  //
  // 实测（Edge，视口 1200×800）：zoom 不影响 vh/vw，zoom=1.4 时 100vh 元素实测 1120px、
  // 100vw 实测 1680px，所以 CSS 里所有 viewport 单位都统一写成 calc(X / var(--ui-zoom)) 抵消，
  // 否则放大后底部和右侧会被裁掉。zoom 也不会给 position: fixed 建立包含块（fixed 元素照旧贴边）。
  //
  // 已知边界：zoom 对媒体查询 / window.matchMedia 不可见 —— 窗口宽度 < ~920px 且字号 ≥ 16px 时，
  // 窄屏断点（Layout 抽屉、ChatPage 内边距）不会触发。桌面端窗口远宽于此，先接受。
  //
  // 若将来把 CSS 全量迁到 rem，可以换回根字号方案；两者不能同时启用，否则字号会被放大两次。
  //
  // 用 useLayoutEffect 而非 useEffect：它在浏览器绘制前跑，非默认字号的用户不会看到
  // 「先是 100% 再跳成缩放后」的一帧闪烁。
  useLayoutEffect(() => {
    const root = document.documentElement
    const uiZoom = fontSize / BASE_FONT_SIZE
    root.dataset.theme = theme
    root.style.setProperty('zoom', String(uiZoom))
    root.style.setProperty('--ui-zoom', String(uiZoom))
    // 数据区字号（用量表 / 答案反馈）：单独一个变量，不跟全局 zoom 混在一起 ——
    // 那两处要的是「一屏多看几行」，与聊天区的「易读」是相反诉求。
    root.style.setProperty('--data-font-size', `${dataFontSize}px`)
    // 页面内容区宽度：页面标题块与所有主卡片共用同一个值（global.css 的 --content-max）。
    // 这是**唯一**的宽度入口 —— 任何页面都不许再自己写一套 max-width，否则会以同等特异性
    // 覆盖它，表现为「设置里调了宽度、卡片纹丝不动」。
    root.style.setProperty('--content-max', `${contentWidth}px`)
    // 面板不透明度 / 磨砂（卡片与页面标题块统一走 --panel-* 三个变量，见 global.css）
    root.style.setProperty('--panel-alpha', String(panelAlpha))
    root.style.setProperty('--panel-blur', `${panelBlur}px`)
  }, [theme, fontSize, dataFontSize, contentWidth, panelAlpha, panelBlur])

  return null
}

export default function App() {
  const checkHealth = useStore((s) => s.checkHealth)
  const auth = useStore((s) => s.auth)
  const clearAuth = useStore((s) => s.clearAuth)
  const loadConversations = useStore((s) => s.loadConversations)
  const toast = useStore((s) => s.toast)

  // 启动时把 persist 恢复的 token 接回 api 层，并校验是否仍有效
  // （30 天过期 / 后端重启清库 → 静默登出，不做多余弹窗）
  //
  // 这里用 getState() 取值而不是订阅：只在挂载时读一次，没必要为它保持订阅 ——
  // 订阅了就会让 App 随任意设置变化重渲染（那正是拖滑块卡顿的来源）。
  useEffect(() => {
    api.setAuthToken(auth?.token ?? '')
    if (!auth) return
    api.me(useStore.getState().settings.apiBase).catch((err) => {
      if (err instanceof api.UnauthorizedError) {
        clearAuth()
        toast('info', '登录已过期，请重新登录')
      }
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // 已登录时拉一次会话列表（会话正文存在服务端、按用户隔离，不再是本地缓存），
  // 并探测后端连通性（每 30s 一次）
  useEffect(() => {
    if (auth) void loadConversations()
    checkHealth()
    const t = setInterval(checkHealth, 30000)
    return () => clearInterval(t)
  }, [auth, loadConversations, checkHealth])

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
      <ThemeVars />
      <Background />
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<ChatPage />} />
          <Route path="/knowledge" element={<KnowledgePage />} />
          <Route path="/usage" element={<UsagePage />} />
          <Route path="/feedback" element={<FeedbackPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
      <Toasts />
    </>
  )
}
