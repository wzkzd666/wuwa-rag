import React from 'react'
import ReactDOM from 'react-dom/client'
import { HashRouter } from 'react-router-dom'
import App from './App'
import './styles/global.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    {/* 两个 v7 future flag 显式打开，控制台从此只剩真消息。
        为什么在本项目里是行为中立（都逐个核过，不是"看起来没事"）：
        · v7_relativeSplatPath 只改变 **splat 路由内部相对链接**的解析基准，
          而全站 `to=` 一律是绝对路径（grep `to="[^/]` 为空），唯一的 splat
          `path="*"` 渲染的是 `<Navigate to="/" replace />`，也是绝对的；
        · v7_startTransition 把路由引起的 state 更新包进 React.startTransition，
          本应用的状态在 zustand 里、导航只有 7 个扁平页面，没有会被并发过渡打断的长列表。 */}
    <HashRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <App />
    </HashRouter>
  </React.StrictMode>,
)
