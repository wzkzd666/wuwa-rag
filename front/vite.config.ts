import { defineConfig, type ProxyOptions } from 'vite'
import react from '@vitejs/plugin-react'

// Vite 的 configure 有多个重载，手写结构类型对不上；直接从它自己的签名里取参数类型。
type ProxyServer = Parameters<NonNullable<ProxyOptions['configure']>>[0]

// 开发代理：前端统一走 /api 前缀，转发到本机 FastAPI（127.0.0.1:8000）
// 后端地址可用环境变量 WUWA_API 覆盖，例如：WUWA_API=http://192.168.1.10:8000 npm run dev
export default defineConfig(() => {
  const target = process.env.WUWA_API || 'http://127.0.0.1:8000'

  // 后端没起时，代理层会对**每个**请求各打一行红字（ECONNREFUSED）——一次页面加载就是
  // 5 条（/health、/auth/me、/conversations、/knowledge/characters、/ingest/records），
  // 浏览器控制台再叠一份同样的红字，非常刷屏。这里做两件事：
  //   ① 首次仍然打出来（保留可诊断性，不能把真错误藏起来）；
  //   ② 之后的同类错误在 5 秒内合并成一行「已合并 N 条」。
  // 目的是「不刷屏」，不是「不报错」——真出问题时第一条一定看得见。
  const recentErrors = new Map<string, { count: number; until: number }>()
  const MERGE_WINDOW = 5000

  const handleProxyError = (proxy: ProxyServer) => {
    proxy.on('error', (err) => {
      const code = (err as NodeJS.ErrnoException).code
      const key = code || err.message
      const now = Date.now()
      const prev = recentErrors.get(key)
      if (prev && prev.until > now) {
        prev.count += 1
        return
      }
      if (prev) {
        console.warn(
          `[代理] ${target} ${code || ''}（同窗口已合并 ${prev.count} 条重复）`.trim(),
        )
      }
      recentErrors.set(key, { count: 0, until: now + MERGE_WINDOW })
      console.warn(`[代理] 无法连接后端 ${target}：${code || err.message}（后续同类错误将在 5 秒内合并）`)
    })
  }

  return {
    plugins: [react()],
    server: {
      port: 5173,
      // 必须 strictPort。Vite 默认在端口被占时**静默**改用 5174/5175，而浏览器的
      // localStorage 是按 origin 隔离的 —— 端口一变就等于换了一套存储，已登录的用户
      // 会「莫名其妙被要求重新登录」，且没有任何报错可供排查。
      // 宁可启动失败并明确报端口占用，也不要悄悄漂到另一个 origin。
      strictPort: true,
      proxy: {
        '/api': {
          target,
          changeOrigin: true,
          rewrite: (p) => p.replace(/^\/api/, ''),
          configure: handleProxyError,
        },
      },
    },
    build: {
      outDir: 'dist',
      sourcemap: false,
    },
  }
})
