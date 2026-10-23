import { fileURLToPath } from 'node:url'
import { defineConfig, loadEnv, type ProxyOptions } from 'vite'
import react from '@vitejs/plugin-react'

// Vite 的 configure 有多个重载，手写结构类型对不上；直接从它自己的签名里取参数类型。
type ProxyServer = Parameters<NonNullable<ProxyOptions['configure']>>[0]

// 端口只在**仓库根 .env** 里定义一次（API_PORT / WEB_PORT），这里读同一个文件，
// 不再复制第二份数字 —— 否则「后端改了、前端还往 8000 打」就是半迁移。
// 用 import.meta.url 而不是 process.cwd()：start.ps1 会把工作目录设成 front/，
// 但 `npm --prefix` / IDE 直接起 vite 时 cwd 未必，按配置文件自身位置找父目录才稳。
const repoRoot = fileURLToPath(new URL('..', import.meta.url))

// 开发代理：前端统一走 /api 前缀，转发到本机 FastAPI。
// 跨机访问时用环境变量临时覆盖，例如：WUWA_API=http://192.168.1.10:28000 npm run dev
// 兜底值与 .env 当前值保持一致：clone 后没配 .env 时也要得到文档里那个端口，
// 而不是 Vite 默认的 3000 / node 惯例的 8000。
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, repoRoot, '')
  const target = process.env.WUWA_API || `http://127.0.0.1:${env.API_PORT || '28000'}`
  const webPort = Number(env.WEB_PORT || '25173')

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
      port: webPort,
      // 默认只绑回环，且**刻意不写 0.0.0.0**：dev server 会把源码与 /api 代理原样发给
      // 任何能连上它的人，局域网里不该默认有这个口子。
      // ⚠️ 一个已知的不对称：`localhost` 在本机解析到的是 IPv6 回环，实测只有 [::1]:25173
      // 在听，`curl http://127.0.0.1:25173` 直接连接被拒（浏览器无感，它会两个都试）。
      // 后端 FastAPI 是两个回环都绑的，所以「API 用 127.0.0.1 通、前端用 127.0.0.1 不通」
      // 会误导排查。Vite 的 host 只能给一个地址，Node 也没有「同时绑两个回环但不绑网卡」的
      // 写法，因此不做默认改动；要用 127.0.0.1 或跨机访问时显式开：
      //   WUWA_WEB_HOST=0.0.0.0 npm run dev   （会暴露到局域网，仅临时调试）
      host: process.env.WUWA_WEB_HOST || 'localhost',
      // 必须 strictPort。Vite 默认在端口被占时**静默**改用下一个端口，而浏览器的
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
