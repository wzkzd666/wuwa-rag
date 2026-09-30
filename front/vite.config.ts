import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 开发代理：前端统一走 /api 前缀，转发到本机 FastAPI（127.0.0.1:8000）
// 后端地址可用环境变量 WUWA_API 覆盖，例如：WUWA_API=http://192.168.1.10:8000 npm run dev
export default defineConfig(() => {
  const target = process.env.WUWA_API || 'http://127.0.0.1:8000'
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
        },
      },
    },
    build: {
      outDir: 'dist',
      sourcemap: false,
    },
  }
})
