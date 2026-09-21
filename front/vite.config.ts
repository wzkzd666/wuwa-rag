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
