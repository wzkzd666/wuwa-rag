import { useState } from 'react'
import { Settings as SettingsIcon, Sun, Moon, Zap, Type, Server, Trash2, Activity, RefreshCw } from 'lucide-react'
import { useStore } from '../store/useStore'
import './SettingsPage.css'

export default function SettingsPage() {
  const settings = useStore((s) => s.settings)
  const setSettings = useStore((s) => s.setSettings)
  const health = useStore((s) => s.health)
  const checkHealth = useStore((s) => s.checkHealth)
  const clearAll = useStore((s) => s.clearAll)
  const conversations = useStore((s) => s.conversations)
  const ingests = useStore((s) => s.ingests)
  const toast = useStore((s) => s.toast)

  const [testing, setTesting] = useState(false)
  const [armed, setArmed] = useState(false)

  const testConnection = async () => {
    setTesting(true)
    await checkHealth()
    setTesting(false)
    const ok = useStore.getState().health === 'ok'
    toast(ok ? 'ok' : 'err', ok ? '后端连接正常' : '无法连接后端，请检查服务与地址')
  }

  const totalMessages = conversations.reduce((n, c) => n + c.messages.length, 0)

  return (
    <div className="page settings-page">
      <div className="page-head">
        <div>
          <h2 className="page-title">
            <SettingsIcon size={19} className="grad-text" /> 设置
          </h2>
          <p className="page-desc">连接后端、外观与数据管理。设置保存在浏览器本地。</p>
        </div>
      </div>

      {/* 后端连接 */}
      <section className="card set-card">
        <h3 className="set-h">
          <Server size={15} /> 后端连接
        </h3>
        <div className="set-row">
          <div className="set-row-main">
            <label>API 地址</label>
            <p>留空则走开发代理 /api（转发到 127.0.0.1:8000）。跨机访问时填完整地址，如 http://192.168.1.10:8000，此时后端需允许 CORS。</p>
          </div>
          <div className="set-row-ctl">
            <input
              className="input api-input"
              value={settings.apiBase}
              placeholder="默认：/api（开发代理）"
              onChange={(e) => setSettings({ apiBase: e.target.value })}
            />
          </div>
        </div>
        <div className="set-row">
          <div className="set-row-main">
            <label>连接状态</label>
            <p>每 30 秒自动探测 GET /health。</p>
          </div>
          <div className="set-row-ctl status-ctl">
            <span className={`tag ${health === 'ok' ? 'tag-ok' : health === 'down' ? 'tag-err' : 'tag-warn'}`}>
              <Activity size={12} />
              {health === 'ok' ? '在线' : health === 'down' ? '离线' : '检测中'}
            </span>
            <button className="btn btn-ghost" onClick={testConnection} disabled={testing}>
              {testing ? <RefreshCw size={14} className="spin" /> : <Zap size={14} />} 测试连接
            </button>
          </div>
        </div>
        <div className="set-row">
          <div className="set-row-main">
            <label>流式输出</label>
            <p>开启走 SSE 逐字吐答案（/ask/stream）；关闭则一次性返回（/ask）。</p>
          </div>
          <div className="set-row-ctl">
            <button
              className={`switch ${settings.stream ? 'switch-on' : ''}`}
              onClick={() => setSettings({ stream: !settings.stream })}
              role="switch"
              aria-checked={settings.stream}
            >
              <span className="switch-knob" />
            </button>
          </div>
        </div>
      </section>

      {/* 外观 */}
      <section className="card set-card">
        <h3 className="set-h">
          <Sun size={15} /> 外观
        </h3>
        <div className="set-row">
          <div className="set-row-main">
            <label>主题</label>
            <p>深空暗色（默认）或浅色模式。</p>
          </div>
          <div className="set-row-ctl">
            <div className="seg">
              <button
                className={`seg-btn ${settings.theme === 'dark' ? 'seg-on' : ''}`}
                onClick={() => setSettings({ theme: 'dark' })}
              >
                <Moon size={13} /> 暗色
              </button>
              <button
                className={`seg-btn ${settings.theme === 'light' ? 'seg-on' : ''}`}
                onClick={() => setSettings({ theme: 'light' })}
              >
                <Sun size={13} /> 浅色
              </button>
            </div>
          </div>
        </div>
        <div className="set-row">
          <div className="set-row-main">
            <label>基础字号</label>
            <p>当前 {settings.fontSize}px，影响全局文字大小。</p>
          </div>
          <div className="set-row-ctl font-ctl">
            <input
              type="range"
              min={12}
              max={18}
              step={1}
              value={settings.fontSize}
              onChange={(e) => setSettings({ fontSize: Number(e.target.value) })}
            />
            <span className="font-val">
              <Type size={13} /> {settings.fontSize}px
            </span>
          </div>
        </div>
      </section>

      {/* 数据 */}
      <section className="card set-card">
        <h3 className="set-h">
          <Trash2 size={15} /> 数据管理
        </h3>
        <div className="set-stats">
          <div className="stat">
            <b>{conversations.length}</b>
            <span>会话</span>
          </div>
          <div className="stat">
            <b>{totalMessages}</b>
            <span>消息</span>
          </div>
          <div className="stat">
            <b>{ingests.length}</b>
            <span>入库记录</span>
          </div>
        </div>
        <div className="set-row danger-row">
          <div className="set-row-main">
            <label>清空全部数据</label>
            <p>删除所有会话、消息与入库记录，并重置为一个新对话。此操作不可恢复。</p>
          </div>
          <div className="set-row-ctl">
            <button
              className={`btn btn-danger ${armed ? 'armed' : ''}`}
              onClick={() => {
                if (armed) {
                  clearAll()
                  setArmed(false)
                  toast('info', '已清空全部本地数据')
                } else {
                  setArmed(true)
                  setTimeout(() => setArmed(false), 4000)
                }
              }}
            >
              <Trash2 size={14} /> {armed ? '确认清空' : '清空数据'}
            </button>
          </div>
        </div>
      </section>

      <div className="set-about">
        潮声智库 · 鸣潮角色知识助手前端 v1.0 · 后端 FastAPI + LangGraph 混合检索 RAG
      </div>
    </div>
  )
}
