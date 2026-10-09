import { useState } from 'react'
import { RefreshCw, Check } from 'lucide-react'
import { useStore } from '../../store/useStore'
import * as api from '../../lib/api'

/**
 * 账号安全：改登录密码。
 *
 * ⚠️ 顺序敏感：后端会**先用旧密码重绑**云端 API-KEY 的密钥（DEK）再更新密码哈希。
 * 少了这一步，改完密码就再也解不开已存的 Key（只能删配置重填），所以这里必须
 * 让用户输入原密码，不能只填新密码。
 */
export function AccountSection() {
  const toast = useStore((s) => s.toast)
  const apiBase = useStore((s) => s.settings.apiBase)
  const [oldPw, setOldPw] = useState('')
  const [newPw, setNewPw] = useState('')
  const [saving, setSaving] = useState(false)

  const doChange = async () => {
    if (!oldPw) { toast('err', '请输入原密码'); return }
    if (newPw.length < 6) { toast('err', '新密码至少 6 位'); return }
    setSaving(true)
    try {
      await api.changePassword(oldPw, newPw, apiBase)
      setOldPw(''); setNewPw('')
      toast('ok', '密码已修改；已存的云端 Key 已同步重绑，下次登录照常自动连上')
    } catch (err) {
      toast('err', err instanceof Error ? err.message : String(err))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="set-row">
      <div className="set-row-main">
        <label>登录密码</label>
        <p>
          改密码时后端会先用旧密码把云端 API-KEY 的密钥重新绑定一遍，
          所以改完之后已存的 Key 依然能自动解开（不用重新填）。
        </p>
      </div>
      <div className="set-row-ctl status-ctl">
        <input className="input api-input" type="password" value={oldPw}
               placeholder="原密码" onChange={(e) => setOldPw(e.target.value)}
               autoComplete="current-password" />
        <input className="input api-input" type="password" value={newPw}
               placeholder="新密码（≥6 位）" onChange={(e) => setNewPw(e.target.value)}
               autoComplete="new-password" />
        <button className="btn btn-ghost" onClick={doChange} disabled={saving}>
          {saving ? <RefreshCw size={14} className="spin" /> : <Check size={14} />} 修改
        </button>
      </div>
    </div>
  )
}
