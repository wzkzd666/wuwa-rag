import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Plus, MessageSquare, Trash2, Pencil, Check, X, Sparkles, Library, Swords, Gem, Coins,
  PanelLeftClose, PanelLeftOpen,
} from 'lucide-react'
import { useStore } from '../store/useStore'
import MessageBubble from '../components/MessageBubble'
import Composer from '../components/Composer'
import ConfirmButton from '../components/ConfirmButton'
import './ChatPage.css'

const EXAMPLES = [
  { icon: Gem, text: '卡卡罗毕业配装用什么声骸？', tag: '配装' },
  { icon: Coins, text: '忌炎六阶突破要多少贝币？', tag: '突破' },
  { icon: Swords, text: '长离的共鸣链效果是什么？', tag: '共鸣链' },
  { icon: Sparkles, text: '今汐怎么玩？', tag: '攻略' },
]

/** 左侧会话列表（数据来自服务端，按登录用户隔离） */
function SessionList() {
  // 会话栏**独立**于主侧栏折叠：主侧栏收起时它还能继续收窄，
  // 两级都收起时聊天区拿到整屏宽度 —— 窄屏（小窗口/分屏）下这是刚需。
  const collapsed = useStore((s) => s.settings.sessionListCollapsed)
  const convs = useStore((s) => s.convs)
  const loadingConvs = useStore((s) => s.loadingConvs)
  const activeThreadId = useStore((s) => s.activeThreadId)
  const select = useStore((s) => s.selectConversation)
  const create = useStore((s) => s.newConversation)
  const setSettings = useStore((s) => s.setSettings)
  const remove = useStore((s) => s.deleteConversation)
  const rename = useStore((s) => s.renameConversation)
  const busy = useStore((s) => s.busy)
  const navigate = useNavigate()

  const [editing, setEditing] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  // 收录入口仅管理员可见（后端 /ingest 也是 admin 守卫）
  const isAdmin = useStore((s) => s.auth?.role === 'admin')

  const startEdit = (id: string, title: string) => {
    setEditing(id)
    setDraft(title)
  }
  const commit = (id: string) => {
    void rename(id, draft)
    setEditing(null)
  }

  return (
    <div className={`session-list ${collapsed ? 'rail' : ''}`}>
      {/* 会话列表的收纳开关：贴在会话栏**顶部**、新建对话之上 ——
          收纳哪个面板的按钮就贴着那个面板的顶端（与侧栏里那个「导航」开关是同一规律）。
          折叠态由 `.session-list.rail button` 统一收成 32px 图标，说明改走 data-tip。 */}
      <button
        className="session-toggle"
        onClick={() => setSettings({ sessionListCollapsed: !collapsed })}
        title={collapsed ? '展开会话列表' : '收起会话列表'}
        data-tip={collapsed ? '展开会话列表' : '收起会话列表'}
        aria-label={collapsed ? '展开会话列表' : '收起会话列表'}
        aria-expanded={!collapsed}
      >
        {collapsed ? <PanelLeftOpen size={15} /> : <PanelLeftClose size={15} />}
        <span className="session-btn-text">会话列表</span>
      </button>

      <div className="session-head">
        {!collapsed && (
          <button className="btn btn-primary new-chat" onClick={() => create()}>
            <Plus size={15} /> 新建对话
          </button>
        )}
      </div>
      {collapsed && (
        <button
          className="btn btn-primary new-chat new-chat-rail"
          onClick={() => create()}
          data-tip="新建对话"
          aria-label="新建对话"
        >
          <Plus size={16} />
        </button>
      )}
      <div className="session-scroll">
        {/* 加载中不许显示「暂无会话」：那是**空状态**，会把「请求还没回来」说成「你没有会话」。
            loadingConvs/loadingMsgs 之前只写不读，这个区分就一直没生效（见 useStore:335/359）。 */}
        {convs.length === 0 && (
          <div className="session-empty">
            {loadingConvs ? (
              <span className="dots">
                <i />
                <i />
                <i />
              </span>
            ) : (
              '暂无会话'
            )}
          </div>
        )}
        {convs.map((c) => (
          <div
            key={c.thread_id}
            tabIndex={0}
            aria-current={c.thread_id === activeThreadId ? 'true' : undefined}
            className={`session-item ${c.thread_id === activeThreadId ? 'session-active' : ''}`}
            // ⚠️ 不能用 `role="button"`：按钮的子孙在无障碍树里是**呈现性**的，
            // 一加上去，行里的改名输入框与删除/重命名按钮就对读屏彻底消失了——
            // 键盘可达性买回来了，读屏可达性反而丢得更多。
            // 保持 div + tabIndex + 下面的 Enter/Space 处理即可（行本身仍可键盘操作）。
            onClick={(e) => {
              // ⚠️ 判据不能写成 `e.target !== e.currentTarget` —— 行是 flex 容器，
              // 面积几乎全被 `.session-icon` / `.session-title`(flex:1) 占满，
              // 那样写等于「只有那 8px 内边距能点」，点标题根本不切会话（实测复核抓到）。
              // 真正要挡的是行内的操作区（改名输入框、重命名/删除按钮），按**容器**判：
              // 它们的 stopPropagation 本来也已经挡了冒泡，这里是第二道闸。
              // 键盘侧不同：焦点落在行本身才算数（见 onKeyDown），那里的 target 判据是对的。
              const t = e.target as HTMLElement
              if (t.closest('.session-ops, .session-edit')) return
              if (!busy) void select(c.thread_id)
            }}
            onKeyDown={(e) => {
              // 原来这是个只有 onClick 的 div —— 切会话是主导航，键盘/读屏完全够不着。
              // 只认焦点落在「行本身」的按键：行里嵌着改名输入框和操作按钮，它们的
              // Enter 会冒泡上来，不挡住就会「敲回车改名 → 顺手又切了一次会话」。
              if (e.target !== e.currentTarget) return
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault() // 空格默认会把滚动容器推到底
                if (!busy) void select(c.thread_id)
              }
            }}
          >
            <MessageSquare size={14} className="session-icon" />
            {editing === c.thread_id ? (
              <span className="session-edit" onClick={(e) => e.stopPropagation()}>
                <input
                  autoFocus
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') commit(c.thread_id)
                    if (e.key === 'Escape') setEditing(null)
                  }}
                />
                <button onClick={() => commit(c.thread_id)} title="保存">
                  <Check size={13} />
                </button>
                <button onClick={() => setEditing(null)} title="取消">
                  <X size={13} />
                </button>
              </span>
            ) : (
              <>
                <span className="session-title">{c.title}</span>
                <span className="session-ops" onClick={(e) => e.stopPropagation()}>
                  <button title="重命名" onClick={() => startEdit(c.thread_id, c.title)}>
                    <Pencil size={12} />
                  </button>
                  {/* 待确认文案刻意比别页短：会话栏只有 248px，
                      「再点一次确认删除」会把标题挤到只剩几个字。 */}
                  <ConfirmButton
                    icon={Trash2}
                    iconSize={12}
                    title="删除"
                    confirmLabel="确认删除"
                    onConfirm={() => void remove(c.thread_id)}
                  />
                </span>
              </>
            )}
          </div>
        ))}
      </div>
      {isAdmin && (
        <button
          className="btn btn-ghost goto-knowledge"
          onClick={() => navigate('/knowledge')}
          data-tip="收录新角色"
          aria-label="收录新角色"
        >
          <Library size={14} /> <span className="session-btn-text">收录新角色</span>
        </button>
      )}
    </div>
  )
}

export default function ChatPage() {
  const messages = useStore((s) => s.messages)
  const busy = useStore((s) => s.busy)
  const send = useStore((s) => s.send)
  const stop = useStore((s) => s.stop)
  const regenerate = useStore((s) => s.regenerate)
  const health = useStore((s) => s.health)
  const loadingMsgs = useStore((s) => s.loadingMsgs)
  const avatarAssistant = useStore((s) => s.settings.avatarAssistant)
  const navigate = useNavigate()

  const scrollRef = useRef<HTMLDivElement>(null)

  // 新消息自动滚到底部
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
  }, [messages.length, messages[messages.length - 1]?.content])

  // 监听侧栏「收录新角色」跳转
  useEffect(() => {
    const handler = (e: Event) => navigate((e as CustomEvent).detail)
    window.addEventListener('wuwa-nav', handler)
    return () => window.removeEventListener('wuwa-nav', handler)
  }, [navigate])

  const isEmpty = messages.length === 0

  return (
    <div className="chat-page">
      <SessionList />

      <div className="chat-main">
        <div className="chat-scroll" ref={scrollRef}>
          {isEmpty && loadingMsgs ? (
            /* 切会话时消息还在路上：先给三个点，别把欢迎页当成「这个会话是空的」 */
            <div className="empty-state">
              <span className="dots">
                <i />
                <i />
                <i />
              </span>
            </div>
          ) : isEmpty ? (
            <div className="chat-welcome">
              {/* 标题与副标题收进一块带底色的面板：自定义背景图下，气泡/卡片外的
                  文字直接压在图片上，深色图里几乎读不出来（见 ChatPage.css .welcome-card）。 */}
              <div className="welcome-card">
                <div className="welcome-logo">
                  {avatarAssistant ? <img src={avatarAssistant} alt="爱弥斯" /> : <Sparkles size={26} />}
                </div>
                <h1>
                  你好，我是<b className="grad-text">爱弥斯</b>
                </h1>
                <p className="welcome-sub">
                  鸣潮角色知识助手 · 声骸配装 / 突破材料 / 共鸣链 / 角色攻略，问我吧
                </p>
                {health === 'down' && (
                  <div className="welcome-warn">
                    后端服务未连接。请确认后端已启动，或在「设置」中检查服务地址。
                  </div>
                )}
              </div>
              <div className="example-grid">
                {EXAMPLES.map(({ icon: Icon, text, tag }) => (
                  <button key={text} className="example-card" onClick={() => void send(text)}>
                    <span className="example-icon">
                      <Icon size={16} />
                    </span>
                    <span className="example-text">{text}</span>
                    <span className="tag">{tag}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="msg-list">
              {messages.map((m, i) => (
                <MessageBubble
                  key={m.listKey ?? m.id}
                  msg={m}
                  canRegenerate={m.role === 'assistant' && i === messages.length - 1 && !busy}
                  onRegenerate={regenerate}
                />
              ))}
            </div>
          )}
        </div>

        <Composer
          onSend={send}
          onStop={stop}
          busy={busy}
          placeholder={health === 'down' ? '后端离线，仍可输入（将报错）…' : undefined}
        />
      </div>
    </div>
  )
}
