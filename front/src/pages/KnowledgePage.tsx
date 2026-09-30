import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ChevronLeft, ChevronRight, CheckCircle2, Clock, Download, Info, Library, Loader2, RefreshCw,
  Search, Trash2, XCircle, Zap,
} from 'lucide-react'
import { useStore } from '../store/useStore'
import {
  ingestStatus, knowledgeCharacters, knowledgeDelete, knowledgeRefresh,
} from '../lib/api'
import type { IngestRecord, IngestStatus, KnowledgeOut } from '../types'
import './KnowledgePage.css'

/**
 * 知识库页 = 「我现在有什么」（已收录列表）+ 「怎么加新的」（提交表单）。
 *
 * ⚠️ 这里**不再有**角色名册常量。改造前前端存着一份与后端 CHARACTER_NAMES
 * 对齐的 ROSTER 数组，加新角色要改两个地方、必然不同步；现在候选名册由
 * `GET /knowledge/characters` 的 `seeded_only` 下发，唯一来源在后端。
 */

const PIPELINE_STEPS = [
  { name: '抓取', desc: 'wuwa-mcp 抓鸣潮 wiki 原文' },
  { name: '分块', desc: '结构感知分块 chunks.jsonl' },
  { name: '入库', desc: '写入 PostgreSQL + S3' },
  { name: '索引', desc: 'BM25 稀疏 + Chroma 稠密' },
  { name: '图谱', desc: '正则抽事实 → Neo4j' },
]

/** 来源标识 -> 展示名。后端 source 目前恒为 `kurobbs`（鸣潮 WIKI）。 */
const SOURCE_LABELS: Record<string, string> = { kurobbs: '鸣潮 WIKI' }

/** 已收录列表每页条数。角色会随库增长（目前 57+），不分页表格会一路拉长。 */
const PAGE_SIZE = 20

function fmtSize(n: number | null): string {
  if (!n) return '—'
  return n >= 1024 ? `${(n / 1024).toFixed(1)} KB` : `${n} B`
}

function fmtTime(iso: string | null): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '—' : d.toLocaleString('zh-CN', { hour12: false })
}

/** 提交记录「状态」列：优先渲染后端实时五步进度，回落到提交回执 */
function renderStatus(r: IngestRecord, st?: IngestStatus) {
  if (r.ok && st && st.found && st.status !== 'pending') {
    return (
      <div className="step-bar" title={st.steps.map((s) => `${s.label}:${s.status}${s.error ? ' ' + s.error : ''}`).join('\n')}>
        {st.steps.map((s) => (
          <span key={s.key} className={`step-dot step-${s.status}`}>
            {s.status === 'running' ? (
              <Loader2 size={10} className="spin" />
            ) : s.status === 'success' ? (
              <CheckCircle2 size={10} />
            ) : s.status === 'failed' ? (
              <XCircle size={10} />
            ) : (
              <Clock size={10} />
            )}
            <em>{s.label}</em>
          </span>
        ))}
        <span className={`tag ${st.status === 'success' ? 'tag-ok' : st.status === 'failed' ? 'tag-err' : 'tag-violet'}`}>
          {st.status === 'success' ? '完成' : st.status === 'failed' ? '失败' : '进行中'}
        </span>
      </div>
    )
  }
  return r.ok ? (
    <span className="tag tag-violet">
      <Loader2 size={11} className="spin" /> 排队中
    </span>
  ) : (
    <span className="tag tag-err" title={r.error}>
      <XCircle size={11} /> {r.state}
    </span>
  )
}

export default function KnowledgePage() {
  const ingests = useStore((s) => s.ingests)
  const ingestCharacter = useStore((s) => s.ingestCharacter)
  const health = useStore((s) => s.health)
  const apiBase = useStore((s) => s.settings.apiBase)
  const toast = useStore((s) => s.toast)
  // 写入类操作（提交/重爬/删除）仅管理员；后端 /ingest 与 /knowledge/refresh|DELETE 都是 admin 守卫
  const isAdmin = useStore((s) => s.auth?.role === 'admin')

  const [name, setName] = useState('')
  const [filter, setFilter] = useState('')
  // 已收录列表页码（从 1 开始）
  const [page, setPage] = useState(1)
  // 角色名 -> 实时进度。3s 轮询 /ingest/status，五步全终态后停轮该角色
  const [progress, setProgress] = useState<Record<string, IngestStatus>>({})
  // 知识库列表（服务端真值，不回 store、不持久化）
  const [kb, setKb] = useState<KnowledgeOut | null>(null)
  const [kbErr, setKbErr] = useState('')
  const [kbLoading, setKbLoading] = useState(false)
  // 逐行操作中的标记：角色名 -> 'refresh' | 'delete'
  const [busy, setBusy] = useState<Record<string, string>>({})

  const loadKb = useCallback(async () => {
    setKbLoading(true)
    try {
      setKb(await knowledgeCharacters(apiBase))
      setKbErr('')
    } catch (err) {
      setKbErr(err instanceof Error ? err.message : String(err))
    } finally {
      setKbLoading(false)
    }
  }, [apiBase])

  useEffect(() => {
    void loadKb()
  }, [loadKb])

  const pendingChars = useMemo(
    () =>
      ingests
        .filter((r) => r.ok)
        .map((r) => r.character)
        .filter((c) => {
          const p = progress[c]
          return !p || p.status === 'pending' || p.status === 'running'
        }),
    [ingests, progress],
  )

  // 待轮询角色的「集合指纹」。用它当 effect 依赖，而不是直接依赖数组本身
  // （每轮 setProgress 都会产生新数组引用，直接依赖会无限重建定时器）
  const pendingKey = pendingChars.join('、')

  useEffect(() => {
    if (pendingChars.length === 0 || health === 'down') return
    let alive = true
    const tick = async () => {
      for (const c of pendingChars) {
        try {
          const st = await ingestStatus(c, apiBase)
          if (!alive) return
          setProgress((prev) => ({ ...prev, [c]: st }))
        } catch {
          /* 单次失败下轮再试 */
        }
      }
    }
    tick()
    const t = setInterval(tick, 3000)
    return () => {
      alive = false
      clearInterval(t)
    }
  }, [pendingKey, health, apiBase])

  // 一批入库跑完就刷新知识库列表 —— 新角色这时才真的出现在「已收录」里
  const wasPending = useRef(false)
  useEffect(() => {
    if (wasPending.current && pendingKey === '') void loadKb()
    wasPending.current = pendingKey !== ''
  }, [pendingKey, loadKb])

  const items = kb?.items ?? []
  const filteredItems = useMemo(() => {
    const kw = filter.trim()
    return kw ? items.filter((it) => it.character.includes(kw)) : items
  }, [items, filter])

  // 翻页。列表顺序沿用后端口径（updated_at DESC），新入库/刚重爬的角色本来就在最上面。
  // 页码只做「越界回第 1 页」的收敛，不额外写 effect —— 筛选变短时可能就落到了空页。
  const totalPages = Math.max(1, Math.ceil(filteredItems.length / PAGE_SIZE))
  const safePage = Math.min(page, totalPages)
  const pageItems = filteredItems.slice((safePage - 1) * PAGE_SIZE, safePage * PAGE_SIZE)

  const submit = (character: string) => {
    const c = (character || name).trim()
    if (!c) return
    ingestCharacter(c)
    setName('')
  }

  const doRefresh = async (character: string) => {
    setBusy((b) => ({ ...b, [character]: 'refresh' }))
    try {
      await knowledgeRefresh(character, apiBase)
      toast('ok', `已提交「${character}」重爬更新，后台先清旧知识再跑五步`)
    } catch (err) {
      toast('err', err instanceof Error ? err.message : String(err))
    } finally {
      setBusy((b) => {
        const n = { ...b }
        delete n[character]
        return n
      })
    }
  }

  const doDelete = async (character: string) => {
    const ok = window.confirm(
      `确认删除「${character}」的知识库？\n\n` +
      `会清掉 PostgreSQL 文档与分块、向量索引、图谱节点，并重建 BM25。\n` +
      `原始 md 对象保留在 RustFS，此操作不影响其他角色。`,
    )
    if (!ok) return
    setBusy((b) => ({ ...b, [character]: 'delete' }))
    try {
      await knowledgeDelete(character, apiBase)
      toast('info', `已提交删除「${character}」，后台清理中…`)
      // 后台清理不是毫秒级（Chroma + Neo4j + BM25 重建），轮询等它从列表消失
      for (let i = 0; i < 30; i++) {
        await new Promise((r) => setTimeout(r, 2000))
        const out = await knowledgeCharacters(apiBase)
        setKb(out)
        if (!out.items.some((it) => it.character === character)) {
          toast('ok', `「${character}」已从知识库移除`)
          break
        }
      }
    } catch (err) {
      toast('err', err instanceof Error ? err.message : String(err))
    } finally {
      setBusy((b) => {
        const n = { ...b }
        delete n[character]
        return n
      })
    }
  }

  return (
    <div className="page knowledge-page">
      <div className="page-head">
        <div>
          <h2 className="page-title">
            <Library size={19} className="grad-text" /> 角色知识库
          </h2>
          <p className="page-desc">
            下面是你**当前拥有**的角色知识；新角色走「抓取 → 分块 → 入库 → 索引 → 图谱」五步异步收录，
            提交后立即返回任务号，后台由 Celery worker 完成。角色名册随数据库自动增长，无需改配置。
          </p>
        </div>
      </div>

      {health === 'down' && (
        <div className="kb-warn">
          <Info size={15} /> 后端未连接。入库/重爬/删除接口需要 FastAPI(:8000) 与 Celery worker 同时在线。
        </div>
      )}

      {/* ============ 收录新角色（仅管理员，置顶） ============
          放在「已收录列表」之上：列表会长到需要翻页，收新角色是这页最常用的动作，
          不该被压在几十行表格下面。 */}
      {isAdmin ? (
        <section className="card kb-form">
          <label className="kb-label">收录新角色（可输入名册外的任意角色名，会自动去 wiki 抓取）</label>
          <div className="kb-form-row">
            <input
              className="input"
              value={name}
              placeholder="输入角色名，如「忌炎」…"
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && submit(name)}
            />
            <button className="btn btn-primary" onClick={() => submit(name)} disabled={!name.trim()}>
              <Download size={15} /> 提交入库
            </button>
          </div>

          {kb && kb.seeded_only.length > 0 && (
            <div className="kb-candidates">
              <span className="kb-candidates-title">
                名册里还没收录的 {kb.seeded_only.length} 个角色（点一下即可入库）
              </span>
              <div className="roster-grid">
                {kb.seeded_only.map((r) => (
                  <button key={r} className="roster-chip" onClick={() => submit(r)}>
                    {r}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="pipeline">
            {PIPELINE_STEPS.map((s, i) => (
              <div key={s.name} className="pipeline-step">
                <span className="pipeline-dot">
                  <Zap size={11} />
                </span>
                <div>
                  <b>
                    {i + 1}. {s.name}
                  </b>
                  <span>{s.desc}</span>
                </div>
              </div>
            ))}
          </div>
        </section>
      ) : (
        <div className="kb-warn">
          <Info size={15} /> 收录 / 重爬 / 删除需要管理员账号（admin）登录。你当前是游客，可以正常问答。
        </div>
      )}

      {/* ============ 已收录角色（核心） ============ */}
      <section className="card kb-owned">
        <div className="kb-owned-head">
          <h3>
            已收录角色
            {kb && <span className="kb-count">{kb.total}</span>}
          </h3>
          <div className="kb-owned-tools">
            <div className="kb-search">
              <Search size={14} />
              <input
                value={filter}
                placeholder="筛选已收录角色…"
                onChange={(e) => {
                  setFilter(e.target.value)
                  setPage(1)
                }}
              />
            </div>
            <button className="btn btn-ghost btn-sm" onClick={() => void loadKb()} disabled={kbLoading}>
              {kbLoading ? <Loader2 size={13} className="spin" /> : <RefreshCw size={13} />} 刷新
            </button>
          </div>
        </div>

        {kbErr && (
          <div className="kb-warn">
            <Info size={15} /> 读取知识库失败：{kbErr}
          </div>
        )}

        {!kbErr && filteredItems.length === 0 ? (
          <div className="empty-state">
            <Library size={26} />
            <span>{items.length === 0 ? '知识库还是空的，先在下面收录一个角色' : '没有匹配的角色'}</span>
          </div>
        ) : (
          <table className="kb-table kb-owned-table">
            <thead>
              <tr>
                <th>角色</th>
                <th>来源</th>
                <th>分块</th>
                <th>最近更新</th>
                <th className="kb-act-col">操作</th>
              </tr>
            </thead>
            <tbody>
              {pageItems.map((it) => (
                <tr key={it.character}>
                  <td className="kb-char">
                    {it.character}
                    {!it.seeded && (
                      <span className="kb-badge" title="不在内置名册里，是自动爬取发现并入库的新角色">
                        自动收录
                      </span>
                    )}
                  </td>
                  <td>
                    <span className="kb-src">{SOURCE_LABELS[it.source] ?? it.source}</span>
                    <span className="kb-src-sub" title={it.raw_uri ?? ''}>
                      原文 {fmtSize(it.raw_size)}
                    </span>
                  </td>
                  <td className="kb-mono">{it.chunks}</td>
                  <td className="kb-time">{fmtTime(it.updated_at)}</td>
                  <td className="kb-act-col">
                    {isAdmin ? (
                      <div className="kb-actions">
                        <button
                          className="btn btn-ghost btn-sm"
                          disabled={!!busy[it.character]}
                          onClick={() => void doRefresh(it.character)}
                          title="清掉旧知识后重新抓取更新（wiki 改版后用）"
                        >
                          {busy[it.character] === 'refresh' ? (
                            <Loader2 size={12} className="spin" />
                          ) : (
                            <RefreshCw size={12} />
                          )}
                          重爬
                        </button>
                        <button
                          className="btn btn-ghost btn-sm btn-danger"
                          disabled={!!busy[it.character]}
                          onClick={() => void doDelete(it.character)}
                          title="从知识库彻底移除该角色"
                        >
                          {busy[it.character] === 'delete' ? (
                            <Loader2 size={12} className="spin" />
                          ) : (
                            <Trash2 size={12} />
                          )}
                          删除
                        </button>
                      </div>
                    ) : (
                      <span className="kb-time">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {totalPages > 1 && (
          <div className="kb-pager">
            <span className="kb-pager-info">
              第 {safePage} / {totalPages} 页 · 共 {filteredItems.length} 个角色
            </span>
            <div className="kb-pager-btns">
              <button
                className="btn btn-ghost btn-sm"
                onClick={() => setPage(safePage - 1)}
                disabled={safePage <= 1}
              >
                <ChevronLeft size={13} /> 上一页
              </button>
              <button
                className="btn btn-ghost btn-sm"
                onClick={() => setPage(safePage + 1)}
                disabled={safePage >= totalPages}
              >
                下一页 <ChevronRight size={13} />
              </button>
            </div>
          </div>
        )}
      </section>

      {/* ============ 提交记录 ============ */}
      <section className="card kb-records">
        <h3>提交记录</h3>
        {ingests.length === 0 ? (
          <div className="empty-state">
            <Clock size={26} />
            <span>还没有提交过入库任务</span>
          </div>
        ) : (
          <table className="kb-table">
            <thead>
              <tr>
                <th>角色</th>
                <th>任务号 chain_id</th>
                <th>状态</th>
                <th>提交时间</th>
              </tr>
            </thead>
            <tbody>
              {ingests.map((r) => (
                <tr key={r.id}>
                  <td className="kb-char">{r.character}</td>
                  <td className="kb-mono">{r.chainId || '—'}</td>
                  <td>{renderStatus(r, progress[r.character])}</td>
                  <td className="kb-time">{new Date(r.createdAt).toLocaleString('zh-CN')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  )
}
