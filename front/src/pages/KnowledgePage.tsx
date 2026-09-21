import { useMemo, useState } from 'react'
import { Library, Search, Download, CheckCircle2, XCircle, Clock, Zap, Info } from 'lucide-react'
import { useStore } from '../store/useStore'
import './KnowledgePage.css'

/** 角色名册（与后端 rag/characters.py CHARACTER_NAMES 对齐，供快捷选择） */
const ROSTER = [
  '丹瑾', '丽贝卡', '仇远', '今汐', '凌阳', '千咲', '卜灵', '卡卡罗', '卡提希娅', '吟霖',
  '嘉贝莉娜', '坎特蕾拉', '夏空', '奥古斯塔', '守岸人', '安可', '尤诺', '布兰特', '弗洛洛',
  '忌炎', '折枝', '散华', '桃祈', '椿', '洛可可', '洛瑟菈', '清宵', '渊武',
  '漂泊者-男-导电', '漂泊者-男-气动', '漂泊者-男-湮灭', '漂泊者-男-衍射',
  '灯灯', '炽霞', '爱弥斯', '珂莱塔', '琳奈', '白芷', '相里要', '秋水',
  '秧秧', '秧秧·玄翎', '穗穗', '绯雪', '维里奈', '莫宁', '莫特斐', '菲比',
  '西格莉卡', '赞妮', '达妮娅', '釉瑚', '鉴心', '长离', '陆·赫斯', '露帕', '露西',
]

const PIPELINE_STEPS = [
  { name: '抓取', desc: 'wuwa-mcp 抓鸣潮 wiki 原文' },
  { name: '分块', desc: '结构感知分块 chunks.jsonl' },
  { name: '入库', desc: '写入 PostgreSQL + S3' },
  { name: '索引', desc: 'BM25 稀疏 + Chroma 稠密' },
  { name: '图谱', desc: '正则抽事实 → Neo4j' },
]

export default function KnowledgePage() {
  const ingests = useStore((s) => s.ingests)
  const ingestCharacter = useStore((s) => s.ingestCharacter)
  const health = useStore((s) => s.health)

  const [name, setName] = useState('')
  const [filter, setFilter] = useState('')

  const filtered = useMemo(() => {
    const kw = filter.trim()
    if (!kw) return ROSTER
    return ROSTER.filter((r) => r.includes(kw))
  }, [filter])

  const submit = (character: string) => {
    const c = (character || name).trim()
    if (!c) return
    ingestCharacter(c)
    setName('')
  }

  return (
    <div className="page knowledge-page">
      <div className="page-head">
        <div>
          <h2 className="page-title">
            <Library size={19} className="grad-text" /> 角色知识库
          </h2>
          <p className="page-desc">
            把新角色塞进异步收录流水线（抓取 → 分块 → 入库 → 索引 → 图谱）。提交后立即返回任务号，后台由 Celery worker 完成。
          </p>
        </div>
      </div>

      {health === 'down' && (
        <div className="kb-warn">
          <Info size={15} /> 后端未连接。入库接口需要 FastAPI(:8000) 与 Celery worker 同时在线。
        </div>
      )}

      {/* 提交表单 */}
      <section className="card kb-form">
        <label className="kb-label">角色名（中文名册标准名，如「忌炎」）</label>
        <div className="kb-form-row">
          <input
            className="input"
            value={name}
            placeholder="输入或从下方名册点选…"
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && submit(name)}
          />
          <button className="btn btn-primary" onClick={() => submit(name)} disabled={!name.trim()}>
            <Download size={15} /> 提交入库
          </button>
        </div>

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

      {/* 名册快捷选择 */}
      <section className="card kb-roster">
        <div className="kb-roster-head">
          <h3>角色名册</h3>
          <div className="kb-search">
            <Search size={14} />
            <input
              value={filter}
              placeholder="筛选角色…"
              onChange={(e) => setFilter(e.target.value)}
            />
          </div>
        </div>
        <div className="roster-grid">
          {filtered.map((r) => (
            <button key={r} className="roster-chip" onClick={() => submit(r)}>
              {r}
            </button>
          ))}
          {filtered.length === 0 && <span className="roster-empty">没有匹配的角色</span>}
        </div>
      </section>

      {/* 提交记录 */}
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
                  <td>
                    {r.ok ? (
                      <span className="tag tag-ok">
                        <CheckCircle2 size={11} /> {r.state}
                      </span>
                    ) : (
                      <span className="tag tag-err" title={r.error}>
                        <XCircle size={11} /> {r.state}
                      </span>
                    )}
                  </td>
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
