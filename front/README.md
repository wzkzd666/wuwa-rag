# 潮声智库 · 鸣潮角色知识助手前端

`wuwa-rag` 后端的配套 Web 前端：Vite 5 + React 18 + TypeScript + Zustand。
深空暗色玻璃拟态 UI，对接 FastAPI（`/health`、`/ask`、`/ask/stream` SSE、`/ingest`）。

## 功能

- **问答**（`/`）
  - SSE 流式逐字输出（打字机光标、可中途停止），也可在设置中切回一次性 `/ask`
  - 「检索中…」状态提示（后端 `status: retrieving` 事件）
  - 答案元数据标签：意图（事实/语义/混合）、命中角色、槽位、引用条数
  - Markdown 渲染（GFM 表格/代码块，DOMPurify 净化防 XSS）
  - 多会话：新建/切换/重命名/两步确认删除；`thread_id` 对应后端 LangGraph 多轮记忆
  - 复制答案、重新生成、欢迎屏示例问题一键提问
- **知识库**（`/knowledge`）：提交角色入库（`POST /ingest`），50+ 角色名册快捷点选（与后端 `rag/characters.py` 对齐）、流水线 5 步说明、提交记录表（chain_id/state/时间）
- **历史**（`/history`）：全会话按时间排序、关键词搜索（标题+正文）、单个/全部导出 JSON、继续对话
- **设置**（`/settings`）：API 地址自定义、连接测试、流式开关、暗/浅主题、基础字号、数据统计与清空（两步确认）
- **其他**：后端连通性指示（每 30s 探测 `/health`）、Toast 通知、会话/设置 localStorage 持久化、移动端响应式（侧栏抽屉）

## 启动

```bash
cd front
npm install          # 首次
npm run dev          # 开发：http://localhost:5173
npm run build        # 生产构建：tsc 类型检查 + vite build -> dist/
npm run preview      # 预览构建产物
```

开发模式下 `/api/*` 由 Vite 代理转发到 `http://127.0.0.1:8000`（即 `uv run python -m wuwa_rag.api.server`）。
后端不在默认地址时：

- 临时：`WUWA_API=http://192.168.1.10:8000 npm run dev`（PowerShell：`$env:WUWA_API="http://…"; npm run dev`）
- 或在「设置」页填完整 API 地址（此时后端需允许 CORS，前端直连不再走代理）

## 目录结构

```
front/
├── index.html              # 入口（内联 SVG favicon，防白闪底色）
├── vite.config.ts          # 代理 /api -> 127.0.0.1:8000（WUWA_API 可覆盖）
├── package.json
├── tsconfig.json
└── src/
    ├── main.tsx            # ReactDOM 入口 + HashRouter
    ├── App.tsx             # 路由表、主题应用、健康探测
    ├── types.ts            # 与后端对齐的数据类型（AskOut/StreamEvent/…）
    ├── lib/
    │   ├── api.ts          # fetch 封装 + SSE 手动解析（POST 流式）
    │   └── markdown.ts     # marked + DOMPurify
    ├── store/useStore.ts   # zustand + persist：会话/流式增量/设置/入库记录
    ├── components/         # Layout、MessageBubble、Composer、Toasts
    ├── pages/              # ChatPage、KnowledgePage、HistoryPage、SettingsPage
    └── styles/global.css   # 设计系统：CSS 变量、暗/浅主题、通用组件类
```

## 与后端的接口约定

| 端点 | 方法 | 前端用途 |
|---|---|---|
| `/health` | GET | 顶栏/侧栏连通性指示灯，30s 轮询 |
| `/ask` | POST `{question, thread_id?}` | 非流式问答（设置里可切换） |
| `/ask/stream` | POST 同上，SSE 响应 | 流式问答：`{status:"retrieving"}` → `{token}` × N → `{done:true, answer, intent, slots, characters, docs}` |
| `/ingest` | POST `{character}` | 角色入库，返回 `{character, chain_id, state}` |

注：SSE 用 `fetch` + `ReadableStream` 手动解析（原生 `EventSource` 不支持 POST body）。
