# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## 项目概览

《鸣潮》(Wuthering Waves) 角色知识助手：LangGraph 混合检索 RAG + 自动爬取流水线 + 爱弥斯(Aemeath)单角色人格微调模型。
Python 3.13、uv 管理、src 布局（`src/wuwa_rag/`）。全链路 async，代码注释以中文为主。

## 常用命令

```bash
uv sync                                # 安装依赖（含 torch cu130 索引）；开发加 --extra dev
                                       # ⚠️ LoRA 训练依赖（torchvision/torchaudio/accelerate/bitsandbytes）
                                       #   2026-10-06 已拆到 `train` extra：只有跑 ms-swift 训练才需要
                                       #   `uv sync --extra train`。torch 本身仍留在主依赖
                                       #   （sentence-transformers 跑 bge-m3 / reranker 运行时需要）。
uv sync --extra dev && uv run pytest    # 单元测试：469 条，**全量离线**，约 5s
                                       # 不依赖 PG / Neo4j / Redis / Chroma / Ollama / 网络，
                                       # 容器没起也能跑 → 可以当提交门禁（不会「假失败」）。
                                       # pytest 在 `dev` extra 里；asyncio_mode=auto 必须显式声明
                                       # （pytest-asyncio 1.x 默认 strict，不声明会让异步用例
                                       #  **静默跳过**——不失败，根本不跑，比没测试更危险）。
docker compose up -d                   # PG(pgvector16) / Neo4j 5.26 / Redis / RustFS，密码读根目录 .env
                                       # （PG_PASSWORD / NEO4J_PASSWORD / S3_ACCESS_KEY / S3_SECRET_KEY；.env 本身 gitignore）
                                       # ⚠️ 宿主机端口同样只认根 .env 的 *_PORT，一律用 2xxxx 不常用段：
                                       #   PG 25432 / Neo4j 27687+27474 / Redis 26380 / RustFS 29000+29001
                                       #   API 28000 / Vite 25173；Ollama 11434 是外部依赖，不改。
                                       #   口径：宿主机端口 = 2 + 标准端口（Redis 例外取 26380，
                                       #   26379 是 Sentinel 默认端口）。容器内部端口保持标准值。
                                       #   为什么：本机 9000 实测被另一个项目的常驻服务占死，
                                       #   RustFS 一起就 port is already allocated；8000/5173 是公共端口。
                                       #   为什么不取 5xxxx+：Windows 动态端口区从 49152 起，
                                       #   落在里面会被系统给出站连接随机占走，偶发且最难复现。
                                       #   start.ps1 会在 compose up 前做端口占用预检并报出占用者 PID/命令行。
# 建表：把 pgsql/001_init.sql 应用到 wuwa 库（documents/chunks/images/crawl_runs/记忆四表 + lg schema）

.\dev.bat                                # 【一键启动】docker 四件套 + celery + FastAPI(:28000) + 前端(:25173)
                                       # 建表 SQL 幂等自动应用；就绪后轮询 /health 并检查 ollama aemeath
                                       # 默认**不弹控制台窗口**：celery / API / 前端在隐藏窗口里跑，各自 stdout+stderr 落到
                                       #   .runtime/<name>.out.log（uvicorn/celery/vite 自己抛的错只在那里，不在 logs/）
                                       #   要实时看输出：.\dev.bat -ShowWindows
                                       # PID 记录到 .runtime/dev.pids.json（运行时状态，与日志分离；stop 连 .out.log 一起清）
                                       # ⚠ 停止必须连子进程树一起杀（stop.ps1 用 taskkill /T + 命令行特征兜底）：
                                       #   只杀窗口会让 uv → celery.exe → python 变孤儿（实测两套 Celery 并存抢队列）
.\dev.bat stop                           # 【一键停止】按 PID 记录关服务窗口 + docker compose stop（保留卷）
                                       # 底层是 scripts/start.ps1 与 scripts/stop.ps1，可直接调用并传参：
                                       # start.ps1 -NoDocker/-NoMigrate/-NoFront/-Open/-ShowWindows；stop.ps1 -KeepDocker/-Down

                                       # 【必备】生成模型 = Ollama 承载的 "aemeath"（QLoRA 微调 Qwen3-8B，爱弥斯），
                                       # 跑前确认 ollama list 里有它；无本地推理服务进程要起
uv run celery -A wuwa_rag.tasks.worker:celery_app worker --pool=solo --loglevel=info
                                       # 【必开】/ingest 与「库外角色自动爬取」都依赖它。
                                       # Windows 必须 --pool=solo 且并发 1：chunk 任务按角色整写 chunks.jsonl，并行会互相覆盖
uv run python -m wuwa_rag.api.server   # FastAPI :28000，端点 /health /ask /ask/stream(SSE) /ingest {"character":"忌炎"}

# 离线全量流水线（各步幂等、可单独重跑）：
uv run wuwa-chunk [src目录 [输出jsonl]]  # 默认 data/raw -> data/chunks/chunks.jsonl
uv run wuwa-ingest                      # chunks.jsonl + raw md -> S3 + PG（documents/chunks）
uv run wuwa-index                       # PG -> Chroma 稠密索引 + bm25.pkl 稀疏索引（全量重建）
uv run wuwa-graph                       # chunks.jsonl -> 规则抽事实 -> Neo4j MERGE
uv run wuwa-ingest-character 忌炎        # 单角色 5 步链入队（等价于 POST /ingest）
uv run wuwa-corpcheck                    # 【只读】语料对账：chunks.jsonl / PG / Chroma / BM25 / Neo4j / raw
                                       #   六方一比，不一致就列出并给补救命令；有 fail 退出码 1（warn 不拦路）。
                                       #   三次真事故的共同指纹都是「派生索引彼此一致、和真源对不上」，
                                       #   这条把当时手写的对账固化下来。见「语料对账」一节。

uv run python -m wuwa_rag.dialog.graph    # RAG 冒烟：跑 3 个内置问题
uv run ruff check src                  # lint（line-length=100）
```

没有测试套件（pytest 在 dev extras 里但无 tests/ 目录）。

## Windows 事件循环约定（最重要的一条）

所有 async 入口必须 `asyncio.run(..., loop_factory=asyncio.SelectorEventLoop)`：psycopg/neo4j 的异步实现默认落在 uvicorn 自起的 Proactor loop 上报 `InterfaceError` / 连接池初始化超时。现有各处（`api/server.py`、worker 的 `_run()`、各 CLI 的 `main()`）都已遵守，新增入口照做。

## 架构

### 存储：PostgreSQL 是唯一真源

- **PG**：`documents`（一角色一篇 wiki md）+ `chunks`（分块文本，含 breadcrumb/module/component/hash）。Chroma、bm25.pkl、Neo4j 都是**可随时全量重建的派生索引**。
- **Chroma**（持久化目录 `data/chroma/chroma/`，即 `VECTOR_DIR / "chroma"`，库文件 `chroma.sqlite3`）：bge-m3 稠密向量，cosine；同属 `data/chroma/` 的 `data/chroma/bm25.pkl`（`VECTOR_DIR / "bm25.pkl"`）：jieba+rank_bm25 稀疏路，pickle 里带 jieba 自定义词典快照，保证建索引/查询分词一致。
- **Neo4j**：结构化事实图谱，全部 MERGE 幂等（约束见 `knowledge/knowledge/graph/neo4j_client.py`）；关系键刻意不含 qty/rank 等易变属性，MERGE 后再 SET。
- **RustFS(S3)**：原文 md + 立绘；`knowledge/s3.py` 是抽象层，将来换后端只改这一层。
- **Redis**：Celery broker/backend。
- **LangGraph checkpointer**：PG `lg` schema（`AsyncPostgresSaver`，多轮记忆按 `thread_id`），DSN 必须带 `search_path=lg`（用 `Settings.PG_DSN_LG`）且连接池 `autocommit=True`（migration 里有 CONCURRENTLY 索引）。

### 摄取流水线（`tasks/worker.py`，Celery chain 5 步）

`crawl_character`（wuwa-mcp 抓鸣潮 wiki → `data/raw/<角色>.md`，记录 crawl_runs）→ `chunk_character`（结构感知分块 → chunks.jsonl，按角色增量重写）→ `ingest_character`（PG+S3，幂等：documents 靠 raw_sha256 DO UPDATE、chunks 靠 **chunk_id** DO NOTHING）→ `index_character`（BM25 全量重建[秒级] + Chroma 只 upsert 该角色）→ `graph_character`（正则抽取 → Neo4j）。

- chunk_id = `角色::H2::H3::H4::sha8`；wiki 结构固定，所以图谱抽取用正则规则（`knowledge/knowledge/graph/extract.py`）而不是 LLM。
- 爬取失败区分：`CharacterNotFound` 不重试直接中止；网络类异常重试 3 次并用 run_id 透传防重复 INSERT。
- **入库进度**：五步任务各自 `_step_update`（Celery backend 上报 + Redis 聚合键 `ingest:progress:<角色>` 双通道，TTL 1h）。`GET /ingest/status?character=xxx` 按角色查（不依赖刷新即丢的 chain_id），返回五步数组；整体 pending|running|success|failed。打点失败只记日志，绝不带崩流水线。前端知识库页对未终态记录 3s 轮询渲染五步进度点。

#### ⚠️⚠️ 2026-09-22 实测并已修：`ON CONFLICT (hash)` 导致跨角色同文本的块被静默丢弃

发现路径：核对「卡卡罗六阶突破材料」时发现答案里**没有一阶突破**，一路查到入库语句。

**根因**：`knowledge/crawl/pipeline.py` 的 chunks 插入写的是 `ON CONFLICT (hash) DO NOTHING`，
而 `001_init.sql` 里有 `CREATE UNIQUE INDEX ux_chunks_hash ON chunks(hash)`。
`hash` 是**纯正文** sha256，`chunk_id` 才是 `角色::H2::H3::H4::hash8`（含角色）。
两者差在「跨角色同文本」：**一阶突破材料表这类小表格的正文不含角色名**，
57 个角色里正文**逐字相同** → 只有第一个角色能插进去，其余全被 DO NOTHING 静默吞掉。

决定性验证（只读、可直接复现）：

| 量 | 值 |
|---|---|
| `chunks.jsonl` 行数 | **6572** |
| 其中不同 `hash` 个数 | **6448** |
| 跨角色同文本的重复数（Σ(每个 hash 的角色数 − 1)） | **124** |
| 修复前 PG / Chroma / BM25 | **6448 / 6448 / 6448** |

124 分毫不差。症状极具迷惑性：**三方索引彼此完全一致（逐角色都对得上），
看上去不像缺数据**；而缺的那块落在别的角色名下，`fetch_chunks` 按 `character` 过滤取不到，
于是「卡卡罗问一阶突破材料」永远答不出。

**修法（已落地）**：
1. `knowledge/crawl/pipeline.py`：`ON CONFLICT (hash)` → **`ON CONFLICT (chunk_id)`**。
2. 线上 DDL：`DROP INDEX ux_chunks_hash` + `CREATE INDEX ix_chunks_hash ON chunks(hash)`
   （必须放宽，否则跨角色同文本会撞唯一约束**直接报错**）。`ux_chunks_chunk_id` 已能保证每角色块唯一。
3. 补插 124 块 + 重建 BM25（全量、秒级）+ **增量** upsert Chroma（只 embed 那 124 块，不必全量重嵌）。
4. 结果：PG/Chroma/BM25 = **6572**；含「一阶突破」的角色 9 → **57**。

**✅ 已修（2026-09-22）**：`pgsql/001_init.sql` 已把该索引改为**非唯一** `ix_chunks_hash`，
并加了 `DROP INDEX IF EXISTS ux_chunks_hash;` 清掉历史库里的遗留唯一索引 ——
**新建库不会再引入这个 bug**，老库重跑这份 SQL 也会被自动纠正（其余句全是 IF NOT EXISTS，幂等）。

**实测（建全新库跑的，非推断）**：应用 SQL 后索引为 `ix_chunks_hash: non-unique` +
`ux_chunks_chunk_id: UNIQUE`；插入两条同 `hash`、不同 `chunk_id` 的块 → **共存 2 条**（旧写法会直接报
唯一约束冲突）；再用同 `chunk_id` 重复插入 → **仍被 `UniqueViolation` 拒**（幂等键没被削弱）。
线上库核对：已是 `ix_chunks_hash`（非唯一），`rows=6572 / distinct_hash=6448 / 含「一阶突破」角色 57`。

**排查提示**：遇到「资料明明在 raw 里、答案却答不出/答不全」，先做**三方对账**
（`chunk_markdown(current raw)` vs PG vs Chroma/BM25，再比 `行数 vs distinct hash 数`），
别急着重跑入库、更别改 prompt 或抽取规则。
（本次中途一度误判成「crawl 跑了但 chunk→ingest→index 没跟上」，靠 `行数 6572 ≠ distinct hash 6448`
这个差值才翻案——**PG 行数必须等于 distinct hash 数**这件事本身就是 hash 全局唯一的指纹。）


#### ⚠️⚠️ 2026-10-23 实测并已修：`chunks.jsonl` 是 0 字节，而 PG 里 6741 块完好

**发现路径**：第三轮审计里子智能体报「中间产物为空」，`ls data/chunks/` 复核 →
`chunks.jsonl` **0 字节**（mtime 2026-10-21 21:32），而 PG `chunks` 表
`count(*)=6741 / count(DISTINCT character)=58`。今天重跑 `uv run wuwa-chunk` 后
文件恢复 **6741 行 = PG 行数**（两边逐字符对上，才算闭环）。

为什么这事危险（三条都是实测口径，不是推测）：

1. **全量 CLI 用 `open("w")` 就地截断再逐行写**（`chunker.py` 的 `main()`）——
   进程被打断或 `data/raw` 读空时，旧文件在第一次 flush 前就已经被清空，
   留下一个「看起来是产物、实际是空壳」的文件。现在改成
   **① 空结果直接 `SystemExit`，绝不用空内容覆盖已有文件；② 先写 `.tmp` 再 `replace()`**。
2. **单角色增量链会把空文件写成「只剩这一个角色」**：`worker._chunk_character_async` 的
   「保留别人」是从**当前文件**里取的，文件空 = 别人全没了，而五步链照样报 success。
   现在 0 块直接 `raise`（账本落 fail，前端进度会红），不再静默通过。
3. **覆盖不全只能 warn 不能拦**：空 jsonl 对**全新安装**是合法起点
   （第一个角色入库时本来就只有它），一刀切拦死会让 clone 即用变成「先手工造文件」。
   所以 `_warn_missing_characters` 拿 PG 的 `DISTINCT character` 比一次，
   把缺的角色名 + 补救命令（`uv run wuwa-chunk`）写进日志，流水线照常跑；
   这个诊断**任何异常都吞掉**（PG 没起不能带崩入库）。

配套的两条同型问题（同一轮一起修）：

- **BM25 的跨进程陈旧**：`retrieve._bm25` 原先是 `@lru_cache(maxsize=1)`，
  而 `bm25.pkl` 是**另一个进程**（Celery worker）在重写 —— API 进程首次查询后
  一直用旧快照，只有重启才好。表现极具误导性：「新角色入库了、五步链全绿，
  但稀疏检索永远召不回它」，而 dense 那条实时查 Chroma，所以看着像偶发召回问题。
  现在按 `(路径, mtime_ns, size)` 指纹记忆：文件没变不重载（缓存的意义），
  变了必重载（这个 bug 的修法）。
- **pickle 就地截断**：`index/bm25.py` 写盘同样是直接覆盖，API 正在 `load` 时
  可能读到半截文件（会直接打挂 `vector_search`）。

**排查提示**：`chunks.jsonl` / `bm25.pkl` / Chroma 目录都是**可重建的派生物**，
怀疑坏了就照上面的命令重生成，不要手改文件；三方对账口径仍是
「jsonl 行数 == PG chunks 行数 == PG distinct character 覆盖数」。

### 语料对账（`knowledge/corpcheck.py`，2026-10-24 新增）

`uv run wuwa-corpcheck`：**只读**，不写任何数据。一次跑清六方 ——
`chunks.jsonl` / PG `chunks` / Chroma（点数 + 存的**文本口径**）/ `bm25.pkl` / Neo4j 名册 / `data/raw` 哈希。

为什么要这条命令（三次事故同一个形状：**派生索引看起来没事，真源其实对不上**，而每次排查都是临时手写一遍脚本）：

| 事故 | 当时看得见的 | 看不见的 | 靠什么发现 |
|---|---|---|---|
| 2026-09-22 `ON CONFLICT (hash)` | PG=Chroma=BM25=6448，逐角色也对得上 | 124 个跨角色同文本的块被静默丢弃 | `jsonl 6572 ≠ PG 6448` |
| 2026-10-23 jsonl 0 字节 | 五步链全绿、库里 6741 块完好 | 产物整角色没了（下一次按角色增量重写会只剩一个角色） | `ls data/chunks/` |
| 2026-10-24 索引文本陈旧 | Chroma 点数 6741、id 全对 | 36 块配队文本存的还是改版前的 `-` 串，与查询侧不同口径 | 本轮临时写的对账脚本（现已固化） |

三条设计判据（改这个模块前先读）：

- **逐角色比，不只看总数**：A 角色多 10 块、B 角色少 10 块时全局计数完全对齐，
  而 B 的答案永远召不回那 10 块 —— 124 块那起事故就是这个形状。
- **缺与多分开报**：「库里有、索引没有」是召不回；「索引里有、库已删」是召回旧知识。
  症状相反，不能合成一句「数量不一致」。
- **warn 不拦路，fail 才给非 0 退出码**；缺席类问题聚合成**一条**（`data/raw` 整目录不在时
  逐角色刷 58 行会把真正要看的 fail 埋掉 —— 对账输出是给人一眼读完的）。
  每条 fail 直接带可抄的补救命令（`uv run wuwa-chunk` / `wuwa-ingest` / `wuwa-index` / `wuwa-graph`）。

判据都是纯函数（吃 dict/set，不碰服务），8 条用例全离线在 `tests/test_corpcheck.py`；
取数那层（PG/Chroma/BM25/Neo4j/raw）任何一路读不了只记 warn 并跳过，**不让对账工具自己变成故障源**。
实测：现网跑 `PG=6741 chunks.jsonl=6741 Chroma=6741 BM25=6741 图谱角色=58 raw 文件=58` → ✓ 退出 0；
把 `CHUNKS_DIR`/`RAW_DIR` 指到空目录 → ✗ 退出 1（1 条 fail + 1 条聚合 warn，不刷屏）。

⚠️ 它**不覆盖 S3/图片**，也不覆盖 `tool_calls`/`llm_usage` 这类运行期表 —— 只管「检索用的语料一致性」。

### 问答链（`dialog/graph.py`，LangGraph StateGraph）

`intent → {chitchat: chitchat | time: time | clarify: clarify | fact: graph→verify | semantic: vector→verify | hybrid: graph→vector→verify}`
⚠️ **`clarify` 是 2026-10-23 才真正接通的分支**（此前恒不可达，见下条）：句子只有
**纯指代/语气词**（`那个 / 继续 / 嗯`，判据 `nlu.needs_clarification`）且主题分类器判 chitchat 时，
才从 chitchat **改判 clarify** 去反问，而不是让模型猜着答。三条硬要求：
① 判据必须挂在**闲聊分流之后**（`classify()` 只返回 hybrid/fact/semantic，挂在它后面判
   `intent == "chitchat"` 是恒假条件——原来就是这么写的，反问一次都没触发过）；
② 只收 ②（主题分类器）那一条出口：音乐回指、`is_identity`、`is_self_intro`、`pure_time`
   都有自己的动作要执行，被改成反问就把功能吞了；
③ 新分支必须同时进 `graph._TEXT_NODES` 与 `_STAGE_LABELS`——**漏了不报错**，
   只会让 `done.answer` 成空串、前端一片空白（token 是旁路抽取的）。
   三条各有一条守卫测试（`test_intent_routing.py`：可达性 + 白名单对账，后者是结构性判据、
   不抄节点名清单，新增产正文分支漏加白名单当场红）。
⚠️ 上面只是**意图层**的分支；图跑到 `graph` 之后还有一层 `_after_graph` 决定要不要补向量，
两条硬规则：**技能类强制补向量** / 指名队伍（≥3 个角色名）强制补向量。
⚠️ **「概括性配队不走向量」不在 `_after_graph` 判** —— 它由 `intent` 承载：`intent_node` 把这类问题的
intent 从 `hybrid` **降级为 `fact`**（见下），`_route` 自然走 fact 分支、`_after_graph` 里
`state["intent"]=="hybrid"` 也不再命中。**intent 是对外字段，必须与实际路径一致**——
绝不能「报 hybrid 却 docs=0」（2026-09-22 起：这个 intent 不能例外，不走向量就应该标记为 fact）。
`verify → {ok/exhausted: generate | retrieval/refreshed: 重检索 | web: web→generate}`

- **追问改写（检索前置，A+B 压缩上下文）**：`nlu.rewrite_query` 用 qwen3:8b 把指代残缺的追问补成自包含问句；**槽位/角色/属性/阶段提取与向量检索全部吃改写句**（存 `RagState.search_query`，history 与展示仍是原句）。输入三路合并：**A 焦点锚点** `focus_anchors`（现成正则从全量历史提角色名，零 LLM 零延迟，救「角色名在长回答 120 字符截断区外」的丢名问题；按**最近提及优先**排序）+ **B 滚动摘要** `summarize_turns`（压缩被滑出 MAX_HISTORY_TURNS 窗口的旧轮次，存 `RagState.context_summary` 由 checkpointer 持久化；`_commit_history` 在 generate/chitchat 截断 history **之前**取 evicted——事后窗口里拿不到；只有真有 eviction 才调，0.3~4s）+ 最近 2 轮短原文。指代消解规则（实测校准，均验证过）：「她/他/那位」**近指代**→话题角色第一个；「开头/之前聊的那位」**远指代**→摘要里的角色——8b 光靠规则句不执行，**必须在 _REWRITE_SYSTEM 里给一条完整示例**（示例即 few-shot）才解析。无历史且无摘要原句返回；输出异常（空/超长）回落原句——改写是增益不是依赖。闲聊判据也用改写句。
- **生成侧禁注入摘要**：实测把 `context_summary` 塞进 `build_context` 会被 aemeath 原样复述进答案（第三人称摘要腔穿帮）。摘要只喂改写器，生成靠窗口内 history + 改写后检索结果。
- **生成侧口吻要求（2026-09-22）**：`prompt.SYSTEM_PROMPT` 里必须**显式**写「用第一人称『我』、活泼亲切，不要第三人称复述资料」。不加这句，模型会输出「她曾在雪原上…爱弥斯挨了一顿训」这种**第三人称资料搬运腔**（实测问「角色故事」→ 530 字 bullet 罗列、人设全丢、用户反馈过）。补上后同一问题变成第一人称叙述（「我的角色故事呀~…我扮演英雄…」709 字），且**数值表完整性不受影响**（技能 13 行 / 材料 11 行照旧逐行、符号原样）。配套保留了「口吻可以活泼，但事实必须严格依据资料，不许发挥」这条防幻觉约束。
  - ⚠️ **第一人称必须限定作用域（同日补）**：原来那句话没说清「我」指谁，问**别的角色**时模型会把「我」套到被问的角色身上（用户报「把清宵当自己了」）。现在明确：**讲爱弥斯自己的事**才用「我」；**资料讲的是别人（清宵/守岸人/尤诺…）时一律用角色名或「他/她」，不许自称「我」、不许冒充那个角色**——你是爱弥斯，不是被问的人。同一条硬要求里另加了「列表/配队条目每条只准出现一次，不许给条目编序号计数器（如『守岸人 + 尤诺*5』）」。
- **「不知道」也要有人设（2026-09-22）**：用户报「不知道的时候回答没有人设感」。两处收口都改了口径——① `graph._unknown_text(names)`（角色不在知识库 + 联网抓取也失败）：由原来越读越像公文的「不知道（知识库里没有这个角色，尝试联网抓取也没找到）。」改成爱弥斯口吻，但**保留三件事实**：翻了本地、联网找过、确实没有；`ask()` 与 `ask_stream()` 共用（流式那条早退的 `done` 事件从「只有 `{"done": True}`」补齐成与正常路径同构的字段，前端 meta/引用面板不再拿到 undefined；**2026-10-23 起同构由构造函数保证**——`graph._done_event(...)` 是 `done` 的唯一形状来源，早退与正常收尾都走它，不再靠「记得把十一个键写两遍」。同一轮还补了 `emotion` / `web_reason` 两键：此前早退那条少这两个，同一个「不知道」在两条路径上形状不一样，前端只能靠 optional 兜着。另有一条跨模块断言钉住「`done` 里有的键，`api/app._meta_of` 必须全部落库」（`_meta_of` 是白名单式的 `evt.get(...)`，加键忘了进它就表现为「流式当场有、刷新历史就没」））。② `build_context` 零资料约束：明确要求「**用你自己的口吻**、一句话说明没有这份记录，可以俏皮一点带点小遗憾」，不再只说「说明你不清楚」（那样会答成干巴巴的公文腔）。
- ⚠️ **澄清一个易误读点**：`generate_node` 只传 `HumanMessage`，`prompt.SYSTEM_PROMPT` 是**拼进该 message 文本**的，**不是** system 参数 —— 所以 aemeath 的 Modelfile SYSTEM 人设**并未被覆盖**。`llm.py` 那句「调用侧不得传 system 参数，否则人设被覆盖」的约束对象是「别传 `SystemMessage`」，与 `prompt.SYSTEM_PROMPT` 这个变量名无关（命名容易误导）。真正让人设丢的是「第三人称复述腔」，不是覆盖。
- **验证升级闭环（verify→重检索→刷新→联网）**：`verify_node`（`services/verify.py`，qwen3:8b 判「资料能否回答问题」，ainvoke 带 `wwa:verify` 标签防流式泄漏）插在检索与 generate 之间——重排低分只挡得住「无资料」，挡不住跑题/脏数据。不匹配按代价从低到高升级：①已识别角色且没刷过 → `worker.build_refresh_pipeline`（**先按角色清 PG documents（chunks 级联）/Chroma 向量/Neo4j 私有节点，再重爬 5 步**——不清就是新旧块并存召回旧知识的根源，chunks 表 DO NOTHING 只挡同 hash；同步等待 `REFRESH_WAIT_TIMEOUT`，`reset_progress` 让前端五步重走）→重检索；②有重试额度 → 用 verifier 给的 refined 检索式重检索（`_rescan` 重算信号，按 intent 回 graph/vector）；③额度尽 → 百度千帆 `web_search`（`services/websearch.py`，v2 chat/completions + web_search 参数按官方文档；**`QIANFAN_API_KEY` 留空=联网兜底整体关闭**，返回 (False,"") 降级，不发任何外部请求）；⚠️ 千帆接线三个坑（2026-09-22 真跑实测后修，见 `config.py` 与 `websearch.py` 注释）：**(a) `QIANFAN_TIMEOUT` 必须 ≥45s**——实测联网请求 6.1s（命中搜索缓存）~26.5s（真搜），多数 21~26s，原来 20s 会**随机 ReadTimeout**，表现成「配了 key 也用不了」；**(b) 溯源要开 `enable_trace: True`**——不开时响应里根本没有 `search_results`（`web_search` 段里那段拼 `- title: url` 的代码此前是死代码，恒 0 条），开了才有（实测 10 条 index/url/title）；**(c) key 用 `AliasChoices("QIANFAN_API_KEY","BAIDUQIANFAN_API_KEY")` 读**——原来写 `os.getenv('BAIDUQIANFAN_API_KEY')` 会在**类体求值**，① 只认进程环境，把 key 写进 `.env` 完全无效（pydantic 的 env_file 不注入 os.environ）；② 变量不存在时默认值是 `None`（字段类型却是 `str`）→ pydantic 抛 ValidationError，而 `ww_logger` 第 10 行 import 时就调 `get_settings()` → **整个服务起不来**，与「留空=优雅降级」的设计相反（已实测复现）。判活判据：响应的 `usage.prompt_tokens_details.search_tokens > 0` 说明真联网了（实测 1905~3430）；④都不行 → exhausted 直接 generate，由 `build_context` 一句话不知道兜底。`web_facts` 以「## 联网搜索资料」并入生成上下文，零资料判定（graph_facts+docs+web_facts 全空）才会触发不知道约束。防死循环三道闸：`retry_count≤VERIFY_MAX_RETRY(1)` + `refreshed` 仅一次 + `used_web` 仅一次，verify 最多进两次。⚠️ 实测抓到的坑：`_purge_async` 跑在 `_run` 循环里，`close_pool` 必须 `await`（再 `_run` 即 loop 套 loop 崩）；重检索分支 stage 必须写死 `"retrieval"`，不能用 `_retry_or_web` 给（它在 MAX_RETRY=1 下只会回 web/exhausted，重检索成死代码——「跑题材料且无角色」的用例第一轮测试没覆盖差点漏网）。⚠️ 部署提醒：worker 重启前旧进程不认识 `purge_character_data`，刷新会失败并安全降级为收口，`.\dev.bat` 重启 worker 后才真正生效。`VERIFY_ENABLED=False` 整闸旁路。
- **摘要模型用 8b 不用 0.6b**：实测 0.6b 合并多轮会**丢角色名**（输出「讨论了声骸组合等话题」空话，摘要价值归零）；8b 能保名（「卡卡罗声骸彻空冥雷，长离两链质变…」）。失败/超长回落：保留原摘要并追加降级标记，下轮压缩完整窗口时自我修复。
- ⚠️ **节点内 ainvoke 的流式事件会泄进答案流**：`classify_topic`（node='intent'）按节点过滤解决；但 `_commit_history` 在 generate/chitchat **节点内部**调摘要，其回调 node='generate' 节点过滤挡不住，实测整句摘要被拼进流式答案尾巴——解法是 ainvoke 传 `config={"tags":["wwa:summary"]}`，ask_stream 按 tags 丢弃（见 graph.py 注释）。**以后凡在生成节点内部调其它 LLM，必须打标签**。
- **前角色注入（远指代兜底）**：`graph._inject_far_characters`——问句/改写句含远指代词（开头|先前|之前|前面|上次|刚才|最初|那位）时，从 `context_summary` 取**文本位置最前**的名册角色并入 `characters`。与改写器互补：改写成功→远指代词已被替换、天然 no-op；改写偶发失败→这里兜住图谱检索不缺角色。多角色原功能保留（比较句自带名字时**追加**不覆盖，如「开头那位和长离谁强」→ [长离,卡卡罗]）。注意：名字匹配正则按**长度降序**拼接（「秧秧」是「秧秧·玄翎」前缀，短的在前会误抢）；known 为空必须挡（空正则到处匹配）。
- **名册 TTL 缓存**：`_known_characters` 60s 缓存（每轮问答 intent_node 与 ensure_characters 至少各查一次 Neo4j，名册只在入库时变）；空结果不缓存；`_crawl_and_wait` 建库成功后 `_invalidate_known_cache()`。实测轮1 耗时 7.6s→2.3s。
- **API 错误处理**：`on_unhandled` 全局处理器把未捕获异常转 `{error(含编号), detail}` JSON + 服务端堆栈日志（request_id 串联）；⚠️ SSE 响应头发出后全局处理器接不住，`/ask/stream` 必须**流内** try/except 转 `{'error','detail'}` 事件下发，前端 store 的 error 分支接住（保留已吐 token、标记 error 态、toast 提示）。
- **规则优先，LLM 只兜底**：`dialog/nlu.py` 正则意图+槽位+属性+阶段（实测 16/16 覆盖、零漏检，且无槽位命中时 `classify()` 回落 hybrid 图谱+向量双跑，本身即安全兜底，**槽位/意图分类不接 LLM**）；但**主题分类接了 qwen3:8b agent**（`nlu.classify_topic`：闲聊 vs 游戏）——规则词典对闲聊句覆盖不全（「家人怎么这么晚才来」无游戏信号却命中语义词「怎么」），故仅在「无角色名 且 无槽位 且 无属性/阶段」时才调 LLM；少样本用单轮补全（末尾 AIMessage 以 `{"` 截停省输出 token，实测首 token 从 6s 降到 0.3s），解析失败一律回落 game（误送 RAG 只是生硬，误判闲聊丢知识，代价不对称）。闲聊走 `chitchat_node`：不挂检索、不带 `prompt.SYSTEM_PROMPT` 术语/答题约束（会诱发拒答独白），带对话历史自然回应，复用复读兜底。`knowledge/entities.py` 静态名册(覆盖率实测 10/11，LLM 兜底触发率 0%)+ qwen3:8b 兜底认角色；`knowledge/retrieve.py` 固定 Cypher 模板（8B 模型 Text2Cypher 幻觉率高）；`tools.py` 三个工具**全部走 L1 确定性直调**——`graph_search`（graph_node）、`vector_search`（vector_node）、`current_time`（time_node）。`agent.py` 的 LLM 自主选工具路径**保留但主链未启用**——已改用 `get_tool_llm()`(qwen3:8b)，接入前务必先解决参数名漂移（见下条）。
- **人格/身份类问题走 chitchat（2026-09-22）**：问「你」的台词/语音/语录/口头禅/名字/身份，是角色人格不是游戏资料——`nlu.is_identity`（正则硬信号）在 `intent_node` 里**优先于主题分类**直接判 chitchat，即使改写出了角色名也不检索。**必须用原句 `q` 判**：改写 sq 会把「你」补成角色名（「你的台词」→「爱弥斯的台词」），第二人称信号丢失。原因：aemeath 人设由模型自带（Modelfile SYSTEM，调用侧不得传 system 否则人设被覆盖），走 RAG 反而召回大段「角色故事/珍贵之物」剧情文案被整段倾倒（实测「你的台词是什么」→ hybrid → 809 字剧情故事，用户报「混入无关内容」）。规则用第二人称词（你的台词/你是谁/你叫什么/你的名字…）锁死，第三人称「爱弥斯是谁」不命中、仍走检索（这类答案尾部可能出现的 `[N]` 引用角标由输出侧 `strip_ref_marks` / `AnswerFilter` 剥离，见下文「输出侧兜底」）。
- **时间类问题走 time 分支（2026-09-30）**：`nlu.is_time_question`（正则硬信号）在 `intent_node` 里**优先于主题分类**判 `intent="time"`，路由到新节点 `time_node` → **确定性调 `tools.current_time`** 取服务端真值 → 交模型用爱弥斯口吻说出。为什么必须有这条分支：模型没有时钟，训练数据里的「今天」永远停在训练期附近；而「现在几点」经 `classify_topic` 会被判 chitchat（确实与游戏无关），落到 `chitchat_node` 后模型只能含糊其辞或自信地报一个错日期（隔离实验实测：不给时间时 4/4 次答「我这边没有钟表/记不太清」）。判据是**白名单 + 句末锚定**，不维护动词黑名单：①「星期几/周几/礼拜几/几月几号」自身无歧义；② 整句就是时间问法（`几点/几点呀`）或带锚点（现在/今天/当前…）且**时间词收尾**——游戏提问里时间词后面总跟着动作词或名词，实测反例「鸣潮几点刷新」「今天几点开服」「卡卡罗几点上线」「日常委托几点」「秧秧共鸣链几号节点要多少材料」全部拦下；回归 33 正例 33/33、29 反例 29/29 全对。条件带 `not slots` 兜底。**判据用原句 `q`**（与 is_identity 一致）。时钟口径：`config.TZ_NAME`（默认 `Asia/Shanghai`，`tools._local_now` 用 zoneinfo；容器 TZ=UTC 时若不按它解释会整整错 8 小时；取不到时回落系统本地并告警，问答不中断）。工具输出**同时给 24 小时制与口语写法**：`2026-09-30 16:52 星期三（下午4点52分，UTC+08:00）`——实测只给 24 小时制时，aemeath 有 3/8 次把「16:5x」心算成「三点五x」；补上口语形态后 10/10 全对（日期 10/10、星期 9/10、小时 10/10）。`chitchat_node` 与 `time_node` 共用 `_chat_turn(state, hint)`（人设注入/流式/复读兜底/情绪/记忆写回完全一致，只差那句 hint），新增「不检索」类分支请复用它而不是复制一份。
- **⚠️ 流式收尾补吐的判据（2026-09-30 修正，勿改回）**：`ask_stream` 收尾处要把后处理（`fix_percent_units` 等）相对**已下发文本**的差量补吐出去，判据必须用「已下发正文 `sent` 做前缀比较 + 只 feed 后缀」。**不能**用 `AnswerFilter.would_append`——它只看列表行的去重集合 `_seen`，而 `_route` 对非列表行是无条件直通的：答案里没有列表行时 `_seen` 恒为空、`would_append` 恒 True，`feed(answer)` 会把**整篇正文原样重吐一遍**。实测（单元级复现 + 端到端）：闲聊/时间这类无列表行的短答案整段重复一遍（48 字答案流出 96 字）；含列表行的事实答案则是开头的非列表行被重吐一遍。前端 `useStore` 最终会用 `done.answer` 覆盖（注释写明是为了盖掉复读文本），所以终稿没错、但流式过程中肉眼可见重复。修正后实测：整段重复 0/10（原 10/10），**流式下发文本 == done.answer 10/10**，且单位修正的后缀补吐、`[n]` 剥离、列表去重三条原设计意图都仍成立（`would_append` 已成死代码，已从 `text.py` 删除）。
- **agent.py 实测坑（接入前必读）**：弱 prompt 下 qwen3:8b 会把参数名写成单数 `character`，pydantic 忽略未知字段 → `characters` 取默认空列表 → 工具返回空字符串 → 模型接着**编造共鸣链名称**（实测编出「追光者」「星海行舟」，均非真实数据）。强化 prompt（显式列参数名+取值域+空结果铁律）后实测：参数名正确、工具返回 409 字真实数据、空结果场景改回「知识库里没有这项资料」不再编造。
- **双路召回**：Chroma dense(30) + BM25 sparse(30) → RRF(k=60) → 前 20 进 bge-reranker-v2-m3 精排 → top6 进 prompt。`RERANK_MIN_SCORE=0.1`：top1 低于阈值视为库内无答案、候选全丢（这是「不知道」的门）。多角色提问每角色各补一轮召回。
  - **调试出口（2026-10-24）**：`retrieve.recall_stages(question)` 返回 dense/sparse/fused/docs
    四步中间量，`vector_search` 只是它的正式出口（**同一份实现**，有源码判据用例钉住
    「不许在调试链里再算一遍 `_rrf`」）。往上就是 `GET /debug/retrieval`（admin 专属、
    独立一档 `RATE_LIMIT_DEBUG=20/hour`），把 `rerank` 的逐条分数与门槛判定一起给出。
    为什么要开这个口：AGENTS 里那三次「资料明明在 raw 里、答案却答不出」，每次都要
    临时手写脚本对账，才能分清是**没召回到** / **被 `TOPK_RERANK_IN` 截掉** /
    **卡在 `RERANK_MIN_SCORE` 门外** —— 三种病因修法互不相干，而生产返回体里只有一句「不知道」。
    ⚠️ `gate.rerank_ready` 与 `gate.passed` 是**两件事**（`rerank.status()` 单独探可用性）：
    精排不可用（权重没就位/HF_HOME 不对）和「精排可用但分数没过门槛」看起来一样（都 `passed=false`），
    但前者修环境、后者是真没答案。
    实测一次调用即看清老问题：问「守岸人声骸怎么配」→ dense=30 sparse=30 fused=20 reranked=6，
    而 top1 是 **`忌炎 › 角色养成推荐 › 配队推荐` 0.9619** —— 精排给跑题材料的分数照样接近满分，
    这正是当初「数值表在 reranker 眼里输给大段机制描述」的同一个低区分度。
  - **精排异常 = 可见的失败**：`rerank()` 现在把模型加载/打分异常**收成「本轮无可用资料」**
    （走既有的不知道/联网兜底）并写 error 日志，附带该查的 `HF_HOME` 与模型名。
    刻意**不是**退回「按 RRF 顺序硬给 top6」——那等于把「不知道」的门拆了还装作正常：
    答案照出、检索照绿、日志无痕，只有用户追问答非所问时才暴露（本仓口径：可见的失败 >
    看起来正常的失败）。两条用例分别钉「异常必须判空且留下原因」与「正常时门槛与放行照旧」。
- **库外角色自动爬取**：`ask()/ask_stream()` 先 `ensure_characters`——问题命中名册但 Neo4j 里没有 → 入队 5 步链并阻塞等待(180s) → 成功继续答、失败直接回「不知道」。这就是 API 必须带 Celery worker 的原因。
- **双模型分工（多 agent）**：`llm.py` 两个客户端——`get_chat_llm(strict=False)` = **aemeath**（chat 专用，只做最终作答；`strict=True` 走 `LLM_REPEAT_PENALTY_STRICT` 低惩罚档，照抄长表格用）；`get_tool_llm()` = **qwen3:8b**（字典抽取 / 工具调用等结构化任务）。两者都 `bind(think=False)` 关思考。⚠️ **不要用 `/no_think` 软开关**：实测对 aemeath 无效（仍 38.3s、输出 3735 字、带 `v` 泄漏前缀，并触发 Ollama 500 `peg-native format` 错误）；改 `bind(think=False)` 后 1.3s 且干净。qwen3:8b 同理：抽取准确率不因开 thinking 提升（多样本均 4/8），耗时却从 0.20s 涨到 6.36s。`TOOL_TEMPERATURE=0`（要稳定 JSON，不要文采）。
- **人格与作答**：整条 prompt 以单条 HumanMessage 发出；人设完全由 Ollama 端 aemeath 微调模型自带（Modelfile SYSTEM），`prompt.SYSTEM_PROMPT` 只做术语对照 + 答题约束 + 防重复，**不再引导口吻**。⚠️ 这些约束必须拼进 HumanMessage——Ollama 的 `system` 参数会**覆盖** Modelfile 内置 SYSTEM，改成 `SystemMessage` 传就会把人设弄丢（但 `knowledge/entities.py`/`agent.py`/`verify.py`/`dialog/nlu.py` 用 qwen3:8b 无人设可覆盖，用 `SystemMessage` 是安全的）。
- **多轮记忆**：`ask()` 与 `ask_stream()` **共用同一张图和 checkpointer**，`thread_id` 在两条路径都生效。`generate_node`/`chitchat_node` 是 **streaming node**（`async for llm.astream()` 内部累积，只在结束时 yield 最终状态——token 由 `llm.astream` 自带回调产出，无需逐 token yield 中间态），流式走 `chain.astream_events(version="v2", config=...)`：从 `on_chat_model_stream` 抽 token，从 `on_chain_end name=="LangGraph"` 拿最终状态（含 intent/slots/docs/truncated）。⚠️ **抽 token 必须按 `metadata.langgraph_node` 过滤只留 generate/chitchat**——intent_node 里的主题分类器也调 LLM，其流式事件同样挂在 `on_chat_model_stream` 上（node='intent'），不过滤会把 `{"topic":"chitchat"}` 当答案吐给前端（实测发生过）。⚠️ streaming node 对 str 字段是**覆盖非拼接**，故节点内必须自己累积全文做复读检测。
- **阶段进度事件**：`ask_stream` 除 token 外还 yield `{"stage", "label"}`，由 `on_chain_start` 的节点名映射 `_STAGE_LABELS`（**取值以该字典为准**，别再抄一份清单——抄过，于是文档里长期少 `time`/`verify`/`web`/`clarify`/`music` 几条）。新增产正文的节点必须同时补 `_STAGE_LABELS` 与 `_TEXT_NODES`，理由见「clarify 三条硬要求」。原因：检索+重排实测约 19s 而生成仅 1~2s，静默期是用户焦虑主因。streaming node 会触发**两次** `on_chain_start`，已用 `emitted_stages` 去重。`LangGraph`/`_route`/`_after_graph` 是图容器与路由函数，对用户无意义，不进映射。
- **闲聊回复的删除线**：aemeath 爱用 `~`/`~~` 当语气破折号，marked 的 GFM 把成对波浪号当删除线定界符（实测 `来~玩呀~` → `来<del>玩呀</del>`）。前端 `lib/markdown.ts` 在渲染前对**代码区外**的波浪号逐个转义（`~~` → `\~\~`，CommonMark 反斜杠转义按单个标点计），代码块/行内代码内的 `~` 保留。本域从不需要删除线，故一律禁用。
- **防复读三道闸**（8B 角色扮演模型在「检索不到资料」时易陷入人设独白循环，整句周期性重复）：① 采样参数 `repeat_penalty`/`repeat_last_n=512`/`top_p`/`top_k`（`config.py` 可调）；② `build_context` 在**无图谱事实且无文档**时注入明确的一句话作答指令——判断必须基于 `graph_facts`/`docs` 而非 join 结果是否为空，否则多轮对话时 history 非空会把「无资料」信号吞掉；③ `dialog/guard.py` 运行时行级检测，命中即中断生成并 `trim_loop` 截断（流式 break 可提前止损，实测省下 76% 无效生成）。表格行与短碎句不参与判重，避免误杀突破材料表。截断状态经 `RagState.truncated` → `AskOut.truncated` / SSE `done.truncated` 透传前端显示「已截断重复内容」。
  - ⚠️ **2026-09-22 补第二类规则（周期块循环）**：原来只判「同一长句出现 >2 次」，实测漏网——问「清宵配队」时模型输出把 6 行一组的目标配队循环了 5 遍，且每行带 `*1`…`*28` 计数后缀（`守岸人 + 尤诺*17`）：① 每行都短于 `LLM_LOOP_MIN_CHARS=12` 被跳过；② 后缀让每行看起来都唯一，精确判重根本抓不到。故新增「周期块循环」判定：一组行（周期 2~`LLM_LOOP_PERIOD_MAX` 行、周期内至少 2 种不同行）整体重复 ≥ `LLM_LOOP_MIN_CYCLES` 遍即判退化。**比较前剥掉行尾计数标记（`*N`/`×N`），但只用于周期序列比较、不参与精确计数**——否则「- 贝币×5000 / ×10000」会被归一成同一行，真实材料表必被误杀。
  - 检测分两档：`feed()`（流式，每个 token 都跑）只查**尾部 `span+p` 行**窗口内所有对齐位置（O(1) 级、与已生成行数无关；留 p 行余量是因为退化块后面常还跟着几行别的内容）；`find_degenerate_start()`（`trim_loop` 里每次生成只跑一次）做**全文扫描**——只查尾部会在「退化块后面还有正常内容」时算不出截断点，trim 什么都不做，流式兜底等于白做（本轮真跑回归抓到的 bug）。切点取所有命中里**最早**的那个，并向前扩展同类周期块。
  - 回归基线（可复用）：退化样本必须命中且截断（455→93 字）、干净答案/真实补料块/技能 13 行/材料 11 行四个负样本必须**不**命中。
- ⚠️ **`repeat_penalty` 别调回 1.3**：1.3 会把「彼此高度相似的成片行」罚到写不下去，症状是**提前收尾而不是报错**——同一张满级数值表只输出前 4 行就收口，还补一句「其他参数未在该列表中出现」；问「爱弥斯共鸣解放伤害倍率」时 7 行稳定丢 1~3 行，**连没有补料块的普通轮次也丢行**。消融结论：`repeat_last_n=64/512/1024` 三档结果一致（**窗口不是主因，惩罚强度才是**）；换 markdown 表格 / 纯文本无项目符号 / 编号列表 / 把数值改成中文表述**全都无效**；`penalty=1.15` 与 `1.0` 均 7 行全出，且「零资料兜底」「闲聊」两个复读高发场景都不见复读（重复句占比 0.00，闲聊反而从 252 字缩到 85 字）。故默认 `LLM_REPEAT_PENALTY=1.15`，要照抄长表格的轮次再用 `LLM_REPEAT_PENALTY_STRICT=1.05`（`get_chat_llm(strict=True)`，由 `generate_node` 在补料块非空时启用）。抗复读靠第③道闸兜，不靠加码惩罚。
- **技能数值 / 突破材料确定性补料（不走检索）**：问技能自动附「满级数值表（Lv 10）」，问培养/突破自动附「突破材料表（一阶~六阶 + 各技能满级一档）」。为什么不用向量召回：技能数值表是 11 列宽表、突破材料每块仅 70~90 字，在 bge-reranker 眼里都输给大段机制描述——实测问「共鸣解放伤害倍率」时 **top6 里数值表 0 条**，且 top6 分数挤在 0.9983~0.9994 完全无区分度。改为按元数据直取：`retrieve.fetch_chunks(character, component)`（Chroma `get(where=…)`，零 embedding、零重排，只多一次本地查询）→ `graph._max_level_rows` / `_max_level_material` / `_material_items` **在 Python 里替模型读表**（8B 读 11 列宽表会串列，实测把 Lv7 读数读成中文数字）→ 拼成块注入 `build_context`。触发：`slots` 含「技能」或问句点名技能页签 → 数值表（优先问句点名的 `tab`，否则用本轮检索命中的 `tab`，都取不到则兜底「共鸣解放+共鸣技能」，保证问技能就有倍率）；问句符合 `nlu.is_material_question` → 材料表。
  ⚠️ 该判据**刻意比 `SLOT_PATTERNS["突破材料"]` 窄**，不是遗漏（2026-10-23 统一）：
  以前 `graph` 里另写了一份 `突破|材料|素材|培养|养成|练度`，实测 8 条问法里 **5 条两边不一致**
  （「60级需要什么」「升满要消耗多少」认槽位不认补料；「练度怎么提升」反之）。
  现在词表只有 `nlu` 那一份，且补料**不认**弱词与声骸/武器共现的句子——
  「声骸需要什么」问的是配装，硬附一张「必须逐条照抄」的突破材料表会把答案带偏。
  两者都只在 `characters` 非空时跑，异常只记日志、绝不影响主链。配套三条：① `build_prompt(..., blocks=…)` 在 prompt **末尾**（近因位）加硬要求——禁止「各需不同数量」这类概述，禁止把参考文档里**分等级展开**的长表拼成混合清单；② `prompt.SYSTEM_PROMPT` 加一条「逐行完整列出、`+`/`*`/`%` 原样照抄」；③ 补料块固定排在 `build_context` **最末**（紧贴 `## 问题`）——同参数下放 `## 资料` 第一节更容易被截断，末尾是第二道保险。实测效果：技能答案从「至于具体效果嘛…我记不清了啦」→ 7 行满级数值齐全；培养答案从跑题讲普攻连段 → 一~六阶 + 5 个技能满级材料逐条列出。

### `dialog/prompt.py`：提示词构建（2026-10-06 从 graph.py 拆出）

`graph.py` 原本 1573 行，混了图编排、节点、路由、验证闭环、补料、prompt 构建、流式控制八种职责。已把**纯字符串构建**这部分拆到 `dialog/prompt.py`：`SYSTEM_PROMPT`（常量）、`build_context`、`build_prompt`、`doc_sources`、`_lock_focus`。

- 拆分只搬代码、**未改任何函数体与提示词文本**（提示词是实测校准过的，逐字节不变很重要）。
- 公开函数去掉了下划线前缀（`_build_prompt` → `build_prompt`），因为跨模块调用了。
- `graph.py` 曾长期**再导出** `doc_sources`，只因 `api/app.py` 从 `dialog.graph` 导入它。2026-10-23 引用来源改成「文档面包屑 + 联网溯源」的统一口径 `prompt.all_sources` 后，app.py 改用它，`doc_sources` 那份纯转发随最后一个消费者一起删了——**留着就是无人使用的间接层**。守卫用例同步改钉 `graph.all_sources is prompt.all_sources`（`test_config_and_prompt.py`）。
- `graph.py` 因此不再需要 `lock_focus` 与 `chunk_text` 之外的部分文本工具导入；改 prompt 逻辑一律去 `prompt.py`，别在 `graph.py` 里加回来（加回去会形成两份实现，本项目对「同一语义两处判据」已有大量踩坑记录）。
- 架构守卫（`scripts/check_layers.py`）当前报 **57 模块 / 173 条依赖边 / 42 个有出边的模块 / 0 违规 / 0 环**，其中已包含 `dialog.prompt`（4 条出边：`config` `dialog.state` `text` `ww_logger`）与 `api.ratelimit`（2 条：`config` `ww_logger`），方向均合规。
  - 📌 2026-10-06 清掉了 5 个 2026-09-30 重构遗留的空壳包（`rag/` `graph/` `ingest/` `retrieval/` `storage/`，只含 `__init__.py`、贡献 0 条依赖边、全项目零引用），模块计数由虚增的 54 回落到真实的 **49**（边数 132 不变）。删除走的是**回收站**（可恢复）。此后随新模块增加，2026-10-07 为 **56 模块 / 164 边**，2026-10-23 复核 **56 模块 / 165 边**；同日 `/music/status` 的失败路径补日志，新增一条 `api.music → ww_logger` 出边 → **166 边**；同日反馈帽同源与 thread_id 判据收口 → **168 边**；2026-10-24 新增 `knowledge/corpcheck.py`（语料对账 CLI，见「语料对账」一节）→ **57 模块 / 173 边**（它的 5 条出边：`config` `core.db` `text` 是 L2→L0，`knowledge.index.bm25` `knowledge.graph.neo4j_client` 是同层，方向合规、无环）。
    - ⚠️ 最大的误删风险点：`wuwa_rag/graph`（空壳，已删）与 `wuwa_rag/knowledge/graph`（活跃包，含 `extract.py` / `neo4j_client.py` / `build_graph.py`）**同名不同物**。清理后已逐个核验 9 个活跃包完好。
    - 核验方式（可复用，零外部依赖）：守卫模块数应等于**分层小计之和**；`pkgutil.walk_packages` 扫描到的模块数应与守卫一致；被删的旧包名 `importlib.import_module` 应抛 `ModuleNotFoundError`；全部 py 文件 `py_compile` 通过。（2026-10-24 复核：**57 模块 = `3+10+18+2+8+9+7`，173 条依赖边，0 违规 0 环**。）
  - ⚠️ 守卫的模块数会把**只含 `__init__.py` 的空包**也计入。将来若见「模块数 > 分层小计之和」，先查是否又留下了空壳目录。

### ⚠️ 提示词铁律：只写正面要求（negative-example contamination）

**绝不要在 `prompt.SYSTEM_PROMPT` / prompt 里写「不要 XX，例如『…』」这类反例**——aemeath 会把点名的反例当成要模仿的样本照抄。三条实证（2026-09-22）：

| 提示词里写了什么 | 输出变成什么 |
|---|---|
| 「不要自行补充『没有提到其他/更多』」 | 结尾**稳定**出现「资料里没有提到其他组合啦。」（字面级一致） |
| 「不许给条目编序号或计数器（如『守岸人 + 尤诺*5』）」 | 开始出现 `[1]`~`[10]` 引用编号 |
| 「开场就是『我呀~』」 | 该开场概率性复现 |

对照实验（直连 Ollama、固定 seed=42、同一份资料）：

- prompt 里**点名** `[1] [2]` 禁止 → 仍写 `[1]`；**放末尾时更糟**（自编到 `[5]`）。
- 改用**泛化**措辞「不要标注来源序号或引用标记」→ **完全不出现**。
- 纯问题、不给资料 → 不写 `[n]` → 说明 `[n]` 是「有资料可依」这件事诱发的微调习惯，不是语料残留（资料里去掉方括号照样写）。

所以：**开发期的反例/踩坑记录写进 Python 注释，不要进 prompt**；硬要求一律用肯定句，并压在 prompt **最末**（近因位，见 `build_prompt` 的 `tail`；同一条规则写在 `prompt.SYSTEM_PROMPT` 里实测无效）。

### 输出侧兜底：来源标记 `[n]`、重复列表项、百分比单位、昵称引号（`graph._clean_answer`）

- **`[n]` 引用编号**：aemeath 只要「依据资料作答」就爱在句末缀 `[1]` `[2]`（会自编到 `[10]`，还把一个 `[1]` 重复用十几次）。提示词侧只能概率压住（实测 3 轮里仍有 2 轮），故由 `text.strip_ref_marks` / `text.AnswerFilter` 在**输出侧**剥离（流式逐 token，会先扣住疑似 `[` / `[1` 的尾巴再吐）。语料正文从不含 `[n]`，游戏术语用的是 `【】`，不受影响。
- **`## 参考文档` 不要加 `[n]` 编号**：加了必被抄进正文。前端引用面板走 SSE `done.sources`（`prompt.all_sources` = 文档面包屑 + `联网 › 标题 · url`），**不依赖模型写编号**；将来真要做「正文引用徽章」请换 `（资料1）` 这类非方括号标记。
- **重复列表项**：图谱事实与参考文档给的是同一批队伍、只是组合里角色名先后不同（图谱 `清宵+莫宁+达妮娅` vs 文档 `清宵+达妮娅+莫宁`），8B 读成「还有一批」又列一遍（4 支队伍列成 6 项）。`text.dedup_list_items` **只丢整行内容完全相同的列表行**——⚠️ 千万别升级成「语义合并」：技能/材料表的「满级 Lv10 一档」5 行材料完全相同、只有行首技能名不同，语义合并必误杀。
- **百分比单位串味（2026-09-30）**：问「六链是什么」时，资料原文 `暴击固定为80%，暴击伤害固定为275%`（`data/raw/爱弥斯.md` 共鸣链六链）被复述成「固定**八十万**暴击伤害」+「**两百七十五**暴击伤害」——三种走形同时出现：① 凭空加量纲（`80%`→`八十万`）；② 丢单位（`275%`→`两百七十五`）；③ 把「暴击率」与「暴击伤害」两个分句并把化。补了两道：
  - **prompt 侧**（`build_prompt` 的输出格式块第 3 条 + `persona.DOMAIN_RULES`）：「带单位或符号的数字一律照原样抄回，百分号必须跟着数字，不擅自加『万』『亿』」。原先这条**只作用于「满级数值表/突破材料表」**，共鸣链不在范围内 —— 这正是漏网的根因。
  - **⚠️ 同时修掉一个更致命的短路**：`build_prompt` 里原有 `if not blocks: return prompt + tail`，导致**无补料场景（共鸣链、剧情、机制问答）拿不到任何输出格式约束**。现在无条件注入；且无补料分支**绝不能点名「满级数值表」**（模型会以为该有那张表，跑去参考文档里翻找并凭空扩列）。
  - **规则侧**（`text.fix_percent_units`）：术语白名单 + 裸数值 → 补 `%`。**改这条正则前必读 `text.py` 里的 6 条注释**，每一条都是踩过才写的：
    - 数值只用 `\d+(?:\.\d+)?`，**上限必须放开**（写过 `[0-9]{1,3}` → `275%` 被回溯切成 `27%5%`）；
    - 数值后排除集必须含 **`.`**（漏了 → 全语料 479 行误伤，`1.20%` 变 `1%.20%`）；
    - 数值后排除集必须含 **`*`**（markdown 加粗会劈开数字，`**2**.80%` 补 `%` 是固化残渣）；
    - 术语与数值之间**只允许空白与冒号，不跨汉字**；「固定为/提升」这类连接后缀**逐个枚举进 `_PERMILL_TERMS`**，不写可选组（写了会让中文数字分支吞掉「定为」）；
    - 数值后**不得紧跟另一个术语**（`_RE_TERM_AHEAD`）：`攻击力：47暴击伤害：10.8%` 里的 `47` 是固定值，不排除就会误补成 `47%`；
    - **不支持中文数字**：模型把 `275%` 写成「两百七十五」本身就是违规改写，补 `%` 只会固化它。
    - 验证口径：全语料 51198 行只应有 6 行改动（`暴击提升2.80`，原文自身缺 `%`），多于此即回归。
- **昵称外层的引号（2026-10-10）**：模型叙述用户时爱给称呼套「」/“”（实测答案「「颗粒」刚刚让我放…」）。按提示词铁律**不写**「别加引号」这类负向指令（negative-example 污染），改为 `text.unquote_mentions` 确定性剥离：只剥**恰好包裹画像昵称**的那一层引号对，歌名《…》、术语【…】、别的「强调」一概不动。昵称词由 `services/profile.nicknames_from_context(注入串)` 现取——画像事实本身是干净的（`用户自称是颗粒`），所以「剥谁的」必须从 `user_context` 反解，不能写死。
- ⚠️ **四步清洗只能有一处实现：`graph._clean_answer(raw, user_context)`**（2026-10-10 收敛）。`generate_node`、`_chat_turn`（chitchat / time / clarify 三条不检索分支的共用出口）与 `ask_stream` 收尾**都要过这四步**，而流式的 `done.answer` 是**旁路重算**的（token 从 `on_chat_model_stream` 抽，节点里那份只进 history 与「截断时覆盖」分支）。本轮评审就是抓到 `unquote_mentions` 只加进了节点、漏在 `ask_stream`，表现成「流式回答还带「」，下一轮 history 里却已剥掉」——同一语义两处判据的必然结局。
  **2026-10-23 又抓到同一个洞的另一半**：`_chat_turn` 只手工做了第 ④ 步，① ② ③ 在闲聊/时间/澄清
  三条分支上**全部缺席**——`[n]` 角标、丢百分号、重复列表行原样进 history 与非流式 `/ask`，
  表现成「检索轮干净、闲聊轮带 [1]」。现在三处都调 `_clean_answer`，并由
  `tests/test_text.py::test_两条输出口都只过唯一清洗实现` 用**源码判据**钉住（出口里单独调任何
  一步就红——造一次 LLM 调用才能验的断言不符合本仓的全量离线约定）。新增清洗规则一律改 `_clean_answer`，并往 `tests/test_text.py::test_清洗管线四步都在` 加一条断言（该用例锁住「四步都在」）。

### 队友槽位：不能输出「队友→队伍」反查表

`graph_search` 的队友槽位**特判**为去重的完整队伍列表：

```
【可组队伍（每行是一支完整队伍，逐行照抄即可）】
  清宵+守岸人+琳奈
  清宵+守岸人+尤诺
  ...
```

**不要退回通用的 `k=v` 反查表**。原因：反查表同时给出「名字池」和「组合池」，模型会做笛卡尔积混搭——实测吐出源数据里根本没有的「清宵+莫宁+尤诺」，并把同一支队伍列两遍（用户报「严重幻觉」）。

**⚠️⚠️ 2026-09-22 深挖（问「守岸人配队」只拿到 2 行占位符的根因）**

1. **不能只查出边**，要把入边一起拿。抽取方向是「本角色 → 标题里提到的队友」，而奶辅类角色大量出现在**别人的页面**里。实测守岸人：**出边 2 条**（`主输出`/`副输出`，即通用模板行「守岸人配队｜主输出+副输出」抽出的占位名）、**入边 36 条**。只查出边 → 模型手里只有 2 行占位符 → 只能自己编，这才是「守岸人 / 维里奈 / 白芷」式幻觉的上游。队友关系本就对称。**落地方式**：Cypher 用两条有向 MATCH + `UNION`（而不是一条无向 `-[r]-`）——因为第 2 点还需要把方向带出来。
2. **补全要用「页面主人」，不是「提问角色」**。旧写法 `CASE WHEN x CONTAINS $n THEN x ELSE $n+'+'+x END` 是**错判据**：`洛可可` 页的标题 `椿+守岸人` 恰好含「守岸人」，被判为「已完整」原样返回，真实队伍 `洛可可+椿+守岸人` 丢了一个人（守岸人一次返回 50 支，一半是 `椿+守岸人` 这种缺主人的两人片段）。故 Cypher 用 `UNION` 分两路把方向带出来（出边主人 `$n` / 入边主人 `t.name`），补全与「或」收窄都在 Python 侧做。
   - 已提到 3 段（`+` 段 ≥ 3）的队伍**不再补主人**——鸣潮只有 3 个位置，3 段即完整。反例：`漂泊者-男-衍射` 页面上的 `守岸人/维里奈/白芷+秧秧+漂泊者-衍射`（主人名与串内写法不同，`owner not in team` 判不出已含主人），硬补会得到 4 人串。
3. **占位串要滤**：`_is_placeholder_team` —— 去掉槽位词（主输出/副输出/其他输出/奶…）后真人名 ≤ 1 个即丢弃，杀掉 `守岸人+主输出+副输出`。
4. **一条 `teams` 值可能是两支队伍**：`折枝` 页存的是 `作为副输出：折枝+今汐/珂莱塔+守岸人/维里奈/白芷；作为主输出：折枝+散华/釉瑚+…`，整串当一支念出来是 6 个人。`entities.split_teams`（2026-10-24 从 `retrieve._split_teams` 挪来，**抽取侧与读取侧共用同一份**）按 `；`/`;`/换行 拆、再取 `：` 后段。
5. **镜像去重（忽略顺序）**：同一支队伍会被两个角色的页面各记一遍（吟霖页写 `灯灯+守岸人`、灯灯页写 `吟霖+守岸人`），补全主人后成为两队镜像。`_team_key` 摊平名字排序作指纹，实测守岸人 57 支里含 10 对镜像。
6. **片段剪枝**：`仇远+守岸人` ⊂ `仇远+嘉贝莉娜+守岸人`，只留完整队。**按「位置组」比较**（`_team_groups`），不能摊平名字——摊平会让 `…+相里要/卡卡罗/今汐/渊武其他输出+…` 看起来覆盖掉真队伍 `卡卡罗+相里要+守岸人`（首版真踩到）。且**只剪不足 3 段的**，3 段队一律保留。

配套两处，改一处必须核对另一处：

- `extract._extract_teammates` 存 `team` 时**补上本角色**：wiki 标题是 `#### 守岸人+尤诺`（整页都是清宵的，标题只写另外两个队友），原样存会让模型把「守岸人+尤诺」读成「守岸人和尤诺是一对」。
- **⚠️ 这条修复只对新爬生效**；存量图谱靠上面第 2 点的「页面主人」补全兜住，**但不重建也能立刻生效**（补全发生在查询期，幂等）。

### 配队「或」语义 + 问谁锁谁（`text.lock_focus`）

鸣潮配队写法 `A/B/C+D/E+F+G`：`/` 是同位置**三选一（或）**，`+` 是不同位置（和），一支队伍只有 3 个人。

- **术语对照保留在 `prompt.SYSTEM_PROMPT`**：只讲 `/`=或、`+`=和、一队 3 人（正面陈述，无污染风险）。
- **「问的是谁，那一组只列他」不要写提示词**——写过，实测**完全不生效**（问守岸人仍输出 `守岸人 / 维里奈 / 白芷`）。与 `[n]` 是同一个教训：8B 对长 `prompt.SYSTEM_PROMPT` 里的规则遵守度低。改为 `text.lock_focus` 的**确定性字符串变换**：进 prompt 之前把含本次角色的「或」组收窄成他本人。
- **必须两侧都锁**，否则形态不一致、锁定等于白做：
  - 图谱侧 → `retrieve.graph_search` 队友分支（拿到 `char`，逐串锁）；
  - 文档侧 → `prompt._lock_focus` 包 `build_context` 里的 `## 参考文档`（否则图谱给锁定形态、文档给原始三项，8B 会挑文档那份抄回去）。
- 实现要点：正则只吃**斜杠短语** `token(/token)+`，token 不跨空白/句读/`+`/`|`；必须 `focus in parts` 精确命中才替换；`focus not in text` 直接短路。幂等、可单测。
- ⚠️ **锁定的范围**：只作用于「图谱事实的队友块」+「## 参考文档」。**绝不要**去动 `_value_blocks` 产出的「满级数值表 / 突破材料表」（那是确定性补料，与配队无关）。

#### 奶辅类角色的队伍会「多到倾倒」→ 已加上限 `TEAM_MAX_SHOWN`

实测（2026-09-22）：问「守岸人的配队」→ 图谱给出 **53 支**（她是奶辅，wiki 上**别人的**页面里到处都有她，
53 支就是真值，不是 bug）→ aemeath **逐行原样输出 47 行**，787 字通篇清单；无幻觉、无 `[n]`，
但拿到的是「一支不漏的原始清单」而不是「配队建议」。

**已落地**：`config.TEAM_MAX_SHOWN`（默认 **12**，`0` = 不限）。

**排序键 `(-t.count("/"), -t.count("+"), t)`**：
① **含「或」的模板优先**——`A+B/C+D+E` 这种形态来自 wiki「配队推荐」表
（`| 卡卡罗配队 | 守岸人/维里奈/白芷+吟霖/长离/散华+卡卡罗 |`），**一支模板覆盖多支具体队，信息量最大**；
不带 `/` 的三人串来自别人的「主流队友」双人格（`| 队伍组成 | 卡卡罗+吟霖+守岸人 |`），低一档；
② 「或」候选越多越先（一个格子能包的位置越多，覆盖的队越多）；③ `+` 段数多（位置定得越全）。

⚠️ **别用「长度」做次序**（首版真踩到）：`守岸人+吟霖/长离/散华+卡卡罗`（16 字）会被一堆 12 字的
双人格串挤出上限之外，而它恰恰是最该出现的那一支（也是当初的样例形态）。
换成当前键后它稳定在**第 4 位**。

#### 泛问 / 多人 / 指名：三种问法对应三种图谱行为（2026-09-22 起）

规则：**优先展示一个队伍可包多个队伍的（多/的）**，比如 `守岸人+吟霖/长离/散华+卡卡罗`；
**有包含在内的队伍比如 `守岸人+吟霖+卡卡罗` 在只问配队情况下不展示，再指明该队伍时展示并向量检索**。

| 问法 | 例子 | 图谱侧行为 | **检索路由** |
|---|---|---|---|
| **泛问** | `守岸人配队`（1 个名字） | 只出模板队；被**逐位**覆盖的条目（具体队**和**模板）**都剪掉**；只锁 `守岸人` | **只走图谱**（不向量，`intent=fact`）+ 答完追问一句 |
| **多人** | `守岸人和吟霖配队`（2 个名字） | 同样剪枝；只出「两人都在其中」的队；跨角色段去重 | **只走图谱**（不向量，`intent=fact`）+ 答完追问一句 |
| **指名** | `守岸人+吟霖+卡卡罗配队`（3 个名字） | **精确**：全锁三名字 → 模板收窄成那一支 → 只出它 | 图谱 **+ 强制向量**（`intent=hybrid`，打法循环在正文） |

判定在 `retrieve.graph_search`（队友那串输出已拆到 `retrieve._teams_block`，判据一行未改，改动需按该函数文档里的逐角色比对验收），用 `extract_characters` 的命中个数切三档：
`has_multi = len(named) >= 2`、`is_exact = len(named) >= TEAM_SLOTS`——**鸣潮一队只有 3 人，
凑满 3 个名字才算「指明一支具体队伍」**。这个 3 自 2026-10-23 起只写在
`knowledge/entities.TEAM_SLOTS` 一处（`retrieve` 的段数判断与 `graph._is_named_team` 都取它）；
此前它在两个模块各有 `_TEAM_SLOTS = 3` + 一处字面量 `>= 3`，而这个阈值真的从 2 改到过 3。
⚠️ 实测踩到过：一开始两者都按 `>=2` 判，问「守岸人和吟霖配队」时会跳过覆盖剪枝，
`守岸人+吟霖+卡卡罗` 和覆盖它的模板同时出现（同一件事说两遍，7 支里混着 1 支冗余）。

- **锁角色范围**：泛问只锁当前 `char`；点名多人时把点到的名字**全锁**——
  模板 `守岸人+吟霖/长离/散华+卡卡罗` 因此被锁成用户点的那支 `守岸人+吟霖+卡卡罗`，
  与数据里真实存在的同串经 `_team_key` 去重合并，答案干净落到那一支（不是「模板 + 具体队」各列一遍）。
- **跨角色段去重**（函数级 `named_shown`）：指名时守岸人/吟霖/卡卡罗会**各查一遍图**，
  同一支队在三段里各出现一遍——实测 3 段共 **25 支、目标队重复 3 遍**。给模型三份重复清单，
  它会照列三遍甚至混搭。点名多人时全局去重，第二、三段只列「新命中」：**25 支 → 1 支**。
  命中为空则回落全量（用户点的组合数据里真没有时，宁可给全景也别给空答案）。
- **覆盖剪枝的判据是「逐位」不是「摊平」**（`_covers`）：`守岸人+吟霖/长离/散华+卡卡罗`
  覆盖 `守岸人+吟霖+卡卡罗`（吟霖正是中间那个位置的备选），但**不**覆盖 `吟霖+守岸人+散华`
  （散华不在模板的任何位置）。若用**摊平的名字集合**判，`守岸人+吟霖+长离` 也会被判成「被覆盖」——
  那组合同位置塞了两人，本就不成立，剪它等于掩盖错误。实现：位置组 `frozenset` +
  `itertools.permutations` 穷举（3 段只有 6 种排法），且只比**位置数相同**的两串
  （「2 段片段 vs 3 段完整队」归片段剪枝管，那是另一条规则）。
- **⚠️ 2026-09-22 扩围：模板之间也剪**（判据：被前者逐位完全覆盖的，覆盖掉）。
  原来这道剪枝的保留条件是「**是模板** or 不被模板覆盖」→ **模板永远豁免**，于是
  `守岸人/维里奈/白芷+洛可可+漂泊者-男-湮灭` 明明被下一条逐位完全覆盖，却仍被列出：
  `守岸人/维里奈/白芷+洛可可+椿/漂泊者-男-湮灭`（第 3 个位置多一个 `椿` 备选）。
  现在**所有条目一律参与判定**，只剪「被别的条目**严格**覆盖」的。
  - **严格 = 对方覆盖我、我不覆盖对方**。判据绝不能写成「被覆盖就剪」：**等价的两条**
    （位置组一一对应）会**互相**覆盖，按「被覆盖就剪」会**两边一起消失**（成对丢数据）。
    等价时只留排序键最优的一条 —— 先把 `parsed` 按排序键排好，再把「位置组**多重集合**」
    （`canon = tuple(sorted(tuple(sorted(s)) for s in g))`）相同的丢掉后面那些。
    （真实图里等价对已被更早的 `_team_key` 摊平去重挡掉，这条是防回归的保险。）
  - **实测收益**：`洛可可` 4→**3** 支；`忌炎` 4→**2** 支 ——
    `守岸人/维里奈/白芷+秋水+忌炎` 与 `…+莫特斐+忌炎` 都被
    `守岸人/维里奈/白芷+忌炎+莫特斐/秋水` 逐位覆盖。
  - **无回归**：`守岸人` 仍 12 支且**全含 `/`**、`白芷` 12 支、`清宵` 4 支、`夏空` 3 支；
    指名 `守岸人+吟霖+卡卡罗` 仍**恰 1 支**（指名跳过本步）；出边归零的角色（白芷）不崩。
  - 覆盖判据自检（合成数据，零 LLM）：严格对 `covers(A,B)=True / covers(B,A)=False` →
    剪 B；等价对 `covers(C,D)=covers(D,C)=True` → **严格判据两边都判 False**（不剪）。
  - ✅ **当时「未解决」的那根 5 段串已于 2026-10-24 解决**，而且根因不在剪枝判据：
    `折枝+今汐/珂莱塔+守岸人/维里奈/白芷+散华/釉瑚+守岸人/维里奈/白芷` 是**抽取侧**造成的 ——
    `extract._extract_teammates` 把整格文本直接丢给 `normalize_team`，而它只认 `+` 与 `/`，
    于是 `；` 和「作为主输出：」在归一途中被吃掉，进图后剩下的一根 5 段串**读取侧再拆也来不及**。
    修法是把 `split_teams` 挪进 `entities`（同一语义只准有一处判据）并**在抽取侧两个来源都先拆再归一**。
    ⚠️ 重跑 `wuwa-graph` 时又抓到第二件事，比这条本身更值钱：**队友关系是 `SET rel.teams = 去重累加`，
    只 upsert 不清边**，所以旧形态与新形态**并存**（实测：拆好的两条进去了，5 段老串还在，
    守岸人于是同时给出「一支 6 人队」和「两支 3 人队」，看上去像抽取没修好）。
    现在 `build_graph._main` 每个角色**先 `delete_character(n)` 再 upsert**（只删**出边**，
    入边属于别人的页面、本轮不会由别人重建 —— 三条铁律见 `_C_DELETE_CHAR`）。
    实测重建前后：Character 58 → 58（不变）、`teams` 条目 631 → **372**（陈旧形态清空）、
    ≥4 段的串 1 → **0**；读取侧回归：`守岸人` 泛问仍 12 支且**全含 `/`**、`折枝` 出两支各 3 段、
    指名 `守岸人+吟霖+卡卡罗` **恰 1 支**。抽取侧新增一条离线用例（判据：抽出的每支队伍
    `+` 段数不得超过 `TEAM_SLOTS`），它单独就能把「只在读取侧拆」这种修法判红。
- **检索路由（2026-09-22 改定）**：
  - **概括性配队只走图谱，且 intent 如实标 `fact`**：`graph._is_team_overview`（slots 恰为
    `['队友']` 且**未指名**）→ **`intent_node` 把 intent 从 `hybrid` 降级为 `fact`**（唯一落点，
    见 `graph.py` 的 `intent_node` 里 `_is_team_overview` 降级处注释）。为什么必须在这里降级而不在 `_after_graph`：① `_after_graph`
    是**路由函数**，只能返回分支名，改不了 state；intent 要如实落进 state 对外（`AskOut.intent` /
    SSE `done.intent`）。② 同一个语义挂两条判据必然迟早不同步。「配队」一词同时命中
    `nlu.SLOT_PATTERNS`（事实槽位）与 `SEMANTIC_PATTERNS`（语义），`classify()` **必然**给出
    `hybrid`（实测确认），所以必须在 `intent_node` 里**改**它，而不是在 `_after_graph` 里加例外。
    为什么不走向量也不丢信息：图谱泛问给的就是含「或」的模板，已完整；走向量只会把正文里**别的**
    队伍的打法描述召回来（答非所问），还白等一次检索 + 重排（实测约 15~19s）。
  - **指名才补向量**：`graph._is_named_team` = **≥3** 个角色名 **且** 含队友槽位 → `_after_graph`
    返回 `need_vector`。⚠️ 阈值 2026-09-22 由 **2 提到 3**：2 个名字时用户还在挑（图谱会列出
    所有「含这两人」的队），补的向量是「这帮人相关的正文」，同样答非所问。鸣潮一队只有 3 个人，
    **凑满 3 个名字才算点名那一支**。理由同技能槽位：**队名在图谱、打法循环在正文描述里**
    （wiki 配队页正文有出招顺序，如「守岸人：AAAA-Z-E-Q-AAAA-Z-R」）。
  - ⚠️ 判据限定 `slots == ['队友']`：同时还有别的槽位时（`守岸人配队和声骸` → `['队友','声骸']`）
    intent **不降级**（仍 `hybrid`）且必须走向量 —— 别一刀切。
  - ⚠️ `_is_named_team` / `_is_team_overview` 是**模块级唯一真相源**（state 版包装 `_named_team` /
    `_team_overview`）。两处消费它、口径必须一致：`intent_node`（命中则降级为 fact）与
    `_should_ask_team`（命中则追问）。**不要再在 `_after_graph` 里加 `if _team_overview(state)`**。
  - ⚠️ `守岸人和吟霖谁更强`这类 2 人问句**不**触发（无队友槽位），已单测覆盖。
- **答完追问一句（2026-09-22 起）**：概括性配队回答完后，答案末尾追加
  `graph._TEAM_FOLLOWUP`＝「你对哪个队伍感兴趣？需要我给你详细介绍一下吗~」，引导用户点名具体队伍。
  - 走 **确定性拼接**，不用提示词求模型生成 —— 这类"元话语"8B 写得千奇百怪、时有时无，
    写进提示词还有 negative-example 污染风险。
  - 触发条件 `graph._should_ask_team`：**① 概括性配队** **且** **② 图谱侧确实给出了队伍**
    （`graph_facts` 里出现「【可组队伍」）—— 角色不在库 / 该角色 wiki 没有配队段时不问，
    否则是无源之水。
  - **截断不加**（复读闸截断时，半截答案后面跟一句追问很突兀）；**LLM 调用失败不加**。
  - ⚠️ **流式必须单独补吐**：`ask_stream` 的 token 是**旁路抽取**（`on_chat_model_stream`），
    只有模型生成的正文，`generate_node` 补的那句不在 token 流里。所以 `ask_stream` 收尾时用
    **同一判据**再 `yield {"token": "\n\n" + _TEAM_FOLLOWUP}`，否则前端少显示一句、
    与 `done.answer` 不一致（用 `endswith` 防重复）。
- **指名时把注意力钉在那一支**：`prompt.build_prompt(..., team_focus=_named_team(state))` 在 prompt
  **末尾**（近因位）加一句「用户已经点名了一支具体队伍，本轮就围绕这一支展开：成员是谁、
  怎么打（出手顺序 / 循环）、为什么这么配。」
  ⚠️ 措辞刻意用「围绕这一支展开」而不是「只介绍这一支」—— 本文件记过教训：「只」字会让模型把
  介绍性口吻整个砍掉、退化成机械罗列。
- **验证脚本形态（可复用，零 LLM）**：`await graph_search(["守岸人"], ["队友"])` 打印逐行，
  与 `await graph_search(["守岸人","吟霖","卡卡罗"], ["队友"])` 对照；断言「泛问结果全部含 `/`」、
  「指名结果恰为 1 支」。**全程不需要 LLM**——`rewrite_query` 无 history 时短路、
  `classify_topic` 有角色时不调用，所以**生成模型没起也能验图谱侧**。
  更完整的一条：手动跑 `intent_node → graph_node → build_context`，直接看**真正进 prompt 的文本**。
- **验证检索路由/intent 一致性（零 LLM，stub 掉 rewrite_query 与 _known_characters）**：
  `await graph.intent_node({"question": q})` 取 `intent`（断言 `守岸人配队`/`守岸人和吟霖配队`
  → `fact`；`守岸人+吟霖+卡卡罗配队`/`守岸人配队和声骸` → `hybrid`），再对同 state 调
  `graph._route` 与 `graph._after_graph`（后者需先塞 `graph_facts` 含「【可组队伍」）验证
  `fact→done`、`hybrid→need_vector`、`_should_ask_team` 的真假。**核心不变量**：
  `intent == "fact" ⇔ docs 为空`（对外字段不得名不副实）。
  ⚠️ **必须同时断言 `characters`，只断言 `intent` 会假绿**：2026-10-06 实测踩过 ——
  `心配队` 的角色识别已经退化成 `chars=[]`，但 `_is_team_overview` 的判据是
  「`slots == ['队友']` 且未指名」，`chars=[]` 反而**更**满足「未指名」，intent 照样是
  `fact`，测试全绿而功能已废。凡涉及角色识别的断言，`intent` 与 `characters` 缺一不可。

#### ⚠️⚠️ 单字角色名（心 / 椿）识别（2026-10-06 已修：`entities.find_mentions`）

**用户报的现象**：新角色「心」（Hsin）爬取入库后，提问仍然「判断为走联网」。

**根因链**（四环，缺一不成立，已逐环实测）：

1. 角色「心」**已在库里**（PG `documents` + 169 个 chunks + Neo4j `Character` 节点都在），
   所以问题不在爬取，在**识别**。
2. `nlu.extract_characters` 有一道 `len(n) >= 2` 过滤（旧注释写「避免『他/她』这类误命中」），
   单字名被一刀切滤掉 → `extract_characters('心的声骸怎么配', known)` 返回 `[]`。
3. `characters` 恒空 → `graph.verify_node` 的「按角色清库重爬」分支要求 `chars` 非空
   （`if chars and not state.get("refreshed")`）**永不触发**；同时 `retrieve.graph_search`
   开头 `if not characters ... return ""`，图谱事实也是空。
4. 资料空 → verifier 判不匹配 → 重检索一轮仍空 → 额度耗尽 → `verify_stage = "web"` →
   **落到联网兜底**（`_web_available()` 为真时）。实测复现：`chars=[]` + `retry_count=1`
   的 state 进 `verify_node`，返回的正是 `web`。

**修法**：单字名不能靠子串匹配（「心」是高频汉字：核心/中心/关心/决心/心情），也不能一刀切滤掉。
改为**单字名要求「独立成词」**，多字名沿用子串匹配 —— 判据集中在 `entities.find_mentions`，
`nlu.extract_characters`、`entities._rule_candidates`、`graph._inject_far_characters` 三处**共用它**
（原先三处各写一套：`nlu` 是 `len>=2` 子串、`entities` 是裸子串、`graph` 是长度降序正则，
同语义三处判据必然不同步，本项目对此已有大量踩坑记录）。

| 判据 | 适用 | 理由 |
|---|---|---|
| 正则子串（长度降序） | 多字名 | 行为已实测稳定；且 `漂泊者-男-衍射` 这类带连字符的名字会被分词器切开，不适合走分词 |
| 分词后**整词相等** | 单字名 | 只有独立成词才算提及，天然排除「核心/关心」 |

⚠️ **三条实现红线**（都是实测踩出来的）：

1. **必须 `HMM=False`**。HMM 是 jieba 的新词发现，会把「单字角色名 + 紧邻普通字」臆造成
   未登录词：`心配队` → `['心配','队']`、`心和椿配队` → `['心和椿','配队']`，而
   **`心配` 根本不在 jieba 词典里**（`tok.FREQ` 查无，纯 HMM 臆造）。关掉后
   `心配队` → `['心','配','队']`，而「核心/关心/开心/中心思想/决心」这些**词典里的真词**
   仍然完整不拆（它们靠 FREQ 权重切分，不依赖 HMM），负样本零退化。
   代价：未登录领域词（`声骸` 不在词典）会被拆成 `['声','骸']`，但无影响 ——
   本函数只关心「切出的词是否**恰好等于**某个单字角色名」，多字名压根不走分词。
2. **绝不能用全局 `jieba.add_word`**。BM25 稀疏路的 pickle 内含**自定义词典快照**，
   建索引与查询必须分词一致（见 `knowledge/index/bm25.py`）；污染全局词典会让已建好的
   BM25 索引与查询侧分词不一致，召回质量静默劣化。故用 `jieba.Tokenizer()` **独立实例**。
3. **分词命中的单字名不参与子串消歧**。消歧规则是「A 是 B 的子串则剔 A」，而 `鉴心` 含 `心`
   —— 参与消歧会把「鉴心和心谁强」里的 `心` 误删（实测两个角色都在比较句中，应同时命中）。
   单字名由「独立成词」保证精确，无需消歧。

**性能**：`jieba.Tokenizer` 首次构建 **302~400ms**（加载词典），之后单次 `cut` **0.042~0.073ms**
（约 6000 倍差距）。`_TOK_CACHE` 以 `frozenset(名册)` 为键缓存，名册只在角色入库/删除时才变，
正常问答路径命中缓存、零开销；相对检索链路的 19s 完全可忽略。⚠️ 别改成每次新建实例。

**回归基线**（可复用，零 LLM、零 Neo4j；`stub` 掉 `rewrite_query` 与 `_known_characters`）：

```
正样本 12 条（chars 必须完全匹配）：
  心配队→[心]  心和椿配队→[心,椿]  心的声骸→[心]  心的队友→[心]
  心+椿+守岸人配队→[心,椿,守岸人]  椿配队→[椿]  心怎么配队→[心]  心的突破材料→[心]
  卡卡罗和心谁强→[卡卡罗,心]  鉴心和心谁强→[鉴心,心]  秧秧·玄翎的武器→[秧秧·玄翎]  守岸人配队→[守岸人]
负样本 10 条（必须全空）：
  核心玩法是什么 / 这个中心思想 / 我很关心剧情 / 开心 / 心情不错 /
  决心要练她 / 用心练她 / 声骸核心词条 / 这个中心很关键 / 担心自己练不好
配队路由（intent + chars 双断言，含原基线 5 条 + 单字名 5 条）：
  守岸人配队→fact[守岸人]  守岸人和吟霖配队→fact[守岸人,吟霖]
  守岸人+吟霖+卡卡罗配队→hybrid[三人]  守岸人配队和声骸→hybrid[守岸人]
  守岸人和吟霖谁更强→hybrid[守岸人,吟霖]
  心配队→fact[心]  心和椿配队→fact[心,椿]  心+椿+守岸人配队→hybrid[三人]
  心的队友→fact[心]  心和守岸人配队→fact[心,守岸人]
实测结果：全部 BAD=0（22 条提及判定 + 10 条配队路由）。
```

#### 纯自我介绍走 chitchat（2026-10-06 已修：`nlu.is_self_intro`）

**用户报的现象**：发「我是颗粒」没进用户画像，而是「走了联网查知识」。

画像那半边其实是**正常的**——查库确认 `用户自称是颗粒` 已入库（`admin`，03:46:47）。
用户感知到「没记住」的真因是：意图走错了 → 答案里全是联网搜来的「颗粒」知识，
**完全没体现昵称**。所以这条与上一条是**同一根因的两个症状**：都源于「识别不出这是闲聊/自我介绍」
→ 走了检索 → 空手 → 升级联网。

**根因**：`classify()` 对「无槽位且无语义词」的句子兜底返回 **`hybrid`**（安全兜底：宁可多查），
于是「我是颗粒」触发全量检索；而 chitchat 原先**完全依赖** `classify_topic` 的 LLM 判定
（只在「无角色名 且 无槽位 且 无属性/阶段」时才调），LLM 判成 `game` 或不可用就一路走到联网。

**修法**：新增 `is_self_intro` 规则硬信号，与 `is_identity` 并列进 chitchat 分流
（`elif (is_identity(q) or is_self_intro(q)) and not slots`）。两者方向相反、语义不重叠：
`is_identity` 问**助手自己**（第二人称「你是谁」），`is_self_intro` 陈述**用户自己**（第一人称「我是X」）。
实测 `is_identity` 的四个样本（你是谁/你叫什么/你的名字/你的台词）都是 `identity=True, self_intro=False`，无重叠。

⚠️ **实测校正过的两处**（写错过一次，别改回去）：

- 昵称字符类要含拉丁字母与数字（`我的游戏ID是颗粒`），且**必须显式排疑问词**：
  `谁/什么/啥/哪/吗/呢/吧/玩家/人/六属性名` 都是合法昵称字符，不排除的话
  「我是谁」「我叫什么」「我是玩家吗」「我是导电属性的吗」全被当成自我介绍
  —— 最后一条是**真游戏提问**，绝不能抢走。
- 昵称位必须是**捕获组**：`is_self_intro` 靠 `m.groups()` 取昵称交给排除表过滤。
  写成非捕获时 `groups()` 恒空、过滤整段变成死代码（首版实测 5/30 失败，
  而注释却写着「已排除疑问句」——**注释里的断言必须实测过再写**）。

**回归基线（30 条，实测 BAD=0）**：正样本 20（我是颗粒/我叫颗粒/我是萌新/我是新来的/我是新人/
你可以叫我颗粒/叫我颗粒就行/大家好我是颗粒/嗯我是颗粒/记住我叫颗粒/我的游戏ID是颗粒/
我的名字是小星/喊我阿明/人家是新手/我是老玩家/我是回归玩家/我叫椿/我是心/我是Hsin/叫我小星吧）；
负样本 10（我是萌新，守岸人怎么玩 / 守岸人怎么玩 / 我是谁 / 你是谁 / 心的声骸怎么配 /
我是来问问题的，卡卡罗配队 / 你觉得我是谁 / 我是导电属性的吗 / 我叫什么 / 我是玩家吗）。

⚠️ 复合句不被抢走靠**双保险**：① 昵称用排除句读的字符类，「我是萌新，守岸人怎么玩」在第一个
逗号处就断掉匹配；② 分流条件带 `and not slots`。实测该句 → `semantic, chars=[守岸人]`，检索照常。

**最坏情形已验证**：把 `classify_topic` stub 成恒返回 `game`（模拟 LLM 判错/不可用），
6 条自我介绍句**仍全部**走 chitchat —— 规则信号独立兜住了，不依赖 LLM。

#### 角色名归一（2026-09-22 已修：`entities.normalize_character_name` / `normalize_team`）

wiki 的「配队推荐」表把「角色名 + 位置标签/注释」挤在同一格，还有一批别名写法。这些串被
`_extract_teammates` 原样存进 Neo4j 的 `teams`、再原样进 prompt，aemeath 就会念出
「渊武其他输出」「维里奈（高熟练度）」这种非角色名；同时 `_team_key` 按名字精确排序去重，
别名变体又让**镜像队躲过去重**（实测问「忌炎配队」同时吐出 `莫特斐+漂泊者·气动` 与
`漂泊者-男-气动+莫特斐`，实质同一支）。

实测盘点（245 条 `SYNERGIZES_WITH` 关系的全部 teams 串）：**23 个非名册 token**，五类 ——

| 类 | 例子 | 计数 |
|---|---|---|
| ① 漂泊者变体（缺 `男-`） | `漂泊者·湮灭` / `漂泊者-湮灭` / `漂泊者·气动` / `漂泊者·衍射` / `漂泊者·导电` | 32 处 |
| ② 括号注释 | `维里奈（高熟练度）`、`夏空（进阶轴，卡提双三剑下落）`、`莫宁（0链爱）`、`千咲（2链绯雪）` | 12 处 |
| ③ 位置标签粘连 | `渊武其他输出`、`凌阳等主输出` | 11 处 |
| ④ 纯位置标签 | `主输出`、`副输出` | 16 处 |
| ⑤ 说明文字 / 全角未拆 | `或者作为奶位配合任意队伍`、`釉瑚＋散华＋折枝` | 2 处 |

⚠️ ③ **不是抽取 bug** —— wiki 原文就这么写（grep `data/raw` 可复现，各出现 1 次）：
`| 吟霖配队 | 守岸人/维里奈/白芷+吟霖+相里要/卡卡罗/今汐+渊武其他输出 |`，
读作「渊武【其他输出】」。

别名口径（`CHARACTER_ALIASES`）：`光主→漂泊者-男-衍射`、`风主→漂泊者-男-气动`、
`暗主→漂泊者-男-湮灭`、`电主→漂泊者-男-导电`、`卡提→卡提希娅`。

**归一链（`normalize_character_name`）**：

```
维里奈（高熟练度）             -> 剥括号               -> 维里奈
漂泊者·湮灭                   -> 补男属性             -> 漂泊者-男-湮灭
渊武其他输出                  -> 剥位置词             -> 渊武
凌阳等主输出                  -> 剥「主输出」再剥「等」  -> 凌阳
折枝 或者作为奶位配合任意队伍    -> 前缀最长匹配（len>4 才启用）-> 折枝
主输出                        -> 全不中               -> None（该格剔除）
```

⚠️⚠️ **唯一的设计红线**：每一步都必须落回名册（`known`）才算数，**绝不「猜着切」**——
wiki 以后出现同形新词也不会被误切。这也是它敢替代旧 `_RE_NAME.fullmatch`
（`[\u4e00-\u9fa5]{2,4}`，已删）的依据：旧规则**同时错在两头**，把含 `-`/`·` 的名字全挡掉
（`漂泊者-男-湮灭` 四个 + `秧秧·玄翎` **从未作为队友出现过**，
`MATCH (:Character)-[:SYNERGIZES_WITH]->(t:Character) RETURN DISTINCT t.name` 可复现），
却又让 `主输出`/`副输出`/`卡提`/`暗主` 漏进图 —— 成了 **4 个垃圾 Character 节点**。

`normalize_team` 是队伍级封装：全角 `＋`/`／` 归一成半角，逐格调
`normalize_character_name`，**认不出的格整格剔除**（队因此变短，刻意为之：留着 `主输出`
模型会把它念成一个角色）。

**两侧接入，形态必须一致**：

- 读取侧 `retrieve.graph_search` 队友分支 —— **排在补主人与剪枝之前**：段数判断要按
  归一后的算，否则 `守岸人/维里奈/白芷+秧秧+主输出` 会先被当成 3 段完整队，既跳过补主人、
  又混进一个假队友。**存量图谱立刻生效，无需重建**。
- 抽取侧 `extract._extract_teammates` / `_extract_team_effects` —— 同一个函数，让新爬数据
  从一开始就干净。⚠️ **只对重建图谱生效**。

**验证（读取侧，存量图谱）**：

| 角色 | 修前 | 修后 |
|---|---|---|
| 忌炎 | 5 支（含一对镜像 `忌炎+莫特斐+漂泊者·气动` / `漂泊者-男-气动+忌炎+莫特斐`） | **4 支**，镜像消失 ★ |
| 散华 | `…+椿/折枝/安可/凌阳等主输出` | `…+椿/折枝/安可/凌阳` |
| 守岸人 | `守岸人+吟霖+相里要/卡卡罗/今汐/渊武其他输出` | `…+渊武` |
| 夏空 | `卡提+千咲（进阶轴，卡提双三剑下落）` | `夏空+卡提希娅+千咲` |

★ **忌炎的 4 支再经「模板逐位互剪」后为 2 支**（见上文配队节的 2026-09-22 扩围）——
本表记的是「**归一**」这一步的效果（5→4），别与最终展示条数（2）混淆。

抽取侧（对 chunks.jsonl 直跑 `_extract_teammates`，不写库）：守岸人从 2 条垃圾
（`主输出`/`副输出`）变 **0 条**；忌炎补回 `漂泊者-男-气动`；全量 57 角色
**非名册队友名为 0**。

**✅ 已清理（2026-09-22）**：4 个垃圾 Character 节点（`主输出`/`副输出`/`卡提`/`暗主`）
及挂在它们上的 9 条 `SYNERGIZES_WITH` 入边已 `DETACH DELETE`，随后跑 `uv run wuwa-graph` 重建。
实测结果：

| 量 | 修前 | 修后 |
|---|---|---|
| Character 节点数 | 61 | **57**（= 57 个真实角色） |
| 垃圾节点残留 | 4 | **0** |
| `SYNERGIZES_WITH` 总数 | 245 | **257**（删 9 条垃圾入边 + 重建补回正确关系） |

重建把被误指向垃圾节点的关系补正：`夏空→卡提` 变成 `夏空→卡提希娅`、
`洛可可→暗主` 变成 `洛可可→漂泊者-男-湮灭`；`守岸人/白芷/维里奈` 的 `→主输出`/`→副输出`
出边归零（那些本就是 wiki 通用模板行的占位，没有队友信息，真正的信息在入边）。
**被指向过的 53 个队友名全部落在 57 名册内**，无一个非角色名。

读取侧回归（零 LLM，直接 `graph_search`）：`守岸人` 泛问 12 支**全含 `/`**；
`守岸人+吟霖+卡卡罗` 指名**恰 1 支**；`忌炎` 4 支**无镜像**；`夏空` 出 `卡提希娅`；
`洛可可` 出 `漂泊者-男-湮灭`；`清宵` 4 支无 `主输出/副输出`。

**仍待做（位置标签不一定准确，暂时先别做）**：按位置合并成
`奶 / 副输出 / 主` 三行式答案。wiki「配队推荐」表就是现成的结构
（`| 奶&辅 | … |` + `| 副输出 | … |` + `| 主输出 | … |`，位置行下面跟候选角色行，
见 `data/raw/卡卡罗.md:530-545`）。

### 鉴权与用户画像（2026-09-22 新增）

**存储**（`pgsql/002_auth.sql`，幂等可重跑）：`users`（username 唯一、`pw_hash`、role=admin/guest）、`auth_tokens`（token PK，30 天过期）、`user_facts`（用户画像）。schema 由 `authdb.ensure_schema()` 在 API lifespan 里自动应用（**种子 admin 口令 2026-10-09 收紧：不再内置弱口令**——`ADMIN_PASSWORD` 显式配置则用它并置 `must_change=TRUE`（登录响应带 `must_change_password` 引导改密），留空则首启随机生成一次性口令、只打一次日志；只在 admin 不存在时插入），池子是 `core/authdb.py` 独立的 `AsyncConnectionPool`（autocommit + dict_row），与 `core/db.py`（worker 用的同步封装）**互不相干——⚠️ 新建 DB 模块时先确认目标文件不存在**（本轮曾把 `shutil.move` 落到已存在的 `core/db.py` 上造成覆盖，靠 VSCode 本地历史恢复）。

**口令与令牌**（`api/auth.py`）：`hashlib.scrypt`（格式 `scrypt$salt$hash`）+ `secrets.token_hex(32)` Bearer token，token 落 `auth_tokens` 表 → **重启 API 不掉线**。登录失败统一报「用户名或密码错误」（不泄露用户名存在性）。

**账号移除（2026-10-24 补，`DELETE /admin/users/{user_id}`）**

开源分发后这是隐私缺口（用户要求删号做不到），也是维护者清理探针账号的唯一正路 ——
此前只能手写 `DELETE FROM users`，其余全留：实测库里躺着 5 位「authdb 里已不存在的主人」的
会话与画像、8 枚孤儿令牌、3 条「主人已删」的 `lg` 记忆。

- **组合而非重写 SQL**：每张表怎么清只有它自己的模块知道（`conv.clear_conversations` /
  `profile.purge_user_facts` / `fb.delete_all_for_user` / `usage.delete_all_for_user` /
  `user_settings.delete_all_for_user` / `llmstore.purge_credentials` / `auth.delete_account`）。
  端点只负责按顺序调用 + 留痕。清单常量 `PURGED_TABLES` / `EXCLUDED_TABLES` 在 `api/app.py`，
  并由 `test_用户回收清单覆盖每张有用户列的表` **从建表语句反推**核对（`pgsql/*.sql` + `src/**/*.py`
  里的 `_DDL` 两处都扫）：以后新增一张带 `user_id`/`submitted_by` 的表而忘了回收 → 当场红
  （变异验证过：临时造一张 `mut_probe_things(user_id)` 恰好让它红）。
- ⚠️ **身份列有两套口径，是本仓最容易静默失效的地方**（实测值）：
  `user_facts.user_id` 存的是**用户名**（`'admin'`），而 `answer_feedback` / `llm_usage` /
  `user_settings` 存的是**字符串化的 `users.id`**（`'1'`），`sessions` / `auth_tokens` /
  `user_llm_configs` 是 **bigint `users.id`**。传错**不报错，只是一条都删不掉**，
  表现为「删了用户，用量看板里还有他的名字」。所以 `purge_user_facts` 的形参刻意叫
  `username` 而不是 `user_id`，端点里每个调用都注明用的是哪一种。
- **模型侧记忆取两路并集**：`sessions` 反查的 key ∪ `conv.memory_keys(user_id)`
  （按 `u{id}:` 前缀**直查 `lg.checkpoints`**）。只取前一路会漏掉「会话行被手写 SQL 删掉、
  checkpointer 还在」的孤儿 —— 实测真有 3 条（u10/u40/u42）。前缀 LIKE 写成 `%{id}%` 会把
  `u10:` 错算到 `u1` 名下，所以钉在 `u{int(id)}:`，并有源码判据用例看着这一行。
- **两道护栏**：不许删自己、不许删最后一个 admin（否则整套鉴权失去入口只能重装）。
  实测：自删 `400 不能删除自己这个账号`、游客 `403 需要管理员权限`。
- `crawl_runs` **刻意不回收**（理由写进 `EXCLUDED_TABLES`）：入库账本记的是**语料来源史**，
  删账号不该改写语料史。
- 实测闭环（6 个靶子账号逐个「删前逐表数 → DELETE → 删后复查」）：服务端回报的回收数与
  独立算出的行数**逐表相等**（如 e2eprobe4385：画像 6=6、用量 125=125；ui_probe3：会话 1、
  记忆 1、用量 3），删后各表残留 0、`lg` 里 `u{id}:` 残留 0。
  前端在 `AdminUsersCard` 走全站唯一的两步确认（待确认文案「再点一次永久移除」，
  自己那行 `disabled` 且 `title` 写「不能删除自己这个账号」），浏览器实测点完出
  「已移除 smoke_target：会话 0 · 画像 0 · 评价 0 · 用量 0」并从列表消失。

**权限矩阵**（FastAPI 依赖）：

| 端点 | 权限 |
|---|---|
| `/auth/register` `/auth/login` | 公开 |
| `/ask` `/ask/stream` `/profile` `/auth/me` `/auth/logout` `/auth/password` | 登录即可（`get_current_user`） |
| `/llm/providers` `/llm/config` `/llm/config/test` `/llm/models` `/llm/unlock` `/llm/lock` `/llm/passphrase` | 登录即可（个人云端模型配置，见下节） |
| `/knowledge/refresh` `/knowledge/characters` DELETE、`/admin/users` GET、`/admin/users/{id}` DELETE、`/debug/retrieval` GET | admin 专属（`require_admin`） |
| `/ingest` `/ingest/status` `/ingest/records` `/ingest/control` | 登录即可（2026-10-23 校正：矩阵曾写 admin 专属，与代码和知识库页的**刻意设计**不符——「提交 / 暂停 / 取消 / 删除**自己**提交的记录」对全体登录用户开放，只有重爬与删知识库才 require_admin；`_may_control` 负责校验只能控自己的） |

⚠️ **FastAPI 依赖链两个坑（实测）**：① 依赖函数的参数必须 `= Depends(...)`，否则类型被当**请求体字段** → 422 `missing body.user`；② 依赖的返回类型必须是 Pydantic `BaseModel`（`AuthUser`），dict 子类报 Invalid args。

### 限流（`api/ratelimit.py`，slowapi，2026-10-06 新增）

限额档位在 `config.py` 的 `RATE_LIMIT_*`：`auth` 10/min 按 IP（挡登录暴力枚举）、`ask` 20/min 按用户（一轮 20s 量级且抢 GPU）、`tts` 10/min 按用户（真实费用）、`outbound` 20/hour 按用户（`/llm/models`、`/llm/config/test` 会向用户填的地址发出站请求，SSRF 面）、**`ingest` 5/hour 按用户**（2026-10-23 补：`/ingest` + `/knowledge/refresh` 跑整条五步链并写**共享语料**，此前只吃兜底档等于没限；`/ingest/control` 不占额度，否则排队时连止损都做不了）、**`debug` 20/hour 按用户**（2026-10-24 补：`/debug/retrieval` 一次要跑 embedding + 20 条精排，抢的是与 `ask` 同一份资源，但它是**另一个能力**——并进 ask 档会让管理员调试把自己问答配额吃光）；**`creds` 复用 `RATE_LIMIT_AUTH` 的数值但单独计数**（改密/解锁/换口令/保存云端与语音凭据五个端点，每次都要跑 PBKDF2 20 万次 ~50ms，`/auth/password` 再加一次 scrypt ~733ms）。兜底 `default_limits` 200/min。来源 IP 统一走 `ratelimit.client_ip`：默认按连接 IP，**不信任** `X-Forwarded-For`（客户端可自报，信了按 IP 限速即作废）；挂反代才设 `TRUST_PROXY_HEADERS=true` 取最左项。

改动这块前必读的四条：

- ⚠️ **`rl.install(app)` 必须在 `add_middleware(CORSMiddleware, ...)` 之前调**。Starlette 里**后**添加的中间件位于**最外层**，CORS 必须在限流外面，否则 429 响应拿不到 `access-control-allow-origin`，前端看到的是一个说不清的跨域错误而不是「请求过于频繁」。已实测：中间件顺序 `[CORSMiddleware, SlowAPIMiddleware]`，带 `Origin` 请求触发 429 时响应头含 `access-control-allow-origin: *`。
- ⚠️ **`/ask` 与 `/ask/stream` 必须用 `shared_limit(scope="ask")`**，不能用 `limit()`。`limit()` 的计数键含路由名 → 两个端点各算一份 → 等于给同一能力开双倍配额，换个端点即可绕过。
- ⚠️ **被 `@limiter.limit(...)` 装饰的端点函数必须有 `request: Request` 形参**（slowapi 靠它取 key），漏写在**请求期**才报错、不是启动期。装饰器顺序是 `@app.post(...)` 在外、`@rl.limit_xxx()` 在内。
- ⚠️ **按用户限速依赖 `request.state.user_id`**，由 `api/auth.get_current_user` 写入。slowapi 装饰器包在端点函数外层，而 FastAPI 依赖在端点之前解析，故装饰器执行时身份已就绪；取不到时 `key_by_user_or_ip` 回落 IP（误用只是退化成按 IP 限速，不会变成完全不限流）。⚠️ 但 `default_limits` 由 `SlowAPIMiddleware` 在**依赖解析之前**生效，所以兜底档的主体永远是 IP。
- `/health` 必须 `@rl.limiter.exempt`：`scripts/start.ps1` 靠轮询它判就绪，被限流会让启动脚本误判失败（实测连打 30 次全 200）。
- `RATE_LIMIT_STORAGE` 留空 = 进程内存（单实例够用）；多实例填 `redis://…` 共享计数。已开 `in_memory_fallback_enabled` + `swallow_errors`：Redis 挂了自动回落内存、检查出错时放行 —— **限流是保护性设施，它自己不该把问答弄挂**。代价是 Redis 故障期跨实例计数各自为政（限流变松），可接受。
- ⚠️ **`headers_enabled` 必须为 `False`**（生产级回归桩，实测踩过）。为 `True` 时 slowapi 会在端点**成功返回后**调 `_inject_headers(kwargs["response"], …)` 写 `X-RateLimit-*` 头，这要求**每个被限流端点都声明 `response: Response` 形参**；真实端点都没有 → 业务明明成功却在写头时抛 `parameter response must be an instance of starlette.responses.Response` → 用户看到 **500（登录成功也 500）**。实测对照：`True`+无形参→抛异常、`True`+有形参→200、`False`+无形参→200。之所以选「关掉」而不是给 7 个端点加形参：① 我们要的 `Retry-After` 由 `_on_rate_limit_exceeded` 自己给，关掉后**依然存在**；② `api_ask_stream` 返回 `StreamingResponse`，加形参更易出错。
- 实测验证方式（无需起 PG/服务）：`TestClient` + `limiter.reset()`，对限额端点连打 12 次 → `[200×10, 429×2]`，429 带 `Retry-After: 60`、中文错误体与 CORS 头。⚠️ 别拿真实 `/auth/login` 打：TestClient 的 anyio portal 与 psycopg 池不兼容（实测 `PoolTimeout`，即使容器 healthy、`asyncio.run` 直连是通的）。已固化为 `tests/test_ratelimit_cors.py`（合成 app 复用真实接线、module 级只建一次以免污染 `_route_limits` 单例注册表）。

**CORS**：默认 `allow_origins` **只放行 Vite 开发端口**（`CORS_ORIGINS` 配置项，2026-10-09 收紧——原来代码默认 `*`，不配就等于全网可跨域调用；要全放开须显式写 `CORS_ORIGINS=*`）。通配符时**自动关掉** `allow_credentials`（`*` + credentials 等于允许任意站点带凭据跨域调用）。本项目鉴权走 `Authorization: Bearer` 头、不依赖 Cookie，通配符场景不需要 credentials；配具体源后 credentials 自动打开。

**用户画像数据流**（`services/profile.py`）：问答完成后 fire-and-forget 抽取（tool LLM、temperature=0、只输出 JSON 数组、单次 ≤3 条、失败回落 `[]` **绝不挡问答**）→ `save_facts` 入库 → 下次提问时 `_user_context(user)` 取活跃事实（≤8 条）经 `RagState.user_context` 注入 prompt。`graph.ask/ask_stream` 的 `user_context` 在 `_fresh_state` 里**显式覆盖空串**——防 checkpointer 把上一轮的画像串带给下一轮。

**触发时机：每轮都抽、每轮都注入，没有「攒够 N 轮」的阈值。** 但**时序是错开的**（`api/app.py::_stream_answer`）：

```
_stream_answer_inner():
  ① user_ctx = await _user_context(user)      ← 先读画像（本轮用的）
  ② async for evt in ask_stream(...)          ← 生成回答
  ③ finally: _spawn_profile_task(...)         ← 这时才抽取本轮说的内容
（不写行号：这块代码一行行在长，行号注释必然腐坏——2026-10-10 复核时它已经错了 85 行）
```

所以**本轮说的话，本轮用不上，下一轮才生效**。这是刻意的：抽取要占 LLM，放在 `finally` 是为了不与生成抢（`OLLAMA_NUM_PARALLEL=1`）。

⚠️ **fire-and-forget 有竞态窗口，但实际无害**：`_spawn_profile_task` 不 await，若用户在写入完成前就发下一句，那句仍读到旧画像。实测窗口 **冷启动 4.33s / 热态 0.13~0.8s**（抽取 4.33s→0.16s，入库仅 5ms）；背靠背请求会漏（第 2 条读到空），**间隔 1s 即正常**。人类打字速度远慢于此，不必加锁或改成同步——同步会把 4s 的抽取塞进响应路径。

**同类覆盖（「可更新」，2026-10-06）**：`user_facts` 新增 `category` 列，`save_facts` 三步走：① 同文去重（连 `created_at` 都不刷新，避免重复说同一句就把事实顶到最前）；② `category` 非空则**软删该用户同类的活跃事实**（`valid_to=now()`，沿用既有软删设计，不物理删除）；③ 插入新事实。

| category | 判据 | 为什么这样处理 |
|---|---|---|
| `nickname` | 自称/名叫/叫做/姓名/昵称/称呼/网名/游戏名/ID | 同一时刻只有一个称呼，覆盖显然正确 |
| `level` | 萌新/新人/新手/老玩家/回归玩家/刚入坑/玩了N年 | 会随时间演进，新值就是当前值 |
| `NULL` | 其余（主玩角色、常用配队、偏好…） | **刻意不覆盖**：一个人可以有多个本命、多支常用队，覆盖等于静默删掉用户真实信息 |

- ⚠️ **匹配顺序敏感**：「用户自称是萌新」同时含「自称是」与「萌新」，`_fact_category` 必须**先判 level 再判 nickname**，否则会与真昵称互相覆盖。
- ⚠️ `_RE_NICKNAME` 的两个分支都必须**停在昵称之前**（不能写成 `叫我[^，。]{1,12}`）：分类靠 `s[m.end():]` 取昵称，把昵称吞进匹配会让 `tail` 恒空 → 「叫我小星」判成 None（实测踩过）。
- ⚠️ **宁可不分类，不可错分类**：错分类会让两条本该并存的事实互相覆盖（静默丢数据），漏分类最多只是多留一条旧事实（原行为，无害）。
- 分类用**规则**而非让抽取模型多输出字段：`_EXTRACT_SYSTEM` 的措辞是多轮实测换来的（见其注释），动它要连解析链路一起重验。
- **老库迁移**：`ALTER TABLE user_facts ADD COLUMN IF NOT EXISTS category TEXT`（`authdb._DDL` 与 `pgsql/001_init.sql` 两处都有，前者才是运行时建表入口）。历史事实 `category=NULL` → 不参与覆盖；需要的话用 `_fact_category` 回填一次，否则用户改昵称时旧的那条仍留着。
- 实测（同一探针用户连说 7 句）：3 个昵称 + 2 个水平 → 收敛为「阿星」+「老玩家」各 1 条；2 个主玩角色（守岸人/今汐）**都保留**；注入串 `用户是老玩家；用户自称是阿星；主玩角色：今汐；主玩守岸人`。

**⚠️ 画像注入覆盖两个分支（2026-10-06 修的缺口）**：`user_context` 原先**只在 `generate_node` → `prompt.build_prompt` 注入**，而闲聊走 `_chat_turn`（`chitchat_node` / `time_node` 共用）**完全拿不到画像**——于是「我是颗粒」的昵称确实入了库，但下一句「你好呀」走闲聊时模型一无所知，表现为「记住了却用不上」。现在 `_chat_turn` 也注入（`_PROFILE_HINT`）。

措辞与位置是**实测选型**的结果（同一画像+同一问句「你好呀」，各抽 8 次，只看「回答里是否出现昵称」与「是否把人称搞反」）：

| 方案 | 结构 / 措辞 | 含昵称 | 人称混淆 |
|---|---|---|---|
| A | hint → 档案 → 原话，「可以自然贴合」 | 4/8 | 0 |
| B | 同 A，「把称呼用在开头，直接喊出」 | 1/3 | **1**（以为用户叫它「颗粒」）|
| C | 档案 → hint → 原话 | 0/3 | 0 |
| D | 同 A，「称呼是这位家人自己希望的叫法」 | 0/8 | 0 |
| **E（采用）** | hint → 原话 → **档案**，「上面这位家人希望你用档案里的称呼喊TA」 | **7/8** | 0 |

三条结论，改动时别违背：① **档案要排在用户原话之后**（近因位），与 `prompt.build_prompt` 把「输出格式」压在末尾是同一条经验；② **措辞越强硬越差**（B/D 都掉，B 还诱发人称混淆）；③ 仍遵守提示词铁律（只写正面要求）。端到端复测 `chitchat_node`：4/6 命中昵称、0 人称混淆，对照组（无画像）0 命中。8B 不是每次都喊，属概率性行为，但已从「完全用不上」变成「大概率用上」。

**前端**：未登录全屏门禁 `AuthPage`（App.tsx 里 `if (!auth)` 拦截）；token persist 在 localStorage，启动时 `api.me()` 静默校验（401 → clearAuth）；`UnauthorizedError` 统一处理；topbar `.user-pill` 徽章（admin=盾牌图标+title「管理员」，游客=人形图标+title「游客」，**角色只以图标区分，徽章文本是用户名**——写自动化断言别找 "guest" 文字）；「收录新角色」/知识库表单仅 admin 渲染；设置页「我的画像」卡片支持查看与单条软删。

### 用户云端 API-KEY 的加密存储（双层密钥 DEK/KEK，2026-09-30）

需求三条必须同时成立：① **加密落盘**；② **下次登录自动连接**；③ **不泄密**。
⚠️ **只用「一个用户口令」做不到这三条的交点** —— 口令只有用户知道，服务端没有它就无法自动解密；而服务端要自行解密，就必然得持有某样东西（这正是 2026-09-29 那版「口令只存内存、重启即上锁」方案无法满足「自动连接」的根因）。故改为**双层密钥**（与 Bitwarden / 1Password 同构）：

```
api_key ──DEK─────> api_key_enc     DEK = 随机数据密钥（`Fernet.generate_key()`），只加密数据
DEK     ──KEK_pwd─> dek_by_pwd      KEK_pwd = 登录密码 + pwd_salt 经 PBKDF2(200k) 派生
DEK     ──KEK_pp──> dek_by_pp       KEK_pp  = 加密口令 + key_salt 经 PBKDF2(200k) 派生（兜底）
```

**密钥存放**：解出的 DEK 原始字节只放在 `_DEK_RAW: dict[int, bytes]`（进程内存）。
⚠️ **绝不写进 LangGraph state** —— checkpointer 会把 state 持久化进 PostgreSQL，进 state 等于把密钥落库到另一张表。

**表列**（`user_llm_configs`，`llmstore.ensure_schema()` 幂等补建）：
`api_key_enc` / `key_hint`（掩码）/ `pwd_salt` / `dek_by_pwd` / `key_salt` / `key_check` / `dek_by_pp`。
`key_check` 是 KEK_pp 加密的固定串，仅用于**给"口令对错"一个明确反馈**（否则只能等解 `dek_by_pp` 失败才知道）。

**四条关键流程**

| 流程 | 位置 | 要点 |
|---|---|---|
| 首次保存 | `llmstore.save_config` + `_bootstrap` | 至少给 `password` 或 `passphrase` 之一；随机 DEK，两条通道各加密一份落盘。`_PENDING` 暂存区保证盐与密文**随同一次事务**落库，不留半成品 |
| 登录自动解锁 | `api/auth.py:login` | 登录请求带明文密码 → `unlock_with_password` → DEK 进内存。**用户零额外输入**，下次登录云端模型直接生效。失败只记日志，绝不影响登录 |
| 改密码 | `auth.change_password` | ⚠️ **顺序敏感：先用旧密码 `rebind_password`，再更新 `pw_hash`**。颠倒会让用户改一次密码就把自己的 Key 永久锁死（只能删配置重填） |
| 加密口令兜底 | `unlock_with_passphrase` / `change_passphrase` | 独立于登录密码；换口令只重加密 `dek_by_pp`，**`api_key_enc` 不动** |

**硬约束（改动前必读）**

- ⚠️ **解不开 = 回落本地，绝不退化明文**：`get_runtime` 返回 `None` → `_chat_client` 走本地 aemeath。任何"取不到密钥"的分支都不得降级为明文存储或明文日志。
- ⚠️ **凭证错一律拒绝写入**：用错误密钥加密写入会把原 Key 变成永久解不开的密文，比拒绝更糟（`save_config` 因此必须先解锁成功再加密）。
- ⚠️ **`cryptography` 是可选依赖**：`crypto_ready()` 为假时云模型配置功能整体关闭，保存直接拒；缺失时不该因缺包而崩（延迟导入 + try/except）。
- ⚠️ **登出 / 删除配置必须 `lock(user_id)`**：否则同一进程内换账号会读到上一个用户的 DEK。
- ⚠️ **改密码后旧 token 仍有效**：服务端内存里若还留着旧会话的 DEK 不受影响，但**新会话必须用新密码登录**才能解锁（重绑只改 `dek_by_pwd`，不改内存）。
- 服务端**不持有任何主密钥**：`.env` 里没有、库里没有（只有 scrypt 单向哈希与三份密文）。拖库者拿不到登录密码也拿不到加密口令 → 解不开。开源 clone 即用，无需先配共享密钥。
- 唯一残留的泄密面是**进程内存**（运行期必有明文 DEK，任何非端到端方案都一样）。

**端点权限**（均在 `get_current_user` 之下，配置只属于自己）

| 端点 | 说明 |
|---|---|
| `GET /llm/config` | 只回掩码；带 `unlocked` / `auto_unlock` / `pp_bound` 三个状态位 |
| `PUT /llm/config` | 保存（`api_key` 留空 = 保留原 Key）；未解锁时需 `password` 或 `passphrase` |
| `POST /llm/unlock` `POST /llm/lock` | 解锁（密码/口令二选一）/ 丢弃内存 DEK |
| `POST /llm/passphrase` | 换加密口令（需原口令） |
| `POST /auth/password` | 改登录密码（后端自动重绑云端密钥） |

**前端约定**：两种凭据只活在组件内存（`password` / `passphrase` state），提交成功即清空，**不写 localStorage**（2026-09-29 那版的「记住口令」已随本方案移除 —— 登录密码通道已经让"自动连接"无需本地留明文）。设置页按 `auto_unlock` 分别渲染「已绑定：登录自动解锁」与「填一次密码即可绑定」两种状态。


- `dialog/nlu.py` SLOT_PATTERNS 的 key ⇄ `knowledge/retrieve.py` CYPHER/SLOT_LABEL 的 key（属性/属性反查/技能/共鸣链/突破材料/声骸/武器/队友，8 槽位一一对应）；`graph_search` 里还有「配装→声骸」remap。加新槽位两边同时加。
- `text.py`：写入端 `embed_input`（build_index）与读取端 `chunk_text`（rerank/chain）共用同一份清洗（面包屑去重、占位表头「列1|列2」删除、**按键图标残渣 `+` 串删除**、**单元格里的长连字符分隔符换成顿号**）——只改一边会导致索引与查询文本不一致。
  - `strip_dash_run`：wiki 把「多个配队」挤在一格、中间用一长串连字符分隔（`清宵+达妮娅+莫宁-----…-----清宵+琳奈+莫宁`），模型会照抄这串 `-`，还误以为「后面另有一条」。门槛必须用 `-{4,}`：markdown 表格分隔行 `| --- |` 正好是 3 个，全语料 `-{3,}` 里 19893 处就是它，而 `-{4,}` 只有 60 处、**100% 是配队分隔**（已实测）。影响 34 个角色的「编队&队伍轴推荐」。
  - ⚠️ **2026-09-22 补 `strip_icon_placeholders`（按键图标残渣 `+`）**：KuroBBS 富文本把「按键图标」写成 `<img>`（`alt` 是没用的 `blob:` URL），转 Markdown 时图标整块消失、只剩原先夹在图标之间的 `+`——语料里于是出现 `·浮声一刹·凌霄：++；+++`、`短按++，获取【静质量能】`，模型会照抄进答案（实测问「清霄的技能是什么」答出 `- 「浮声一刹·凌霄」：++；+++`）。规则**只吃「连续 ≥2 个 `+`」且两端非数字/百分号**（紧邻数字是伤害式 `26.92%+40.38%+67.29%`，必须保留）；另清「紧跟 `或/：/，/、/·` 且后接空白或句读的孤立 `+`」（`+++或+` 的尾巴）。**单个 `+` 一律不动**。全语料 6572 块实测：连续串 8 处——数学型 **0**、非数学型 8 处且 100% 是残渣（这正是「只吃 ≥2 连续」安全的依据：连着的 `+` 在语料里从不是数学式）；单个 `+` 7265 处——数学型 6648 处保留，非数学型 617 处全是有效内容（配队 `布兰特+长离`、连招 `【锯环·疾攻】+【锯环·终结】`、词条 `攻击+攻击`）。最终只改动 4 块（弗洛洛/清宵/琳奈/莫宁 的「技能说明」）、删 21 个 `+`。
  - ⚠️ 判空坑：`"" in "%."` 恒为 `True`，`is_math` 必须显式写 `left and …` / `right and …`，否则「行尾 `+` 串」（`：++；+++` 的尾巴）判定成数学式清不掉（首版真跑踩到，`：++；+++` → `：；+++`）。
  - 回归基线（可复用）：5 个正样本（含 `++；+++`、`+++或+`、`短按++，` 三种）必须被清；8 个负样本（`26.92%+40.38%+67.29%`、`200.80%+267.74%*3`、`布兰特+长离`、`长离+维里奈`、`1cost：攻击+攻击`、`【锯环·疾攻】+【锯环·终结】`、`热熔伤害加成+攻击`、`14.28%+16.66%*2`）必须一字不动。
  - ✅ **「旧索引未重建」这笔欠账已于 2026-10-24 清掉**（此前记的「影响面仅 4 个 chunk」是**错的**，只数了图标残渣那一轮）。
    实测对账（只读脚本比 `Chroma.documents` vs 当前 `embed_input(PG 行)`）：**6741 块里 36 块文本口径不一致**，
    全是「Character Strategy › 编队&队伍轴推荐 › 主流队友」，涉及 33 个角色 —— 索引里还是一长串 `-`，
    PG 与查询侧早已换成顿号。也就是说**配队这一类最常被问的块，稠密检索一直在拿旧文本算相似度**。
    根因不是某次改动漏跑，而是 `wuwa-index` 只有「无脑全量重嵌」一档：6741 块 CPU 重嵌要几十分钟，
    于是每次改清洗规则都先欠着，欠久了没人记得。
    修法（`build_index.plan_reconcile`，已落地）：重建时按**文本口径**对账 ——
    ① 只重嵌「索引里没有」或「存的文本 ≠ 当前口径算出来的文本」的块（一致就跳过，bge-m3 干脆不加载）；
    ② 顺手删掉「索引里有、PG 里已经没有」的孤儿 id。②这条以前根本不存在：upsert 只增不删，
    所以「全量重建」其实并不会清掉 wiki 改版后换了 chunk_id 的旧块（按角色刷新是另一条路，两者都要）。
    怀疑索引本身坏了仍可用 `wuwa-index --full` 强制全量；未知参数**直接退出**（`--force` 这种笔误
    被静默当成增量跑，差的就是「改版后旧块还在」这类最难查的故障）。
    重跑后复测：**文本不一致 0 / 6741**，缺 id 0，BM25 同步重建（695 术语）。
    4 条用例在 `tests/test_ingest_control.py`（不一致要重嵌 / 缺失要补 / 孤儿要删 / `--full` 名副实），
    变异验证过：把比对退化成「只看 id 在不在」，恰好第一条红。
- `knowledge/knowledge/graph/extract.py` 的抽取规则依赖 `knowledge/crawl/chunker.py` 产出的 module/component/breadcrumb 结构；wiki 改版先坏的是抽取。⚠️ **不要因为字段覆盖率低就接 LLM 补抽**——57 角色实测：skills/chains/materials 均 100%，weapons 96.5%，其余缺口经逐项辨明**全部源于 wiki 数据源本身缺失**，LLM 抽不出原文没有的信息：① `echoes` 29.8% 是设计使然（38 个角色的声骸被并入 `echo_builds`，总覆盖 55/57）；② 丽贝卡/露西缺 weapons/teammates 因其原文「声骸」0 次、「配队」0 次；③ `attrs` 84.2% 缺的 9 个角色原文**没有「基础资料」段**（对比卡卡罗有 `- 属性：导电`），其中 4 个漂泊者是刻意设计。
- 检索/切块/重排所有参数集中在 `config.py` Settings（.env 可覆盖，`get_settings()` lru_cache 单例）。

### 语音朗读 + 情绪标签（Qwen-Audio-3.1-TTS-Flash，2026-09-30 开启）

链路：答案生成 → 情绪判定（`services/emotion.py`）→ 朗读稿清洗（`services/tts.py::_to_speech_text`）→
指令 + 情感标签 → POST `SpeechSynthesizer` → 24h 音频 URL → 前端 `<audio>` 播放。

- **模型**：`qwen-audio-3.1-tts-flash`（2026-09-19 发布）。相对 3.0 多两项对角色扮演有用的能力：
  **指令控制**（`instruction`）与**细粒度情感/富语言标签**。
- **端点**（已核实，与 3.0 相同、无需改）：`https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer`，**仅北京地域**，API Key 也必须同地域。Qwen-TTS 系列用的是另一个端点（`aigc/multimodal-generation/generation`），**不可混用**。
- ⚠️ **音色与模型强绑定**：3.1 只认带 `_v3.1` 后缀的音色；填 3.0 的 `longanhuan_v3.6` 会返回 `InvalidParameter`（`[cosyvoice:]Engine error: TTS speak operation failed`）。这是换模型时最容易踩的一脚，`tts.py` 已针对该错误串给出点名提示。
- **默认音色** `longanlingxi_v3.1`（龙安灵希 · 可爱甜美 · 社交陪伴）：贴爱弥斯「爱笑、话多、活泼亲切」的人设。备选见 `config.py` 注释（longhua / qiaoxiaojiao / xiaxiaochen / xuxiaoqiao / yuxiaoyun）。
- **指令控制** `TTS_INSTRUCTION`（现为代码默认值，用户可在设置页覆盖）：官方上限 **100 字符且汉字按 2 字符计**（超长会整单被拒，不是忽略）。`_clip_instruction()` 本地按该口径截断后再发。
  - ⚠️ **铁律：只描述声音特质，不写「模仿某某」**。官方明确说明模型不支持模仿特定人物，且真人/角色配音的模仿可能涉及版权风险（详见「声音设计」文档「原创而非模仿」）。当前文案描述的是「活泼爱笑的少女感」，不指向任何具体作品角色。
- **情感标签**（`services/emotion.py::EMOTION_TAGS`）：`[excited]` `[amazed]` `[serious]` `[empathetic]` `[mischievously]` —— 已逐一核对**均仍在 3.1 官方控制类标签清单内**，无需改动。可选富语言标签（拟声）：`[giggles]` 咯咯笑、`[laughing]` 大笑、`[sighing]` 叹息等。
  - 标签只作用于**其后文本**，故只在开头插一个；两层控制叠加不冲突——标签管段落情绪，`instruction` 管整体音色性格。
- **凭据来源：用户自持（2026-09-30 追加）** —— 开源分发的关键设计，改动这条链路前先读本节。
  - `services/tts.py::resolve(user_id)` 是唯一入口：**用户自持凭据 > 全局 `.env` 兜底**，返回 `{api_key, workspace_id, model, voice, instruction, source}`。`source ∈ {user, global}`。部署者**不需要**替所有用户垫额度；`.env` 里那几项只留给「自部署者统一配一份给所有人用」的场景。
  - **存储复用 `user_llm_configs` 表 + 同一把 DEK**（见 `llmstore._DDL` 里的说明）：不新增密钥体系，于是「登录自动解锁 / 改密码自动重绑 / 加密口令兜底」三条能力全部自动继承，用户只维护一套口令。表名是历史命名，语义已是「用户自持的外部服务凭据」。
    - ⚠️ 因此 `configured` 必须按**各自的密文字段**判断（`api_key_enc` vs `tts_api_key_enc`），不能只看行是否存在——否则「只配了 TTS」的用户会被显示成「已配置云端模型」。
    - 保存走 `_ensure_dek()`（与云端模型 `save_config` 共用），新增凭据类型时不要另写一套建/解锁分支。
  - **`TTS_ENABLED` 是真总闸**：在 `resolve()` 里**先于凭据判定**。关掉后连用户自持的凭据也不生效——否则「关了开关却还能用」就是个说不清的 bug。`_global_ready()` 只管「兜底凭据齐不齐」，不管总闸。
  - **`workspace_id` 会被拼进请求 URL 的 host 段** → 走 `validate_workspace_id()`（`^[A-Za-z0-9-]{1,64}$`）防 SSRF，与 `validate_base_url` 同类。`model`/`voice` 只进请求体，走 `_RE_TTS_TOKEN` 限字符与长度。
  - **空串 = 用默认值**（`_fill` 三项统一用 `or`）：不要给 instruction 单独做「空串=显式清空」，否则用户会撞上「清空后声音没变」这种说不清的状态。
  - 刻意**不设** per-user 语音总开关：前端已有一个本地偏好开关（`settings.ttsEnabled`，只管渲染朗读按钮），后端再来一个「停用后回落全局兜底」既重名又语义不直观。不用自己这份凭据就直接删（`delete_tts_config`）。
  - `graph.py` 用 `await tts.available(user_id)` 决定「要不要花一次情绪判定调用」；它**内部吞异常返回 False** —— TTS 是可选增强，不能反过来把主问答链弄挂。
- **可用性状态**：凭据缺失/未解锁/密文损坏时，`/tts` 返回 `200 + ok=false` + 可读原因（**配置缺失不是服务故障**，不报 5xx）。
  - `/tts/status` 返回**当前生效**的 `model`/`voice`（附 `model_label`/`voice_label` 中文展示名）、`source`、`configured`/`unlocked`，以及 `defaults`（代码默认值，供前端写 placeholder）——直接渲染 `longanlingxi_v3.1` 这类内部 id 对用户毫无意义。
  - `resolve()` 的 reason 文案会**直接展示给用户**，故按「该做什么」分档写：没配→去设置页填；配了但未解锁→重新登录即可自动恢复；解不开→需重填。
- **前端**：`settings.ttsEnabled` 默认 true（本机显示偏好，关掉只是不渲染朗读按钮）；设置页「**语音合成**」独立卡片承载用户凭据（API Key / 业务空间 ID / 音色 / 模型 / 语气指令 + 「已就绪 / 未就绪」徽章 + 解锁入口），音色用 `datalist` 列出 `VOICE_LABELS` 避免手填踩版本坑；气泡按钮三态 title（正在合成 / 停止朗读 / 朗读这条回答）。
- ⚠️ **待验证**：音频 URL 能否跨域播放（后端返回的是阿里云 OSS 直链，24h 有效）。若浏览器拦截，改为服务端转发字节流。
- **声音设计 / 声音复刻**（未接入，可选进阶）：官方支持用自然语言描述创建专属音色（`POST .../audio/tts/customization`，`model: qwen-voice-design`），返回 `voice_id` 可直接填 `TTS_VOICE`。**声音设计音色不支持方言**；复刻需 10~20 秒音频样本且**有版权风险，开源项目慎用**。

### 音乐播放（QQ音乐 MCP，2026-10-07 新增）

**形态**：`tools/qqmusic_mcp/` 是**独立的自包含 MCP server**（子进程 + stdio），
`services/music.py` 是它的 stdio 客户端。默认关闭（`config.MUSIC_ENABLED=false`），
用户在设置页开启后**即时生效，不需要重启服务**。

**这四层判定必须分开，别合回去**：

| 函数 | 判什么 | 用途 |
|---|---|---|
| `is_enabled(user_id)` | **开关**：个人设置 ∪ 部署默认（取「或」—— 这是本机能力，不是账号数据） | 所有入口 |
| `_env_ready()` | **环境**：server 文件存在 + `mcp` 依赖已装 | 建会话前 |
| `available()` | 同步版 = 部署开关 + 环境 | 仅执行环境的自检 |
| `available_async(user_id)` | **权威判定** = `is_enabled` + `_env_ready` | 一切真正的调用路径 |

⚠️ **踩过的三个 bug 是同一处的表亲，修的时候要一起看**：
1. `available_async` 曾在 `is_enabled` 为真时**无条件** `return (True, ...)`，把环境不合格也算作可用；
2. `_ensure_session()` 与 `_run_music()` 却调**同步** `available()` —— 只看 `.env`，完全无视个人开关。
   两者叠加的后果：用户在设置页开了开关，系统仍回「音乐功能未启用」，表现如同「必须改 .env 再重启」；
3. 「开关」与「环境」原本混在同一个返回值里，调用方**无法区分**是哪一种失败，只能整条否定。

⚠️ **`as_user` 是隐式契约**：`_ensure_session` 只能从 contextvar 取当前用户（会话是全局单例）。
漏包 `as_user` 的后果**不是报错**，而是静默退回部署默认。
凡可能触发建会话的入口都要包：`play` / `control` / `status`，以及 `_run_music` 整段 ——
**status 分支尤其容易漏**，它是直接走 `call()` 的。

**MCP 会话必须常驻在同一个 task 里**：anyio 的 `CancelScope` 只能在其创建所在的 task 中退出。
旧写法把 `AsyncExitStack` 存成模块级全局、在「第一个发起请求的 task」里 enter，
那个 task 一销毁，之后任何在别的 task 里发生的 close 都抛
`RuntimeError: Attempted to exit a cancel scope that isn't the current tasks's current cancel scope`
（实测出现在 `/ask/stream` 流式收尾）。现在由 `_session_main` 长驻持有，跨 task 只传「工具名 + 参数」。

- `wait_for(shield(ready), ...)` 的 **`shield` 是必需的**：否则超时会取消 `ready`，宿主稍后 `set_result` 撞 `InvalidStateError`
- 循环里用 **`break` 而非 `return`**，让 `async with AsyncExitStack()` 仍在宿主 task 内收尾
- `set_exe()` 改播放器路径后要 `aclose()` 重建会话（子进程 env 是启动时注入的）

**stop ≠ pause（2026-10-10 实测，用户纠正「暂停和关闭不是一个事情」）**：
QQ音乐**不实现** SMTC 的 StopRequested——命令被接受、状态仍 PLAYING。三层修法：
① smtc 对 stop 的确认口径改为「STOPPED 或会话消失」（此前误用切歌的「歌名变了」判据，
   永远假阴性，「关闭音乐」看起来像卡住）；② 没停就**关进程**：taskkill 发 WM_CLOSE，
   3s 没退再 /F 强杀（QQ音乐可能弹确认或缩托盘），文案如实说「已强制关闭，下次点歌自动冷启动」；
③ 客户端本来没跑 → 直接「已确认（本来就没在跑）」，不假装执行。
配套：同曲但**暂停中**再点同一首 → 走 SMTC play **恢复**而非重投（重投会从头播，
「唱到一半突然重来」）。实测：PAUSED→恢复 ✓；stop 关进程后 client_running=False、会话消失 ✓。

**音量走应用级 Core Audio**（只调整 QQ音乐）。⚠️ COM 属 STA 且为同步阻塞调用，
必须在 **MCP server 进程**内执行 —— 放进本服务的事件循环会阻塞整个 API
（实测 `/music/status` 超时，且**不产生任何日志**，属最难排查的一类故障）。

**NLU 的高频坑：差一个字就永远命中不了，且失败是静默的**（表现为「说了没反应」）：

- 词表按**用户会怎么说**穷举，不按程序怎么叫：`关掉音乐` ≠ `关闭音乐`，后者一度整句落空
- 音量类动作**必须分方向两支写**，别用 `[大小]`（分不出方向）；方向词要挂在
  「音量 / 声音 / 音乐 / 音响」这个锚上，否则「大招伤害高不高」会被判成调音量
- 点歌正则**锚在 `^`** 时要格外小心：中文习惯在动词前加「我想 / 我要 / 麻烦 / 能不能」，
  前缀漏了就是整句落空；量词还要兜「一下」（否则「听一下晴天」会把「一下」算进歌名）
- 「取消静音」含「静音」子串，必须排在它前面
- 音量动词组是 `(调|放|开|提|增|加|升)(大|高)` / `(调|放|开|降|减|压)(小|低)` 两支——
  2026-10-10 实测补的：原来只有 `(调放开)`，「**降低**音量呢」「**提高**音量」「切歌」
  全部落空（用户截图复现）。补完必须连负样本一起跑（「提高共鸣效率」「大招伤害高不高」）
- **回指分支禁用裸指代词**（那个/还是/一样）：`_MUSIC_RETRY_RE` 只认明确重试词
  （再试|再放|刚刚|上次|继续|还是那…）。曾把「那个」收进回指表 → 点过歌后问
  「那个声骸怎么配」会**把上一首再放一遍**（「带入前面点的歌」根因之一，已修）
- ⚠️ **回指还必须躲开「回忆型问句」**（2026-10-10 端到端重跑实测抓到）：
  「我刚才**做什么来着**？你还记得我吗」里的「刚才」是时间锚，整句在问
  「我说过什么」，不是命令重放。命中回指后 `intent_node`（`graph.py:340`，它拿得到
  `state.history`）把上一首的动作与歌名算进 state，`chitchat_node` 就真的再投一遍。
  修法不是继续删重试词，而是加**反向排除** `_MUSIC_RECALL_ASK_RE`
  （做什么/叫什么/是什么/哪首/来着…）。
  ⚠️ 排除表**不能**宽写成「有疑问词就不重放」——「刚刚那首再放一遍**好吗**」是礼貌祈使句，
  带疑问词也必须照办（用例成对写：5 条回忆问句 → None，6 条明确重放 → `('play', 歌名)`）。
  ⚠️ **别去 API 层找这条 bug**：`/ask/stream` 调 `_stream_answer` 时**不传 history**，
  故 `_is_music_turn` 永远走不到回指分支——回指重放**只可能在图内发生**。
  （API 预执行那层的 `not slots and not chars` 守卫仍然必要，但它只管**直接命令**。）
- 改完**必须重跑负样本**（游戏问句 + `我想听歌` 这类无具体歌名的说法），确认未误伤

**音乐轮的两条上下文红线（2026-10-10 修，用户报「带入前面点的歌，影响操作和回答」）**：

1. **API 预执行必须与 `intent_node` 同口径守卫**（`app._is_music_turn`）：
   `not detect_slots(...) and not extract_characters(...)`。预执行发生在图守卫**之前**，
   漏守卫时哪怕最终判成游戏提问，歌已经被放出去了——副作用拦不住。
2. **音乐轮生成不带对话历史**（`chitchat_node` → `_chat_turn(..., use_history=False)`）：
   history 里躺着前几轮点歌的执行文案（旧歌名），aemeath 会把它复述进本轮
   （实测「关闭音乐」答成「无间渡还在原地打转」）。本轮工具返回本身含歌名与状态，够了；
   回指重放的歌名走 `music_keyword`→工具，不靠 prompt 历史。⚠️ 只掐**进 prompt 的**历史，
   `_commit_history` 仍写全量——记忆不能因为一轮音乐就断。

**搜索降级**：QQ 的 smartbox 接口**不认「歌手 + 的 + 歌名」**
（实测「周杰伦的青花瓷」返回 0 条，而「周杰伦 青花瓷」返回 4 条 —— 前者恰是用户最自然的说法）。
`search_song()` 在原串落空时依次按「的 → 空格」「仅保留歌名」重试。
⚠️ 裸调该接口**必须带 `Referer: https://y.qq.com/`**，否则恒返回 0 条（会被误判成接口失效）。

**去重**：`api/app.py` 为了让歌早点开始放，会在进图之前先跑一次 `_run_music_best_effort`；
图里的 `chitchat_node` 也会处理音乐动作。⚠️ 必须把前者的结果经
`ask_stream(music_result=...)` 带进 `RagState.music_result`，否则同一句点歌会**搜索两次、投放两次**。
`music_result` 为**空串**表示「本轮没有实际执行」（例如没听清歌名），此时图里照常执行 ——
所以空串**不能**当成「已执行过」。

### 模型资源分配（GPU 只有 8G）

- GPU：生成模型 aemeath 与 qwen3-vl:8b 立绘 VLM 都由 Ollama 承载（`LLM_URL`；VLM 链路未接入主链），8G 显存互抢，跑批前先 `ollama stop`。
- CPU：bge-m3 embedding（`max_seq_length` 压到 512）与 reranker（bs=8、torch 线程 8 是实测最优，且线程数是临时设置后恢复的——全局设会污染同进程 embedding）。

### 工程加固（2026-10-06）

一批可靠性/安全性改动，改动相关代码前先读本节。

**`config.py` 三个默认值调整**

- `DEBUG`：`True` → **`False`**（生产友好；本机需要时在 `.env` 显式开，本仓 `.env` 就写着 `DEBUG=True`，属本地配置不入库）。
- `HF_HOME`：删掉硬编码的 `D:/hf_cache/huggingface`，改为 `Path.home()/".cache"/"huggingface"`。⚠️ 优先级是**环境变量 > `.env` > 代码默认**，所以本机若设了系统级 `HF_HOME`，仍走系统值（这也是「默认值改了但本机路径没变」的正常现象，不是没生效）。
- `CLOUD_ALLOW_PRIVATE_NET`：`True` → **`False`**。原来默认放开等于给注册用户开了内网探测口；要指向本机 Ollama/vLLM 时才显式设 `true`。云元数据端点（`169.254.169.254` 等）无论此项如何都无条件拦截。

**`_known_characters` 名册缓存加了 `asyncio.Lock`（`dialog/graph.py`）**

原来是裸 `global` 读改写，多个协程同时判空会同时打 Neo4j、同时写缓存。现在是**双重检查**：先无锁快路径读缓存，未命中才进锁，进锁后再查一次（等锁期间可能已被别的协程刷新）。空结果仍不缓存（Neo4j 抖动时避免脏 60s）。

**同步阻塞调用改走有界线程池（`dialog/graph.py` 的 `_run_io`、`dialog/tools.py` 的 `_run_in_executor`）**

`asyncio.to_thread` 用的是默认 `ThreadPoolExecutor`（`max_workers=40`），并发高时会与 Celery worker / Ollama 抢线程。改为各模块自持 `ThreadPoolExecutor(max_workers=8)`。⚠️ `_run_io` 用 `functools.partial` 包装，因为 `run_in_executor` 不接受关键字参数，而 `r.get(timeout=…)` 需要传 kwarg。

**⚠️⚠️ `InvalidToken` 不是 `ValueError` 子类（`core/llmstore.py`）**

这是本轮实测推翻的一个想当然假设，**改解密相关代码前必读**。`cryptography.fernet.InvalidToken` 直接继承 `Exception`：

```
MRO: ['InvalidToken', 'Exception', 'BaseException', 'object']
```

所以「捕获 `ValueError` 就能覆盖密文解不开」**是错的**，会漏掉**口令/密码不对**这个最常见场景——后果是登录时自动解锁抛未捕获异常，把登录整个弄挂。已定义模块级 `_DECRYPT_ERRORS = (ValueError, InvalidToken)`（`cryptography` 缺失时降级为 `(ValueError,)`，此时 `crypto_ready()` 已判 False、云模型整体停用）。**五处**解密统一用它：`unlock_with_password` / `unlock_with_passphrase` / `rebind_password` / `get_runtime` / `get_tts_runtime`。
验证方式（可复用，零外部依赖）：`Fernet(k1).encrypt(...)` 后用 `k2` 解，断言被 `_DECRYPT_ERRORS` 接住。

**异常捕获口径（全仓 ~35 处 `except Exception` 已逐个定性）**

不是一律收窄，而是分三类处理，`# noqa: BLE001` 标注有意的宽捕获并写明理由：

| 类别 | 处理 | 例子 |
|---|---|---|
| 预期异常明确 | **收窄**到具体类型 | 解密 → `_DECRYPT_ERRORS`；`verify_password` → `(ValueError, AttributeError, TypeError)`；时区 → `(ZoneInfoNotFoundError, ValueError)`；S3 → `ClientError` 且再判 `_is_not_found` |
| 增益不阻塞（降级有明确回落） | **保留宽捕获 + 标注** | 画像抽取、滚动摘要、查询改写、情绪判定、TTS 可用性、补料块 |
| 必须宽捕获（收窄会放大故障） | **保留 + 写明为什么不能收窄** | SSE 流内、`generate_node`/`_chat_turn` 的 `astream`、`services/verify.py` 的 fail-open 闸门、Celery 任务体的失败落账 |

三条要记住的理由：

- ⚠️ **`services/verify.py` 是 fail-open 闸门，绝不能收窄**：「任何异常都降级为匹配」是它的承重不变量。意外异常冲出去会打断整张 LangGraph（用户收不到答案），比「放行一份跑题资料」严重得多——降级方向的代价不对称。
- ⚠️ **流式生成（SSE / `astream`）绝不能收窄**：响应头已发出、全局异常处理器接不住；漏掉一种异常类型，用户的流式回答就中途裸崩、前端永远停在「生成中」。
- ⚠️ **Celery 任务体是「先落账再抛」**：`_step_update(..., "fail", ...)` 后 `raise`。收窄会让某些异常跳过落账，前端五步进度永远卡在 `running`。
- `knowledge/s3.py` 原来吞掉全部异常有真实缺陷：鉴权失败/网络不通时也去 `create_bucket`（真病因被换成更莫名的报错），`exists()` 在基础设施故障时返回 `False` → `put_raw` 误判「没传过」而重传。现在只有确认「桶/对象不存在」（`_NOT_FOUND_CODES`）才走原逻辑，其余如实抛出。

**`docker-compose.yml` 四个容器都加了 `deploy.resources.limits`**：PG 1G/1.0 CPU、Neo4j 2G/1.0（另有原有的 heap 1G + pagecache 512M）、Redis 512M/0.5、RustFS 512M/0.5。Redis 额外加 `--maxmemory 512mb --maxmemory-policy noeviction`（2026-10-23 校正：此处曾写 256mb/allkeys-lru，与 compose 实际值不符——这是**durability 关键旗标**，allkeys-lru 会在内存触顶时连 Celery 队列消息和任务结果一起淘汰，表现成「提交了入库却永远不跑、进度卡住」且无报错；noeviction 让故障变响亮）。⚠️ `deploy.resources.limits` 在 `docker compose up`（非 swarm）下生效于 Compose v2；若用的是老版 `docker-compose` v1 需改用 `mem_limit`/`cpus` 顶层键。

**第三轮加固（2026-10-23 深夜，同一轮审计的收尾）**

- ⚠️ **云端 chat 客户端必须禁跟随重定向**：`llmstore.validate_base_url` 只核**第一跳**，
  而 openai SDK 建 httpx 客户端时写的是 `kwargs.setdefault("follow_redirects", True)`
  （实测 openai 3.13 `_base_client.py`，且运行时确认默认客户端 `follow_redirects=True`）——
  用户填的 base_url 一条 302 就能把服务端请求甩进内网/云元数据端点（SSRF 第二跳）。
  现在 `core/llm.py::_build_openai` 自己造两个客户端塞给 `ChatOpenAI`
  （`http_client` 同步 / `http_async_client` 异步，都 `follow_redirects=False`），
  回归桩除了断言这两个字段，还断言**SDK 真拿的就是这两个对象**
  （`llm.root_client._client is llm.http_client`；不注入时那两个字段是 None，
  只断言字段值会写出一个恒真的假桩）。
  附一条减轻严重度的事实：httpx 在**跨源**重定向时会剥掉 `Authorization`
  （实测 `_redirect_headers`），所以 key 不跟着外泄，但请求仍打进去、响应仍回落到模型上下文。
  `llmstore.list_models` 那边本来就 `follow_redirects=False`，**两处必须一起保持**。
- `validate_base_url` 的**残留缺口如实写进了 docstring**（别再当它是闭合的）：
  ① 「先解析后判断」，真正发请求时 httpx 会再解析一次 DNS，两次之间换记录（重绑定）
     仍能落到私网——彻底封堵要把解析出的 IP 钉进请求（自建 transport + 改 Host 头），
     只有公网多租户才值得付这个复杂度；缓解是 `CLOUD_ALLOW_PRIVATE_NET` 默认 False。
- **`AskIn.question` 长度帽 512**（`api/app.py`）：问句进 embedding、进 prompt、进 PG 与
  checkpointer 三处存储，限流只数请求数不看体积，不设上限等于给注册用户一张
  提示词注入 + 存储放大票；再长的内容属于「资料」，该走 `/ingest`。
  ⚠️ 跨语言成对：`front/src/components/Composer.tsx` 的 `QUESTION_MAX = 512` 与它同口径，
  前端先挡是**手感**（超限时浏览器不让再输入，比发出去收一个 422 好），改一边必须改另一边。
  同一轮里**没有**给凭据字段加 `max_length`——`api/auth.py` 已经有「2~24 / ≥6」的判据，
  模型层再加一份就是本项目最忌讳的第二判据；TTS 也**不加**，它已有 `TTS_MAX_CHARS`
  的句末优雅截断，加 422 会把好手感换成报错。
- **`/music/status` 的失败不再静默**：宽捕获保留（它是 5 秒一次的轮询读，卡住必须降级成
  `available=false`，不能拖住事件循环），但补一条 `log.warning`——这条路径坏掉的形态
  恰恰是「播放条悄悄不见了、什么错都没有」，本文件已经记过同类难排查故障（无日志的超时）。
  顺手改掉 `except (TimeoutError, Exception)`：`Exception` 本就覆盖 `TimeoutError`，
  并列写会让人以为两类判据不同。
- **Vite 只绑一种回环**（`front/vite.config.ts`）：`host: 'localhost'` 在本机解析到 IPv6，
  实测只有 `[::1]:25173` 在听，`curl http://127.0.0.1:25173` 直接连接被拒
  （浏览器无感，它两个都试；但后端 FastAPI 是两个回环都绑的，
  于是「API 用 127.0.0.1 通、前端不通」会误导排查）。Vite 的 `host` 只能给一个地址，
  Node 也没有「同时绑两个回环但不绑网卡」的写法，所以**不改默认值**（默认绑 0.0.0.0
  等于把 dev server 与 `/api` 代理暴露到局域网），要跨机访问显式 `WUWA_WEB_HOST=0.0.0.0`。

**第四轮加固（2026-10-23 凌晨，三个只读子智能体审计 + 逐条复核）**

先记一条方法论：子智能体的清单里 **7 条有 2 条是错的或路径不存在**
（`tools/qqmusic_mcp` 报在 `src/wuwa_rag/tools/` 下、`TtsSection.tsx` 报在 `components/` 下、
「MusicBar 的 `setSt` 缺 alive 保护」在 React 18 下是无害的）。
所以每条都先读代码再动手，判据是「这条改动会不会让某个真实故障变可见/变不发生」。

- **入库必须有自己的一档**：`/ingest` 与 `/knowledge/refresh` 此前只吃 200/minute 的兜底档
  （等于没限），而一次提交要跑「爬外部 wiki → 分块 → 写 PG/S3 → BM25 全量重建 → 写 Neo4j」
  并改动**全体用户共享的语料**。现在两端口共用 `shared_limit(RATE_LIMIT_INGEST, "ingest")`
  （默认 5/hour，按登录用户）。⚠️ 与 `ask` 档同一条铁律：**分开计数就能换端点绕过**。
  `/ingest/control`（暂停/继续/取消）**故意不占额度**——排队时连止损都做不了就本末倒置了。
- ⚠️ **口令类 CPU 必须离事件循环**（本轮实测数字）：`hashlib.scrypt` 单次 **733ms**、
  PBKDF2-HMAC-SHA256 20 万次单次 **50ms**（本机 python3.13 实测）。这些是**同步**函数，
  原先直接在 async 处理函数里调 —— 一次登录就把整个 API 钉住 0.73 秒，
  并发登录等于全站停摆，而且改密/解锁/保存凭据这五个端点当时**没有任何专属限额**。
  现在 `api/auth.py` 四处 + `core/llmstore.py` 五处 `await asyncio.to_thread(...)` /
  `_derive_async(...)`（`_bootstrap` 是同步函数，在**调用它的那一步**整体丢线程池），
  五个端点共用 `shared_limit(RATE_LIMIT_AUTH, "creds")`。
- **三条判据从「多处各写一遍」收成一处**：
  ① 角色名：`knowledge/entities.CHARACTER_NAME_PATTERN` 是唯一判据，四个入口共用
     （`IngestIn` / `IngestControlIn` / `GET /ingest/status?character=` /
     `DELETE /knowledge/characters/{character}`）；
     后两个此前**完全不校验**，名字会拼进文件路径、S3 key、Redis 键与 Cypher 参数。
     （写这条的同时我就把常量名拼错过一次 `CHACTER`，靠导入期 AttributeError 才发现——
     说明「字符串写四处」的漂移不止是理论问题。）
  ② `thread_id`：`core/conversations.THREAD_ID_RE` 是唯一判据，
     `api/app._client_thread_id` 与 `core/feedback.save_feedback` 都引用它
     （此前 `/feedback` 绕开了校验，可以写入任意超长/带特殊字符的 thread 串）。
  ③ 反馈各列长度帽：常量只定义在 `core/feedback.py`
     （`COMMENT_MAX/QUESTION_MAX/ANSWER_MAX/TARGET_ID_MAX/NAME_MAX`），
     `api/usage.FeedbackIn` 用 `Field(max_length=fb.*)` 引用而不是再抄一遍数字，
     并由 `test_反馈帽只定义在core一处` 钉住。
     ⚠️ **键列（thread_id/target_id）必须拒绝而不是截断**：截断会把两个不同的 id
     折叠成同一个 `(user_id, target_id)`，表现是「给 A 点的赞跑到 B 上」——
     数据不报错、只是悄悄错位。只有快照三列（comment/question/answer）允许截断。
- **入库失败原文不再发给全体登录用户**：`crawl_runs.error` / 进度里的 error 是
  worker 写的 `f"fetch: {exc}"` 之类**异常原文**，可能带 URL、内部路径。
  项目既有口径是「客户端只给中文一句话，原文进日志」（`on_unhandled`、SSE 的 error 事件都这样），
  而 `/ingest/records` 与 `/ingest/status` 这两个出口漏了。现在非管理员拿到的是
  「『抓取』这一步失败了，详情请联系管理员」（用五步快照里第一个 failed 的步骤判，
  不另起一套判据），管理员照旧看原文；`CANCEL_ERROR` 哨兵**必须原样透出**，
  因为下游靠它显示「已取消」。
- 会话搜索转义 LIKE 元字符（`conversations._like_pattern`）：不转义时搜 `100%` 命中全部会话、
  搜 `a_b` 命中「AXBY」。不是注入（参数化早就挡住了），是**结果不对 + 全表扫**。
- `api/ratelimit.py` 的文档段曾写「`/docs`、`/redoc`、`/openapi.json` 同样豁免」——
  **代码里从来没有那段**（全文件只有 `/health` 一处 `@limiter.exempt`）。已改成如实描述：
  它们走 200/minute 兜底档；要收紧就用 `FastAPI(docs_url=None, ...)` 关掉，
  而不是留一句假承诺（读的人会以为已经处理过）。
- `DELETE /ingest/records/{id}` 里那段裸 `psycopg.connect` 补了 `to_thread`：
  它是同文件里唯一还在事件循环里做同步 PG 往返的地方（其余三处都是 `to_thread` 口径）。

**审过但决定不改的两条**（不改的理由也要留，否则下一个人会再来一遍）：

- `MusicControlIn.action` **不加** `pattern` 白名单。动作词表（12 个 + `volume_set_<float>`）
  的权威定义在 MCP server 里（`tools/qqmusic_mcp/qqmusic_local_mcp.py:player_control`），
  它自己会拒绝未知动作并给可读原因；在 API 再写一份就是**第二判据**，
  而且那个 server 是独立进程、刻意自包含（导入它会把 mcp/comtypes 依赖拖进 API）。
  已核实没有注入面：`act` 只进 `smtc.control(act)` 与 `float(...)` 解析，
  `taskkill` 那条分支只在硬编码的 `act == "stop"` 下走。
- `/auth/register` 的用户名密码**不加** pydantic `max_length`。`api/auth.py` 已经有
  「2~24 / ≥6」的判据（且已覆盖体积问题），模型层再写一份就是两处判据。
  TTS 同理：它已有 `TTS_MAX_CHARS` 的句末优雅截断，加 422 会把好手感换成报错。

### 测试套件（`tests/`，2026-10-06 新增）

`uv run pytest` → **469 条用例、全量离线、约 5s**。pytest 在 `dev` extra（`uv sync --extra dev`）。

十五个文件，各自守一类「功能没坏，只是没被验证」的高危判据：

| 文件 | 守什么 |
|---|---|
| `test_entity_names.py` | 角色名提及判定：单字名正样本 12 条 + 常用词负样本 10 条 |
| `test_intent_routing.py` | 意图分流（双断言 intent+chars）、`is_self_intro` 30 条、`is_identity` 边界 |
| `test_profile_facts.py` | 画像 `_fact_category` 22 条、注入串拼接与 `MAX_FACTS_IN_PROMPT` 截断 |
| `test_ratelimit_cors.py` | 限流各档、豁免、429 头、CORS 顺序、真实 app 接线完整性（含 **`ask`/`ingest`/`creds` 三组必须各自共用一个计数器** —— 分开计就等于换个端点把配额翻倍） |
| `test_architecture_layers.py` | 分层方向、循环依赖、计数基线、空包清理与同名子包未误删 |
| `test_config_and_prompt.py` | config 默认值口径、prompt 拆分后 `all_sources` 再导出与提示词构建、**联网溯源去重与失败原因**（含 `web_*` 三字段每轮清零）、**端口四处对账**（compose 宿主机侧只准引用 `${VAR}`；`.env` 的 `*_PORT` == `config.py` 默认；`ports.ps1` 与 `vite.config.ts` 的兜底值必须等于代码默认；一律不许退回常用端口或落进 Windows 动态区 49152+）、**SSE `done` 事件与落库 meta 同键**（`_done_event` 是唯一构造器，加键忘了进 `_meta_of` 就红）、**日志大类名单与 `get_logger` 调用点双向一致** |
| `test_ingest_control.py` | 入库暂停/继续/取消旗标、状态聚合与操作留痕、**分块 0 块必须失败**（且失败时不许先把 jsonl 写坏）、**覆盖不全只 warn 不拦路**（PG 挂了也不能带崩入库）、**BM25 缓存随磁盘指纹失效**（文件没变不重载、变了必重载）、**稠密索引按文本口径对账**（口径不一致才重嵌、索引里没有要补、PG 已不存在的孤儿要删、`--full` 必须名副其实、未知参数直接退出） |
| `test_music.py` | 音乐 NLU 正/负样本（差一字即漏的动词锚定）、可用性四层判定、as_user 契约 |
| `test_qqmusic_mcp.py` | MCP server 侧：常驻会话、工具面收敛、搜索降级重试 |
| `test_security_fixes.py` | 2026-10-09 安全修复回归桩 + **账号回收对账**（从建表语句反推「带用户列的表」核对 `PURGED_TABLES`/`EXCLUDED_TABLES`、两道护栏、记忆两路并集与前缀口径）：默认凭据/CORS/XFF/verify 分片/TTS workspace/错误体不泄原文/must_change/`_DECRYPT_ERRORS` 接住 InvalidToken、**反馈键列拒而不截断 + 帽只定义在 core 一处**、**ILIKE 元字符转义**；**云端 chat 客户端禁跟随重定向**（openai SDK 默认 True，会把 `validate_base_url` 只核过第一跳的地址 302 进内网 = SSRF 第二跳） |
| `test_text.py` | **`text.py` 全部清洗规则的实测基线**：图标残渣 5 正 8 负、`fix_percent_units` 六条红线（`27%5%`/`1%.20%`/术语前瞻/量纲）、`[n]` 剥离与 COST 还原、AnswerFilter 流式（含「短答案整段重复」回归）、`lock_focus`、dedup 只丢整行相同、**两条输出口都只过 `_clean_answer`**（源码判据，防再出现「④ 补了 ①②③ 漏」） |
| `test_guard.py` | 防复读闸：整句重复三次必中/两次放行、周期块循环（计数后缀形态）从最早块截断、退化块后跟正常内容也能算出切点、材料表/技能 13 行/短碎句负样本不命中、流式逐 token 触发 |
| `test_team_prune.py` | 名册归一五类形态（漂泊者补男属性/剥括号/剥位置词/前缀最长匹配/纯槽位→None）+「绝不猜着切」红线；`_covers` 逐位不摊平、等价对互 True、占位串判定、镜像指纹；**抽取出的每支队伍 `+` 段数不得超过 `TEAM_SLOTS`**（一格两支队伍必须在进图前拆开） |
| `test_corpcheck.py` | **语料对账的八条判据**：总数相同但逐角色不同要报、缺/多两个方向分开报、稠密索引**文本口径**不一致要报（只比 id 抓不到那 36 块）、raw 哈希漂移 fail 而缺席只 warn（聚合一条，不刷屏）、图谱名册缺角色 fail / 多垃圾名 warn、退出码只看 fail（warn 不拦路）、fail 必须自带可抄的补救命令 |
| `test_offline_gate.py` | **守卫的守卫**：conftest 的非回环 socket/DNS 拦截必须真的能红（fixture 写错条件恒假 = 离线约束静默失效） |

**全量离线是硬约定**：不依赖 PG / Neo4j / Redis / Chroma / Ollama / 网络。已实证——把五个外部服务全部指向不可达端口，全部用例仍全绿（2026-10-06 首轮实测：139 条 / 0.99s；2026-10-09 复核：392 条 / 4.6s；2026-10-10 复核：**394 条 / 4.4s**；同日第二轮端到端重跑后再补 19 条 → **413 条 / 4.3s**（同一轮复跑 5.3s，差异是 jieba/reranker 预热）；2026-10-22 端口迁移补 5 条端口对账 → **418 条 / 4.68s**；2026-10-23 质量审计修复补 3 条守卫 + 13 条材料判据用例 → **434 条 / 3.3s**；同日联网溯源与失败原因收口再补 2 条 → **436 条 / 3.2s**；第二轮审查（假阳性回归 + 安全加固）再补 3 条 → **439 条 / 3.3s**；第三轮（done 事件与落库 meta 同键、日志大类名单、云端重定向二跳）再补 3 条 → **442 条 / 4.67s**；第四轮（分块零结果守卫、BM25 跨进程缓存失效、反馈帽同源、ILIKE 转义）再补 6 条 → 448 条 / 4.74s；第五轮（复核自己今天的 diff，抓到诊断函数的元组行 bug + 掩码措辞收口）再补 2 条 → **450 条 / 4.9s**；第八轮（GitHub 同类项目调研后落地：`build_index.plan_reconcile` 索引口径对账，见「text.py 写入端/读取端共用同一份清洗」那条）再补 4 条 → 454 条 / 5.09s；同轮再落地 `knowledge/corpcheck.py`（把三次事故的「三方对账」固化成一条命令） + 8 条判据用例 → 462 条 / 5.6s；第十轮（账号回收：清单反推对账 + 两道护栏 + 记忆两路并集）再补 3 条 → 465 条 / 5.7s；第十一轮（一格两支队伍：`split_teams` 收进 entities 供抽取/读取两侧共用 + `wuwa-graph` 改为先清出边再写）再补 1 条 → 466 条 / 5.21s；第十二轮（精排异常判空 + 召回调试出口共用同一份实现）再补 3 条 → **469 条 / 5.0s**）。2026-10-09 起这条约定由 `conftest._block_external_network` **强制执行**（autouse：非回环 `socket.connect` / `getaddrinfo` 直接 RuntimeError）；回环必须放行——Windows 的 asyncio 把 `socketpair` 模拟成 127.0.0.1 回连，全禁会让所有异步用例假失败。⚠️ 偷偷连本机容器的行为仍拦不住，那类依赖靠「容器没起也必须全绿」的人工抽查兜底。这条约定让它能当**提交门禁**：不会因为 `dev.bat` 没起而「假失败」，久了没人信等于没有。

✅ **2026-10-24 这条约定真的变成机器门禁了**（`.github/workflows/ci.yml`，学自 uv/vite 项目的通行写法）：
`backend`（`uv sync --frozen --extra dev` → ruff → `scripts/check_layers.py` → pytest）+
`frontend`（`npm ci` → vitest → `tsc --noEmit && vite build`）。runner **不起 docker**，
所以「离线」必须是可证的，本地核了四件事：
① `uv lock --check` 同步（`--frozen` 才不会悄悄改锁）；
② `torch` 主依赖走 `pytorch-cu130` 索引，lock 里**确有** `manylinux_2_28_x86_64` 的 cp313 wheel（Linux 装得起来，代价是几个 GB）；
③ Windows-only 的 `pycaw`/`winrt-*` 只在 `music` extra 里，CI 不装 → 主依赖没有平台坑；
④ 把 PG/Neo4j/Redis/S3/Ollama **全部指向不可达端口**跑一遍 → **453 通过**，
   唯一红的 `test_端口_env_是唯一真源且与各消费方一致` 断言的正是「这些变量没被改过」，
   是配置一致性不是连库依赖，CI 里不设这些变量所以也绿。
⚠️ `requires-python = ">=3.13,<3.14"` **上界也锁死**，workflow 里 Python 版本只写一处（setup-uv 的 input）。
⚠️ **诚实说明**：workflow 文件本身没法在本地验证（Actions 只在 push 后真跑）。
上面四条是本地能核的全部；首推之后要看一眼 Actions 是否真绿。
`eval_retrieval.py` 刻意不进 CI：它要真实 Chroma/BM25 与 reranker 权重，且文件名不以 `test_` 开头，
pytest 收不到（缺索引时不会假失败）。

❌ **顺手记一条「调研说有、实测证伪」**（免得下一个人再照着改一遍）：本轮子智能体报告称
`test_offline_gate.py`（守卫的守卫）里「断言在被污染的环境条件下恒成立 = 又一类结构假绿」。
直接做变异测试复核：把 `_block_external_network` 的 **connect 分支**改成空转 →
`test_非回环连接必须被拦` 恰好红；再把 **getaddrinfo 分支**改成直落真解析 →
`test_非回环DNS解析必须被拦` 恰好红。两条分支各自都能红，**结论不成立**，那条不改。
（同一份报告里另有两条也不成立：`tools/qqmusic_mcp` 与 `TtsSection.tsx` 的路径写错了目录、
「MusicBar 的 setSt 缺 alive 保护」在 React 18 下无害 —— 记这个是为了重申老规矩：
**子智能体的每一条都要先读代码再动手**。）

三条写用例时必须遵守的经验（都是实际踩过的）：

- ⚠️ **断言要验 `characters`，不能只验 `intent`**。单字名 bug 期间 `intent_node('心的声骸怎么配')` 返回 `intent='hybrid'`、`chars=[]`——只看 intent 全绿，功能其实已废（空 chars 反而**更容易**满足「未指名」判据，intent 照样合理）。所以每条路由用例都双断言。
- ⚠️ **写断言前先实测键名/返回值形态**。`check_layers.modules()` 的键**不带** `wuwa_rag.` 前缀（是 `dialog.prompt`）。若按直觉写成 `assert "wuwa_rag.rag" not in modules`，该断言**恒真**——空包回来了也不会红。这类假绿靠跑测试发现不了，只能靠「变异测试」：临时造回一个空包（`src/wuwa_rag/storage/__init__.py`），确认对应用例真的变红，再清理。
- ⚠️ **`limiter` 是模块级单例**，合成 app 必须 **module 级只建一次**（`scope="module"` fixture）。每个用例都新建 app 会把同名端点反复注册进 `_route_limits`，导致①真实 app 接线断言看到多余项、②同一路由挂上多份限额使 429 提前。断言真实 app 时还要按 `wuwa_rag.api.app.` 前缀过滤掉合成端点。

**覆盖边界（诚实说明）**：`tests/` 里的 469 条单测守的是**判据**（规则层），不是**效果**（检索质量）。检索质量由另外两个工具覆盖：

- **`tests/eval_retrieval.py`**：检索质量评测脚本。依赖真实向量索引（Chroma + BM25）与本地 reranker 权重，所以**不离线**、需 `data/chroma/` 完好；文件名不以 `test_` 开头，pytest 不会收集（缺索引不会假失败）。手动跑：`uv run python tests/eval_retrieval.py [--topk N] [--stage recall|rerank] [--json]`，17 条约 1~2 分钟（每条要跑一次 CrossEncoder）
- **`tests/retrieval_eval_dataset.json`**：17 条带标注 query（6 人工标注 + 8 声明式选择器 + 3 真域外负样本），覆盖 fact / semantic / multi / single_char / value_table / negative。两种标注可混用：
  - `relevant_chunk_ids`：显式 id 列表，精度最高，适合单目标 query；
  - `relevant_selector`：`{"characters":[…],"components":[…],"tabs":[…]}`，评测时按语料实时解析成 id 集合，语料增长后自动跟上，比手写 id 可维护；
  - 两者都空 = 负样本，期望精排闸门判空。脚本启动时会**体检**：显式 id 若指向不存在的 chunk 直接退出码 2（否则指标恒 0 且看不出来）。
- **指标**：`recall@k` / `precision@k` / `MRR` / `gate_empty_rate`，外加 **`recall_ceiling`（天花板）与 `recall_gap`（真缺口）**。
  - ⚠️ **天花板 = `min(topk, 相关数)/相关数`**：相关块多于 topk 时召回不全**不是缺陷**（一次只能送 topk 条），是评测口径上限。只有 `recall < ceiling` 才是真缺口、才值得调参数。不区分这两者会把「口径上限」误读成「检索坏了」（首版就误报了：`长离常态攻击` 9 个相关块、topk=6，recall=0.67 已是满分）。
  - ⚠️ **人工标注与选择器标注分开报**：前者可信，后者是宽松上界，混在一起会互相污染。做参数决策看 manual 那行。
- **基线（2026-10-06，走完整生产链路 vector_search→rerank，topk=6）**：

  | 分组 | n | recall | 天花板 | precision | MRR |
  |---|---|---|---|---|---|
  | 人工标注 | 6 | **1.0000** | 1.0000 | 0.1667 | 0.8889 |
  | 选择器标注 | 8 | 0.7875 | 0.9583 | 0.5208 | 0.7750 |
  | 正样本合计 | 14 | 0.8786 | 0.9762 | 0.3691 | 0.8238 |
  | 负样本闸门判空率 | 3 | — | — | **1.0000** | — |

  真缺口 **4/14**（其余 10 条已达天花板）。其中 3 条是相关块落在 RRF 第 22 位、被 `TOPK_RERANK_IN=20` 切掉；1 条（「守岸人怎么玩」的 `战斗风格` 块）是**标注偏松**——该块平均只有 64 字有效正文 + 3.5 个 `[图]` 占位符（全语料 58 个同类块都这样），reranker 排在 top6 外是合理的。
  ⚠️ 人工标注 precision=0.1667 恰等于 `1/6`，是 topk=6 下的**数学上限**（6 条里只有 1 条相关），不是检索差。
- **`tests/ab_topk_rerank_in.py`**：`TOPK_RERANK_IN` 的 A/B 实测脚本（一次性诊断用，非测试）。2026-10-06 跑过 20/25/30/40 四档，**结论是维持 20**——详见 `config.py` 该参数的注释（含完整数据表与「首版 A/B 因把 `TOPK_RERANK_IN` 误当 `topk` 而得出相反结论」的教训）。
- ⚠️ 绝对值受评测集规模与标注松紧影响，**只用于同一评测集下不同参数的相对比较**，别跨版本比绝对值。

**前端也有测试基线了（2026-10-23 新增，`front/`）**：`cd front && npm test` → **vitest 37 条 / 4 个文件，纯 node、不起浏览器、不碰后端**。
装的是 `vitest@^2`（**不是最新**：vitest 5 的 peer 要求 vite ≥6，而本项目是 vite 5.4；
升 vite 会牵动构建与没人目视验证过的界面，所以按 vite 5 选大版本）。

| 文件 | 守什么 |
|---|---|
| `store/useStore.test.ts`（14 条） | 见下面 ①~④（断流落 error、两条路径同判据、loading 态被消费、clearAll 失败不误报已清空）+ **三条出口都把 `message_id` 换进气泡 id**（流式 send / regenerate / 非流式 `/ask`），并断言非流式那份 meta 与流式**键集合相同**（它原先真的少了 `web_reason`）+ **`convsErr` 一写一清**（列表读失败必须记下原因、读成功必须清零——「读不到 ≠ 没有」在 store 层的那一半） |
| `styles/tokens.test.ts`（12 条） | **配色令牌的可读性契约**：9 个「文字角色」令牌在两主题的每块平底下都 ≥4.5:1；**同一条规则里「文字色 × 半透明底」的组合**按 alpha 合成后也要过；语义饱和色（`--accent/--ok/--warn/--err/--accent-2/3`）不许再被直接当 `color:`（只准 `-text` 档）；浅色主题必须真的覆盖那六个 `-text` 令牌。详见「主题与配色令牌」一节 |
| `lib/format.test.ts`（4 条） | 时间/字节格式化的三条判据：**null / 空串 / 畸形串一律「—」**（裸写 `toLocaleString` 时这些会印成 "Invalid Date"）、24 小时制不许出现「上午/下午」、KB 一位小数且 0 走破折号 |
| `lib/feedbackHydrate.test.ts`（7 条） | 评价回填的**同帧合并**（三条气泡只发一个 `/feedback/mine`）、非服务端 id 不发、按 apiBase 分批、成功落缓存/失败不落缓存（否则永远回填不上）、`rememberRating` 必须覆盖旧缓存、取消只丢回调不丢别人的回调 |

`useStore.test.ts` 守的是**这一批真实缺陷**：
① SSE 干净断流（流结束但没收到 `done`）必须落 `error` 态而不是永停「帮家人找资料…」；
② **同一条判据在 `send` 与 `regenerate` 两条路径都成立**——修复时只补 `send` 就漏了它，
   所以两条各有一例（`metaOfDone` 也由此收成一份，后端加字段只改一处）；
③ `loadingConvs` 在请求期间为真（这个字段曾只写不读，于是加载期拿空状态冒充「没有数据」）；
④ `clearAll` 失败返回 `false` 且不许清本地列表（旧写法吞掉异常还照报「已清空」）。
替身纪律：整个 `../lib/api` 用 `vi.mock` 顶掉；`listConversations` 在「新建会话」用例里
**必须返回那条会话**，否则会撞上 `loadConversations` 里「当前会话已不存在 → 退回草稿态并清空
messages」这条**正确**逻辑，测出来的是替身不像真而不是代码有问题（踩过）。
带模块级状态的用例（`feedbackHydrate` 的缓存）每条都要 `vi.resetModules()` + 动态 `import()`
取一份干净实例，共享一份模块会让用例互相污染；计时器用 `await vi.advanceTimersByTimeAsync()`
而不是 `advanceTimersByTime()`——后者不排微任务，`await` 掉的响应永远落不到断言里（踩过）。
变异验证：把两条断流判据改回旧写法 → 恰好那 2 条红，还原即绿；
配色那条同理做过三次（把 `--accent-text` 退回 #0f766e → `.kb-count` 那组红；
往 `_done_event` 加一个新键 → done/meta 那条红；把 `cached.set` 删掉 → 缓存那条红）。

`pyproject.toml` 的 `[tool.pytest.ini_options]` 里 `asyncio_mode = "auto"` **不能省**：pytest-asyncio 1.x 默认 strict，不声明会让异步用例**静默跳过**（不失败，根本不跑），比没测试更危险。

### 前端交互收口（2026-10-23）

界面这一侧的几条**规则**（不是改动清单，改这些组件前先读）：

- **破坏性操作的两步确认只有 `components/ConfirmButton.tsx` 一处实现**。全站 5 个调用点。
  四条不可回退的行为：① 待确认态定时器挂在组件 ref 上并在卸载时清掉（列表行删完组件就没了，
  旧写法会把回调留到已消失的行上）；② 外部把按钮置灰（`disabled`/`busy`）必须同时解除待确认，
  否则恢复可点那一瞬间用户看到的还是「再点一次」，等于默认放行了一次破坏操作；
  ③ **`title` 在待确认态不许抑制**——知识库两处删除把后果（清文档/索引/图谱、只删账本不动流水线）
  只写在 `title` 上，按钮脸上的文案只是「再点一次确认删除」不含后果，而那一刻正是用户
  最需要读到它的瞬间（触屏没有 tooltip，这条已经是仅存的提示通道）；
  ④ 自动解除时长只有一个模块常量 `AUTO_DISARM_MS = 4000`，
  曾经挂在组件上的 `autoDisarmMs` prop 没有任何调用点用过它，已删（要按站点调手感时再按需求加回来，
  不要为假想需求留开关）。
- **`GET /feedback/mine` 的回填走 `lib/feedbackHydrate`，不许在气泡里直接发请求**。
  同帧挂载的 id 合并成一个请求、按 apiBase 分批、成功落缓存/失败**不**落缓存。
  三条判据都有用例（见测试套件表）。为什么不「把 rated 从 ChatPage 往下传」：
  回填是每个气泡自己的显示增益，调度权不该交给列表；合并层放在 lib 里也才测得了
  （组件级测试要引 React 测试环境，而本项目 vitest 跑的是纯 node）。
  ⚠️ 提交成功后必须 `rememberRating(id, rating)`：缓存里可能已有这条的 `0`（挂载时问过、
  那时还没评），不回写就会出现「点完 👍 切走再切回，按钮显示没评过」。
- **层级与尺寸只从令牌取**：`--z-gate` / `--z-toast`（登录门禁与 toast 谁压谁都写死在令牌里，
  不要再在组件里写 `z-index: 9999`）、`--fs-*` / `--space-*`、内容宽度 `--content-max`
  （唯一入口在 `App.tsx` 写 CSS 变量，页面自己再写一套 max-width 就会「设置里调了没反应」）。
  同一条弱化提示底色全站共用 `.kb-warn, .set-warn` 一个原语，别再复制第二份黄底。
- **`done.message_id` = 这条回答的 `messages` 行号，前端必须拿它换掉气泡的本地 uid**
  （`store/useStore.ts` 的 `send` 与 `regenerate` 两处 + 非流式 `/ask` 的 `AskOut.message_id`，
  三条出口都要换）。原因：满意度反馈按 `msg.id` 提交，而本轮新气泡出生时只有 `uid()` 随机串——
  不换就会把评价挂到一个「刷新后再也不会出现」的 id 上，表现为
  「刚点过 👍，刷新回来变成没评过」+「同一条回答能被评两次（一次 uid、一次数字 id）」。
  评价记录本身没丢，丢的是**归属**，所以这类 bug 靠后端查是查不出来的，只能前端自证。
  键的归属：`message_id` 与 `done`/`thread_id`/`answer` 一样属于**事件信封**，不进 `messages.meta`
  （`_meta_of` 的白名单里没有它，契约用例按这个集合断言）。
- **展示格式化只有一份：`lib/format.ts` 的 `fmtTime / fmtDate / fmtSize`**。
  原来 `fmtTime` 在 `KnowledgePage.tsx` 与 `AdminUsersCard.tsx` 里**逐字重复两份**，
  而 `FeedbackPanel.tsx`、`HistoryPage.tsx` 干脆裸写 `new Date(x).toLocaleString('zh-CN')` ——
  裸写的两处遇到 null/空串会把 **"Invalid Date"** 直接印在界面上（时间戳为空的记录是真实存在的）。
  两处重复 + 两处裸写 = 四种口径；现在四处都调用同一份，异常值统一成「—」。
- **图表的空态文案不许写死天数**：用量面板可以选 1/7/30 天，
  而 `UsageCharts` 的两个子图原来硬编码「近 7 天没有用量」——选 30 天时看到这句话就是假信息。
  现在 `days` 由 `UsagePanel` 传进来（两个子图各自接一次），与表格里已有的 `近 {days} 天…` 同口径。
- **`@media (prefers-reduced-motion: reduce)` 只写一条全局规则**（`global.css`）。
  原先只有 `ConfirmButton.css` 自己处理，`blink`（在线点）、`dot-bounce`（进度点）、
  `fade-up`（气泡入场）、`spin`（加载圈）照旧无限转。
  ⚠️ 写法是「时长压到 0.01ms + 只跑一轮」而**不是** `animation: none`：
  转圈是「正在进行」的反馈，彻底抹掉会让用户以为界面卡死。
- **输入框去掉默认描边后必须自己补聚焦态**：`.kb-search input` 与 `.session-edit input`
  都写了 `outline: none`（内部无边框的观感需要它），但没有任何替代 —— 键盘用户看不出光标在哪。
  现在按 `.hist-search:focus-within` 的既有口径补上（外壳描边 / `:focus-visible` 描边）。
- ⚠️ **这台机器上的浏览器实测有一个硬限制**：in-app 浏览器视口是 0×0，
  `getBoundingClientRect()` 全部返回 0、截图工具直接报 `NATIVE_BROWSER_VISPORT_UNAVAILABLE`。
  所以「尺寸类/视觉类」的结论不能靠量 DOM 得到；改成的口径是
  ① 颜色对比：遍历 DOM 用 `getComputedStyle` 按 alpha 合成底色后算比值（这个不需要视口）；
  ② 规则是否真的生效：查**构建产物** `dist/assets/*.css` 里有没有那条选择器
     （`grep -o "kb-search:focus-within{[^}]*}"` 这种），比读源码更接近真相。

- **列表行不是按钮**：会话行用 `div` + `tabIndex` + `keydown`，不要加 `role="button"`
  （行内有删除按钮，嵌套可点击元素对读屏是非法结构）；`onClick` 里必须判
  `e.target !== e.currentTarget` 才返回，否则点行内按钮会连带切会话；
  当前会话行用 `aria-current` 而不是只靠颜色。
- **toast 容器常驻挂载**（`aria-live="polite"`）：条件渲染会让读屏错过第一条播报。

- **行内操作不能只在 `:hover` 出现**：会话栏的「重命名 / 删除」原本只有
  `.session-item:hover` 一支显示规则，而 `display:none` 是把按钮整个移出 tab 序的——
  键盘用户**永远够不到**，触屏用户（没有 hover 事件）也永远看不见。
  现在是 `:hover` / `:focus-within` / `:focus-visible` 三支 + `@media (hover: none)` 常显
  （行本身 `tabIndex=0` 可聚焦，所以顺序是「聚焦行 → 按钮出现 → Tab 进入按钮」）。
  ⚠️ 不要退回「常显」：会话栏只有 248px，图标常驻会把标题挤成三五个字。
  图标按钮同时补了 `min-width/min-height: 24px`（WCAG 2.5.8 最小可点区；
  用 min-size 而不是加 padding，因为加 padding 会把整行撑高）。
  实测口径：这些规则是否真的下发，看 `dist/assets/*.css` 里的
  `session-item:focus-within` 与 `@media (hover: none)` —— 本机的 in-app 浏览器
  视口是 0×0，`getBoundingClientRect()` 全是 0，**量不了尺寸**，只能查 CSSOM/产物。
- **「读不到」绝不能渲染成「没有」**：新增 `components/LoadFail.tsx`，
  用在画像卡、云端模型、语音合成三处。原来的形态是三处各写了一遍同一句谎话——
  `load()` 里 `catch { setFacts([]) }` / `st`/`cfg` 留 null，渲染分支把 null 当
  「还没配过」，于是后端一挂界面就说「你还没有画像 / 尚未配置」，
  ✅ **2026-10-24 把剩下的逐个核完了**（原先那句「还没核」是真的没核，一核就核出 9 处显示位、7 个根因）：
  - **最大的一个根因在 store 不在组件**：`loadConversations` 的 catch 只处理 401，其它错误**不留任何痕迹**
    （列表读挂了 → `convs` 还是空 → 会话栏说「暂无会话」、历史页说「还没有任何会话」、设置页数据管理
    统计说「0 会话 / 0 消息」，而同一屏的「入库记录」是本地值、看着同样可信）。
    现在单一出口是 `useStore.convsErr`（成功清零、失败记原因、**列表照旧保留上一次的**），
    三个消费点各自只判这一次；两条用例在 `useStore.test.ts`（失败要记下原因 / 成功必须清零）。
  - 音乐设置：`mset` 留 null 会落到那条三元链的**兜底分支**，等于断言「音乐功能没开」——
    现在失败先渲「状态未知：这次没读到，下面的勾选与路径都不作数」+ `LoadFail`（注释原本写着
    「不假装有设置」，实际正在假装）。
  - 云端模型「当前生效」：读失败时原来会肯定地报「本地默认（aemeath）」，真实状态可能是云端自定义 → 报「状态未知」。
  - TTS 折叠**标题**：内容区早改成三态了，标题还写「（尚未配置）」，同屏自相矛盾 —— 修一半等于没修。
  - 反馈汇总芯片：`{st?.up ?? 0}` 把读不到显示成「0 赞 / 0 踩 / 全部反馈（0）」；同屏的差评率本来就是 `—`，
    照它对齐即可（`—` 与 `0` 的区别就是这条规则本身）。
  - 历史页全文搜索失败：回落本地标题/预览过滤是**对的**（不打断输入），但落空时那句「没有匹配的会话」是假的
    —— 改成「标题与预览里没有匹配（全文搜索这次没读到）」。
  - 画像卡重试窗口：`loadFacts` 只清错误不清数据，重试那几百毫秒 `facts=[]` 且 `factsErr=''`
    → 界面闪回「还没有画像，多聊几轮…」。开头补 `setFacts(null)` 回到「读取中」。
  ⚠️ 判据口径：**失败后保留上一次可用数据 ≠ 可以不记失败原因**；两者都要，缺一个就退化成谎话。
  `LoadFail` 因此多了一个可选 `note`：默认句「这不代表没配过」是给**配置类**卡片写的，
  会话列表/反馈列表要的是「这不是没有会话 / 这不是没有反馈」——硬套默认值就成了新的谎话。
  实测口径（in-app 浏览器，把 `apiBase` 指到不可达端口 `http://127.0.0.1:1` 触发真失败，不动任何在跑的服务）：
  会话栏出「会话列表读取失败：Failed to fetch。这不是没有会话 ——重试」、历史页头一句「会话数未知（列表这次没读到）」、
  数据管理统计 `—/—/0`、反馈芯片 `—`、三张设置卡各自 `LoadFail`；恢复 apiBase 后全部回落正常态，
  并真发一句「守岸人配队」确认问答与列表刷新没被这批改动碰坏。
- **删除会话要成功了才报成功**：`deleteConversation` 改成返回 `Promise<boolean>`
  （与 `clearAll` 同口径），`HistoryPage` 只在 `true` 时 toast「会话已删除」；
  原来无条件报成功，失败时同一屏会出现「删除失败」与「已删除」互相打脸。
- **`target="_blank"` 必须钉 `rel="noopener noreferrer"`**：`markdown.ts` 里
  `ADD_ATTR: ['target']` 只是**放行**这个属性，答案里的链接来自检索语料与模型输出，
  写不写 rel 不由我们控制 —— 所以在 `afterSanitizeAttributes` 钩子里统一补，
  而不是拿正则去改已经净化好的 HTML 字符串。
- **react-router 的两个 v7 future flag 显式打开**（控制台从此只剩真消息）：
  打开前逐个核过行为中立性 —— 全站 `to=` 都是绝对路径（grep `to="[^/]` 为空），
  唯一的 splat `path="*"` 渲染的是 `<Navigate to="/" replace />`，
  所以 `v7_relativeSplatPath` 是 no-op；`v7_startTransition` 只影响路由引起的
  state 更新批次，本应用状态在 zustand 里、导航只有 7 个扁平页面。
  实测：六页来回点正常、控制台从 5 条降到 3 条（剩下的是 vite debug 与 DevTools 提示）。

**第五轮：复核自己今天的 diff（两条真实回归，都值得记）**

- ⚠️ **「静默降级」的诊断函数必须留痕，测试必须能验它「出声」**。`worker._warn_missing_characters`
  里我写了 `r["character"]` —— 但 `core/db.py` 的池**没有** dict_row factory（那是 authdb 池才有的），
  psycopg3 默认返回**元组行**，于是每次都是 TypeError → 被这个诊断自己的宽 except 吞掉 →
  函数变成「永远不会出声的死代码」，而它的 docstring 写着会写日志。
  我的第一条用例还把 `get_cursor` stub 成「直接抛」，只验了「不拦路」，完全没验「能 warn」——
  这就是 AGENTS 里反复警告的**假绿**。修法两件事一起做：
  ① 按位置取列（`{r[0] for r in await cur.fetchall() if r[0]}`，与本仓其余 `core.db` 用法一致）；
  ② 补一条**返回元组行**的用例，断言 `log.warning` 真的被调用（`test_覆盖不全时真的写出warning`）。
  变异验证：把判据改回 `r["character"]` → 恰好这条用例红。
  推广出来的规矩：**宽捕获可以，但要 `log.debug("…跳过: %s", type(exc).__name__, exc)` 留痕**；
  只 stub 成抛异常的测试等于没测这条分支。
- **给「行」加内联按钮时，防冒泡不能写成 `e.target !== e.currentTarget`**。
  会话行是 flex 容器、面积几乎全被图标与标题（`flex:1`）占满，那条判据等于
  「只有 8px 内边距能点」——点标题不切会话（我第二轮引入的真回归，第五轮才被抓到）。
  正确判据是按**容器**：`if ((e.target as HTMLElement).closest('.session-ops, .session-edit')) return`。
  键盘侧（`onKeyDown`）保留 `e.target !== e.currentTarget` 是**对的**：那里要的就是
  「焦点落在行本身才算数」，两者语义不同，别再统一成一份。
- **采纳服务端 id 必须同时把列表 key 与 id 解耦**。`done.message_id` 换 id 之后
  `key={m.id}` 变了 → React 卸载重挂那条气泡，答案刚收尾的一瞬间**重播 0.32s 入场动画**（闪一下）。
  现在 `Message.listKey` 出生即定、永不改写，列表 key 用 `m.listKey ?? m.id`。
  推广：凡是「前端临时 id → 服务端真 id」的替换，都要问一句「有没有什么以 id 为身份」。
- **焦点环也是非文字对比度，门槛 3:1**：`ConfirmButton.css` 的 `outline` 原先用饱和
  `--accent-2`（浅底实测 2.98:1，刚好不达标），换成 `--accent-2-text`；
  它自带的那份 `prefers-reduced-motion{animation:none}` 也删了，只留 global.css 那一条
  （两份实现不但重复，写法还互相矛盾：全局刻意不写 `none`）。
- 顺带收掉一处措辞重复：入库失败原因的掩码文案原先在 records 里内联写一遍、
  `_public_step_error` 里又写一遍，现在只有后者一处，且它不再自己打日志
  （那个端点被前端 3s 轮询，按条写会刷满；改成整批汇总一条）。

### 主题与配色令牌（2026-10-23）

**浅色主题是事后补的，只补了 `--text-*` 三档 —— 语义色整族在浅底上全是洗掉的。**
`--accent / --accent-2 / --accent-3 / --ok / --warn / --err` 在深色底上是 7~13:1，
同一份色值放到白底实测只有 **1.29~2.98:1**，而全站有 **36 处** `color:` 直接引用它们：
浅色主题里错误按钮、「已收录」标签、markdown 链接、点赞/点踩选中态、入库进度点文字
全部洗成一片淡彩。默认主题是深色，人眼扫一遍很容易漏，所以它活到今天。

三条口径，改配色前必读：

1. **饱和色只用于装饰，文字/图标一律走 `-text` 档**。不在浅色主题里整体覆盖 `--err`：
   状态圆点、toast 左边框、开关滑块要的是「信号灯」，加深反而像脏了。
   `--accent-2-text` 是这个家族最早的一个（早有注释说明「两者是不同的颜色，不是别名」），
   本轮把它补齐成六个：`--accent-text / --accent-3-text / --ok-text / --warn-text / --err-text`。
   深色主题这些基本就是原色别名，唯一例外 `--err-text: #fca5a5`（见下面第 3 条）。
2. **逐个底色核，不许只看白底**：`--bg-deep` 浅色是 #eceff8，**比白底更深**，
   所以「白底 4.74:1 合格」的 `--text-muted` #69738c 在它上面只有 **4.12:1**、
   在 `--bg-base` #f4f6fc 上 4.39:1 —— 三块浅底里两块不达标（正是搜索框、空态、进度点
   走的路径）。现值 #5f6a83：5.42 / 4.71 / 5.01。
   浅色 `-text` 一族取值：teal `#0d655c`、indigo `#5a4bd6`、pink `#a21caf`、
   green `#047857`、amber `#92400e`、red `#b91c1c`。
   常见的 `#0d9488 / #dc2626 / #b45309` 在纯白上是 3.74 / 4.83 / 5.02，
   「看着像过了」，落到 `--bg-deep` 只有 3.26 / 4.20 / 4.37 —— 别按白底选色。
3. **半透明芯片比平底更深**：`--grad-soft` 是两层 0.16 alpha 的品牌渐变，
   叠在白底上合成出 rgb(211,232,248)。`--accent-text` 当时取 #0f766e：五块平底全过
   （最差 4.76），落到 `.kb-count` 这个芯片上只有 **4.34:1**。现值 #0d655c（芯片 5.50、
   平底 6.02~6.92）。深色同理，`--err-text` 的待确认态芯片（`--err-tint-14` 叠在
   `--bg-hover` 上）让 #f87171 掉到 4.34，提到 #fca5a5 后 6.33。
   ⚠️ 算这类底**必须按 alpha 逐层合成**，拿令牌原色或拿平底代替都会得出错的比值。

`::selection` 也是同一类：白字压在浅靛选中芯片（`--accent-2-tint-38` 叠白底）上实测
1.46:1 —— 划选一段文字等于整段看不见；深色主题同一组合是 9.90:1，所以只在浅色主题里
把选中字换成墨色档（`[data-theme='light'] ::selection`）。

**静态契约**：`front/src/styles/tokens.test.ts` 把这些变成断言（9 个文字令牌 × 两主题 ×
五块平底、以及「同一条规则里的 文字色 × 半透明底」组合、语义饱和色不许再直接当 `color:`、
浅色主题必须真的覆盖六个 `-text` 令牌）。它按 `themeScope/tailOf` 跳过被主题作用域覆盖过的
基础规则，否则 `.markdown code` 那种「深色给 #b9c2ff、浅色给 #5a4bd6」会报假案。

**浏览器实测口径**（一次性核对，非门禁）：遍历 `body *` 里**有自有文字节点**的元素，
把自身与祖先的 `backgroundColor` 和渐变 stop 按 alpha 依次合成，再按字号/字重取
4.5 或 3.0 的门槛。两个必须知道的坑：
① 隐藏页面里 **CSS 过渡不推进**（`transition: all` 让 `getComputedStyle().color` 停在
   切换主题前的值），要先注入 `*{transition:none!important}` 再测；
② 渐变色底不能用「平均色当不透明底」，会把 `.btn-primary` 的深墨字误判成 1.0:1。
实测结果：6 个页面 × 2 主题、790 个文字元素，除上述合成口径造成的假阳性外为 0 项不达标。

### 日志

`ww_logger.get_logger(name)` → `logs/<name>.log`，**一个大类一个文件**（名单在 `ww_logger._MODULE_LOGS`：app 是 root 兜底，独立文件 17 个 —— upload/rag/neo4j/vec/bm25/celery/auth/conv/usage/feedback/profile/llm/llmstore/tts/emotion/music/settings），每日轮转留 7 天，另有 `app.log` 兜底。⚠️ **`_MODULE_LOGS` 漏一个名字不报错，只会让那个 logger 静默落进 `app.log`** —— 排查时按大类去翻 `<name>.log` 就会读成「这条链路没跑」（`music` 与 `auth` 长期如此，2026-10-23 补全并由 `test_config_and_prompt.py::test_日志大类名单与调用点一致` 双向钉住：调用点用了却没进名单 = 红；名单里留着没人引用的名字 = 也红）。扫描调用点时两个坑：名字带数字（neo4j/bm25），以及引号两种风格混用（`get_logger('upload')` 与 `get_logger("rag")`）。每行格式 `时间 | 级别 | role | logger | 消息`，**role 列**逐行标注是谁写的：`api`（FastAPI）/ `worker`（Celery）/ `main`（兜底）。⚠️ **文件不再按 role 拆分**（曾产出 `rag.api.log` / `rag.worker.log` / `rag.main.log` 一堆文件，2026-09-22 已废止）：api 与 worker **共写同一个 `rag.log`**，靠 role 列区分，跨进程轮转安全由 `concurrent-log-handler` 的文件锁保证。**`start.ps1` 启动的两个窗口会显式注入 `WUWA_LOG_ROLE`**（Celery=worker、FastAPI=api），argv 推断只是兜底——注意 `python -m wuwa_rag.api.server` 时 `sys.argv[0]` 是模块**文件路径**（反斜杠），必须先归一化斜杠才能匹配 `wuwa_rag/api`（2026-09-22 首轮线上就因漏了这步，API 进程的 role 被记成 `main`）。

✅ **跨进程日志轮转（已修，2026-09-22）**：曾因 FastAPI 与 Celery worker 共用同一个 `rag.log`，`doRollover` 的「先 close 自己的 stream、再 os.rename」被另一进程占用句柄 → `PermissionError [WinError 32]`；且 `shouldRollover()` 此后会逐条重试这个失败的 rename，**该 logger 当天所有日志一条都写不进去**（实测指纹：`app.log` / `neo4j.log` 都产出了 `.2026-09-21` 轮转文件，唯独 `rag.log` 没有且停更）。
修法（两层）：① **基类换成 `concurrent-log-handler` 的 `ConcurrentTimedRotatingFileHandler`**（即 `_SafeTimedRotatingFileHandler`）—— 靠**跨进程文件锁**协调轮转 + 基于文件 mtime 智能判断是否真需轮转，这才是根治，也让「api 与 worker 共写同一个 `rag.log`」重新成立（这正是按 role 拆文件的做法被废止的前提）。实测两进程对齐同一时刻并发 `doRollover()`：双双成功、无异常、只产出**一份**轮转产物、12 行日志一条不丢，且没有兜底 warning（说明是锁机制本身起效，不是靠兜底救场）。② 兜底：`doRollover` 抛 `OSError` 时重开 stream + 把 `rolloverAt` 推到次日，只留一条 warning，避免该 logger 整天哑掉。`_process_tag()` 现在只用于给每行日志打 role 标签（见上条），**不再参与文件名**。
⚠️ **改完必须重启 API 与 Celery 两个进程**（各自会 import 一次 `ww_logger`）。排查时别把「rag 日志没更新」误读成「链路没跑」：看 `rag.log` 末尾有无该 role 的行（**别再去翻已废止的 `rag.api.log` 之类**）。

📌 **轮转后 `rag.log` 会短暂不存在**：`ConcurrentTimedRotatingFileHandler` 是 **lazy 重建**（下一次写日志时才 `_open()`）。实测两进程对齐同一时刻并发 `doRollover()`：双双成功、只产出 **1 份**轮转产物、14 行一条不丢；随后 `rag.log` 本体消失属正常行为，**别误判成「日志系统坏了」**。同理，整夜不写日志的进程其文件会一直缺席，直到再次写入。

## 领域术语（`prompt.SYSTEM_PROMPT` 同款对照，答问与写 prompt 都靠它）

声骸=角色装备（资料里写作「套装」，COST 是声骸费用点数组合如 43311）；共鸣链=命座（1~6 链）；贝币=货币；突破阶段=一阶~六阶。角色名册与别名（`光主→漂泊者-男-衍射`、`风主→气动`、`暗主→湮灭`、`电主→导电`、`卡提→卡提希娅`）在 `knowledge/entities.py`；**图谱/队伍串里的名字归一也走同一个模块**（`normalize_character_name` / `normalize_team`，见上文）。
