import os
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_BASE_DIR = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = _BASE_DIR / "data"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---------- 路径（.env 可覆盖；默认基于项目根，不依赖 cwd） ----------
    DATA_DIR: Path = _DEFAULT_DATA_DIR
    RAW_DIR: Path = _DEFAULT_DATA_DIR / "raw"
    CHUNKS_DIR: Path = _DEFAULT_DATA_DIR / "chunks"
    VECTOR_DIR: Path = _DEFAULT_DATA_DIR / "chroma"    # Step 5 Chroma 持久化
    LOG_DIR: Path = _BASE_DIR / "logs"
    
    CHUNKS_FILENAME: str = "chunks.jsonl"

    @property
    def CHUNKS_JSONL(self) -> Path:
        return self.CHUNKS_DIR / self.CHUNKS_FILENAME

    # ---------- 日志设置 ----------
    LOG_LEVEL:str = "INFO"
    DEBUG:bool =True
    
    # ---------- PostgreSQL（唯一真源） ----------
    PG_HOST: str = "localhost"
    PG_PORT: int = 5432
    PG_DB: str = "wuwa"
    PG_USER: str = "wuwa"
    PG_PASSWORD: str = ""

    # ---------- Neo4j（结构化事实 + 组队关系） ----------
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = ""

    # ---------- Redis（缓存 + Celery broker） ----------
    REDIS_URL: str = "redis://localhost:6379/0"

    # ---------- RustFS / S3（原文 + 立绘） ----------
    S3_ENDPOINT: str = "http://localhost:9000"
    S3_ACCESS_KEY: str = "rustfsadmin"
    S3_SECRET_KEY: str = "rustfsadmin"
    S3_BUCKET_RAW: str = "wuwa-raw"
    S3_BUCKET_IMAGES: str = "wuwa-images"   # 预留：立绘桶，images 表已建、VLM 链路未接入主链

    # ---------- LLM：多模型分工（多 agent）----------
    # aemeath = chat 专用（角色扮演微调，人设由模型自带 Modelfile SYSTEM）
    # qwen3:8b = tool 模型（字典抽取 / 工具调用 / 滚动摘要）
    LLM_MODEL: str = "aemeath"
    TOOL_MODEL: str = "qwen3:8b"
    VLM_MODEL: str = "qwen3-vl:8b"      # 预留：立绘视觉描述（未接入主链）
    LLM_URL: str = "http://localhost:11434"       # ChatOllama
    LLM_API_KEY: str = "wuwa"                   # 不校验，随便填
    LLM_TEMPERATURE: float = 0.3
    MAX_TOKENS: int = 2048
    LLM_NUM_CTX: int = 8192
    # tool 模型参数：抽取要稳定可解析的 JSON，不要文采，故 temperature=0、输出压短。
    # 实测抽取平均 0.2s（think=False）vs 6.36s（think=True），准确率同为 4/8，
    # 所以抽取侧关 thinking 是纯收益。
    TOOL_TEMPERATURE: float = 0.0
    TOOL_MAX_TOKENS: int = 512
    # 滚动摘要（intent.summarize_turns 压缩滑出窗口的旧轮次）复用 TOOL_MODEL：
    # 实测 0.6b 合并多轮会丢角色名，而摘要的价值恰恰在保住名字。
    # 这里只留长度硬帽：超长视为模型跑偏，丢尾部并标记降级。
    SUMMARY_MAX_CHARS: int = 160

    # ---------- 采样 / 防复读（aemeath 是 8B 角色扮演模型，容易陷入整句复读）----------
    # ⚠️ 别再调回 1.3：实测 1.3 会让 aemeath「不敢继续写相似内容」而**提前收尾**——
    #    同一张满级数值表，1.3 下只输出前 4 行就收口（还补一句「其他参数未在该列表中
    #    出现」），问共鸣解放倍率时 7 行里稳定丢 1~3 行；连**没有补料块**的普通轮次也
    #    会丢行。降到 1.15 后同一 prompt 7 行全出，零资料/闲聊两个复读高发场景都没见
    #    复读（重复句占比 0.00，闲聊反而从 252 字缩到 85 字）。last_n=64/512/1024 三档
    #    结果一致，说明**惩罚强度是主因、窗口不是**。
    #    抗复读不靠加码惩罚：运行时另有 LoopGuard（复读即中断 + 截断）兜底。
    LLM_REPEAT_PENALTY: float = 1.15    # >1 惩罚重复 token，1.0=不惩罚
    # 要照抄长表格的轮次（满级数值表 / 突破材料表）再降一档：表里各行彼此高度相似，
    # 最容易被重复惩罚误伤成「只抄前几行」。
    LLM_REPEAT_PENALTY_STRICT: float = 1.05
    LLM_REPEAT_LAST_N: int = 512        # 惩罚回看的 token 窗口；太小压不住长段复读
    LLM_TOP_P: float = 0.9              # 核采样；收窄候选，减少跑偏进人设独白
    LLM_TOP_K: int = 40
    # 关闭思考模式，走 .bind(think=False)（见 llm.py）。
    # ⚠️ 不要改回 prompt 里的 /no_think：实测对 aemeath 无效（仍 38.3s、输出带 'v'
    # 泄漏前缀，并触发 Ollama 500 peg-native format 错误）。bind 方式 1.3s 且干净。
    LLM_NO_THINK: bool = True
    LLM_SEED: int = -1                  # -1 = 随机；调试复现时可固定
    # 运行时复读兜底：命中即中断生成 + 截断尾巴（采样参数压不住时的最后一道闸）
    LLM_LOOP_MAX_REPEAT: int = 2        # 同一句子最多允许出现的次数
    LLM_LOOP_MIN_CHARS: int = 12        # 「长句」门槛：≥ 此长度才做精确重复计数
    # ⚠️ 2026-09-22 加：光有上面的长句规则会**漏网**。实测「清宵配队」的输出把 6 行
    #    一组的目标配队循环了 5 遍，且每行带 `*1`…`*28` 计数后缀（`守岸人 + 尤诺*17`），
    #    ① 每行都短于 12 字被 MIN_CHARS 跳过；② 后缀让每行看起来都唯一，精确判重抓不到。
    #    故补「周期块循环」检测：一组行整体重复 ≥ MIN_CYCLES 遍即判退化（比较前剥掉
    #    行尾计数标记，仅用于序列比较，不做精确计数——否则真实材料表会误杀）。
    LLM_LOOP_MIN_ITEM: int = 4          # 参与判重的单行最短长度（归一化后），滤掉「嗯~」
    LLM_LOOP_PERIOD_MAX: int = 12       # 周期长度上限（行）
    LLM_LOOP_MIN_CYCLES: int = 3        # 同一周期至少重复几遍才判退化

    # ---------- 检索模型（走 CPU，GPU 被 VLM 占满） ----------
    EMBED_MODEL: str = "BAAI/bge-m3"
    EMBED_DEVICE: str = "cpu"
    EMBED_BATCH_SIZE: int = 64
    EMBED_MAX_SEQ_LENGTH: int = 512   # bge-m3 默认 8192，CPU 上必须压到 512
    RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANK_DEVICE: str = "cpu"

    # ---------- 检索参数 ----------
    TOPK_DENSE: int = 30     # Chroma 稠密召回
    TOPK_SPARSE: int = 30    # BM25 稀疏召回
    TOPK_RERANK: int = 6     # 重排后送进 LLM

    # 图谱「队友」槽位最多展示几支队伍。奶辅类角色（守岸人）全 wiki 到处都有她，
    # 实测 53 支 → aemeath 会 47 行原样倒出来（787 字通篇清单）。按「`+` 段数多、
    # `/` 候选少」（越确定）排序后取前 N 支，答案才回到可读。0 = 不限（全量）。
    TEAM_MAX_SHOWN: int = 12

    # ---------- 知识验证 / 重新检索（qwen3:8b verifier agent）----------
    # generate 前用 tool 模型判「检索到的资料是否真能回答问题」；判不匹配则
    # 重检索→按角色刷新重爬→仍不足走千帆联网搜索兜底（见 rag/websearch.py）。
    # verifier 每次问答多一次 8b 调用（实测 0.3~0.9s）；EMPTY_SKIP=True 时
    # 材料全空直接判不足、省这次调用（空材料无需模型也该刷新）。
    VERIFY_ENABLED: bool = True
    VERIFY_EMPTY_SKIP: bool = False
    VERIFY_MAX_RETRY: int = 1         # 验证不通过→重检索的最多次数（防图内死循环）
    REFRESH_WAIT_TIMEOUT: int = 180   # 刷新链（清库+重爬5步）同步等待上限，同自动爬取
    # 百度千帆联网搜索（v2 chat/completions + web_search，API 直连不走本地 SDK）。
    # ⚠️ API key 留空 = 联网兜底整体关闭，链路降级为「不知道」，不报配置错误。
    # 2026-09-22 修：原写法 `os.getenv('BAIDUQIANFAN_API_KEY')` 有两个坑——
    #   ① 它在**类体求值**，读的是进程环境；pydantic 的 env_file 只走自己的 source，
    #      不会把 .env 注入 os.environ → 把 key 写进 .env 完全无效（配了也不生效）。
    #   ② 变量不存在时默认值是 None，而字段类型是 str → pydantic 校验直接抛
    #      ValidationError；ww_logger 第 10 行就 import 时就调 get_settings()
    #      → **整个服务 import 阶段就崩**，与「留空 = 优雅降级」的设计正好相反（已实测）。
    # 改用 AliasChoices：QIANFAN_API_KEY / BAIDUQIANFAN_API_KEY 两名都认，
    # 且 .env 与进程环境都能配（爸爸现有的 BAIDUQIANFAN_API_KEY 环境变量不受影响）。
    QIANFAN_API_KEY: str = Field(
        "", validation_alias=AliasChoices("QIANFAN_API_KEY", "BAIDUQIANFAN_API_KEY")
    )
    QIANFAN_CHAT_URL: str = "https://qianfan.baidubce.com/v2/chat/completions"
    QIANFAN_WEB_MODEL: str = "ernie-4.5-turbo-128k"   # 支持 web_search 的对话模型
    # 实测（2026-09-22 真跑）：联网请求 6.1s（命中搜索缓存）~25.6s（真搜），
    # 多数落在 21~26s。原来 20s 会随机 ReadTimeout → 表现成「联网不可用」。
    QIANFAN_TIMEOUT: float = 45.0

    # ---------- 切块参数 ----------
    MIN_CHARS: int = 60
    MAX_CHARS: int = 500
    OVERLAP_LINES: int = 2      # 强制切分时，下一片重复上一片末尾几行

    # ---------- chromadb名称 ----------
    CHUNK_COLLECTION: str = "wuwa_chunks"

    # ---------- HF本地目录 ----------
    HF_HOME: str = "D:/hf_cache/huggingface"

    # ---------- 重排 ----------
    RERANK_MAX_LENGTH: int = 512     
    RERANK_BATCH_SIZE: int = 8       # 实测 bs=8 比 bs=32 快 35%
    RERANK_THREADS: int = 8          # 实测 8 线程最快，16 反而慢（线程抢资源）
    RERANK_MIN_SCORE: float = 0.1    # logit，超低分=明显不相关，直接丢
    TOPK_RERANK_IN: int = 20         # 送进 reranker 的候选数（不是全部 30 条）

    # ---------- API（Step 9） ----------
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    MAX_HISTORY_TURNS: int = 3      # 带进 prompt 的历史轮数

    @property
    def PG_DSN_LG(self) -> str:
        """LangGraph checkpointer 专用
        不指定就会落到 public
        """
        return f"{self.PG_DSN}?options=-csearch_path%3Dlg"

    @property
    def PG_DSN(self) -> str:
        return (
            f"postgresql://{self.PG_USER}:{self.PG_PASSWORD}"
            f"@{self.PG_HOST}:{self.PG_PORT}/{self.PG_DB}"
        )

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

def ensure_dirs() -> None:
    """建好所有目录。在 CLI 入口调用一次，不放模块级。"""
    s = get_settings()
    for d in (s.DATA_DIR, s.RAW_DIR, s.CHUNKS_DIR, s.VECTOR_DIR, s.LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

