from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

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
    S3_BUCKET_IMAGES: str = "wuwa-images"

    # ---------- LLM ----------
    LLM_MODEL: str = "qwen3:8b"
    VLM_MODEL: str = "qwen3-vl:8b"
    LLM_URL: str = "http://localhost:11434"       # ChatOllama 用（不带 /v1）
    LLM_URL_V1: str = "http://localhost:11434/v1" # ChatOpenAI 用
    LLM_API_KEY: str = "ollama"                   # Ollama 不校验，随便填
    LLM_TEMPERATURE: float = 0.3
    MAX_TOKENS: int = 2048

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

    # ---------- 切块参数 ----------
    MIN_CHARS: int = 60
    MAX_CHARS: int = 500
    OVERLAP_LINES: int = 2      # 强制切分时，下一片重复上一片末尾几行

    # ---------- chromadb名称 ----------
    CHUNK_COLLECTION: str = "wuwa_chunks"

    # ---------- HF本地目录 ----------
    HF_HOME: str = "D:/hf_cache/huggingface"

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

