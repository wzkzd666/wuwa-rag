import logging
import sys
from logging.handlers import TimedRotatingFileHandler

from .config import get_settings, ensure_dirs

s=get_settings()

# ── 格式 ──────────────────────────────────
FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED=False

def _make_handler(filename: str, level: int = logging.INFO) -> logging.Handler:
    """每七天清理旧日志"""
    handler = TimedRotatingFileHandler(
        s.LOG_DIR / filename,
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(FMT, datefmt=DATEFMT))
    return handler


def _setup_logging() -> None:
    """
    初始化日志：
    - 控制台：所有 INFO 以上
    - app.log：全局兜底
    - 各模块独立日志文件
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    
    _CONFIGURED=True
    ensure_dirs()
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # 根级别设 DEBUG，各 handler 自己过滤

    # ── 控制台 ──────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, s.LOG_LEVEL.upper(), logging.INFO))
    console.setFormatter(logging.Formatter(FMT, datefmt=DATEFMT))
    root.addHandler(console)

    # ── 全局兜底 ────────────────────────────
    root.addHandler(_make_handler("app.log"))

    # ── 各模块独立文件 ──────────────────────
    # 上传相关
    upload_handler = _make_handler("upload.log")
    upload_logger = logging.getLogger("upload")
    upload_logger.addHandler(upload_handler)
    upload_logger.setLevel(logging.INFO)
    upload_logger.propagate = False  

    # RAG 检索相关
    rag_handler = _make_handler("rag.log")
    rag_logger = logging.getLogger("rag")
    rag_logger.addHandler(rag_handler)
    rag_logger.setLevel(logging.INFO)
    rag_logger.propagate = False

    # 模型调用相关
    ai_handler = _make_handler("ai.log")
    ai_logger = logging.getLogger("ai")
    ai_logger.addHandler(ai_handler)
    ai_logger.setLevel(logging.INFO)
    ai_logger.propagate = False

    # pg数据库相关
    pg_handler = _make_handler("pg.log")
    pg_logger = logging.getLogger("pg")
    pg_logger.addHandler(pg_handler)
    pg_logger.setLevel(logging.INFO)
    pg_logger.propagate = False

    # 图谱数据库相关
    neo4j_handler = _make_handler("neo4j.log")
    neo4j_logger = logging.getLogger("neo4j")
    neo4j_logger.addHandler(neo4j_handler)
    neo4j_logger.setLevel(logging.INFO)
    neo4j_logger.propagate = False

    # 向量数据库相关
    vec_handler = _make_handler("vec.log")
    vec_logger = logging.getLogger("vec")
    vec_logger.addHandler(vec_handler)
    vec_logger.setLevel(logging.INFO)
    vec_logger.propagate = False

    # BM25相关
    bm25_handler = _make_handler("bm25.log")
    bm25_logger = logging.getLogger("bm25")
    bm25_logger.addHandler(bm25_handler)
    bm25_logger.setLevel(logging.INFO)
    bm25_logger.propagate = False

    # BM25相关
    celery_handler = _make_handler("celery.log")
    celery_logger = logging.getLogger("celery")
    celery_logger.addHandler(celery_handler)
    celery_logger.setLevel(logging.INFO)
    celery_logger.propagate = False

    # ── 第三方库降噪 ────────────────────────
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if s.DEBUG else logging.WARNING
    )
    logging.getLogger("alembic").setLevel(logging.WARNING)

    # ── 启动标记 ────────────────────────────
    logging.getLogger(__name__).info("Logging initialized | log_dir=%s", s.LOG_DIR)


def get_logger(name: str) -> logging.Logger:
    _setup_logging()
    return logging.getLogger(name)

