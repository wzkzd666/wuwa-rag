import logging
import sys
from logging.handlers import TimedRotatingFileHandler

from .config import ensure_dirs, get_settings

s = get_settings()

# ── 格式 ──────────────────────────────────
FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False

# 业务模块独立日志文件：一个名字一个文件，propagate=False 避免重复落 app.log。
# 只保留真实有写入的 logger（曾经的 ai / pg 两个从未被引用，已移除）。
_MODULE_LOGS = ("upload", "rag", "neo4j", "vec", "bm25", "celery")

# 第三方库降噪：这些库在 root=INFO 下仍会刷大量 INFO/DEBUG（模型加载进度条、
# bolt 握手、HF 缓存探测），对排查业务问题无价值。
_NOISY = (
    "httpx", "httpcore", "urllib3", "asyncio", "torch", "filelock", "PIL",
    "sentence_transformers", "transformers", "huggingface_hub",
    "neo4j.bolt", "neo4j.graph", "neo4j.pool",   # 只压驱动子 logger，业务用的 "neo4j" 不动
    "chromadb", "sqlalchemy.engine", "alembic", "openai", "multipart",
)


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
    """初始化日志：控制台 + app.log 兜底 + 各模块独立文件 + 第三方降噪。

    root 用 INFO 而不是 DEBUG：root=DEBUG 时未显式设级别的第三方库会把 DEBUG
    全量灌进控制台与 app.log（实测启动一次刷上百行模型加载日志），既淹没有效
    信息又拖 IO。需要深挖时改 LOG_LEVEL=DEBUG 或单独调某个库。
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    _CONFIGURED = True
    ensure_dirs()
    # root 放到 DEBUG 只为放行；实际过滤在各级 handler。未显式设级别的第三方库
    # 会生成大量 DEBUG 记录，全部由下面 _NOISY 的 WARNING 压制拦截。
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # DEBUG=True（.env）时控制台/app.log 放行 debug；模块文件恒为 INFO
    active = logging.DEBUG if s.DEBUG else getattr(logging, s.LOG_LEVEL.upper(), logging.INFO)

    # ── 控制台 ──────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(active)
    console.setFormatter(logging.Formatter(FMT, datefmt=DATEFMT))
    root.addHandler(console)

    # ── 全局兜底 ────────────────────────────
    root.addHandler(_make_handler("app.log", active))

    # ── 各模块独立文件 ──────────────────────
    for name in _MODULE_LOGS:
        lg = logging.getLogger(name)
        lg.addHandler(_make_handler(f"{name}.log"))
        lg.setLevel(active)
        lg.propagate = False          # 不重复写进 app.log

    # ── 第三方库降噪 ────────────────────────
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)

    # ── 启动标记 ────────────────────────────
    logging.getLogger(__name__).info(
        "Logging initialized | log_dir=%s level=%s", s.LOG_DIR, s.LOG_LEVEL)


def get_logger(name: str) -> logging.Logger:
    _setup_logging()
    return logging.getLogger(name)
