import logging
import os
import sys
import time

from concurrent_log_handler import ConcurrentTimedRotatingFileHandler

from .config import ensure_dirs, get_settings

s = get_settings()

# ── 格式 ──────────────────────────────────
# role 列逐行标注进程角色（api / worker / main）。**不再按 role 拆文件**：同机多个
# 进程共写一个 rag.log，靠这一列区分是谁写的；跨进程轮转安全由 concurrent-log-handler
# 的文件锁保证。文件只按业务「大类」（logger 名：rag/app/celery…）分开。
FMT = "%(asctime)s | %(levelname)-8s | %(role)-6s | %(name)s | %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False

# 业务模块独立日志文件（即「大类」，一个名字一个文件，**不按进程角色再拆**），
# propagate=False 避免重复落 app.log。是 api 还是 worker 写的，看行内 role 列。
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


def _process_tag() -> str:
    """当前进程角色，仅用于**每行日志的 role 列标注**（可用 WUWA_LOG_ROLE 覆盖）。

    历史背景（曾导致 rag 日志整天全丢的 bug）：FastAPI 与 Celery worker 是同机两个
    独立进程，都会走 _setup_logging()。早先两者共用 logs/rag.log，midnight 的
    TimedRotatingFileHandler.doRollover 会「先 close 自己的 stream、再 os.rename」，
    而另一个进程此刻仍持有该文件句柄 —— Windows 直接抛 PermissionError [WinError 32]；
    更要命的是此时 stream 已关，shouldRollover() 会逐条重试这个失败的 rename，于是该
    logger 当天剩余记录一条都写不进去（现象：rag.log 停更、没有当天轮转产物，而只有
    API 进程在写的 app.log 看起来一切正常）。

    当时的临时修法是「按 role 拆文件」，现在**已废止**——那会产出一堆
    rag.api.log / rag.worker.log / rag.main.log（用户明确要求不要）。真正根治靠
    ConcurrentTimedRotatingFileHandler 的跨进程文件锁 + 基于 mtime 的智能轮转判断，
    多个进程（含同一 role 的父子进程）共写一个文件也不会互踩。所以本函数现在只干
    一件事：给每行日志打上 role 标签。
    """
    role = os.environ.get("WUWA_LOG_ROLE")
    if role:
        return role.strip().lower()
    # 反斜杠要归一化：`python -m wuwa_rag.api.server` 启动时 sys.argv[0] 是模块
    # **文件路径**（实测 ...\src\wuwa_rag\api\server.py），不是模块名 —— 不归一化
    # 就匹配不上，会掉进 main 兜底桶（2026-09-22 首轮线上实测正是如此：
    # API 进程被记成 role=main）。start.ps1 也会显式注入 WUWA_LOG_ROLE，
    # 这里的推断只作兜底。
    argv = " ".join(sys.argv).lower().replace("\\", "/")
    if "celery" in argv:
        return "worker"
    if "uvicorn" in argv or "wuwa_rag.api" in argv or "wuwa_rag/api" in argv:
        return "api"
    return "main"


def _formatter() -> logging.Formatter:
    """带 role 列的行格式。role 是本进程常量，用 defaults 注入 —— 第三方库
    （uvicorn/celery 等）自己造的 LogRecord 上没有 role 属性，defaults 会自动补齐，
    不会 KeyError。"""
    return logging.Formatter(FMT, datefmt=DATEFMT, defaults={"role": _process_tag()})


class _SafeTimedRotatingFileHandler(ConcurrentTimedRotatingFileHandler):
    """跨进程安全轮转 + 「轮转失败也不整天丢日志」兜底。

    基类用**跨进程文件锁**协调轮转，从根上解决「两个进程同时持句柄 → os.rename 抛
    PermissionError [WinError 32]」的问题（2026-09-22 修掉的真实 bug；FastAPI 与
    Celery、以及同一 role 的父子进程都属于这种情形）。这里再叠一层兜底：万一锁或
    权限仍然出问题，doRollover 抛 OSError 时重开 stream 并把 rolloverAt 推到次日，
    只留一条 warning —— 不让该 logger 当天所有日志一条都写不进去。
    """

    def doRollover(self) -> None:
        try:
            super().doRollover()
        except OSError as exc:            # WinError 32 / 5 都归 OSError
            if self.stream is None:
                self.stream = self._open()
            self.rolloverAt = self.computeRollover(int(time.time()))
            logging.getLogger(__name__).warning(
                "日志轮转失败，继续写原文件（本次不切分）| file=%s | %s",
                self.baseFilename, exc)


def _make_handler(filename: str, level: int = logging.INFO) -> logging.Handler:
    """一个大类一个文件（如 rag.log），按 midnight 切分、保留 7 天。

    文件名**不带 role**：api / worker / main 共写同一个文件，谁写的看行内 role 列。
    跨进程安全轮转由 `ConcurrentTimedRotatingFileHandler` 的文件锁保证。
    """
    handler = _SafeTimedRotatingFileHandler(
        s.LOG_DIR / filename,
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(_formatter())
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
    console.setFormatter(_formatter())
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
    # role 已在 FMT 的 role 列里，message 不再重复写一次。
    logging.getLogger(__name__).info(
        "Logging initialized | log_dir=%s level=%s", s.LOG_DIR, s.LOG_LEVEL)


def get_logger(name: str) -> logging.Logger:
    _setup_logging()
    return logging.getLogger(name)
