"""配置默认值与 prompt 模块拆分。

① 配置默认值
-----------
2026-10-06 收紧了三个默认值，都是「clone 即用」场景下的坑：
  · `DEBUG=True` → `False`：生产环境不该默认开调试；
  · `HF_HOME="D:/hf_cache/huggingface"` → `~/.cache/huggingface`：硬编码 D 盘，
    换台机器直接失效（本机看到的 D 盘路径来自**系统环境变量**，不是代码默认值）；
  · `CLOUD_ALLOW_PRIVATE_NET=True` → `False`：这是 SSRF 防护开关，
    默认放开等于给所有 clone 的人留了个口子。开发者自用再在 .env 里打开。

⚠️ 必须断言 `model_fields[...].default`，**不能**断言 `get_settings()` 的实例值
--------------------------------------------------------------------------
`get_settings()` 会读 .env，而 .env 是本地文件（gitignore、不入库）。
断言实例值会让「有 .env 的机器绿、CI 上红」，或反过来——两头都不可信。
`model_fields[...].default` 是代码里写死的默认值，才是这条测试要守的东西。

② prompt 模块拆分
----------------
`graph.py` 原本 1573 行混了八种职责，把纯字符串构建拆到了 `dialog/prompt.py`。
拆分最大的风险不是「没拆干净」，而是**再导出断链**：`api/app.py` 一直
`from wuwa_rag.dialog.graph import doc_sources`，拆分后 graph 必须把它再导出，
否则 app 导入期就炸。所以下面专门有一条断言「graph.doc_sources is prompt.doc_sources」
——验的是同一个对象，不只是同名。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from wuwa_rag.config import Settings
from wuwa_rag.dialog import graph as graph_mod
from wuwa_rag.dialog import prompt as prompt_mod


def _default(name: str):
    """取代码里写死的字段默认值（不读 .env）。"""
    return Settings.model_fields[name].default


# ---------- ① 配置默认值 ----------

def test_DEBUG_默认关闭() -> None:
    assert _default("DEBUG") is False


def test_HF_HOME_不硬编码盘符且位于用户目录() -> None:
    """断言「在用户目录下」而非具体字符串——它由 Path.home() 在类定义时求值，
    不同机器/不同用户名下值不同，写死字符串会让测试只在作者机器上绿。
    """
    got = Path(_default("HF_HOME"))
    assert got == Path.home() / ".cache" / "huggingface"
    # 回归钉桩：绝不能再退回硬编码 D 盘那个路径
    assert got != Path("D:/hf_cache/huggingface")
    assert "hf_cache" not in str(got), f"HF_HOME 又退回硬编码路径了：{got}"


def test_SSRF_防护默认收紧() -> None:
    """CLOUD_ALLOW_PRIVATE_NET 默认必须是 False——放开等于给所有 clone 的人留 SSRF 口子。"""
    assert _default("CLOUD_ALLOW_PRIVATE_NET") is False


# ---------- 限流配置（档位数字被 test_ratelimit 用作断言依据）----------

@pytest.mark.parametrize(
    "field,expected",
    [
        ("RATE_LIMIT_ENABLED", True),
        ("RATE_LIMIT_STORAGE", ""),            # 留空 = 进程内存，不强制依赖 Redis
        ("RATE_LIMIT_AUTH", "10/minute"),
        ("RATE_LIMIT_ASK", "20/minute"),
        ("RATE_LIMIT_TTS", "10/minute"),
        ("RATE_LIMIT_OUTBOUND", "20/hour"),    # SSRF 面，按小时收得最紧
        ("RATE_LIMIT_DEFAULT", "200/minute"),
    ],
)
def test_限流配置默认值(field: str, expected) -> None:
    assert _default(field) == expected


# ---------- ② prompt 模块拆分 ----------

@pytest.mark.parametrize("name", ["SYSTEM_PROMPT", "build_context", "build_prompt", "doc_sources"])
def test_prompt_模块导出齐全(name: str) -> None:
    assert hasattr(prompt_mod, name), f"prompt.py 缺 {name}"


def test_graph_再导出_doc_sources_是同一对象() -> None:
    """api/app.py 依赖 `from ...graph import doc_sources`，拆分后必须再导出且是同一对象。

    用 `is` 而非「都能调通」：若哪天变成两份实现，同名同签名也能各自跑，
    但行为迟早分叉——那正是项目文档反复警告的「双判据必然不同步」。
    """
    assert graph_mod.doc_sources is prompt_mod.doc_sources


def test_graph_不再残留已迁出的私有实现() -> None:
    """旧私有名必须真的删掉，不能「新模块有了、旧函数还留着」——留着就是死代码 + 分叉源。

    历史上真出过这个状态：prompt.py 建好了、graph.py 也导入了，但旧 `_build_prompt`
    还在且仍引用已删除的 `_SYSTEM`，主链路一跑就 NameError。
    """
    for stale in ("_build_prompt", "_build_context", "_lock_focus", "_SYSTEM"):
        assert not hasattr(graph_mod, stale), f"graph.py 仍残留已迁出的 {stale}"


def test_SYSTEM_PROMPT_含云端不需要的本地约束() -> None:
    """SYSTEM_PROMPT 是给**本地 aemeath** 的（术语对照 + 作答约束）。

    云端模型走另一条路径不带它。这里只钉一个不变量：它非空且是字符串——
    措辞本身经多轮实测校准（约束与理由见 `dialog/prompt.py` 的模块注释），
    不适合在单测里逐字断言。
    """
    assert isinstance(prompt_mod.SYSTEM_PROMPT, str)
    assert prompt_mod.SYSTEM_PROMPT.strip()
