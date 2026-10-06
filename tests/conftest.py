"""pytest 公共 fixture 与离线约定。

**全量离线**：所有用例不得依赖 PG / Neo4j / Redis / Chroma / Ollama / RustFS，
也不得联网。跑测试的前提只有 `.venv` 本身。

为什么这条约定是硬的
------------------
项目的真实故障模式是「**功能没坏，只是没被验证**」——历史上出现过
① 角色名识别失败导致问答一路升级到联网兜底（`characters` 恒空，verify 的
   按角色重爬分支永不触发）；
② 画像同类事实无限叠加（连说三个昵称，注入串自相矛盾）。
这两个 bug 都能在**纯规则层**被断言拦住，不需要任何外部服务。反过来，
一旦测试依赖 PG，`scripts/start.ps1` 没跑或容器停了，测试就会「假失败」，
久了没人信，等于没有。

⚠️ 断言必须验 characters，不能只验 intent（血泪教训）
------------------------------------------------
单字名 bug 期间，`intent_node('心的声骸怎么配')` 返回 intent='hybrid'、chars=[]。
只看 intent 断言全绿，功能其实已经废了——空 chars 反而更容易满足「未指名」判据，
intent 照样合理。所以每条路由用例都同时断言 intent 与 characters。
"""
from __future__ import annotations

import logging

import pytest

from wuwa_rag.dialog import graph as graph_mod

# 测试期静音日志：jieba 首次加载、限流 warning 都会刷屏，盖住真正的失败信息。
logging.disable(logging.CRITICAL)

# 角色名册：取自真实库的形态（含单字名「心」「椿」、多字名、带连字符的形态变体）。
# ⚠️ 必须含单字名——那正是历史上被 len(n) >= 2 一刀切滤掉的那类。
ROSTER = [
    "心", "椿", "鉴心", "卡卡罗", "守岸人", "秧秧", "秧秧·玄翎",
    "今汐", "长离", "散华", "吟霖", "漂泊者-男-衍射",
]


@pytest.fixture
def roster() -> list[str]:
    return list(ROSTER)


@pytest.fixture
def offline_intent(monkeypatch, roster):
    """intent_node 的离线替身环境：stub 掉 Neo4j 名册、LLM 改写与主题分类。

    三个 stub 的理由：
      - `_known_characters()` 查 Neo4j（带 60s TTL 缓存）→ 直接返回固定名册；
      - `rewrite_query` 仅在**有历史/摘要**时才调 LLM，空历史会短路返回原句。
        这里仍一并 stub，是为了让「带历史」的用例也不触网；
      - `classify_topic` 会调 qwen3:8b → **stub 成一律返回 'game'**（最坏情形）。

    把主题分类器 stub 成 game 是刻意的：闲聊分流若只在 LLM 帮忙时才成立，
    那就不叫规则兜底。stub 成 game 之后仍判 chitchat 的用例，才证明
    `is_identity` / `is_self_intro` 这两道硬信号真的独立生效。

    ⚠️ monkeypatch 是**函数级** fixture，所以本 fixture 也必须是函数级。
    会话级 fixture 拿不到 monkeypatch（pytest 不提供 session 版），
    写成 session 会在收集期直接报 fixture 未找到。

    返回一个可 await 的函数：(question, history=None, summary=None) -> dict
    """

    async def _fake_known() -> list[str]:
        return roster

    async def _fake_rewrite(question, history, *, known=None, summary=""):
        return question

    async def _fake_topic(question) -> str:
        return "game"

    monkeypatch.setattr(graph_mod, "_known_characters", _fake_known)
    monkeypatch.setattr(graph_mod, "rewrite_query", _fake_rewrite)
    monkeypatch.setattr(graph_mod, "classify_topic", _fake_topic)

    async def _ask(question: str, history=None, summary: str = "") -> dict:
        state = {
            "question": question,
            "history": history or [],
            "context_summary": summary,
        }
        return await graph_mod.intent_node(state)

    return _ask
