"""LLM 封装（Ollama / qwen3:8b）。"""
from __future__ import annotations

from functools import lru_cache

from langchain_ollama import ChatOllama

from ..config import get_settings


@lru_cache(maxsize=1)
def get_chat_llm() -> ChatOllama:
    """ChatOllama 而不是 ChatOpenAI：/v1 端点不认 num_ctx，
    extra_body 会被静默忽略，上下文会被 Ollama 默认的 2048 截断。"""
    s = get_settings()
    return ChatOllama(
        model=s.LLM_MODEL,
        base_url=s.LLM_URL,          # 不带 /v1
        temperature=s.LLM_TEMPERATURE,
        num_predict=s.MAX_TOKENS,
        num_ctx=8192,                # 关键：默认 2048 放不下 RAG 上下文
    )
