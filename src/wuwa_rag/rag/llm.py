"""LLM 封装（amis OpenAI 兼容服务 / qwen3_8b_amis）。"""
from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from ..config import get_settings


@lru_cache(maxsize=1)
def get_chat_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.LLM_MODEL,            # "amis"
        base_url=s.LLM_URL_V1,        # "http://127.0.0.1:18000/v1"
        api_key=s.LLM_API_KEY,        # amis 服务不校验，随便填
        temperature=s.LLM_TEMPERATURE,
        max_tokens=s.MAX_TOKENS,
        streaming=True,
        extra_body={"enable_thinking": False},  # 保险；服务端已硬编码关思考
    )
