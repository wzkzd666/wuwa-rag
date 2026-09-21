"""LLM 封装"""
from __future__ import annotations

from functools import lru_cache

from langchain_ollama import ChatOllama

from ..config import get_settings


@lru_cache(maxsize=1)
def get_chat_llm() -> ChatOllama:
    s = get_settings()
    return ChatOllama(
        model=s.LLM_MODEL,            # "aemeath"
        base_url=s.LLM_URL,           # Ollama：人设由模型自带(Modelfile SYSTEM)
        api_key=s.LLM_API_KEY,        # Ollama 不校验，随便填
        temperature=s.LLM_TEMPERATURE,
        num_predict=s.MAX_TOKENS,
        num_ctx=8192,
        # ---- 防复读：8B 角色扮演模型在「无资料可答」时极易整句循环 ----
        repeat_penalty=s.LLM_REPEAT_PENALTY,
        repeat_last_n=s.LLM_REPEAT_LAST_N,
        top_p=s.LLM_TOP_P,
        top_k=s.LLM_TOP_K,
        seed=None if s.LLM_SEED < 0 else s.LLM_SEED,
        # 注意：不要加 streaming=True。ChatOllama 没有该字段且 pydantic
        # extra='ignore'，写了会被静默丢弃（误导性死参数）；流式与否只由
        # 调用 .astream() 还是 .ainvoke() 决定。
    )


def no_think_marker() -> str:
    """Qwen3 思考模式软开关。

    langchain_ollama 1.1.0 的 ChatOllama 没有 think / reasoning_effort 字段，
    无法用参数关思考；Qwen3 官方约定是在 prompt 里写 /no_think。
    原先由 serve_amis.py 服务端强制关（CoT 会降低角色扮演质量），该服务弃用后
    改由这里接手。返回空串表示保持模型默认行为。
    """
    return "/no_think" if get_settings().LLM_NO_THINK else ""
