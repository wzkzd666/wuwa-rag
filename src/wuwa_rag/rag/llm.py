"""LLM 封装：双模型分工。

- aemeath（chat 专用）：角色扮演微调模型，只负责最终作答。人设由模型自带
  (Modelfile SYSTEM)，所以调用侧不得传 system 参数，否则人设被覆盖。
- qwen3:8b（tool 模型）：负责字典抽取、工具调用等结构化任务。

thinking 关闭方式（实测结论，勿改回 prompt 软开关）：
  /no_think 对 aemeath 无效——实测仍耗时 38.3s、输出 3735 字、带 'v' 泄漏前缀，
  并触发 Ollama 500「output does not match the expected peg-native format」。
  改用 .bind(think=False) 后 1.3s、输出干净、无泄漏，且与 astream_events 兼容。
  qwen3:8b 同样关：抽取准确率不因此下降（多样本均 4/8），但耗时从 6.36s 降到 0.20s。
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama

from ..config import get_settings


def _build_ollama(model: str, temperature: float, num_predict: int) -> Runnable:
    """构造 ChatOllama 并 bind(think=False) 固化关思考。

    bind() 返回 _ChatModelBinding（仍有 bind_tools，可与 agent 链式组合）。
    注意：不要写 streaming=True —— ChatOllama 无该字段且 pydantic extra='ignore'，
    会被静默丢弃（误导性死参数）；流式与否只由调 .astream() 还是 .ainvoke() 决定。
    """
    s = get_settings()
    llm = ChatOllama(
        model=model,
        base_url=s.LLM_URL,
        api_key=s.LLM_API_KEY,        # Ollama 不校验，随便填
        temperature=temperature,
        num_predict=num_predict,
        num_ctx=s.LLM_NUM_CTX,
        # ---- 防复读：8B 角色扮演模型在「无资料可答」时极易整句循环 ----
        repeat_penalty=s.LLM_REPEAT_PENALTY,
        repeat_last_n=s.LLM_REPEAT_LAST_N,
        top_p=s.LLM_TOP_P,
        top_k=s.LLM_TOP_K,
        seed=None if s.LLM_SEED < 0 else s.LLM_SEED,
    )
    # think=False 只能作为调用级参数（ChatOllama 无 think 字段），bind 固化到实例上
    return llm.bind(think=False) if s.LLM_NO_THINK else llm


@lru_cache(maxsize=1)
def get_chat_llm() -> Runnable:
    """chat 专用：aemeath，负责最终作答（人设自带）。"""
    s = get_settings()
    return _build_ollama(s.LLM_MODEL, s.LLM_TEMPERATURE, s.MAX_TOKENS)


@lru_cache(maxsize=1)
def get_tool_llm() -> Runnable:
    """tool 模型：qwen3:8b，负责字典抽取、工具调用、会话滚动摘要。

    temperature 取 TOOL_TEMPERATURE（默认 0）——抽取要的是稳定可解析的 JSON，
    不是文采；num_predict 也压小，抽取输出本就短。
    摘要也用它的理由见 intent.summarize_turns（0.6b 合并多轮会丢角色名）。
    """
    s = get_settings()
    return _build_ollama(s.TOOL_MODEL, s.TOOL_TEMPERATURE, s.TOOL_MAX_TOKENS)
