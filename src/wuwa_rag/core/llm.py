"""LLM 封装：双模型分工 + 双 provider。

模型分工（不随 provider 变化）：
- chat 模型：负责最终作答。**默认本地 aemeath**；用户可改用自己的云端大模型。
- qwen3:8b（tool 模型）：字典抽取、工具调用、会话滚动摘要、资料审查。
  tool 模型始终走本地，不跟随用户 provider：结构化任务需要稳定的 JSON 输出，
  且不该把用户的 key 花在内部任务上（省钱，也缩小 key 的泄漏面）。

人设注入的两条相反规则（搞反就丢 chat 效果）：
- 本地 aemeath：人设烧在 Modelfile SYSTEM 里，**调用侧不得传 system**，否则覆盖人设。
- 云端通用模型：不认识爱弥斯，**必须**传 system 注入人设（persona.cloud_system()），
  注入点在 chain（消息层），不在这里——本模块只负责造客户端。

thinking 关闭方式（实测结论，勿改回 prompt 软开关）：
  /no_think 对 aemeath 无效——实测仍耗时 38.3s、输出 3735 字、带 'v' 泄漏前缀，
  并触发 Ollama 500「output does not match the expected peg-native format」。
  改用 .bind(think=False) 后 1.3s、输出干净、无泄漏，且与 astream_events 兼容。
  qwen3:8b 同样关：抽取准确率不因此下降（多样本均 4/8），但耗时从 6.36s 降到 0.20s。
  think=False 是 Ollama 专有参数，云端 OpenAI 兼容 API 不支持，传入会被视为
  未知参数发给服务商（可能 400）。故 _build_openai 绝不 bind think。
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from wuwa_rag.config import get_settings
from wuwa_rag.core import llmstore
from wuwa_rag.ww_logger import get_logger

log = get_logger("llm")

# 云端请求超时（秒）：没有它，服务商挂起会把整条 SSE 流卡死到客户端超时。
_CLOUD_TIMEOUT = 60
_CLOUD_MAX_RETRIES = 1


def _build_ollama(
    model: str, temperature: float, num_predict: int, repeat_penalty: float | None = None,
) -> Runnable:
    """构造 ChatOllama 并 bind(think=False) 固化关思考。

    bind() 返回 _ChatModelBinding（仍有 bind_tools，可与 agent 链式组合）。
    注意：不要写 streaming=True —— ChatOllama 无该字段且 pydantic extra='ignore'，
    会被静默丢弃（误导性死参数）；流式与否只由调 .astream() 还是 .ainvoke() 决定。

    repeat_penalty 留 None 取 settings 默认；照抄长表格的轮次会传更低的 STRICT 档
    （见 config.LLM_REPEAT_PENALTY 的实测说明：惩罚过高会把相似行罚到写不下去）。
    """
    s = get_settings()
    llm = ChatOllama(
        model=model,
        base_url=s.LLM_URL,
        # Ollama 侧不校验该字段，填任意占位串即可
        temperature=temperature,
        num_predict=num_predict,
        num_ctx=s.LLM_NUM_CTX,
        # ---- 防复读：8B 角色扮演模型在「无资料可答」时极易整句循环 ----
        repeat_penalty=s.LLM_REPEAT_PENALTY if repeat_penalty is None else repeat_penalty,
        repeat_last_n=s.LLM_REPEAT_LAST_N,
        top_p=s.LLM_TOP_P,
        top_k=s.LLM_TOP_K,
        seed=None if s.LLM_SEED < 0 else s.LLM_SEED,
    )
    # think=False 只能作为调用级参数（ChatOllama 无 think 字段），bind 固化到实例上
    return llm.bind(think=False) if s.LLM_NO_THINK else llm


@lru_cache(maxsize=4)
def _chat_llm(repeat_penalty: float) -> Runnable:
    """按惩罚档位缓存实例，避免每轮重建。"""
    s = get_settings()
    return _build_ollama(s.LLM_MODEL, s.LLM_TEMPERATURE, s.MAX_TOKENS, repeat_penalty)


_CLOUD_CACHE: dict[tuple, Runnable] = {}   # (base_url, model, key指纹, strict) → 实例


def _build_openai(cfg: dict, strict: bool) -> Runnable:
    """构造云端 OpenAI 兼容客户端（ChatOpenAI）。

    cfg 来自 llmstore.get_runtime（含明文 api_key，生命周期止于返回的实例）。
    构造参数使用 ChatOpenAI 的别名（model/api_key/base_url/timeout）——这些
      别名正确绑定到 model_name/openai_api_key/openai_api_base/request_timeout。
    不要 bind(think=False)：该参数仅适用于 Ollama，发给 OpenAI 兼容服务会被视为未知
      参数（可能 400）。云端模型的思考控制交给服务商默认或模型名（如带 -thinking 后缀）。
    抑制重复应使用 frequency/presence_penalty（OpenAI 系），而非 Ollama 的 repeat_penalty。
    """
    s = get_settings()
    return ChatOpenAI(
        model=cfg["model"],
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        temperature=s.CHAT_CLOUD_TEMPERATURE,
        max_tokens=s.CHAT_CLOUD_MAX_TOKENS,
        frequency_penalty=(s.CHAT_CLOUD_FREQ_PENALTY_STRICT if strict
                           else s.CHAT_CLOUD_FREQ_PENALTY),
        presence_penalty=s.CHAT_CLOUD_PRESENCE_PENALTY,
        timeout=_CLOUD_TIMEOUT,
        max_retries=_CLOUD_MAX_RETRIES,
    )


def _cloud_chat_llm(cfg: dict, strict: bool) -> Runnable:
    """云端客户端缓存：按 (base_url, model, key指纹, strict) 缓存。

    缓存键使用 key 的指纹而非明文——明文不得进入任何长生命周期容器的键位。
    """
    fp = llmstore.fingerprint(cfg["api_key"])
    ck = (cfg["base_url"], cfg["model"], fp, strict)
    hit = _CLOUD_CACHE.get(ck)
    if hit is None:
        hit = _build_openai(cfg, strict)
        _CLOUD_CACHE[ck] = hit
    return hit


def get_chat_llm(strict: bool = False, cloud_cfg: dict | None = None) -> Runnable:
    """chat 模型：负责最终作答。

    - cloud_cfg 非 None → 走用户自定义云端模型（OpenAI 兼容 API）；
    - cloud_cfg=None → 走**本地默认 aemeath**（需求①：默认仍用项目自带 agent）。

    cloud_cfg 由调用方（chain）从 `llmstore.get_runtime(user_id)` 取，含明文 key。
    本函数不读库、不碰 state——明文 key 的生命周期止于返回的客户端实例。

    strict=True：本轮要照抄「满级数值表 / 突破材料表」这种成片高相似度行，用更低重复惩罚。
    """
    if cloud_cfg:
        return _cloud_chat_llm(cloud_cfg, strict)
    s = get_settings()
    return _chat_llm(s.LLM_REPEAT_PENALTY_STRICT if strict else s.LLM_REPEAT_PENALTY)


@lru_cache(maxsize=1)
def get_tool_llm() -> Runnable:
    """tool 模型：qwen3:8b，负责字典抽取、工具调用、会话滚动摘要。

    temperature 取 TOOL_TEMPERATURE（默认 0）——抽取要的是稳定可解析的 JSON，
    不是文采；num_predict 也压小，抽取输出本就短。
    摘要也用它的理由见 intent.summarize_turns（0.6b 合并多轮会丢角色名）。
    """
    s = get_settings()
    return _build_ollama(s.TOOL_MODEL, s.TOOL_TEMPERATURE, s.TOOL_MAX_TOKENS)
