"""Qwen-Audio-3.0-TTS 语音合成（预留接口，默认关闭）。

为什么走 httpx 直连而不用 dashscope SDK
--------------------------------------
`dashscope` 不在依赖里（实测 `import dashscope` → ModuleNotFoundError）。
为"预留一个默认关闭的接口"而引入一个 SDK 不划算，且 SDK 的 SpeechSynthesizer
是同步阻塞调用，放进 FastAPI 还要 to_thread 包一层。直接用 httpx POST 更透明。
（与 `rag/websearch.py` 同一风格：能查证参数就直接调 REST，不依赖厂商 SDK。）

端点与地域（官方「非实时语音合成」2026-09 版，已核实）
----------------------------------------------------
注意：Qwen-Audio-TTS 的端点不是 dashscope.aliyuncs.com，而是带 WorkspaceId 的
maas 域名，且**仅北京地域**可用；API Key 也必须是北京地域的。端点不可与
Qwen-TTS 系列混用（那条是 `/api/v1/services/aigc/multimodal-generation/generation`）。
  POST https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer
  body: {"model","input":{"text","voice","format","sample_rate"}}

三重开关（任一不满足即安全关闭，返回 TtsUnavailable，绝不抛异常打断问答）
--------------------------------------------------------------------
① `TTS_ENABLED=False`（默认）—— 当前阶段仅预留接口，音色方案与计费策略尚未定稿；
② `DASHSCOPE_API_KEY` 为空；
③ `TTS_WORKSPACE_ID` 为空（端点必需，缺了拼不出 URL）；
④ dashscope 相关网络不可达 / 4xx / 5xx —— 返回带原因的错误对象。

响应字段存在不确定性
------------------------------
官方文档只写"非流式模式下响应中包含合成音频的 URL，有效期 24 小时"，
**没有给出确切 JSON 路径**。故 `_extract_audio_url` 做多路径兼容解析
（output.audio.url / output.url / output.audio_url / url / data.url …），
并在全部落空时返回 None → 上层给出可读错误，而不是静默返回空音频。
真实开通后若字段不符，看日志里记录的顶层键名即可一处修正。

朗读稿清洗（`_to_speech_text`）
------------------------------
送进去合成的是**界面上的答案原文**（markdown），直接念会把 `##`、`**`、`|`、
反引号这些版式符号一个个读出来。合成前统一转成纯文本朗读稿；
引用标记在此再剥一次（答案出炉时已剥过，这里是幂等兜底 —— 见 text.strip_ref_marks），
确保任何调用入口送来的文本都不含 `[n]` 类标记。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from ..config import get_settings
from ..text import strip_ref_marks
from ..ww_logger import get_logger
from .emotion import emotion_tag

log = get_logger("tts")

# 合成文本长度帽之外的安全余量：标签本身占字符（`[mischievously]` = 15）
_TAG_RESERVE = 24

# 朗读稿清洗规则（顺序不可调换：先删块级容器，再处理行内符号）。
# 注意：`*` 只处理成对强调与行首列表项两类形态：答案里的 `*` 还可能是乘号
#    （如伤害倍率 `26.92%*3`），全局删除会吃掉语义。
_RE_FENCE = re.compile(r"```.*?```", re.S)          # 围栏代码块
_RE_INLINE_CODE = re.compile(r"`([^`\n]*)`")        # 行内代码：只摘反引号，保留内容
_RE_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")     # 图片：连 alt 一起删
_RE_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")     # 链接：只留链接文字
_RE_TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$", re.M)   # 表格分隔行（`| --- |`）
_RE_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)          # 标题井号
_RE_QUOTE = re.compile(r"^\s{0,3}>\s?", re.M)                 # 引用符
_RE_EMPHASIS = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")   # 成对加粗/强调
_RE_BULLET = re.compile(r"^\s*[-*•]\s+", re.M)                # 行首列表符号
_RE_MULTISPACE = re.compile(r"[ \t]+")
_RE_BLANKS = re.compile(r"\n{3,}")


def _to_speech_text(text: str) -> str:
    """把答案原文转成朗读稿：剥引用标记 + 去 markdown 版式符号（幂等）。

    表格不整行丢弃（那是答案的主要内容），把分隔符换成中文顿号顺成一句话，
    否则 11 列数值表会被念成一串竖线。纯文本输入不会被改动（无符号可去）。
    """
    t = strip_ref_marks(text or "")
    t = _RE_FENCE.sub(" ", t)
    t = _RE_INLINE_CODE.sub(r"\1", t)
    t = _RE_IMAGE.sub("", t)
    t = _RE_LINK.sub(r"\1", t)
    t = _RE_TABLE_SEP.sub("", t)
    t = _RE_HEADING.sub("", t)
    t = _RE_QUOTE.sub("", t)
    t = _RE_EMPHASIS.sub(r"\2", t)
    t = _RE_BULLET.sub("", t)
    # 表格残留分隔符 -> 顿号（先于最后的空白折叠，避免留下连续逗号）
    t = "\n".join(line.replace("|", "、").strip("、 ").strip() if "|" in line else line
                  for line in t.split("\n"))
    t = t.replace("**", "").replace("##", "")
    t = _RE_MULTISPACE.sub(" ", t)
    t = _RE_BLANKS.sub("\n\n", t)
    out = "\n".join(line.strip() for line in t.split("\n"))
    return out.strip()


@dataclass
class TtsResult:
    """合成结果。ok=False 时 error 给可读原因（前端直接展示）。"""

    ok: bool
    url: str = ""          # 音频 URL（24h 有效）
    error: str = ""
    model: str = ""
    voice: str = ""
    emotion: str = ""
    elapsed_ms: int = 0


class TtsUnavailable(Exception):
    """TTS 未开启或配置不全（由调用方转成 200 + ok=False，不当 500 处理）。"""


def tts_ready() -> tuple[bool, str]:
    """TTS 是否可用，返回 (可用, 不可用原因)。供 /tts 端点与前端开关展示。"""
    s = get_settings()
    if not s.TTS_ENABLED:
        return False, "TTS 功能未开启（TTS_ENABLED=false）"
    if not s.DASHSCOPE_API_KEY.strip():
        return False, "未配置 DASHSCOPE_API_KEY（北京地域）"
    if not s.TTS_WORKSPACE_ID.strip():
        return False, "未配置 TTS_WORKSPACE_ID（业务空间 ID）"
    return True, ""


def _endpoint() -> str:
    s = get_settings()
    return s.TTS_URL_TEMPLATE.format(workspace_id=s.TTS_WORKSPACE_ID.strip())


def _extract_audio_url(payload: Any) -> str:
    """从响应里挖音频 URL：官方未给确切路径，故多路径兼容（见模块 docstring）。"""
    if not isinstance(payload, dict):
        return ""
    # 常见候选路径，按可能性排序
    out = payload.get("output")
    candidates: list[Any] = []
    if isinstance(out, dict):
        candidates += [
            (out.get("audio") or {}).get("url") if isinstance(out.get("audio"), dict) else None,
            out.get("url"),
            out.get("audio_url"),
            out.get("audio"),          # 有的实现直接给字符串 URL
        ]
    candidates += [payload.get("url"), payload.get("audio_url")]
    data = payload.get("data")
    if isinstance(data, dict):
        candidates += [data.get("url"), data.get("audio_url")]
    elif isinstance(data, str):
        candidates.append(data)
    for c in candidates:
        if isinstance(c, str) and c.startswith(("http://", "https://")):
            return c
    return ""


async def synthesize(text: str, emotion: str | None = None) -> TtsResult:
    """把文本合成语音，返回音频 URL。

    - `emotion` 为情绪枚举名（见 rag/emotion.py），映射成官方控制类标签插到文本开头；
      控制类标签作用于其后全部文本，所以只插一个。
    - 入参先过 `_to_speech_text` 转朗读稿（剥引用标记 + 去版式符号），随后才做长度截断。
    - 文本超长按 `TTS_MAX_CHARS` 截断（官方 Qwen-TTS 系上限 512 token，中文约 1 字 1 token
      量级，留足余量）；截断优先在句末标点处，避免把话切一半。
    - 标签**只进合成文本**，不进任何用户可见字段。
    """
    ready, why = tts_ready()
    if not ready:
        return TtsResult(ok=False, error=why)

    s = get_settings()
    body_text = _to_speech_text(text)
    if not body_text:
        return TtsResult(ok=False, error="待合成文本为空")

    limit = max(64, s.TTS_MAX_CHARS - _TAG_RESERVE)
    if len(body_text) > limit:
        cut = body_text[:limit]
        # 在最后一个句末标点处收口，避免半句话
        for p in ("。", "！", "？", "~", "；", "\n", "，"):
            idx = cut.rfind(p)
            if idx > limit // 2:
                cut = cut[: idx + 1]
                break
        body_text = cut
        log.info("TTS 文本超长，截断到 %d 字", len(body_text))

    synth_text = f"{emotion_tag(emotion)}{body_text}"
    payload = {
        "model": s.TTS_MODEL,
        "input": {
            "text": synth_text,
            "voice": s.TTS_VOICE,
            "format": s.TTS_FORMAT,
            "sample_rate": s.TTS_SAMPLE_RATE,
        },
    }
    headers = {
        "Authorization": f"Bearer {s.DASHSCOPE_API_KEY.strip()}",
        "Content-Type": "application/json",
    }

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=s.TTS_TIMEOUT) as cli:
            rsp = await cli.post(_endpoint(), json=payload, headers=headers)
    except httpx.HTTPError as exc:
        log.warning("TTS 请求失败: %s", type(exc).__name__)
        return TtsResult(ok=False, error=f"语音服务连接失败：{type(exc).__name__}",
                         model=s.TTS_MODEL, voice=s.TTS_VOICE, emotion=emotion or "")

    elapsed = int((time.perf_counter() - t0) * 1000)
    if rsp.status_code in (401, 403):
        return TtsResult(ok=False, error="语音服务鉴权失败（401/403），请检查北京地域 API Key",
                         elapsed_ms=elapsed)
    if rsp.status_code >= 400:
        # 常见：WorkspaceId 不对(404)、音色与模型版本不匹配(400)
        log.warning("TTS 返回 %s: %.300s", rsp.status_code, rsp.text)
        return TtsResult(ok=False, error=f"语音服务返回 {rsp.status_code}",
                         elapsed_ms=elapsed)

    try:
        data = rsp.json()
    except ValueError:
        log.warning("TTS 响应非 JSON: %.200s", rsp.text)
        return TtsResult(ok=False, error="语音服务响应不是合法 JSON", elapsed_ms=elapsed)

    url = _extract_audio_url(data)
    if not url:
        # 字段路径与文档不符：把顶层键名记进日志，便于开通后一处修正
        log.warning("TTS 响应里没找到音频 URL，顶层键=%s（若字段有变，改 _extract_audio_url）",
                    list(data.keys())[:12])
        return TtsResult(ok=False, error="语音服务未返回音频地址（响应结构可能已变，详见服务端日志）",
                         elapsed_ms=elapsed)

    log.info("TTS 合成成功 model=%s voice=%s emotion=%s 耗时=%dms",
             s.TTS_MODEL, s.TTS_VOICE, emotion or "-", elapsed)
    return TtsResult(ok=True, url=url, model=s.TTS_MODEL, voice=s.TTS_VOICE,
                     emotion=emotion or "", elapsed_ms=elapsed)
