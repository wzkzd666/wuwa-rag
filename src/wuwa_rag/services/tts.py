"""Qwen-Audio-TTS 语音合成（qwen-audio-3.1-tts-flash）。

为什么走 httpx 直连而不用 dashscope SDK
--------------------------------------
`dashscope` 不在依赖里（实测 `import dashscope` → ModuleNotFoundError）。
为"一个可选的语音接口"而引入一个 SDK 不划算，且 SDK 的 SpeechSynthesizer
是同步阻塞调用，放进 FastAPI 还要 to_thread 包一层。直接用 httpx POST 更透明。
（与 `rag/websearch.py` 同一风格：能查证参数就直接调 REST，不依赖厂商 SDK。）

端点与地域（官方「非实时语音合成」2026-09 版，已核实）
----------------------------------------------------
注意：Qwen-Audio-TTS 的端点不是 dashscope.aliyuncs.com，而是带 WorkspaceId 的
maas 域名，且**仅北京地域**可用；API Key 也必须是北京地域的。端点不可与
Qwen-TTS 系列混用（那条是 `/api/v1/services/aigc/multimodal-generation/generation`）。
  POST https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer
  body: {"model","input":{"text","voice","format","sample_rate","instruction"}}

模型选型：qwen-audio-3.1-tts-flash
----------------------------------
2026-09-19 发布的 Flash 版，面向实时交互，相对 3.0 多两项对角色扮演直接有用的能力：
  · **指令控制**（`instruction`，≤100 字符）——自然语言控制音调/语速/情感/音色特点，
    系统音色与复刻音色均可传任意指令；
  · **细粒度情感与富语言标签**——`[excited]`/`[serious]` 等控制类标签 +
    `[giggles]`/`[laughing]` 等拟声类标签（见 rag/emotion.py）。
⚠️ 音色与模型**强绑定**：3.1 的音色一律带 `_v3.1` 后缀，填 3.0 的音色名（如
`longanhuan_v3.6`）会返回 `InvalidParameter`。换模型务必同步换音色。

凭据来源（开源分发：用户自持优先）
----------------------------------
密钥**不由部署者统一垫付**——每个用户在设置页填自己的阿里云百炼凭据
（API Key + 业务空间 ID），密文入库（见 `rag/llmstore.py` 的 DEK/KEK 两段式加密）。
解析顺序见 `resolve()`：**用户自持凭据 > 全局 `.env` 兜底**；`.env` 里那几项只留给
自部署者统一配一份的场景，开源分发时留空即可。

任一不满足即安全关闭（返回 TtsUnavailable 语义的 TtsResult，绝不抛异常打断问答）
--------------------------------------------------------------------------------
① `TTS_ENABLED=False`；
② 该用户没配、且全局 `DASHSCOPE_API_KEY` 也为空；
③ 生效配置的 `workspace_id` 为空（端点必需，缺了拼不出 URL）；
④ 配了但服务进程还没解锁（DEK 不在内存）→ 提示「重新登录即可自动恢复」；
⑤ dashscope 相关网络不可达 / 4xx / 5xx —— 返回带原因的错误对象。

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

from wuwa_rag.config import get_settings
from wuwa_rag.core import llmstore
from wuwa_rag.services.emotion import emotion_tag
from wuwa_rag.text import strip_ref_marks
from wuwa_rag.ww_logger import get_logger

log = get_logger("tts")

# 合成文本长度帽之外的安全余量：标签本身占字符（`[mischievously]` = 15）
_TAG_RESERVE = 24

# 指令控制（instruction）长度上限：官方口径 100 字符，且**汉字按 2 字符计**
# （见「非实时语音合成 · 指令控制」2026-09 版）。超长会被服务端拒绝整单请求。
_INSTRUCTION_MAX_CHARS = 100

# 音色 id → 中文展示名（含官方标注的声线特质）。
# 用途：设置页要显示「当前音色」，直接摆 `longanlingxi_v3.1` 对用户毫无意义。
# 只收常用候选，未收录的 id 原样回落显示（`voice_label`），因此换任何音色都不会报错。
VOICE_LABELS: dict[str, str] = {
    "longanlingxi_v3.1": "龙安灵希 · 可爱甜美（社交陪伴）",
    "longhua_v3.1":      "龙华 · 元气甜美（社交陪伴）",
    "longhuohuo_v3.1":   "龙火火 · 顽皮少年（角色音）",
    "qiaoxiaojiao_v3.1": "乔小娇 · 俏丽可爱",
    "xiaxiaochen_v3.1":  "夏小晨 · 元气明亮",
    "xuxiaoqiao_v3.1":   "徐小俏 · 自然俏皮",
    "yuxiaoyun_v3.1":    "于小云 · 元气亲切",
    "anxiaolan_v3.1":    "安小岚 · 清甜纯净",
    "baiqinglan_v3.1":   "白清岚 · 明亮清纯",
    "anyuqing_v3.1":     "安语晴 · 甜妹",
    "longanhuan_v3.1":   "龙安欢（多语种）",
    "longanfengyue_v3.1": "龙安风悦（多语种）",
}

# 模型 id → 展示名（设置页展示用）
MODEL_LABEL = "Qwen-Audio-3.1-TTS-Flash"


def voice_label(voice: str) -> str:
    """音色 id → 中文展示名；未收录的 id 原样返回（不报错、不隐藏）。"""
    v = (voice or "").strip()
    return VOICE_LABELS.get(v, v)

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


def _global_ready() -> tuple[bool, str]:
    """全局 `.env` 兜底配置是否可用，返回 (可用, 不可用原因)。

    这是**自部署者统一给所有用户配一份**的兜底通道；正常开源分发场景下应当留空，
    由每个用户自己填（见 `resolve`）。原因文案会直接展示给用户，
    故写成「缺什么 + 去哪儿补」的可操作表述。

    注意：这里**不管** `TTS_ENABLED`（那是全功能总闸，在 `resolve` 里先行判定），
    只管「兜底凭据本身齐不齐」——两件事混在一个函数里会让「为什么不可用」说不清。
    """
    s = get_settings()
    if not s.DASHSCOPE_API_KEY.strip():
        return False, "缺少语音服务密钥（DASHSCOPE_API_KEY，需北京地域）"
    if not s.TTS_WORKSPACE_ID.strip():
        return False, "缺少业务空间 ID（TTS_WORKSPACE_ID），可在百炼控制台「业务空间」中获取"
    return True, ""


# 用户未填的字段回落到代码默认值时的取值来源标记
_SRC_USER = "user"
_SRC_GLOBAL = "global"


async def resolve(user_id: int | None = None) -> tuple[dict | None, str]:
    """解析实际生效的语音配置。返回 (runtime, 不可用原因)。

    runtime 形如 ``{api_key, workspace_id, model, voice, instruction, source}``，
    其中 `api_key` 是**明文**（只用于拼请求头，不落日志、不回前端）。
    **优先级：用户自持凭据 > 全局 `.env` 兜底** —— 开源分发下部署者不该替用户
    垫额度，所以用户级永远优先；自部署者想统一配一份时再填 `.env`。

    不可用原因分得很细，因为对用户来说「没配」和「配了但没解锁」要做的事完全不同：
      · 没配 → 去设置页填
      · 配了但未解锁 → 重新登录即可自动恢复
      · 配了、解锁了但解不开 → 密文坏了，需重填

    `TTS_ENABLED=False` 是**全功能总闸**（先于凭据判定）：部署者关掉后，
    用户自持的凭据也不生效——否则「关了开关却还能用」就是个说不清的 bug。
    """
    s = get_settings()
    if not s.TTS_ENABLED:
        return None, "语音朗读功能未启用（部署者在服务端关闭了该功能）"
    if user_id is not None:
        rt = await llmstore.get_tts_runtime(user_id)
        if rt:
            return _fill(rt, _SRC_USER), ""
        cfg = await llmstore.get_tts_masked(user_id)
        if cfg.get("configured"):
            if not cfg.get("unlocked"):
                return None, "语音配置已保存，但服务进程尚未解锁（重新登录即可自动恢复）"
            return None, "语音密钥无法解密，请在设置页重新填写"

    ok, why = _global_ready()
    if not ok:
        # 用户没配、全局也没配 → 这才是最常见的「未配置」态
        if user_id is not None:
            return None, "尚未配置语音服务，请在设置页填入阿里云百炼的 API Key 与业务空间 ID"
        return None, why
    return _fill({
        "api_key": s.DASHSCOPE_API_KEY.strip(),
        "workspace_id": s.TTS_WORKSPACE_ID.strip(),
        "model": s.TTS_MODEL,
        "voice": s.TTS_VOICE,
        "instruction": s.TTS_INSTRUCTION,
    }, _SRC_GLOBAL), ""


def _fill(rt: dict, source: str) -> dict:
    """给用户配置里留空的字段补上代码默认值（用户只想改 key + 空间 ID 时少填几项）。

    ⚠️ 三项都用 `or` 而非 `is not None`：**空串一律视为「用默认值」**，与
    `model` / `voice` 保持一致。若 instruction 单独走「空串=显式清空」，用户就会遇到
    「清空后声音没变化（其实是回落到默认）」这种说不清的状态；统一成
    「留空=用默认」，前端用 placeholder 提示默认值即可，语义只有一条。
    """
    s = get_settings()
    return {
        "api_key": rt.get("api_key") or "",
        "workspace_id": rt.get("workspace_id") or "",
        "model": rt.get("model") or s.TTS_MODEL,
        "voice": rt.get("voice") or s.TTS_VOICE,
        "instruction": rt.get("instruction") or s.TTS_INSTRUCTION,
        "source": source,
    }


async def available(user_id: int | None = None) -> bool:
    """该用户当前能否做语音合成。供问答链决定「要不要花一次情绪判定调用」。

    ⚠️ 内部吞掉异常并返回 False：TTS 是**可选增强**，判定它可用与否的路上若出问题
    （例如连接池抖动），也绝不能把整条问答链带崩——那等于语音功能反过来把主功能弄挂。
    失败只记一条 warning，后果就是这一轮不判情绪、不朗读。
    """
    try:
        rt, _ = await resolve(user_id)
    except Exception as exc:  # noqa: BLE001 —— 见上：可选增强不得反噬主链路
        log.warning("TTS 可用性判定失败（按不可用处理）：%s", type(exc).__name__)
        return False
    return rt is not None


def _clip_instruction(text: str) -> str:
    """按官方口径截断指令文本：上限 100 字符，**汉字按 2 字符计**。

    超长会被服务端整单拒绝（不是忽略指令），所以宁可本地先截断也不冒险发送。
    截断点取「已用字符数」累计，避免把一个汉字劈成半个。
    """
    budget = 0
    out: list[str] = []
    for ch in (text or "").strip():
        cost = 2 if "\u4e00" <= ch <= "\u9fff" else 1
        if budget + cost > _INSTRUCTION_MAX_CHARS:
            log.warning("TTS 指令超长（>%d 字符口径），已截断：%.40s…",
                        _INSTRUCTION_MAX_CHARS, text)
            break
        out.append(ch)
        budget += cost
    return "".join(out)


def _endpoint(workspace_id: str) -> str:
    """业务空间 ID 直接拼进域名，故只接受已通过 `validate_workspace_id` 的取值。

    调用方一律传 `resolve()` 出来的 workspace_id（用户级或全局兜底），
    不在这里回读 `get_settings()`，避免「解析用用户值、拼 URL 用全局值」的不一致。
    """
    return get_settings().TTS_URL_TEMPLATE.format(workspace_id=workspace_id)


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


async def synthesize(text: str, emotion: str | None = None,
                     user_id: int | None = None) -> TtsResult:
    """把文本合成语音，返回音频 URL。

    - `user_id` 决定用谁的凭据：先试该用户自持的加密配置，无则回落全局 `.env`
      兜底（见 `resolve`）。传 None 等价于「只用全局兜底」。
    - `emotion` 为情绪枚举名（见 rag/emotion.py），映射成官方控制类标签插到文本开头；
      控制类标签作用于其后全部文本，所以只插一个。
    - 入参先过 `_to_speech_text` 转朗读稿（剥引用标记 + 去版式符号），随后才做长度截断。
    - 文本超长按 `TTS_MAX_CHARS` 截断（官方 Qwen-TTS 系上限 512 token，中文约 1 字 1 token
      量级，留足余量）；截断优先在句末标点处，避免把话切一半。
    - 指令控制（instruction）取自生效配置（用户填的优先，留空回落默认），超长本地截断。
    - 标签与指令**只进合成文本/请求体**，不进任何用户可见字段。
    """
    rt, why = await resolve(user_id)
    if rt is None:
        return TtsResult(ok=False, error=why)

    # 传输侧参数（超时/格式/采样率/长度帽）是服务端行为，不随用户变，仍读全局设置；
    # 只有「谁的 key / 哪个空间 / 什么模型音色」这几项才来自用户配置。
    s = get_settings()
    model = rt["model"]
    voice = rt["voice"]
    body_text = _to_speech_text(text)
    if not body_text:
        return TtsResult(ok=False, error="待合成文本为空", model=model, voice=voice)

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
        "model": model,
        "input": {
            "text": synth_text,
            "voice": voice,
            "format": s.TTS_FORMAT,
            "sample_rate": s.TTS_SAMPLE_RATE,
        },
    }
    # 指令控制（item 级可选）：描述音调/语速/音色基调。留空则不发该字段。
    # 与 `[excited]` 这类**句级**情感标签是两层控制 —— 标签管这一段的情绪起伏，
    # 指令管整体声音底色（音色性格），二者叠加不冲突（官方示例同款用法）。
    instruction = _clip_instruction(rt["instruction"])
    if instruction:
        payload["input"]["instruction"] = instruction
    headers = {
        "Authorization": f"Bearer {rt['api_key']}",
        "Content-Type": "application/json",
    }

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=s.TTS_TIMEOUT) as cli:
            rsp = await cli.post(_endpoint(rt["workspace_id"]), json=payload, headers=headers)
    except httpx.HTTPError as exc:
        log.warning("TTS 请求失败: %s", type(exc).__name__)
        return TtsResult(ok=False, error=f"语音服务连接失败：{type(exc).__name__}",
                         model=model, voice=voice, emotion=emotion or "")

    elapsed = int((time.perf_counter() - t0) * 1000)
    if rsp.status_code in (401, 403):
        return TtsResult(ok=False, error="语音服务鉴权失败，请确认使用的是北京地域的 API Key",
                         elapsed_ms=elapsed)
    if rsp.status_code >= 400:
        # 服务端不返回结构化错误码时，只能按状态码 + 响应体关键词给出可操作的建议。
        # `InvalidParameter` 几乎总是「音色与模型版本不匹配」——3.1 只认 `_v3.1` 后缀音色，
        # 这是换模型时最容易踩的一脚，故单独点名。
        log.warning("TTS 返回 %s: %.300s", rsp.status_code, rsp.text)
        detail = rsp.text or ""
        if "InvalidParameter" in detail or "Engine error" in detail:
            msg = (f"语音参数不被接受（音色 {voice} 可能不适用于模型 "
                   f"{model}）；同一代模型的音色不可跨版本混用")
        elif rsp.status_code == 404:
            msg = "语音服务地址不存在，请在设置页核对业务空间 ID（Workspace ID）是否正确"
        elif rsp.status_code == 429:
            msg = "语音服务请求过于频繁，请稍后重试"
        else:
            msg = f"语音服务返回错误（HTTP {rsp.status_code}）"
        return TtsResult(ok=False, error=msg, elapsed_ms=elapsed)

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

    log.info("TTS 合成成功 model=%s voice=%s emotion=%s 指令=%s 来源=%s 耗时=%dms",
             model, voice, emotion or "-", "有" if instruction else "无",
             rt.get("source", "-"), elapsed)
    return TtsResult(ok=True, url=url, model=model, voice=voice,
                     emotion=emotion or "", elapsed_ms=elapsed)
