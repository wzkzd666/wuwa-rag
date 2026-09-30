"""情绪标签 agent：文本 LLM 生成答案后顺带判定情绪，供 TTS 合成使用。

设计要点
--------
1. **为什么不让 chat 模型（aemeath / 云端模型）自己在答案里输出标签**：
   标签会污染用户可见文本（`[excited]卡卡罗毕业配装…`），而本仓已有实测教训——
   `text.strip_ref_marks` 就是为了兜住模型自创的 `[图谱]`/`[1]` 标记。再开一个
   "请输出标签"的口子等于自找清洗负担。
   改用 **tool 模型（qwen3:8b）对已生成的答案做一次独立判定**，答案文本零污染。

2. **枚举必须映射到官方真实标签**（阿里云「非实时语音合成 · 情感与富语言标签」2026-09 版）：
   注意：官方未提供 [cheerful]/[gentle]/[happy] 这类标签名——自行命名会被服务端
   当成普通文本念出来（静默失效，比报错更难发现）。下列 5 个枚举逐个对照官方清单：
     cheerful  → [excited]        兴奋（爱弥斯日常语体：轻快、话多、爱分享）
     amazed    → [amazed]         惊叹（"这个真的很强！"式推荐）
     serious   → [serious]        严肃（数值/机制讲解、警示类结论）
     empathetic→ [empathetic]     共情（用户沮丧、或答不上来时的致歉）
     playful   → [mischievously]  调皮（闲聊、开玩笑、给自己起外号）
   控制类标签作用于**其后全部文本**，故只在开头插一个即可（官方示例同款用法）。

3. **判定模型的分工**：
   - **默认走本地 tool 模型 qwen3:8b**：零远程依赖、零额外费用，
     即使请求落到了用户自配的云端 provider 也不受影响。
   - **可选由答题的那个模型兼任**：配了自定义云端 LLM 且该用户开启
     `emotion_enabled` 时，`chain` 会把本轮的 chat 客户端传进来。
     两条路径共用同一套白名单校验与回落逻辑，差异不会传导到问答链路。

4. **失败一律回落 `cheerful`**：情绪只是语音佐料，判错顶多语气不够贴，
   绝不能因为它挡住问答主链路。

4. 写法沿用 `intent.classify_topic` 已实测有效的范式：SystemMessage 定义 +
   少样本单轮补全（Human/AI 交替，末轮 AI 前缀吃掉 `{"`）+ 正则取 JSON + 白名单校验。
   本仓铁律：qwen3:8b 对抽象规则钝感、**对示例敏感**，故必须给 few-shot。
"""
from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from wuwa_rag.config import get_settings
from wuwa_rag.core.llm import get_tool_llm
from wuwa_rag.ww_logger import get_logger

log = get_logger("emotion")

# 情绪枚举 → Qwen-Audio-TTS 官方控制类标签（见模块 docstring 第 2 点）
EMOTION_TAGS: dict[str, str] = {
    "cheerful":   "[excited]",
    "amazed":     "[amazed]",
    "serious":    "[serious]",
    "empathetic": "[empathetic]",
    "playful":    "[mischievously]",
}
DEFAULT_EMOTION = "cheerful"

# 判定输入的答案长度帽：情绪看开头语气就够了，长答案（满级数值表）整篇喂进去
# 既慢又容易被大段数字带偏成 serious。
_JUDGE_MAX_CHARS = 600

_EMOTION_SYSTEM = """你是《鸣潮》角色「爱弥斯」问答助手的语气标注器。读一段助手的回答，判断它最适合用哪种语气朗读，只输出一个 JSON：
{"emotion": "cheerful"} 
可选值（只能从中选一个）：
- cheerful：日常讲解、介绍配装/声骸/玩法，语气轻快活泼（默认，拿不准就选它）
- amazed：带明显赞叹或强烈推荐（"这个真的很强""特别好用"）
- serious：严肃说明数值机制、给出警示或否定结论（"不推荐萌新练""必须先满级"）
- empathetic：安慰、共情、致歉或坦承不知道（"这个我不太清楚""别灰心"）
- playful：闲聊、开玩笑、撒娇、给自己起外号等俏皮内容
不要输出 JSON 以外的任何文字。"""

# few-shot（本仓铁律：8b 靠示例不靠规则）。覆盖 5 类 + 一个"长数据表"陷阱样本：
# 数据表看着严肃，但爱弥斯讲数据仍是轻快语气 → 应判 cheerful，不是 serious。
_EMOTION_EXAMPLES = (
    ("卡卡罗毕业配装用的是彻空冥雷，COST 组合 43311，主词条选暴击伤害~", "cheerful"),
    ("这个声骸套装真的超强！泛用性拉满，闭眼刷就完事啦~", "amazed"),
    ("注意：长离的共鸣链二链才是质变点，未满二链前不建议投入过多资源。", "serious"),
    ("诶…这个我翻遍小本本都没找着，确实不清楚呀，家人再帮我确认下名字嘛~", "empathetic"),
    ("我？我可是拉海洛车神兼人气歌手飞行雪绒哦~豹豹说想你了！", "playful"),
    ("- 攻击：2100\n- 暴击：15%\n- 生命：11400\n满级数值就是这些啦，需要我再讲讲词条吗~", "cheerful"),
)


# 定向提取「emotion: 值」，容忍键值之间出现裸换行/缩进/中文冒号。
# 注意：仅依赖 json.loads 并不可靠，实测 qwen3:8b 会输出开头带裸换行的 JSON
# （`{"` 前缀 + `\n"emotion": "serious"}`），换行落在字符串字面量内 →
# 严格解析抛 `Invalid control character` → 走 except **静默回落**默认情绪。
# 那次样本期望恰好是 cheerful 才「蒙对」（日志里只有一条 WARNING）；
# 若期望 serious 就会判错，且回落路径看起来像"判对了"，极难发现。
_RE_EMOTION = re.compile(r'"?emotion"?\s*[:：]\s*"?([A-Za-z_]+)"?', re.I)


def _parse_emotion(text: str) -> str:
    """从模型输出提取情绪枚举；解析不到或不在白名单一律回落 DEFAULT_EMOTION。"""
    raw = (text or "").strip()
    if not raw:
        return DEFAULT_EMOTION
    m = _RE_EMOTION.search(raw)
    if m:
        emo = m.group(1).strip().lower()
        # 白名单校验：模型可能编造枚举外的值（如 happy/tender），一律回落默认
        if emo in EMOTION_TAGS:
            return emo
        log.warning("情绪判定：值不在白名单（%r），回落 %s", emo, DEFAULT_EMOTION)
        return DEFAULT_EMOTION
    # 兜底再试宽松 JSON（strict=False 容忍控制字符）
    try:
        obj = json.loads(raw, strict=False)
        if isinstance(obj, dict):
            emo = str(obj.get("emotion", "")).strip().lower()
            if emo in EMOTION_TAGS:
                return emo
    except Exception:
        pass
    log.warning("情绪判定：输出无法解析，回落 %s：%r", DEFAULT_EMOTION, raw[:60])
    return DEFAULT_EMOTION


async def detect_emotion(answer: str, chat_client: Runnable | None = None) -> str:
    """判定答案情绪，返回 EMOTION_TAGS 的键；任何异常回落 DEFAULT_EMOTION。

    `chat_client` 为空（默认）时用本地 tool 模型 qwen3:8b；非空则用传入的客户端
    —— 传入即表示「本次用答题的那个模型兼任判定」，由 rag/chain.py 按用户配置决定。
    兼任路径不调整温度等采样参数：情绪是五选一的轻量分类，输出经白名单校验，
      不合格一律回落，不需要为此再维护一份客户端实例。
    """
    text = (answer or "").strip()
    if not text:
        return DEFAULT_EMOTION
    if not get_settings().EMOTION_ENABLED:
        return DEFAULT_EMOTION

    msgs: list = [SystemMessage(content=_EMOTION_SYSTEM)]
    for a, e in _EMOTION_EXAMPLES:
        msgs.append(HumanMessage(content=f"回答：{a}"))
        msgs.append(AIMessage(content=f'{{"emotion": "{e}"}}'))
    msgs.append(HumanMessage(content=f"回答：{text[:_JUDGE_MAX_CHARS]}"))
    msgs.append(AIMessage(content='{"'))
    try:
        # 注意：tags 必须通过 config={"tags":[...]} 传入，不能写成 ainvoke(msgs, tags=[...])：
        #    get_tool_llm() 返回的是 bind(think=False) 后的 RunnableBinding，
        #    再传 tags= 关键字会与之冲突 → TypeError: got multiple values for 'tags'，
        #    情绪判定会**静默全量回落** cheerful（实测踩过：7/7 全失败，准确率看着像 3/7）。
        #    与 intent.summarize_turns / verify.verify_knowledge 保持同一写法。
        # 打标原因：detect_emotion 在 generate/chitchat 节点**内部**被调用，其 ainvoke
        #    的流式回调 node='generate'，节点过滤挡不住 → 必须靠标签在 ask_stream 里丢弃，
        #    否则 {"emotion":"..."} 会拼进答案尾巴（与 wwa:summary 同一类泄漏）。
        client = chat_client or get_tool_llm()
        resp = await client.ainvoke(msgs, config={"tags": ["wwa:emotion"]})
        return _parse_emotion(str(resp.content or ""))
    except Exception as exc:
        log.warning("情绪判定失败，回落 %s: %s", DEFAULT_EMOTION, exc)
        return DEFAULT_EMOTION


def emotion_tag(emotion: str | None) -> str:
    """情绪名 → TTS 控制标签；未知/空值回落默认标签（绝不返回空串导致无情感合成）。"""
    return EMOTION_TAGS.get((emotion or "").strip().lower(), EMOTION_TAGS[DEFAULT_EMOTION])


def with_emotion_tag(text: str, emotion: str | None) -> str:
    """把控制标签插到文本开头（官方：控制类标签作用于其后所有文本）。

    标签**只用于 TTS 合成文本**，绝不进用户可见的答案字段——
    否则前端会把 `[excited]` 直接渲染出来。
    """
    body = (text or "").strip()
    return f"{emotion_tag(emotion)}{body}" if body else body
