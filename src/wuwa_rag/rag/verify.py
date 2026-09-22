"""答案检查 agent：qwen3:8b 判定「检索到的资料能否回答这个问题」。

检索链路里「重排低分→丢弃候选」只能发现**完全没有**资料，发现不了
**资料跑题/脏数据**（旧爬取残留、召回到别的角色等）——那种情况会拿着
不匹配的材料硬答。verify_knowledge 在生成前把关：不匹配则给出更精确的
检索式（refined），供上层重检索/刷新。

风格与 classify_topic 一致（实测校准）：单轮补全、末尾 AIMessage 以 `{"`
截停省输出 token、严格 JSON、解析失败降级为「匹配」（宁可漏报不误伤正常
问答路径——降级成多花一次刷新不如把对的答错）。
"""
from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..ww_logger import get_logger
from .llm import get_tool_llm

log = get_logger("rag")

_VERIFY_SYSTEM = """你是《鸣潮》问答助手的资料审查员。判断给定的资料能否回答用户问题，只输出一个 JSON：
{"match": true} 或 {"match": false, "refined": "更适合检索的问句"}
- match=true：资料里有足够信息回答该问题（数值、配装、机制等，对得上角色和方面即可）。
- match=false：资料跑题（是别的角色、别的方面）、为空、或与问题无关；此时 refined 给出一个更精确的中文检索问句（补全角色名+方面关键词）。
示例1：
资料：【卡卡罗】声骸=彻空冥雷 COST43311
问题：卡卡罗毕业配装用什么声骸
输出：{"match": true}
示例2：
资料：【长离】突破材料=贝币x18万
问题：卡卡罗毕业配装用什么声骸
输出：{"match": false, "refined": "卡卡罗 毕业 声骸 配装 COST"}
不要输出 JSON 以外的任何文字。"""

_EXAMPLES = (
    ("【卡卡罗】声骸=彻空冥雷，COST43311", "卡卡罗毕业配装用什么声骸", "true", ""),
    ("【长离】共鸣链一链=抗打断提升", "卡卡罗毕业配装用什么声骸", "false", "卡卡罗 毕业 声骸 配装"),
)


def _fmt_materials(graph_facts: str, docs: list[dict], cap: int = 1200) -> str:
    """图谱事实优先，向量块取面包屑+正文头；总长设帽防 token 爆。"""
    parts: list[str] = []
    if graph_facts:
        parts.append(graph_facts[:cap])
    budget = max(0, cap - sum(len(p) for p in parts))
    for d in docs:
        if budget <= 0:
            break
        line = f"[{(d.get('breadcrumb') or '')[:40]}] {(d.get('text') or '')[:120]}"
        parts.append(line)
        budget -= len(line)
    return "\n".join(parts) if parts else "（无资料）"


async def verify_knowledge(question: str, graph_facts: str, docs: list[dict]) -> tuple[bool, str]:
    """返回 (是否匹配, refined 检索式)。任何异常都降级为匹配（不阻塞正常回答）。"""
    materials = _fmt_materials(graph_facts, docs)
    if materials == "（无资料）":
        # 空材料无需模型也是不匹配，直接给重检索式
        return False, question

    msgs: list = [SystemMessage(content=_VERIFY_SYSTEM)]
    for mat, q, mt, rf in _EXAMPLES:
        msgs.append(HumanMessage(content=f"资料：{mat}\n问题：{q}"))
        out = '{"match": true}' if mt == "true" else f'{{"match": false, "refined": "{rf}"}}'
        msgs.append(AIMessage(content=out))
    msgs.append(HumanMessage(content=f"资料：{materials}\n问题：{question}"))
    msgs.append(AIMessage(content='{"'))

    try:
        resp = await get_tool_llm().ainvoke(
            msgs, config={"tags": ["wwa:verify"]})   # 打标防流式泄漏（同 wwa:summary）
        txt = '{"' + (resp.content or "")
        m = re.search(r"\{.*?\}", txt, re.S)
        if not m:
            log.warning("资料审查: 输出无 JSON，放行：%r", txt[:60])
            return True, ""
        data = json.loads(m.group(0))
        if data.get("match") is True:
            return True, ""
        refined = str(data.get("refined") or "").strip()[:120]
        log.info("资料审查: 判不匹配 refined=%r", refined or question)
        return False, refined or question
    except Exception as exc:
        log.warning("资料审查失败，放行: %s", exc)
        return True, ""
