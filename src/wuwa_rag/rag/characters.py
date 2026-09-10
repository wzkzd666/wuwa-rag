"""角色名解析：静态角色名册 + LLM 兜底。
用于问答时识别「知识库里还没有」的角色，以便自动触发爬取+建库。
规则优先：先在名册(CHARACTER_NAMES / CHARACTER_ALIASES)里做匹配；名册没命中再走一次轻量 LLM 抽取。
"""
from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage

from ..ww_logger import get_logger
from .llm import get_chat_llm

log = get_logger('rag')

# 全量角色名
CHARACTER_NAMES: set[str] = {
    "丹瑾", "丽贝卡", "仇远", "今汐", "凌阳", "千咲", "卜灵", "卡卡罗", "卡提希娅", "吟霖",
    "嘉贝莉娜", "坎特蕾拉", "夏空", "奥古斯塔", "守岸人", "安可", "尤诺", "布兰特", "弗洛洛",
    "忌炎", "折枝", "散华", "桃祈", "椿", "洛可可", "洛瑟菈", "清宵", "渊武",
    "漂泊者-男-导电", "漂泊者-男-气动", "漂泊者-男-湮灭", "漂泊者-男-衍射",
    "灯灯", "炽霞", "爱弥斯", "珂莱塔", "琳奈", "白芷", "相里要", "秋水",
    "秧秧", "秧秧·玄翎", "穗穗", "绯雪", "维里奈", "莫宁", "莫特斐", "菲比",
    "西格莉卡", "赞妮", "达妮娅", "釉瑚", "鉴心", "长离", "陆·赫斯", "露帕", "露西",
}

# 别名对应规则表
CHARACTER_ALIASES: dict[str, str] = {
    "光主": "漂泊者-男-衍射",
    "风主": "漂泊者-男-气动",
    "暗主": "漂泊者-男-湮灭",
    "电主": "漂泊者-男-导电"
}

# 漂泊者：性别维度归一为[男]
_POVER_ATTRS = ("导电", "气动", "湮灭", "衍射", "热熔", "冷凝")
_POVER_DEFAULT = "漂泊者-男-导电"  # 用户只说「漂泊者」不带属性时的默认分支；改这里换默认


def _pover_resolve(question: str) -> str | None:
    """漂泊者特例：性别强制男；属性取用户提到的，没有则用 _POVER_DEFAULT。"""
    if "漂泊者" not in question:
        return None
    attr = next((a for a in _POVER_ATTRS if a in question), None)
    return f"漂泊者-男-{attr}" if attr else _POVER_DEFAULT


_EXTRACT_PROMPT = (
    "你是《鸣潮》wiki 的角色名抽取器。从用户问题里提取提到的游戏角色中文标准名。\n"
    "只输出一个 JSON：{\"characters\": [\"角色名\", ...]}，没有具体角色就 {\"characters\": []}。\n"
    "不要编造；问题没出现具体角色就返回空列表。"
)


async def _llm_candidates(question: str) -> list[str]:
    try:
        resp = await get_chat_llm().ainvoke(
            [HumanMessage(content=_EXTRACT_PROMPT + "\n\n问题：" + question)]
        )
        txt = resp.content.strip()
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            return []
        data = json.loads(m.group(0))
        return [str(c) for c in data.get("characters", []) if c]
    except Exception as exc:
        log.warning("LLM 角色抽取失败: %s", exc)
        return []


def _rule_candidates(question: str) -> list[str]:
    """名册规则匹配。漂泊者走特例（性别归男）；其余按 CHARACTER_NAMES / CHARACTER_ALIASES 匹配。"""
    hits: set[str] = set()
    p = _pover_resolve(question)
    if p:
        hits.add(p)
    for n in CHARACTER_NAMES:
        if n and n in question:
            hits.add(n)
    for alias, std in CHARACTER_ALIASES.items():
        if alias in question:
            hits.add(std)
    out = [s for s in hits]
    return out


async def resolve_candidates(question: str) -> list[str]:
    rule = _rule_candidates(question)
    if rule:
        log.info("角色抽取: 规则命中 %s", rule)
        return rule
    llm = await _llm_candidates(question)
    log.info("角色抽取: LLM兜底 %s", llm)
    return llm
