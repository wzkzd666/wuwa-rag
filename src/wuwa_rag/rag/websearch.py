"""百度千帆联网搜索兜底（v2 chat/completions + web_search）。

检索链路最后一级：本地 RAG 重检索/刷新后 verifier 仍判不匹配时，用百度实时
搜索结果补答。参数按官方文档「联网搜索」（qianfan-docs, 2026-07 版）：
  web_search: {enable, search_mode, search_number, reference_number, ...}
  模型需 ERNIE 4.5+ 系（search_mode 仅支持 auto——ernie 不支持 required 强制搜索）。

⚠️ API key 留空 = 整体关闭：返回 (False, "")，上层降级「不知道」，不报配置错。
⚠️ 数据出境：会把「问题+本地资料摘要(≤800字)」经 HTTPS 发百度；未配 key 不发生。
"""
from __future__ import annotations

import httpx

from ..config import get_settings
from ..ww_logger import get_logger

log = get_logger("rag")


def _enabled() -> bool:
    return bool(get_settings().QIANFAN_API_KEY.strip())


async def web_search(query: str) -> tuple[bool, str]:
    """联网搜索：返回 (是否成功, 资料文本)。答案由调用方交 aemeath 组织（保人设）。"""
    s = get_settings()
    if not _enabled():
        return False, ""
    body = {
        "model": s.QIANFAN_WEB_MODEL,
        "messages": [{"role": "user", "content": query}],
        # 官方文档：ernie 系列 search_mode 只支持 auto（模型判意图），无 temperature 强制
        "web_search": {
            "enable": True,
            "search_mode": "auto",
            "search_number": 10,
            "reference_number": 8,
        },
        "temperature": 0.2,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=s.QIANFAN_TIMEOUT) as client:
            r = await client.post(
                s.QIANFAN_CHAT_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {s.QIANFAN_API_KEY.strip()}",
                },
                json=body,
            )
            if r.status_code != 200:
                log.warning("千帆联网搜索 HTTP %s: %s", r.status_code, r.text[:200])
                return False, ""
            data = r.json()
            choice = (data.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content", "").strip()
            # 非流式溯源在 choice.search_results / data.search_results（文档流式示例均出现过）
            sources = choice.get("search_results") or data.get("search_results") or []
            lines = [content] if content else []
            for it in sources[:8]:
                if isinstance(it, dict):
                    lines.append(f"- {it.get('title', '')}: {it.get('url', '')}")
            text = "\n".join(lines).strip()
            if not text:
                log.warning("千帆返回空内容: keys=%s", list(data.keys())[:6])
                return False, ""
            log.info("千帆联网搜索成功: %d 字 / 溯源 %d 条", len(text), len(sources))
            return True, text
    except Exception as exc:
        log.warning("千帆联网搜索失败(%s): %s", type(exc).__name__, str(exc)[:160])
        return False, ""
