"""百度千帆联网搜索兜底（v2 chat/completions + web_search）。

检索链路最后一级：本地 RAG 重检索/刷新后 verifier 仍判不匹配时，用百度实时
搜索结果补答。参数按官方文档「联网搜索」（qianfan-docs, 2026-07 版）：
  web_search: {enable, search_mode, search_number, reference_number, enable_trace, ...}
  模型需 ERNIE 4.5+ 系（search_mode 仅支持 auto——ernie 不支持 required 强制搜索）。

实测数据（请求 qianfan.baidubce.com）：
  - 鉴权 = `Authorization: Bearer <千帆控制台 API Key>`（形如 bce-v3/ALTAK-…），直连 v2 即可。
  - 是否真联网看 `usage.prompt_tokens_details.search_tokens > 0`（实测 1905~3430）。
  - 延迟 6.1s（命中搜索缓存）~25.6s（真搜），多数 21~26s → QIANFAN_TIMEOUT 必须 ≥45s，
    否则随机 ReadTimeout，表现为「联网不可用」。

API key 留空即整体关闭：返回 (False, "")，上层降级为「不知道」，不报配置错误。
外部依赖：会将「问题 + 本地资料摘要（≤800 字）」经 HTTPS 发送至百度；未配置 key 时不发生。
"""
from __future__ import annotations

import httpx

from wuwa_rag.config import get_settings
from wuwa_rag.ww_logger import get_logger

log = get_logger("rag")


def _enabled() -> bool:
    return bool(get_settings().QIANFAN_API_KEY.strip())


async def web_search(query: str) -> tuple[bool, str, list[dict[str, str]]]:
    """联网搜索：返回 **(是否成功, 资料文本或失败原因, 溯源列表)**。

    第二个槽位在失败时放**给用户看的一句话原因**（不是日志用的堆栈）：
    以前所有失败都返回空串，调用方只能笼统降级，用户看到的是「我翻了本地也上网找了，
    确实没有」——而真相可能是「Key 过期了，根本没找成」。这两种情况对用户的意义完全不同，
    所以失败也要如实说，且只说类别（HTTP 状态/没连通/解析失败），不外泄响应体。

    第三个槽位是 `[{title, url}]`：以前溯源只拼进 prompt 文本（`- title: url`），
    `done.sources` 里没有它，于是**联网答出来的内容在引用面板里是空的**。
    答案由调用方交 aemeath 组织（保人设）。
    """
    s = get_settings()
    if not _enabled():
        # 整条联网兜底关闭：不发任何外部请求，也如实说明「这次没法上网查」
        return False, "本部署未启用联网搜索", []
    body = {
        "model": s.QIANFAN_WEB_MODEL,
        "messages": [{"role": "user", "content": query}],
        # 官方文档：ernie 系列 search_mode 只支持 auto（模型判意图），无 temperature 强制
        # enable_trace 必开：实测不开时响应里**没有** `search_results` 字段（溯源恒 0 条），
        # 开了才有（实测 10 条 index/url/title）。enable_citation 效果相同，二选一即可。
        "web_search": {
            "enable": True,
            "search_mode": "auto",
            "search_number": 10,
            "reference_number": 8,
            "enable_trace": True,
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
                # 4xx 里 401/403 是凭据问题、429 是额度，其余归到「服务端拒绝」；
                # 只报类别，响应体留在日志里（可能含请求上下文）。
                why = {401: "API Key 无效或已过期", 403: "API Key 无权访问该模型",
                       429: "联网搜索额度已用尽"}.get(r.status_code, f"服务端返回 HTTP {r.status_code}")
                return False, f"联网搜索没成功（{why}）", []
            data = r.json()
            choice = (data.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content", "").strip()
            # 溯源位置实测在 **choice.search_results**（开了 enable_trace 才有）；
            # data.search_results 兜底保留。每项含 index/url/title。
            raw_sources = choice.get("search_results") or data.get("search_results") or []
            sources = [
                {"title": str(it.get("title") or "").strip(), "url": str(it.get("url") or "").strip()}
                for it in raw_sources
                if isinstance(it, dict) and str(it.get("url") or "").strip()
            ]
            lines = [content] if content else []
            for it in sources[:8]:
                lines.append(f"- {it['title']}: {it['url']}")
            text = "\n".join(lines).strip()
            if not text:
                log.warning("千帆返回空内容: keys=%s", list(data.keys())[:6])
                return False, "联网搜索没有返回内容", []
            log.info("千帆联网搜索成功: %d 字 / 溯源 %d 条", len(text), len(sources))
            return True, text, sources
    except httpx.HTTPError as exc:
        # 只吞网络/超时/协议类故障——联网兜底本就是「有就加分、没有就降级」，
        # 不该因为千帆挂了而中断问答。响应体解析失败（KeyError/ValueError）属
        # 可预期，也一并降级；其余编程错误应如实暴露。
        log.warning("千帆联网搜索失败(%s): %s", type(exc).__name__, str(exc)[:160])
        return False, f"联网搜索没连通（{type(exc).__name__}）", []
    except (KeyError, IndexError, ValueError) as exc:
        log.warning("千帆联网搜索响应解析失败(%s): %s", type(exc).__name__, str(exc)[:160])
        return False, "联网搜索结果解析失败", []
