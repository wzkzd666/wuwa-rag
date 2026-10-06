"""用量看板与答案反馈的 HTTP 接口（L6）。

权限分两条线，**都由后端强制**，不靠前端藏按钮：
- 用量：管理员看**所有人**；普通用户只看**自己**（`only_user=str(user.id)` 强制带上，
  前端传什么都拦得住）。
- 反馈：任何登录用户都能**提交**自己那条回答的评价；只有管理员能**读别人/全体**的反馈。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from wuwa_rag.api import auth as authn
from wuwa_rag.core import feedback as fb
from wuwa_rag.core import usage

router = APIRouter()


class FeedbackIn(BaseModel):
    thread_id: str = Field(..., description="所属会话（thread）id")
    target_id: str = Field(..., description="被评价的那条回答 id（前端消息 id）")
    rating: int = Field(..., description="1 满意 / -1 不满意")
    comment: str = Field("", max_length=1000, description="可选补充说明")
    question: str = Field("", description="快照：当时的问题")
    answer: str = Field("", description="快照：当时的回答")
    provider: str = Field("", description="local / cloud")
    model: str = Field("", description="回答所用模型")


@router.get("/usage/summary")
async def api_usage_summary(days: int = 7, user: str = "",
                            auth: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """近 N 天 token 用量，按人 × provider(local/cloud) 汇总。

    `user` 非空时只看这一个人（管理员用：先看全员总量，再单独盯某个人的开销）。
    ⚠️ **普通用户传了也无效**：非管理员一律强制 `only_user=自己`，
    越权必须在后端拦，前端藏选项不算防护。
    """
    days = max(1, min(days, 90))
    if auth.is_admin:
        only = user.strip() or None
    else:
        only = str(auth.id)
    out = await usage.usage_summary(days=days, only_user=only)
    out["scope"] = "all" if auth.is_admin else "self"
    if auth.is_admin:
        out["feedback"] = (await fb.list_feedback(days=days))["summary"]
    return out


@router.post("/feedback")
async def api_feedback(body: FeedbackIn,
                       user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """对一条回答点赞/点踩（可带文字）。重复提交 = 改主意，覆盖而非追加。"""
    if body.rating not in (fb.RATING_UP, fb.RATING_DOWN):
        raise HTTPException(status_code=400, detail="rating 只能是 1 或 -1")
    fid = await fb.save_feedback(
        user_id=str(user.id), username=user.username, thread_id=body.thread_id,
        target_id=body.target_id, rating=body.rating, comment=body.comment,
        question=body.question, answer=body.answer,
        provider=body.provider, model=body.model,
    )
    return {"id": fid, "rating": body.rating}


@router.get("/feedback")
async def api_feedback_list(days: int = 7,
                            user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """反馈列表。普通用户只看自己的；管理员看全员（含问答快照与差评原因）。"""
    only = None if user.is_admin else str(user.id)
    out = await fb.list_feedback(days=max(1, min(days, 90)), only_user=only)
    for it in out["items"]:
        it["can_delete"] = user.is_admin or it["user_id"] == str(user.id)
    return out


@router.delete("/feedback/{record_id}")
async def api_feedback_delete(record_id: int,
                              user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """删掉一条反馈。管理员可删任意；普通用户只能删自己的（与提交记录同一套口径）。"""
    # 按 id 精确取一条做归属判定 —— 不能「拉一大段列表再遍历」：列表有 LIMIT 200，
    # 反馈多了以后目标根本不在里面，删自己刚提交的反馈会 404。
    target = await fb.get_feedback(record_id)
    if target is None or (not user.is_admin and target["user_id"] != str(user.id)):
        # 不是自己的（或不存在）—— 一律 404，不泄露「这条存在但不属于你」
        raise HTTPException(status_code=404, detail="反馈不存在")
    thread_id = await fb.delete_feedback(record_id)
    return {"id": record_id, "thread_id": thread_id}


@router.get("/feedback/mine")
async def api_feedback_mine(target_ids: str,
                            user: authn.AuthUser = Depends(authn.get_current_user)) -> dict:
    """我点过哪些回答（逗号分隔的 target_id）→ {target_id: rating}，前端据此标记。"""
    ids = [t for t in (target_ids or "").split(",") if t][:200]
    return {"ratings": await fb.my_feedback(str(user.id), ids)}


def install(app) -> None:  # noqa: ANN001
    app.include_router(router, prefix="", tags=["usage"])
