"""答案级质量检查（确定性，零 LLM）—— 把「已经踩过的坑」变成一条命令能反复跑的门。

为什么要这一层：`tests/` 里的五百多条单测守的是**判据**（规则层对不对），
`tests/eval_retrieval.py` 守的是**检索效果**；而「答案本身坏掉」的那一类
（第三人称搬运腔、`[n]` 角标漏剥、百分比丢单位、列表重复、清洗只补了三个出口）
此前每次都靠人肉发现 + 事后记进 AGENTS。本模块把它们钉成可复跑的规则。

⚠️ 三条设计约束（都是本仓的老规矩）：
1. **判据只准一处**：所有检查都复用 `text.py` 与清洗链**同一批函数**
   （`strip_ref_marks` / `fix_percent_units` / `dedup_list_items` / `unquote_mentions`），
   判据写成「清洗后还变不变」的**幂等对账** —— 清洗规则一改，这里自动跟着改，
   不存在第二份正则会漂。
2. 检查项**只报事实，不改答案**。这里是质量闸门，不是第三道清洗。
3. 「没触发」不等于「没问题」：本模块查的是**已知故障形态**，新故障形态仍要先复现再加。
"""
from __future__ import annotations

from wuwa_rag.services.emotion import EMOTION_TAGS
from wuwa_rag.text import (
    dedup_list_items,
    fix_percent_units,
    strip_ref_marks,
    unquote_mentions,
)


def _emotion_tags_in(answer: str) -> list[str]:
    """答案正文里出现的情绪标签（它们只该进朗读指令，不该出现在文字里）。

    ⚠️ `EMOTION_TAGS` 是 `{情绪名: "[tag]"}` —— 要查的是**值**，别写成 `[{key}]`
    （第一版就踩了：查出来永远是空，检查形同不存在）。
    """
    return sorted({tag for tag in EMOTION_TAGS.values() if tag in answer})


def check_answer(answer: str, *, team_followup: bool = False,
                 nicknames: list[str] | None = None) -> list[str]:
    """返回问题清单（空 = 这轮答案没踩到已知故障形态）。

    参数 `team_followup`：这轮是**概括性配队**（`_should_ask_team` 为真），
    答案末尾必须有那句追问 —— 它由 `graph` 确定性拼接，流式侧还要求单独补吐，
    两条出口漏一条就是「前端少一句、done.answer 却有」。
    """
    out: list[str] = []
    text = answer or ""

    if not text.strip():
        return ["答案为空（done.answer 是空串：多半是节点正文没进 _TEXT_NODES）"]

    # ①~④ 幂等对账：输出口已经过 `_clean_answer` 四步，**任何一步再改变得动说明
    # 那条出口漏了这一步**（AGENTS 记过：④ 补进节点却漏在 ask_stream，
    # 表现成「流式带「」、下一轮 history 已剥掉」）。
    for name, fn in (("引用角标 [n] 没剥净", strip_ref_marks),
                     ("百分比单位串味没修", fix_percent_units),
                     ("重复列表行没去", dedup_list_items)):
        if fn(text) != text:
            out.append(name)

    names = [n for n in (nicknames or []) if n]
    if names and unquote_mentions(text, names) != text:
        out.append("昵称外层引号没剥")

    tags = _emotion_tags_in(text)
    if tags:
        out.append(f"情绪标签泄进正文：{tags}")

    if team_followup:
        from wuwa_rag.dialog.graph import _TEAM_FOLLOWUP  # 常量只此一份，不抄字符串
        if not text.rstrip().endswith(_TEAM_FOLLOWUP):
            out.append("概括性配队缺那句追问（流式侧要单独补吐）")

    return out
