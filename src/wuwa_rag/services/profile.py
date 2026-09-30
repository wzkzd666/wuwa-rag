"""用户画像：启用 001_init.sql 中定义的 user_facts 表。

数据流：
  用户提问 → (异步、不阻塞回答) tool 模型从消息里抽「稳定偏好事实」→ 去重入库
  下次提问 → 取该用户仍有效的事实 → 拼成 user_context 注入生成 prompt（个性化）

事实示例：「主玩角色是守岸人」「萌新，需要基础讲解」「偏爱湮灭队」。
约束：
- 只抽**稳定的长期偏好**，不抽一次性问题（「今汐几级突破」这种不能进画像）；
- 每条 ≤40 字，单次最多 3 条，防止低质量事实堆积；
- **角色名必须有原文依据**（`_grounded`，双检）：① 句式正则抓「主玩/本命/主力 + 名字」，
  ② 名册（`rag/characters.py` 的 57 个角色名 + 别名）抓「已知角色名无依据出现」。
  模型会照模板编造用户没提过的角色，而事实一旦入库就会注入之后每一轮 prompt，
  编造代价远大于漏抽一条，所以两道检查都偏严（宁可丢，不可留）。
- 同文去重：活跃事实（valid_to IS NULL）里已有完全相同的就跳过；
- 删除是软删（valid_to=now），与表设计的「不物理删除」一致；
- 抽取走 get_tool_llm()（qwen3:8b，temperature=0）；模型挂了/解析失败一律回落 []，
  画像失败绝不影响问答主链路。
"""
from __future__ import annotations

import json
import re

from wuwa_rag.core.authdb import get_pool
from wuwa_rag.core.llm import get_tool_llm
from wuwa_rag.knowledge.entities import CHARACTER_ALIASES, CHARACTER_NAMES
from wuwa_rag.ww_logger import get_logger

# 事实长度 / 条数硬帽
FACT_MIN_CHARS = 4
FACT_MAX_CHARS = 40
MAX_FACTS_PER_MSG = 3
# 注入 prompt 的事实条数上限（太多会挤占注意力）
MAX_FACTS_IN_PROMPT = 8

log = get_logger("profile")

_EXTRACT_SYSTEM = (
    "你从用户与游戏助手的对话里提取该用户的长期偏好事实，供后续多轮对话记住这个人的情况。"
    "① 用户常常在同一句话里既讲自己的情况、又提出请求；遇到请求只忽略请求那一部分，"
    "仍然照常提取其中关于用户自己的信息。"
    "② 可以把用户的话归纳成更短的表述，但不得写出用户没提到过的具体名字"
    "（角色名、配队名、数值）—— 凡是要落到具体名字的，那个名字必须来自用户原话。"
    "一次性问题（具体数值、某次突破或材料）不要提取。"
    "没有可提取的就返回空数组。只输出 JSON 字符串数组，不要任何解释，"
    "每条是不超过40字的短句。"
)
# prompt 措辞是**几次实测**换来的，别合并：
# - 老版本给了「（主玩角色/常用配队/……）」类别清单 + `形如：["主玩角色是守岸人"]` 具名示例，
#   等于递了份填空题：用户只说「一直在玩湮灭队、不喜欢守岸人」时，模型照模板编出
#   「主玩角色是刻晴」（刻晴是别的游戏的角色）。所以类别清单与具名示例都必须去掉。
# - 但也别只写「只提取明确说出口的、不要推断」：真实用户惯用「我一直在玩湮灭队，
#   帮我看看怎么配」这种「陈述 + 请求」句式，该措辞下模型会整句不抽。所以要**分开说**：
#   允许归纳成短句，只禁止写出用户原话里没有的**具体名字**。
# - 另外，措辞会改变模型的**输出形态**（纯字符串，或 `[{"text": ...}]` 这类对象；且没有
#   固定对应关系）。本次一度以为「召回从 18/21 掉到 6/21」是 prompt 丢了信息，打印原始
#   输出才发现元凶在下游：解析器只认字符串，把对象形态整批静默丢掉。所以**改 prompt
#   必须连解析链路一起重验**，只看 prompt 效果会得出完全错误的结论。
# 注意：本项目 prompt 一致**不写反例**（反例会被当成样本照抄，见 chain.py 的实测结论）。

# 「某个角色是这位用户的主力」这类断言。断言里的角色名必须能在用户原话里找到，
# 否则判定为模型编造。只拦角色身份，不拦配队/水平等其它类别——那些常需要归纳
# （用户说「忌炎和今汐」，事实可能是「常用配队是忌炎+今汐」），硬拦会误伤召回。
#
# 「是/为」必须写成可选：实测换个 prompt 措辞后，模型的输出从「主玩角色是守岸人」
# 变成了「主玩守岸人」——而后者才是更可能的编造形态，**不能漏**。
_ROLE_ASSERT = re.compile(
    r"(?:主玩|本命|常玩|最爱玩|专精|最常玩|主力)"
    r"(?:的)?(?:角色|干员|英雄)?"
    r"(?:是|为|：|:)?\s*([\u4e00-\u9fffA-Za-z0-9·]{2,12})"
)
# 名字前后的通用词要剥掉，否则「主玩角色和配队都很固定」会被当成角色名。
# 注意这条会**宁可多丢**：剥不完就按「无依据」丢弃——编造的代价远大于漏抽一条。
_HEAD_NOISE = re.compile(r"^(?:角色|的|和|与|及|、|配队|阵容|玩法|体系)+")
_TAIL_NOISE = re.compile(r"(?:角色|定位|流派|打法|阵容|配队|玩法|体系)$")

# 名册（rag/characters.py）：已知角色名 + 别名，含 1 字名（如「椿」）。
# 编造出来的「刻晴」「行吟诗人」都不在名册里，所以名册检查抓的是**另一类**问题：
# 事实里提到了某个真实角色、但用户根本没提过（例如用户说守岸人、事实却写成忌炎）。
# 与上面的句式正则互补 —— 正则抓「任意名字被塞进角色位」，名册抓「已知名字无依据出现」。
_KNOWN_NAMES: tuple[str, ...] = tuple(sorted(set(CHARACTER_NAMES) | set(CHARACTER_ALIASES)))


def _grounded(fact: str, question: str) -> bool:
    """事实里的角色名必须有原文依据，否则丢弃。

    编造的代价远大于漏抽一条：事实会持久化，并注入之后**每一轮** prompt 的
    「这位用户的小档案」，一条假事实会长期扭曲模型对这位用户的认识。
    因此两道检查都刻意偏严：宁可丢，不可留。
    """
    for name in _KNOWN_NAMES:
        if name in fact and name not in question:
            return False
    for m in _ROLE_ASSERT.finditer(fact):
        name = _TAIL_NOISE.sub("", _HEAD_NOISE.sub("", m.group(1)))
        if len(name) >= 2 and name not in question:
            return False
    return True


async def extract_facts_safe(question: str) -> list[str]:
    """从一条用户消息抽事实；任何异常回落 []（画像绝不挡问答）。"""
    if not question or len(question.strip()) < 4:
        return []
    q = question[:500]
    try:
        rsp = await get_tool_llm().ainvoke(
            [("system", _EXTRACT_SYSTEM), ("user", q)]
        )
        content = rsp.content
        if isinstance(content, list):
            # 某些返回形态是 [{"type":"text","text":...}] 分片
            content = "".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        text = str(content or "")
        m = re.search(r"\[.*\]", text, re.S)
        if not m:
            return []
        arr = json.loads(m.group(0))
        if not isinstance(arr, list):
            return []
        out: list[str] = []
        for item in arr[:MAX_FACTS_PER_MSG]:
            # 模型两种吐法都要吃：纯字符串，或 [{"text": "..."}] 这类对象形态。
            # 实测 qwen3:8b 的输出形态**随 prompt 措辞变化，且没有固定对应**：同一句话在
            # 一种措辞下连打 8 次全是纯字符串，换个措辞就整批变成对象，连 key 都是模型自己
            # 发明的（见过 text / value / fact / type + value 嵌套）。只认字符串的话，整批
            # 事实会被静默丢掉，表现是「聊半天画像也长不出来，日志里一条线索都没有」——
            # 这类静默失败极难排查（本次就是靠打印原始输出才定位到）。所以常见 key 都取一遍，
            # 取不到就交给下面那条 warning 留痕，绝不静默吞掉。
            if isinstance(item, dict):
                item = item.get("text") or item.get("value") or item.get("fact") or ""
            if not isinstance(item, str):
                continue
            fact = re.sub(r"\s+", " ", item).strip()
            if not (FACT_MIN_CHARS <= len(fact) <= FACT_MAX_CHARS):
                continue
            if fact in out:
                continue
            if not _grounded(fact, q):
                log.info("画像丢弃无原文依据的角色断言：%r（原话 %r）", fact, q[:40])
                continue
            out.append(fact)
        # 解析出非空数组却一条都没留下 = 形态又变了（或全被闸门拦下），必须留痕，
        # 否则下次再遇到就是又一次「静默无输出」。
        if arr and not out:
            log.warning("画像抽取结果全部未采用（形态变化或均无依据）：%r", text[:200])
        return out
    except Exception as exc:
        log.warning("画像抽取失败（不影响问答）：%s", exc)
        return []


async def save_facts(user_id: str, session_id: str, facts: list[str]) -> int:
    """去重入库。返回实际新增条数。同文活跃事实已存在则跳过。"""
    if not facts:
        return 0
    pool = await get_pool()
    added = 0
    async with pool.connection() as conn:
        for fact in facts:
            cur = await conn.execute(
                "SELECT 1 FROM user_facts WHERE user_id = %s AND fact = %s AND valid_to IS NULL",
                (user_id, fact),
            )
            if await cur.fetchone() is not None:
                continue
            await conn.execute(
                "INSERT INTO user_facts (user_id, session_id, fact, confidence, source)"
                " VALUES (%s, %s, %s, %s, 'chat')",
                (user_id, session_id, fact, 0.8),
            )
            added += 1
    if added:
        log.info("画像入库 user=%s 新增 %d 条", user_id, added)
    return added


async def get_facts(user_id: str, limit: int = 100) -> list[dict]:
    """某用户的活跃画像事实，新的在前。"""
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT id, fact, confidence, source, created_at FROM user_facts"
            " WHERE user_id = %s AND valid_to IS NULL"
            " ORDER BY created_at DESC LIMIT %s",
            (user_id, limit),
        )
        return list(await cur.fetchall())


async def soft_delete_fact(user_id: str, fact_id: int) -> bool:
    """软删（valid_to=now）。只能删自己的：带 user_id 条件。"""
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "UPDATE user_facts SET valid_to = now()"
            " WHERE id = %s AND user_id = %s AND valid_to IS NULL",
            (fact_id, user_id),
        )
        return cur.rowcount > 0


async def all_users_stats() -> list[dict]:
    """管理员视角：每个用户的画像条数（users 左连 facts，0 条也在列）。"""
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            SELECT u.username, u.role, u.created_at,
                   COUNT(f.id) FILTER (WHERE f.valid_to IS NULL) AS facts
            FROM users u LEFT JOIN user_facts f
              ON f.user_id = u.username
            GROUP BY u.id ORDER BY u.created_at
            """
        )
        return list(await cur.fetchall())


def facts_to_context(facts: list[dict]) -> str:
    """把画像事实拼成注入生成 prompt 的一段话（供 chain._build_prompt 使用）。"""
    lines = [str(f["fact"]) for f in facts[:MAX_FACTS_IN_PROMPT]]
    if not lines:
        return ""
    return "；".join(lines)
