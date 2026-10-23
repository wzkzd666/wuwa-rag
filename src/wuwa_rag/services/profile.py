"""用户画像：启用 001_init.sql 中定义的 user_facts 表。

数据流：
  用户提问 → (异步、不阻塞回答) tool 模型从消息里抽「稳定偏好事实」→ 去重入库
  下次提问 → 取该用户仍有效的事实 → 拼成 user_context 注入生成 prompt（个性化）

事实示例：「主玩角色是守岸人」「萌新，需要基础讲解」「偏爱湮灭队」。
约束：
- 只抽**稳定的长期偏好**，不抽一次性问题（「今汐几级突破」这种不能进画像）；
- 每条 ≤40 字，单次最多 3 条，防止低质量事实堆积；
- **角色名必须有原文依据**（`_grounded`，双检）：① 句式正则抓「主玩/本命/主力 + 名字」，
  ② 名册（`knowledge/entities.py` 的库内角色名 + 别名，Neo4j 实时查、带 TTL 缓存）
  抓「已知角色名无依据出现」。
  模型会照模板编造用户没提过的角色，而事实一旦入库就会注入之后每一轮 prompt，
  编造代价远大于漏抽一条，所以两道检查都偏严（宁可丢，不可留）。
- 同文去重：活跃事实（valid_to IS NULL）里已有完全相同的就跳过；
- **同类覆盖（可更新）**：称呼/昵称、玩家水平这两类天然单值的事实，新值软删旧值后入库
  （`category` 列），不会像「只增不改」那样堆出自相矛盾的多个昵称；
  主玩角色/配队等可多值的事实**刻意不覆盖**，仍叠加保留（见 `_fact_category`）；
- 删除是软删（valid_to=now），与表设计的「不物理删除」一致；
- 抽取走 get_tool_llm()（qwen3:8b，temperature=0）；模型挂了/解析失败一律回落 []，
  画像失败绝不影响问答主链路。
"""
from __future__ import annotations

import json
import re

from wuwa_rag.core.authdb import get_pool
from wuwa_rag.core.conversations import _like_pattern as conv_like
from wuwa_rag.core.llm import get_tool_llm
from wuwa_rag.knowledge.entities import CHARACTER_ALIASES, current_names
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
# 注意：本项目 prompt 一致**不写反例**（反例会被当成样本照抄，见 `dialog/prompt.py`
# 的 SYSTEM_PROMPT 注释里记录的实测结论）。

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

# 名册（knowledge/entities.py）：已知角色名 + 别名，含 1 字名（如「椿」）。
# 编造出来的「刻晴」「行吟诗人」都不在名册里，所以名册检查抓的是**另一类**问题：
# 事实里提到了某个真实角色、但用户根本没提过（例如用户说守岸人、事实却写成忌炎）。
# 与上面的句式正则互补 —— 正则抓「任意名字被塞进角色位」，名册抓「已知名字无依据出现」。
# 名册取**动态快照**（种子 ∪ 库内角色）：新角色入库后自动纳入检查范围，
# 不必回来改代码。这里同步读快照，不 await（profile 走的是同步过滤链路）。
def _known_names() -> tuple[str, ...]:
    return tuple(sorted(current_names() | set(CHARACTER_ALIASES)))


def _grounded(fact: str, question: str) -> bool:
    """事实里的角色名必须有原文依据，否则丢弃。

    编造的代价远大于漏抽一条：事实会持久化，并注入之后**每一轮** prompt 的
    「这位用户的小档案」，一条假事实会长期扭曲模型对这位用户的认识。
    因此两道检查都刻意偏严：宁可丢，不可留。
    """
    for name in _known_names():
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
    # 调用层：LLM 不可用（Ollama 没起/网络断）不能挡住问答主链。
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
    except Exception as exc:  # noqa: BLE001 —— 画像是增益，任何失败都不能阻塞问答
        log.warning("画像抽取调用失败（不影响问答）：%s", exc)
        return []

    # 解析层单独收窄：只认「输出畸形」这几类可预期异常。
    try:
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
    except (ValueError, TypeError, AttributeError) as exc:
        # 只认「模型输出畸形」这几类：JSON 解析失败（JSONDecodeError 是 ValueError 子类）、
        # 元素形态非预期导致的取值错误。其余异常属编程错误 —— 而且外层
        # `api/app._spawn_profile_task` 的 job 已有 `except Exception` 兜底（画像是
        # fire-and-forget，绝不挡问答），所以这里漏掉的异常也不会影响用户。
        log.warning("画像抽取输出无法解析（不影响问答）：%s", exc)
        return []


# ---------- 同类事实的「可更新」覆盖 ----------
#
# 需求来源（实测复现）：用户先后说「我是颗粒」「我叫小明」「其实我是阿星」，
# 三条事实全部入库并一起注入 prompt，得到自相矛盾的
# `用户自称是阿星；用户姓名：小明；用户自称是颗粒` —— 画像是**只增不改**的。
# 用户要的是「可更新的那种」：同类事实新值覆盖旧值。
#
# 实现：给 `user_facts` 加 `category` 列（见 authdb._DDL 与 pgsql/001_init.sql，
# 老库靠 `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` 幂等补列）。入库时先按 category
# 软删旧的活跃事实（`valid_to=now()`），再插新的 —— **沿用既有的软删设计**，
# 不物理删除，历史可追溯。
#
# ⚠️ 分类用**规则**而不是让抽取模型多输出一个字段：`_EXTRACT_SYSTEM` 的措辞是
# 多轮实测换来的（见下方注释），动它就得连解析链路一起重验；而分类只依赖事实文本
# 自身的形态，规则完全够用且零额外延迟、零模型不确定性。
#
# ⚠️ **只对语义上天然单值的类别做覆盖**，其余一律返回 None（= 退回叠加）：
#   · `nickname` 称呼/昵称/游戏ID —— 一个人同一时刻只有一个称呼，覆盖显然正确；
#   · `level`    玩家水平（萌新/老玩家/回归） —— 会随时间演进，新值就是当前值。
# 刻意**不**覆盖「主玩角色」「常用配队」：一个人可以有多个本命、多支常用队，
# 强行覆盖等于静默删掉用户的真实信息 —— 丢数据的代价远大于多留一条。
#
# 匹配顺序敏感：「用户自称是萌新」同时含「自称是」与「萌新」，必须先判 level
# 再判 nickname，否则会被归到昵称类，与真正的昵称互相覆盖。
_RE_LEVEL = re.compile(
    r"(?:萌新|新人|新手|新来的|老玩家|回归玩家|刚入坑|入坑不?久|玩了\s*\d+\s*年)")
# 昵称/称呼类：自称是X、名叫X、叫做X、姓名：X、称呼为X、游戏ID是X、叫我X
#
# ⚠️ **标记词表只此一份**（`_NICK_MARKERS`）：取昵称词（`nicknames_from_context`）与
#    分类（`_fact_category`）都从它构造。以前是两条各写一遍的正则，实测已经漂出两个方向：
#    · 抽取产物「用户名为小探」「用户**希望被**称呼为小满」——取词的那条认、分类的不认
#      → 事实不参与覆盖，用户改昵称时旧的仍留着（正是 2026-10-06 要修的形态）；
#    · 「主玩角色名为今汐」——分类的裸 `search` 认、取词的（锚在子句开头）不认
#      → 若照它覆盖，会把**真昵称**软删掉（错分类比漏分类严重得多）。
#    合成一份后两种形态由同一个锚点判：标记必须在子句开头（可带「用户」主语与
#    「希望/被」这类垫词），昵称词紧跟其后。
# 明确排除：这些词出现在昵称位说明它是「水平自述」而非真昵称（如「我是玩家」）。
_RE_NICKNAME_NOISE = re.compile(r"^(?:玩家|人|萌新|新人|新手|老玩家|回归玩家)$")


# 标记词表（唯一一份，见上面 `_RE_NICKNAME_NOISE` 处的说明）。
# ⚠️ 顺序由 `_nicks_by_len_desc` 排：**长的必须排在短的之前**（正则交替从左到右第一个命中）——
#    「叫我」若排在「叫」之后，「用户叫我小星」会先撞上「叫」，捕获组剩成「我小星」（实测红过一次）。
_NICK_MARKERS = (
    "自称是", "自称叫", "自称", "名叫", "名字叫", "名为", "叫做", "叫作", "被称为", "称为",
    "昵称是", "昵称为", "昵称", "称呼为", "称呼是", "称呼我", "称呼",
    "姓名是", "姓名为", "姓名", "名字是", "名字为", "名字",
    "游戏名是", "网名是", "游戏ID是", "游戏ID为", "ID是", "ID为", "id是", "id为",
    "叫我", "喊我", "叫",
)
_MARKER_ALT = "|".join(sorted((re.escape(m) for m in _NICK_MARKERS),
                              key=lambda e: (-len(e), e)))
# 判据**锚在子句开头**（可选「用户」主语 + 可选「希望/想/请 + 被」垫词），不能全文搜标记词：
# 「主玩角色名为今汐」里的「名为」不是昵称（端到端实测踩过，误剥比不剥糟）。
# `facts_to_context` 就是按「；」连接各条事实的，所以按子句切分与写入端同源。
_CTX_NICK = re.compile(
    r"^(?:用户|TA|他|她)?\s*"
    # 抽取模型实测会写「用户**希望被**称呼为小满」（2026-10-24 真链路抓到），
    # 只认紧跟主语的标记词就取不到昵称词，`unquote_mentions` 等于没接线。
    # 垫词只放这几个字，不放 `.*?`：放开了「用户主玩角色被定为今汐」也会被当昵称句。
    r"(?:希望|想要|想|愿意|请)?\s*(?:被)?\s*"
    r"(?:" + _MARKER_ALT + r")"
    r"\s*([^；;，。！!？?]{1,12})")


def _nick_word(clause: str) -> str:
    """子句里被标记词引出的昵称词；不是昵称句 / 命中泛称噪声词则返回 ''。"""
    m = _CTX_NICK.match((clause or "").strip())
    if not m:
        return ""
    nick = m.group(1).strip("是为：: ")
    return "" if _RE_NICKNAME_NOISE.match(nick) else nick


def _fact_category(fact: str) -> str | None:
    """推断一条事实的「同类覆盖」分类键；返回 None 表示不参与覆盖（叠加保留）。

    ⚠️ 宁可不分类，也不要错分类：错分类会让两条**本该并存**的事实互相覆盖，
    静默丢掉用户信息 —— 而漏分类最多只是多留一条旧事实（原行为，无害）。
    """
    s = (fact or "").strip()
    if not s:
        return None
    if _RE_LEVEL.search(s):
        return "level"
    # 分类与取词共用上面那一条锚定判据（`_nick_word`）：能取出昵称词的句子才算昵称类。
    # 用整句 `search` 的老写法会把「主玩角色名为今汐」判成昵称，进而把**真昵称**软删掉。
    return "nickname" if any(_nick_word(c) for c in re.split(r"[；;]", s)) else None


# 否定式自述：「我不主玩今汐了」「不喜欢维里奈」「没在用散华」。
# 这类话**不是**一条新的可覆盖事实，而是要把旧事实撤掉。不处理就会出现
# 「用户都说不喜欢 X 了，档案里还挂着『主玩 X』并继续注入 prompt」——
# 同类覆盖只对 nickname/level 生效（这两类同一时刻只有一个值），
# 而「主玩角色/常用配队」刻意不覆盖（一个人可以有多个本命），所以否定语义必须单独走。
# ⚠️ 刻意**不含**「换成/改成」：那后面跟的是**新的**对象（「主玩换成长离了」里的
# 长离是要留下的那个），拿它去反查删除正好删错。旧值没被说出来，安全做法是不删。
# 「没在练」「不再玩」「不太喜欢」都要认：中间那层副词/介词是可选的，
# 但**枚举**着写，不给 `.*`（给了就会把「不谋而合地选了同一支队伍」判成否定句，
# 实测这类误判会静默删掉用户的真实事实 —— 撤销比不撤销危险）。
_RE_NEGATION = re.compile(
    r"(?:不|别|没|无)(?:再|太|怎么|那么|在|想|愿意)?(?:喜欢|爱|主玩|玩|用|练|抽|做|打)")
# 对象词后面常挂语气/时态字（「不喜欢今汐**了**」），要剥掉才能拿去 LIKE。
# 顺序是**先按句读切开、再剥尾巴**：带着「了」去匹配，档案里那条「用户主玩今汐」
# 根本对不上（实测撤不掉）；而先剥尾巴会把句读后面别的句子也吞进来当对象。
_OBJ_TAIL_CHARS = "了呀呢吧啊的过啦哦噢嗯.。,，、!！~～ \t"


def negated_objects(text: str) -> list[str]:
    """一句话里**所有**被否定掉的对象词（「不喜欢今汐，也不想练散华」→ 两个）。

    对象词过一遍 `CHARACTER_ALIASES` 归一（卡提 → 卡提希娅）：档案里存的是全名，
    拿简称去 LIKE 会命中「卡提希娅」这条**更长的名册名**，下面的前缀判据会把该撤的
    判成「认不出是谁」而整批不撤。归一放在这一处，两个消费方（撤销与拦截正面事实）
    拿到的都是同一个口径。
    """
    out: list[str] = []
    for m in _RE_NEGATION.finditer(text or ""):
        obj = _object_after(m.end(), text)
        obj = CHARACTER_ALIASES.get(obj, obj)
        if obj and obj not in out:
            out.append(obj)
    return out


def _object_after(pos: int, text: str) -> str:
    """从否定动词之后截出对象词；截不出来返回 ''。

    只做字符级切分，不碰名册也不叫模型：无论输入是用户原话还是抽取出来的一句短陈述
    （「用户说不喜欢今汐了」），否定动词后面那一段就是对象。
    宁可返回 ''（什么都不撤），也不要去猜 —— 猜错就是静默删掉用户真实信息
    （本模块的第一条老规矩：宁可不分类/不覆盖，不可错覆盖）。
    """
    tail = (text or "")[pos:].strip()
    if not tail:
        return ""
    # ⚠️ 先切标点、后剥语气字：顺序反了会把「今汐了」整个当对象（实测 LIKE 撤不掉，
    # 因为档案里那条写的是「主玩今汐」）。切开之后才看得清真正的名字。
    obj = re.split(r"[，。！？、；;\s]", tail)[0].strip(_OBJ_TAIL_CHARS)[:8]
    # 下限 2 字是刻意的：单字名（椿/心）走 LIKE 会连带命中「鉴心」「开心」这类，
    # 撤错比不撤危险。宁可不撤销（多留一条旧事实，无害），也不误撤（静默丢信息）。
    return obj if len(obj) >= 2 else ""


def negated_object(fact: str) -> str:
    """单句版本：取被否定的那一个对象词（多个对象时只看第一个）。"""
    got = negated_objects(fact)
    return got[0] if got else ""


def _prefix_collides(fact: str, obj: str) -> bool:
    """`obj` 在这条事实里**每一次**出现都只是某个更长名册名的前缀 → 判不出用户说的是谁。

    名册里 `秧秧` 与 `秧秧·玄翎` 是**两个角色**。用户说「不喜欢秧秧」，而档案里那条写的是
    「用户主玩秧秧·玄翎」——裸 `fact LIKE '%秧秧%'` 会把它一起软删，撤错的是另一个角色。
    本模块的口径是「宁可不撤（多留一条旧事实，无害），不可误撤（静默丢用户真实信息）」，
    所以这种一律不动。「用户主玩秧秧」里秧秧独立出现，照撤。
    同理兜住「不喜欢漂泊者」对「主玩漂泊者-男-衍射」（四个属性版本都在名册里，指代不明）。
    """
    longer = [n for n in _known_names() if n != obj and n.startswith(obj)]
    if not longer:
        return False
    pos = fact.find(obj)
    if pos < 0:
        return False
    while pos >= 0:
        if not any(fact.startswith(n, pos) for n in longer):
            return False        # 这一处是 obj 自己，不是长名的前缀
        pos = fact.find(obj, pos + 1)
    return True


async def _retract_matching(conn, user_id: str, obj: str, fact: str) -> int:
    """软删该用户「提到 `obj` 且不是否定式」的活跃事实，返回条数。

    读出来在 Python 里判否定式，而不是在 SQL 里写 `NOT LIKE '%不%'`：
    后者是第二套否定判据，与 `_RE_NEGATION` 迟早漂移（会把「不谋而合」这类误判）。
    同理，「obj 是不是真的指这个角色」也在这儿判（`_prefix_collides`）——LIKE 只管粗筛。
    """
    cur = await conn.execute(
        "SELECT id, fact FROM user_facts WHERE user_id = %s AND valid_to IS NULL AND fact LIKE %s",
        (user_id, conv_like(obj)),
    )
    rows = await cur.fetchall()
    doomed = [r["id"] for r in rows
              if r["fact"] != fact
              and not _RE_NEGATION.search(r["fact"])
              and not _prefix_collides(r["fact"], obj)]
    if doomed:
        await conn.execute("UPDATE user_facts SET valid_to = now() WHERE id = ANY(%s)", (doomed,))
        log.info("画像否定撤销 user=%s 对象=%s 软删 %d 条", user_id, obj, len(doomed))
    return len(doomed)


async def save_facts(user_id: str, session_id: str, facts: list[str],
                     utterance: str = "") -> int:
    """去重入库，**同类覆盖**。返回实际新增条数。

    三道处理，顺序不可换：
      ① 同文去重：活跃事实里已有完全相同的就跳过（连 `created_at` 都不刷新，
         避免用户重复说同一句话就把事实顶到列表最前）；
      ② 同类覆盖：`category` 非空时，先把该用户同 category 的活跃事实软删
         （`valid_to=now()`）—— 这就是「可更新」；`category` 为 None 则跳过本步，
         行为与旧版一致（叠加）；
      ②b 否定撤销：句子是否定式（「不喜欢今汐了」）时，把**提到同一对象**的
         肯定式活跃事实软删（`negated_object` / `_retract_matching`）。
         为什么②不够：nickname/level 同一时刻只有一个值，覆盖天然正确；
         而「主玩角色/常用配队」刻意不参与覆盖（一个人可以有多个本命），
         于是「我不主玩 X 了」以前**什么都不会发生**，档案里那条继续注入；
      ③ 插入新事实。

    参数 `utterance` = **用户这一轮的原话**。撤销必须以它为主：实测抽取器对否定句
    三种表现都有（`我不喜欢今汐了，之前说主玩今汐` → 只回正面的「主玩今汐」、
    `我没在用莫宁，换成琳奈了` → 只回「使用角色琳奈」、`我最近不主玩维里奈了` →
    正确回否定句）。只看抽取结果就等于**六成场景撤不掉**。有了原话，
    正面那条也会被顺手拦下（下面 skipped 那段），不必去改抽取提示词 ——
    试过加「撤回也要成为一条」的规则，实测反而把否定抽取整批压成空数组（4/6 变 0），
    已回退；这条经验记在 AGENTS「画像」一节，别用同样的措辞再试一遍。
    """
    # 光「facts 为空」不足以提前返回：抽取器实测会把整句否定丢成空数组，
    # 而这时用户原话里的「不喜欢 X」仍然要求撤掉旧事实。
    if not facts and not negated_objects(utterance):
        return 0
    pool = await get_pool()
    added = 0
    replaced = 0
    retracted = 0
    skipped = 0   # 抽取器回成正面、被原话里的否定拦下的条数
    async with pool.connection() as conn:
        # 先按**用户原话**里的否定对象撤旧：抽取器可能已经把否定丢掉了（见 docstring），
        # 这时只有原话还留着「不喜欢今汐」这个信号。
        objs = negated_objects(utterance)
        for o in objs:
            retracted += await _retract_matching(conn, user_id, o, "")
        for fact in facts:
            cur = await conn.execute(
                "SELECT 1 FROM user_facts WHERE user_id = %s AND fact = %s AND valid_to IS NULL",
                (user_id, fact),
            )
            if await cur.fetchone() is not None:
                continue
            if objs and not _RE_NEGATION.search(fact) \
                    and any(o in fact and not _prefix_collides(fact, o) for o in objs):
                # 这句已经把 X 撤掉了，抽取器却还回一条正面的「主玩 X」——正面那条不许入库
                # （`_prefix_collides` 同一份判据：「不喜欢秧秧，主玩秧秧·玄翎」里
                #   后半句讲的是**另一个角色**，它是新信息，不能被前一个对象词顺带拦掉）
                skipped += 1
                log.info("画像否定撤销 user=%s 丢弃正面事实 %r（对象 %s 已被否定）",
                         user_id, fact, "、".join(objs))
                continue
            cat = _fact_category(fact)
            if cat:
                cur = await conn.execute(
                    "UPDATE user_facts SET valid_to = now()"
                    " WHERE user_id = %s AND category = %s AND valid_to IS NULL",
                    (user_id, cat),
                )
                if cur.rowcount:
                    replaced += cur.rowcount
                    log.info("画像同类覆盖 user=%s category=%s 软删旧事实 %d 条",
                             user_id, cat, cur.rowcount)
            # ②b 否定撤销：「不喜欢今汐了」要把「主玩今汐」那条撤掉。
            # 放在插入**之前**：新事实自己也在待插入队列里，先删干净再插，
            # 否则同一句话刚插进去就被自己的对象匹配删掉（否定式那条要留着——
            # 「用户不喜欢 X」本身就是有用的口吻线索）。
            obj = negated_object(fact)
            if obj:
                retracted += await _retract_matching(conn, user_id, obj, fact)
            await conn.execute(
                "INSERT INTO user_facts (user_id, session_id, fact, category, confidence, source)"
                " VALUES (%s, %s, %s, %s, %s, 'chat')",
                (user_id, session_id, fact, cat, 0.8),
            )
            added += 1
    if added:
        log.info("画像入库 user=%s 新增 %d 条%s%s%s", user_id, added,
                 f"（覆盖旧 {replaced} 条）" if replaced else "",
                 f"（否定撤销 {retracted} 条）" if retracted else "",
                 f"（拦下正面事实 {skipped} 条）" if skipped else "")
    return added


async def get_facts(user_id: str, limit: int = 100) -> list[dict]:
    """某用户的活跃画像事实，新的在前。

    `category` 一并返回：`/profile` 端点会把它透给前端，排查「同类覆盖是否生效」
    时不必再直连库。为 None 表示这条不参与覆盖（可多值事实，如主玩角色）。
    """
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT id, fact, category, confidence, source, created_at FROM user_facts"
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
            SELECT u.id, u.username, u.role, u.created_at,
                   COUNT(f.id) FILTER (WHERE f.valid_to IS NULL) AS facts
            FROM users u LEFT JOIN user_facts f
              ON f.user_id = u.username   -- ⚠️ 画像按**用户名**挂，不是 users.id（列名叫 user_id 但存的是名字）
            GROUP BY u.id ORDER BY u.created_at
            """
        )
        return list(await cur.fetchall())


def facts_to_context(facts: list[dict]) -> str:
    """把画像事实拼成注入生成 prompt 的一段话。

    两处消费：`prompt.build_prompt`（generate_node 主链路）与 `graph._chat_turn`
    （闲聊/时间分支的注入）。category 是给 `save_facts` 的覆盖逻辑用的，
    **不参与**拼接（模型不需要知道分类）。
    """
    lines = [str(f["fact"]) for f in facts[:MAX_FACTS_IN_PROMPT]]
    if not lines:
        return ""
    return "；".join(lines)


# ---------- 输出侧配合：从注入串里取昵称 ----------
# 模型叙述用户时爱给称呼套上「」/“”（实测答案「颗粒」刚刚让我放了…），
# 而画像事实本身是干净的（`用户自称是颗粒`）。按提示词铁律不写「别加引号」
# 的负向指令（negative-example 污染），改由 text.unquote_mentions 确定性剥离，
# 这里负责告诉它**剥谁的**。
#
# 标记词表与锚定判据在文件上半部（`_NICK_MARKERS` / `_nick_word`）——
# 与 `_fact_category` **共用同一条**，不在这里写第二份。
def nicknames_from_context(ctx: str) -> list[str]:
    """从画像注入串（`facts_to_context` 的输出）里取昵称词。取不到返回空表。"""
    out: list[str] = []
    for clause in re.split(r"[；;]", ctx or ""):
        nick = _nick_word(clause)
        if nick and nick not in out:
            out.append(nick)
    return out


async def purge_user_facts(username: str) -> int:
    """硬删该用户的全部画像事实（账号被移除时回收）。

    ⚠️ 参数是**用户名**不是 users.id：`user_facts.user_id` 这列名存实亡，
    库里实测存的是 `'admin'` / `'e2eprobe4385'` 这样的**用户名**，
    而 `answer_feedback.user_id` 存的是字符串化的 `users.id`。
    同一个 schema 里两套身份，传错**不会报错、只会一条都删不掉** —— 所以签名按真实语义命名。

    与 `soft_delete_fact` 分开是有意的：软删是「这条不再注入 prompt，但历史还在」，
    而账号已经不存在时留着「用户自称是××」没有任何用途，
    只会变成一堆永远关联不上主人的文本（实测库里就躺着已删用户的会话与画像）。
    """
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute("DELETE FROM user_facts WHERE user_id = %s RETURNING id", (username,))
        return cur.rowcount
