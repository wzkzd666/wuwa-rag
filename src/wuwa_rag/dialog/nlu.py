"""意图识别 + 实体抽取。

槽位/意图分类规则优先（实测 16/16 覆盖、零漏检，无命中时 classify() 回落 hybrid
图谱+向量双跑，本身即安全兜底）。主题分类（闲聊 vs 游戏）走 qwen3:8b agent——
这类判断规则词典覆盖不全（「家人怎么这么晚才来」无任何游戏信号也无闲聊词典），
但只在规则全无信号时才调用，有信号时零额外延迟。
另有两条**规则硬信号**不经过主题分类器、直接定性：
  · `is_identity`（问「你」的台词/名字/身份）→ chitchat；
  · `is_time_question`（「现在几点」）→ time（由 chain.time_node 调 current_time 取真值）。
两者都用**原句**判（改写器会把「你」补成角色名，第二人称信号会丢）。
另有一条**弱信号** `mentions_time`（只是顺带提了时间，例如「现在几点，顺便说说今汐的
共鸣链」）：不改变路由，只让 chain 在照常检索/闲聊之外**额外**把真值时间交给模型
（state.need_time，由 chain 写入）。

注意 slots 的 key 必须与 retrievers.py 的 CYPHER / SLOT_LABEL 一一对应，
若将来接入 LLM 抽取槽位，必须用白名单过滤非法槽位名。
"""
from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from wuwa_rag.config import get_settings
from wuwa_rag.core.llm import get_tool_llm
from wuwa_rag.knowledge.entities import mentioned_names
from wuwa_rag.ww_logger import get_logger

log = get_logger("rag")

# 事实型槽位：能直接查图谱
SLOT_PATTERNS: dict[str, str] = {
    "属性":    r"属性|元素|武器类型|什么武器|性别|出生|哪里人|稀有度",
    "属性反查": r"导电|冷凝|热熔|气动|衍射|湮灭",
    "技能":    r"技能|常态攻击|普攻|共鸣技能|共鸣解放|变奏|延奏|共鸣回路|大招",
    "共鸣链":  r"共鸣链|几链|命座|一链|六链|满链",
    "突破材料": r"突破|材料|素材|需要什么|消耗|多少个",
    "声骸":    r"声骸|配装|套装|cost|COST|词条|毕业|主词条|副词条",
    "武器":    r"武器|专武|装备",
    "队友":    r"队友|组队|配队|和谁|一起|阵容|队伍",
}

# 语义型：需要文档描述支撑
SEMANTIC_PATTERNS: tuple[str, ...] = (
    r"怎么|如何|怎样|为什么|原因|理由|依据|优势|好处|队友|组队|配队|和谁|一起|阵容|队伍|思路|玩法|攻略|讲解|介绍|分析|评价|强吗|值得|机制|原理",
)

# 人格/身份类硬信号：问「你」的台词/口头禅/名字/身份，是角色人格不是游戏数值资料。
# aemeath 人设由模型自带（Modelfile SYSTEM），这类应走 chitchat 让人设自由发挥；
# 走 RAG 反而召回大段「角色故事/珍贵之物」剧情文案被整段倾倒（实测「你的台词是什
# 么」→ hybrid → 809 字剧情故事，答案里混入无关内容）。注意必须用**原句**匹配：
# 改写器会把「你」补成角色名（「你的台词」→「爱弥斯的台词」），第二人称信号丢失。
_IDENTITY_PATTERNS: tuple[str, ...] = (
    r"你的台词", r"你的语音", r"你的语录", r"你的口头禅", r"你的口头语",
    r"你是谁", r"你叫什么", r"你叫啥", r"你的名字", r"你的身份",
)

# ---------- 自我介绍硬信号 ----------
# 需求来源：用户发「我是颗粒」这类纯自我介绍，期望走**闲聊**并把昵称写进画像；
# 实际却因为 `classify()` 对「无槽位且无语义词」的句子兜底返回 `hybrid`，走了全量
# 检索 → 查不到资料 → `characters` 为空、`verify_node` 的按角色重爬分支不触发 →
# 额度耗尽落到**联网兜底**（实测复现：`verify_stage='web'`）。
#
# 与 `is_identity` 是同一类判据（规则硬信号、零 LLM、判 chitchat），只是方向相反：
#   · `is_identity`    —— 问**助手自己**的身份（第二人称「你是谁」）；
#   · `is_self_intro`  —— 陈述**用户自己**的身份（第一人称「我是X」）。
# 两者语义不重叠，共用同一个 chitchat 分支。
#
# ⚠️ 判据用**原句 q**（与 is_identity 一致）：改写器可能把「我是颗粒」改得面目全非。
# ⚠️ 昵称用**排除句读**的字符类，而不是贪婪的 `.+?`：「我是萌新，守岸人怎么玩」里
#    `.+?` 会一路吃到句末（结尾锚定允许中间有逗号），把一句真游戏提问误判成自我介绍、
#    整轮丢掉检索。排除句读后昵称在第一个逗号处就断掉，该句不再命中。
# ⚠️ **必须显式排除疑问句**：`谁` / `什么` / `吗` / `哪` 都是合法昵称字符，
#    不排除的话「我是谁」「我叫什么」「我是玩家吗」会被当成自我介绍。这类问句
#    本身也属闲聊（该走 chitchat），但混进来会让 `is_self_intro` 的语义变得含混，
#    且「我是导电属性的吗」是**真游戏提问**，绝不能被抢走。
# ⚠️ 昵称字符类要含拉丁字母与数字：「我的游戏ID是颗粒」的 ID、以及英文昵称都属常见写法。
#
# 回归用例见 `tests/test_intent_routing.py`（正负样本成组参数化）：
#   正样本 = 陈述用户自己（我是颗粒 / 我的游戏ID是颗粒 / 我是新来的 …）；
#   负样本 = 真游戏提问与疑问句（我是萌新，守岸人怎么玩 / 我是导电属性的吗 / 我是谁 …），
#   这类绝不能被抢走 —— 抢走就等于整轮丢掉检索。
# 与 `is_identity` 无重叠：你是谁 / 你叫什么 问的是助手，is_self_intro 陈述的是用户自己。
_INTRO_NICK = r"([^，。！？、；：?,!;\s]{1,12})"
# ⚠️ `_INTRO_NICK` **必须是捕获组**：`is_self_intro` 靠 `m.groups()` 把昵称取出来交给
#    下面的 `_INTRO_BAD_NICK` 过滤。写成非捕获（无括号）时 `groups()` 恒为空、
#    过滤整段变成死代码——首版就是这么错的（实测「我是谁」「我是玩家吗」全被误判成
#    自我介绍，而注释却写着「已排除疑问句」）。改正则时别把这层括号弄丢。
# 昵称位不允许出现的疑问词/虚词/属性名：出现在这里说明整句是疑问或游戏提问，不是陈述。
_INTRO_BAD_NICK = re.compile(
    r"^(?:谁|什么|啥|哪|吗|呢|吧|玩家|人|导电|冷凝|热熔|气动|衍射|湮灭)")
_INTRO_TAIL = r"[了呀呢啊吧嘛哦~～]*[？?。!！~～·.…]*$"
# 陈述句尾：只收句号/感叹号/语气词，**不收问号**（「我是玩家吗」这类要靠它挡掉）
_INTRO_TAIL_DECL = r"[了呀呢啊吧嘛哦~～]*[。!！~～·.…]*$"
_SELF_INTRO_PATTERNS: tuple[str, ...] = (
    # ① 我是X / 我叫X（可带「大家/你好」类招呼前缀与「记住」类动词）
    r"^(?:大家好|你好|哈喽|嗨|记住|那个|嗯+)?[，,]?\s*"
    r"(?:我|人家|本人)(?:就|也|还)?(?:是|叫|名叫|叫做|的?名字(?:是|叫))\s*"
    + _INTRO_NICK + _INTRO_TAIL_DECL,
    # ② 你可以叫我X / 叫我X就行 / 喊我X
    r"^(?:你可以|可以|请|就)?(?:叫我|喊我|称呼我)\s*" + _INTRO_NICK +
    r"(?:就行|吧|好了|呗)?" + _INTRO_TAIL_DECL,
    # ③ 我的名字是X / 我的ID是X / 我的游戏ID是X / 我的游戏名是X
    #    「游戏」是可加可不加的前缀，故写成 `(?:游戏)?`；实测漏掉它就是
    #    「我的游戏ID是颗粒」判不出来的原因（`游戏名` 匹配不上 `游戏ID`）。
    r"^我的(?:游戏)?(?:名字|昵称|ID|id|Id|名|网名|称呼)(?:是|叫|为)\s*"
    + _INTRO_NICK + _INTRO_TAIL_DECL,
    # ④ 我是新来的 / 我是新人 / 我是萌新（无昵称的水平自述，同属自我介绍）
    r"^(?:我|人家|本人)是(?:新来的|新人|萌新|新手|老玩家|回归玩家)" + _INTRO_TAIL,
)


def is_self_intro(question: str) -> bool:
    """整句是不是一个**纯自我介绍**（「我是颗粒」「叫我小星就行」）。

    命中即走 chitchat：这类句子的正确答案只可能来自人设闲聊 + 画像记录，
    检索游戏资料必然空手而归，而空手而归在 `verify_node` 里会一路升级到联网兜底
    （用户实测报的现象）。与 `is_identity` 同理，规则能全覆盖、零 LLM 零延迟。

    两段判定，顺序不可换：
      ① 先用 `_SELF_INTRO_PATTERNS` 抓句式（**陈述句尾**，问号已被 `_INTRO_TAIL_DECL` 挡掉）；
      ② 再取出昵称部分，用 `_INTRO_BAD_NICK` 排掉「谁/什么/吗/属性名」这类
         看着像昵称实则是疑问词或游戏术语的命中（「我是导电属性的吗」）。

    ⚠️ 复合句不命中：「我是萌新，守岸人怎么玩」在第一个逗号处就断了昵称匹配，
    仍照常走检索（用户的主诉求是问守岸人，自我介绍只是顺带）。
    """
    s = (question or "").strip()
    if not s:
        return False
    for p in _SELF_INTRO_PATTERNS:
        m = re.search(p, s)
        if not m:
            continue
        # 取句式末尾那段作为「昵称」候选：正则最后一个分组就是 _INTRO_NICK 或水平自述词。
        nick = ""
        for g in reversed(m.groups()):
            if g:
                nick = g
                break
        # ④ 分支（新来的/萌新…）没有昵称分组，`nick` 取不到，属正常命中。
        if nick and _INTRO_BAD_NICK.match(nick):
            continue
        return True
    return False

# ---- 「问用户自己是谁」（与 is_self_intro 同族的**疑问**形态）----
#
# 为什么需要：`is_self_intro` 只认**陈述句**（「我是颗粒」「叫我小星」），
# 疑问形态「颗粒是谁」一条都不命中 → 走向量检索 → 游戏库里没有这个人 → 答案只可能
# 由人设编出来。用户实测到的正是这个：问「颗粒是谁」，答的却是爱弥斯自述
# （「粉发金瞳、话多又爱笑的女孩」）。
#
# 昵称从**画像文本**里取（`facts_to_context` 的输出，形如「用户自称是颗粒」），
# 而不是硬编码任何名字：换个用户、换个昵称都自动生效，没有维护成本。
# 昵称为空（还没画像）时返回 False —— 此时判定不了也不该硬猜。
_PROFILE_NICK_RE = re.compile(
    r"用户(?:自称|的名字)?(?:是|叫做?|叫)\s*([^\s，。！？、,!?：:]{1,12})"
)
# 疑问形态：问身份/称呼。刻意**不**收「是不是」「对不对」这类是非问 —— 那是确认，
# 不是问「你是谁」。
_ASK_NICK_RE = re.compile(r"(是谁|叫什么|怎么称呼|是哪位|是什么人|什么身份|本人是谁)")


def asks_about_own_nickname(question: str, profile_context: str = "") -> bool:
    """问句是不是在问「<画像里的昵称> 是谁」——也就是在问**用户自己**。

    命中即走 chitchat：正确答案只可能来自画像（实测 `user_facts` 里存着
    「用户自称是颗粒」），检索游戏资料必然空手，而空手在 `verify_node` 里会一路
    升级到联网兜底。判据零 LLM 零延迟，与 `is_self_intro` 同族。
    """
    s = (question or "").strip()
    if not s or not _ASK_NICK_RE.search(s):
        return False
    nick = next(iter(_PROFILE_NICK_RE.findall(profile_context or "")), "")
    return bool(nick) and nick in s


# ---------- 音乐意图（纯规则，零 LLM 零延迟）----------
#
# 为什么不用 LLM 判：这些句式高度固定（放/播放/暂停/下一首…），规则能全覆盖；
# 而判错的后果很糟 —— 游戏问句「卡卡的声骸」里也有「卡」，一旦误判就会去放歌。
# 判据要求**动词或控制词明确出现**，且不含任何角色名 —— 双保险。
_MUSIC_CTRL: tuple[tuple[str, str], ...] = (
    (r"暂停|停一下|先停", "pause"),
    (r"继续播放|接着放|继续放", "play"),
    (r"下一首|换一首|下首", "next"),
    (r"上一首|前一首|回到上一", "prev"),
    (r"停止播放|关掉音乐|别放了", "stop"),
    (r"现在.?在放|在听什么|放的什么|谁在唱", "status"),
)
# 点歌：动词 + （可选量词）+ 歌名。动词必须在**句首附近**，避免「卡的声骸怎么配」这类误判。
_MUSIC_PLAY_RE = re.compile(
    # ⚠️ 动词必须**按长度降序**排列：正则交替分支是从左到右第一个命中，
    # 「播放」若写成「播」在前，「播放稻香」会被切成「播」+「放稻香」，歌名多一个「放」字。
    r"^(?:帮我|给我|你)?\s*(?:播放|放|播|来|听|唱|点|要)\s*(?:一?[首支个]\s*)?"
    r"(?P<kw>[^，。！？,.!?]{1,40})$"
)


# ---------- 「没理解」判定：该不该反问用户 ----------
#
# 为什么要这条：判定落空时（太短、纯指代、槽位与角色都没认出来）系统原先**照样硬答**，
# 于是用户看到的是一本正经的编造内容（实测：问「刚刚的任务你再试试看」，助手开始讲
# 另一个角色的故事）。反问一句成本远低于答错 —— 答错了用户还得再纠正一次。
#
# 判据刻意**保守**：只覆盖「明确没有可答内容」的情况。宁可漏判（照常答）也不可
# 误判（把正常问题反问回去），后者比前者更烦人。

# 纯指代/回指，句子里没有任何可答内容
_ONLY_REF = re.compile(
    r"^(那个|这个|它|他|她|刚刚的|刚才的|上面那个|之前那个|嗯+|啊+|额+|唔+|"
    r"再(试|来|说|问|放|唱)(一(次|遍|下|首))?|重来|继续|还是|一样|同样|再等等)")


def needs_clarification(question: str, slots: list[str] | None = None,
                         characters: list[str] | None = None) -> bool:
    """这句话是否**没有足够信息**去回答 —— 该反问而不是硬答。

    只在两种情况返回 True：
      ① 极短且识别不出槽位与角色（「卡卡呢？」这种反问，回答它需要反问）；
      ② 纯指代/回指句（`再试`、`那个`），本身不含任何可答内容。
    其余一律 False —— 宁可漏判也不误判。
    """
    q = (question or "").strip()
    if not q:
        return True
    # ⚠️ 判据只覆盖**纯指代/回指/语气词**。曾经还加过「≤6 字且无槽位角色 → 澄清」，
    # 实测立刻误判：「讲个故事」「你好」「谢谢」语义完整，却被打回去反问 ——
    # 把「短」等同于「不明」是错的：中文里大量完整意图只有 4 个字。
    # 宁可漏判（照常答），也不要把正常请求反问回去。
    if _ONLY_REF.match(q):
        return not (slots or characters)
    return False


# 回指/重试型说法：用户不是在点新歌，而是让你**再做一次刚才那件事**。
# 没有它，多轮里说「刚刚的任务你再试试看」会完全落空 —— 句子里既没有动词也没有歌名，
# 而「刚刚的任务」只有在**历史**里才解析得出来（用户实测：这个场景下音乐完全失效）。
_MUSIC_RETRY_RE = re.compile(
    r"(再试|再放|再来|重试|再唱|再听|那个|刚刚|刚才|上次|继续|还是|一样|同样|刚才那)")


def music_action(question: str, history: list[dict] | None = None) -> tuple[str, str] | None:
    """识别音乐指令，返回 `(action, keyword)`；不是音乐指令返回 None。

    action ∈ play / pause / next / prev / stop / status（keyword 仅 play 用）。
    判据是**动词锚定**：控制词必须出现；点歌句必须以播放动词开头。
    「卡卡的声骸怎么配」这类游戏问句不含这些动词，不会命中。

    `history` 用于**回指**：「刚刚的任务你再试试看」这种句子本身没有音乐动词，
    只有回指词；此时从历史里倒着找最近一条能判成音乐的用户消息，沿用它的动作与歌名。
    找不到就返回 None —— **不猜**。
    """
    q = (question or "").strip()
    if not q:
        return None
    got = _music_action_direct(q)
    if got is not None:
        return got
    if history and _MUSIC_RETRY_RE.search(q):
        for m in reversed(history):
            if m.get("role") != "user":
                continue
            prev = _music_action_direct((m.get("content") or "").strip())
            if prev is not None:
                return prev              # 沿用上一条的动作与歌名
            break                        # 只看最近一条用户消息，不跨多轮回溯
    return None


def _music_action_direct(q: str) -> tuple[str, str] | None:
    """只判当前这一句（不看历史）。"""
    if not q:
        return None
    for pat, act in _MUSIC_CTRL:
        if re.search(pat, q):
            return act, ""
    m = _MUSIC_PLAY_RE.match(q)
    if m:
        kw = m.group("kw").strip()
        # 「放首歌」「放点音乐」这类没有具体歌名 → 只报可用性，不去瞎搜
        # 「放首歌 / 放点音乐」这类**没有具体歌名**：返回 ("play", "")，
        # 上层会回「没听清要放哪首」而不是拿「点音乐」去搜 —— 宁可问一句，不要瞎搜。
        if kw and len(kw) >= 2 and not re.fullmatch(r"[点来放要]*[歌音乐节曲]+", kw):
            return "play", kw
    return None


# 时间类硬信号：问「现在几点 / 今天几号 / 星期几 / 当前日期」。
# 这类问题规则能全覆盖（问法就那么几种），答案来自服务端真实时钟（tools.current_time
# 工具），既不检索也不该让模型猜——模型没有时钟，凭空作答必然给出训练期附近的日期。
#
# 判据是**白名单 + 句末锚定**，不维护动词黑名单（黑名单永远漏，且与项目一贯口径相反）：
#   ① 「星期几 / 周几 / 礼拜几 / 几月几号」自身无歧义，任意位置命中；
#   ② 整句就是时间问法（可选礼貌前缀/锚点，时间词收尾）：「几点」「几点了」「请问现在几点」；
#   ③ 「几点 / 几号」太常见，**只有落在句末**才算问时间，且需要有「现在/今天/当前」这类
#      锚点佐证——锚点让「爱弥斯现在几点」「帮我看看现在几点」照样命中。
#      游戏提问里时间词后面总跟着动作词或名词，实测反例：「鸣潮几点刷新」「今天几点开服」
#      「卡卡罗几点上线」「日常委托几点」（无锚点）「秧秧共鸣链几号节点要多少材料」。
#      句末 + 锚点两条一起用就切掉了整类，无需枚举动词。
#      ⚠️ 锚点不能省：省了「今天有时间吗」「我今天有点累」会被误吞。
# ⚠️ 不要往里加「什么时候」：它太宽，「秧秧什么时候上线」「这个声骸什么时候出的」都是
# 游戏提问，误判会把真问题踢出检索链路。与 classify_topic 的回落口径同理——代价不对称。
# 判据用**原句**（与 is_identity 一致），改写器可能把「现在几点」改得面目全非。
_TAIL = r"[了呀呢啊吧]*[？?。!！~～·.…]*$"          # 收尾语气词 + 标点
_TIME_ANCHOR = r"(现在|此刻|这会儿|眼下|当前|今天|今日)(是|的)?(什么|多少|几)?"
_TIME_TAIL_WORD = r"(时间|日期|时刻|星期|周几|几点(几分)?|几分)"
_TIME_PATTERNS: tuple[str, ...] = (
    # ① 星期/日期问法自身无歧义，但**仍要句末锚定**：句中出现「星期几」就抢走的写法
    #    会把「今天星期几呀，长离怎么玩」整句踢进时间分支，长离那半句直接没人答。
    #    收尾锚定后，只有「…星期几(了呀)？」这种整句问法才算纯时间问题。
    r"(星期几|周几|礼拜几|几月几[日号])" + _TAIL,
    r"^(请问|家人|你)?(现在|今天|今日)?(几点|几号|几分)" + _TAIL,
    _TIME_ANCHOR + _TIME_TAIL_WORD + r"(是多少|是几|是什么|多少)?" + _TAIL,
    r"(日期|时间)[是到](多少|几|什么)" + _TAIL,
)

# 时间**提及**（比 _TIME_PATTERNS 宽一档）：句子里除了别的诉求还顺带问了时间。
# 与上面的区别只在**不要求句末收尾**——「现在几点，顺便说说今汐的共鸣链」整句不是
# 时间问法，但里面那半句是真的在问时间。
# ⚠️ 依然只收带锚点的写法，裸「几点 / 几号」一概不收（理由见 mentions_time docstring）。
_TIME_MENTION_PATTERNS: tuple[str, ...] = (
    r"(现在|此刻|这会儿|眼下|当前|如今)(是|的)?(什么|多少|几)?"
    r"(时间|日期|时刻|星期|周几|几点|几号|几分)",
    r"(今天|今日)(是|的)?(几号|几月几|星期几|周几|礼拜几|什么日子)",
    r"星期几", r"周几", r"礼拜几", r"几月几[日号]",
)

# 属性值（鸣潮共 6 种）——「导电角色有哪些」里没有「属性」二字，只有属性值
ELEMENTS = ("导电", "冷凝", "热熔", "气动", "衍射", "湮灭")
ELEMENT_RE = "|".join(ELEMENTS)
STAGE_RE = r"([一二三四五六]阶突破|[一二三四五六]阶)"


def detect_stage(question: str) -> str:
    m = re.search(STAGE_RE, question)
    return m.group(1) if m else ""


def detect_element(question: str) -> str:
    m = re.search(ELEMENT_RE, question)
    return m.group(0) if m else ""


def detect_slots(question: str) -> list[str]:
    """根据字典提取图谱关键词"""
    return [slot for slot, pat in SLOT_PATTERNS.items() if re.search(pat, question)]


def is_semantic(question: str) -> bool:
    """提取语义关键词"""
    return any(re.search(p, question) for p in SEMANTIC_PATTERNS)


def is_identity(question: str) -> bool:
    """是否在问角色人格（台词/口头禅/名字/身份），而非游戏数值资料。"""
    return any(re.search(p, question) for p in _IDENTITY_PATTERNS)


def is_time_question(question: str) -> bool:
    """是否在问当前时间/日期。

    命中即走 chain.time_node：**确定性调 tools.current_time** 取服务端真值，
    再让爱弥斯用自己的口吻说出这个时间。放在 classify_topic 之前判——主题分类器
    会把「现在几点」判成 chitchat（它确实与游戏无关），落到 chitchat_node 后模型
    对真实时间一无所知，只能含糊其辞或编一个出来。
    """
    return any(re.search(p, question) for p in _TIME_PATTERNS)


def mentions_time(question: str) -> bool:
    """句子里是否**顺带**问了时间（不要求整句都是时间问法）。

    与 `is_time_question` 的分工（两者是包含关系，纯时间问题必然也命中这里）：
      · `is_time_question` = 整句就是一个时间问法 → 走 time 分支：不检索，只调工具；
      · `mentions_time`    = 除了别的诉求还提到时间（「现在几点，顺便说说今汐的共鸣链」）
                          → 照常检索/闲聊，**另外**把服务端真值一并交给模型。

    ⚠️ 判据同样只用**带锚点**的写法，绝不放进裸的「几点 / 几号」：
    裸词会把「鸣潮几点刷新」「秧秧共鸣链几号节点要多少材料」这类真游戏问题也算成
    「问了时间」，于是往资料答案里平白塞一句「现在是…」，把刷新时刻/节点号的答案带偏。
    宁可漏（少说一句时间），不可滥（污染资料答案）——与 classify_topic 的回落口径同理。

    实现上是 `is_time_question` 的**真超集**（纯时间问题必然也命中这里），调用方据
    `is_time_question` 决定走不走 time 分支、据本函数决定要不要带真值时间。
    """
    return is_time_question(question) or any(
        re.search(p, question) for p in _TIME_MENTION_PATTERNS)


def classify(question: str, slots: list[str]) -> str:
    """根据关键词提取情况，返回预计处理方式，混合兜底"""
    has_fact, has_sem = bool(slots), is_semantic(question)
    if has_fact and has_sem:
        return "hybrid"
    if has_fact:
        return "fact"
    if has_sem:
        return "semantic"
    return "hybrid"         


def extract_characters(question: str, known: list[str]) -> list[str]:
    """返回问句里提到的角色名（支持「卡卡罗和吟霖谁更强」这类多角色提问）。

    判据统一走 `entities.mentioned_names`：**多字名**子串匹配 + 消歧（「秧秧」是
    「秧秧·玄翎」前缀，同句命中留长的）；**单字名**要求独立成词。

    ⚠️ 这里曾写 `len(n) >= 2` 过滤单字名（注释说是防「他/她」误命中），后果是
    单字角色名**心**（Hsin）与**椿**永远识别不出来：`characters` 恒空 →
    `graph.verify_node` 的「按角色清库重爬」分支永不触发 → 额度耗尽落到**联网兜底**。
    「他/她」本就是代词不在名册里，真正的单字名风险是「核心/关心」这类常用词误命中，
    已由分词判据解决（实测：心的声骸→命中；核心玩法/关心/开心/心情→不误判）。
    """
    return mentioned_names(question, known)


# ---------- 歧义角色名的 LLM 裁决（第 ④ 层；开关默认关闭，见 config.NAME_LLM_ADJUDICATION）
#
# 规则层已经处理掉绝大多数情况，剩下的疑难只有一类：**单字名**既被分词切成了独立
# token（`心算`/`心累` 明明在 jieba 词典里却仍被切成 `心/…`），句内又没有领域词证据。
# 理论上可以交给模型裁决。**实测 qwen3:8b 在这件事上不可用，两个方向都不可用**：
#
# · 问法 A「这句里的『心』指的是该角色吗」：69 条判对 47，22 条错**全是假阳性**
#   （核心属性/开心/心田/心愿/心中有数/心太软 一律答「是角色」）→ 只能确认、不能否决；
# · 问法 B「先说出该字所在的词，再判断」：12 条只对 4，错**全是假阴性**
#   （心值得练吗/帮我配个心/心/心连招 一律判成不是角色）→ 会把真阳性全杀掉。
#
# 所以默认关闭。留着它的意义是：换模型 / 改问法之后改一行配置就能启用，不必再动代码。
# ⚠️ 启用前请重跑上面两组实测，别凭感觉开：它会**否决**规则层给出的名字，
# 假阴性高的模型会把本来认对的角色一起丢掉。
_ADJUDICATE_SYSTEM = (
    "你是《鸣潮》游戏的角色名消歧器。给你一句用户提问和其中一个单字，"
    "判断这个单字在该句中指的是该游戏角色，还是普通词语。"
    "只输出一个 JSON 对象，键为 refers_to_character，值是 true 或 false。不要输出别的内容。"
)


async def veto_ambiguous_names(question: str, candidates: list[str]) -> list[str]:
    """用 tool LLM **否决**规则层给出的单字角色名。

    只处理单字名（多字名规则层几乎不会错，交给模型反而引入风险）。
    只有裁决明确说「不是角色」才移除；模型不可用、输出畸形、一切异常都保留规则层结论
    —— 裁决是增益，绝不能反过来把问答主链弄挂。
    """
    keep = list(candidates)
    for name in candidates:
        if len(name) != 1 or name not in question:
            continue
        try:
            resp = await get_tool_llm().ainvoke([
                ("system", _ADJUDICATE_SYSTEM),
                ("user", f"角色名：{name}（这是《鸣潮》里的一个角色）\n"
                         f"用户提问：{question}\n\n"
                         f"上面这句提问里的「{name}」字，指的是这个角色吗？只输出 JSON。"),
            ])
            m = re.search(r"\{.*\}", str(resp.content or ""), re.S)
            if not m:
                continue
            if json.loads(m.group(0)).get("refers_to_character") is False:
                keep = [c for c in keep if c != name]
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as exc:
            log.warning("角色名 LLM 裁决输出无法解析（保留规则结论）: %s", exc)
        except Exception as exc:  # noqa: BLE001 —— 模型不可用不该挡问答主链
            log.warning("角色名 LLM 裁决失败（保留规则结论）: %s", exc)
    return keep


# ---------- 主题分类：闲聊 vs 游戏（qwen3:8b agent）----------

_TOPIC_SYSTEM = """你是《鸣潮》问答助手的意图分类器。判断用户这句话属于哪一类，只输出一个 JSON：
{"topic": "game"} 或 {"topic": "chitchat"}
- game：询问《鸣潮》游戏内容（角色、声骸、配装、共鸣链、突破材料、技能、属性、配队、版本、玩法攻略等），或明显想查资料的问题。
- chitchat：日常寒暄、问候、情绪表达、与游戏无关的闲聊（如问好、问近况、撒娇、天气、玩笑）。
拿不准时输出 game。不要输出 JSON 以外的任何文字。"""

_TOPIC_EXAMPLES = (
    ("卡卡罗毕业配装用什么声骸", "game"),
    ("秧秧怎么玩", "game"),
    ("家人怎么这么晚才来，今天过得怎么样", "chitchat"),
    ("你好呀，今天心情不错", "chitchat"),
    ("我有点累了，陪我聊会儿", "chitchat"),
)


async def classify_topic(question: str) -> str:
    """判断主题是闲聊还是游戏提问。返回 "chitchat" 或 "game"。

    调用时机见 chain.intent_node：无角色名且无槽位/属性/阶段信号时才调，
    游戏提问大多被规则直接拦下，不承担额外延迟。解析失败一律回落 game——
    把闲聊误送 RAG 顶多答得生硬，把真问题误判闲聊则直接丢知识，代价不对称。

    少样本用单轮补全（Human 提问 / AI 答 JSON 交替），最后一轮 AI 前缀吃掉
    `{"`，模型只需续写 `"topic": "..."}` —— 比把完整 JSON 塞进 Human 消息
    省输出 token，分类实测更快。
    """
    msgs: list = [SystemMessage(content=_TOPIC_SYSTEM)]
    for q, t in _TOPIC_EXAMPLES:
        msgs.append(HumanMessage(content=f"问题：{q}"))
        msgs.append(AIMessage(content=f'{{"topic": "{t}"}}'))
    msgs.append(HumanMessage(content=f"问题：{question}"))
    msgs.append(AIMessage(content='{"'))
    try:
        resp = await get_tool_llm().ainvoke(msgs)
        txt = '{"' + (resp.content or "")
        m = re.search(r"\{.*?\}", txt, re.S)
        if not m:
            log.warning("主题分类: 输出无 JSON，回落 game：%r", (resp.content or "")[:60])
            return "game"
        topic = json.loads(m.group(0)).get("topic", "")
        return topic if topic in ("game", "chitchat") else "game"
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        log.warning("主题分类：JSON 解析失败，回落 game: %s", exc)
        return "game"
    except Exception as exc:  # noqa: BLE001 —— LLM 调用本身的异常也回落，不能阻塞问答
        log.warning("主题分类失败，回落 game: %s", exc)
        return "game"


# ---------- 追问改写：把指代残缺的追问补成自包含问句（qwen3:8b agent）----------

_REWRITE_SYSTEM = """你是《鸣潮》问答助手的查询改写器。用户的问题可能带指代（她/它/他/那个/再/那/换成/开头那位…），需要结合对话上下文把它改写成一个不需要上下文就能看懂、适合拿去检索的自包含问句。
规则：
- 只输出改写后的一句话，不要引号、不要 JSON、不要解释。
- 把「她/它/他/那位」替换成具体的角色名；省略主语的要补上。
- 上下文里有角色锚点（话题角色）时，指代优先解析为锚点角色。
- 「话题角色」按最近提及排序：「她/他/它/那位」默认指**列表第一个**（最近讨论的角色）。
- 「开头/之前/前面聊的那位」这类**远指代**，解析为**摘要**里提到的角色（摘要按谈话顺序保留角色名，「开头聊的」= 摘要里最先出现的名字），不是最近话题。
- 如果问题本身已自包含（无指代、无省略），原样输出。
- 不改写主题，不加信息，不回答问题。

示例：
上下文：
更早对话摘要：卡卡罗毕业配装推荐彻空冥雷。
话题角色: 长离、今汐
最近对话：
用户: 长离的共鸣链效果
助手: 第一链提高抗打断
问题：开头聊的那位武器推荐什么
改写为：卡卡罗的武器推荐是什么"""


def focus_anchors(history: list[dict], known: list[str]) -> str:
    """A·结构化焦点压缩：从历史里提炼「话题角色 + 聊过的槽位」微型锚点。

    零 LLM、零延迟（全现成正则）。存在的意义：rewrite_query 里每轮原文截 120
    字符，角色名出现在长回答的深处就会被截丢；锚点用全量文本提名字，不受截断影响。
    角色按**最近提及优先**排序（倒序遍历历史）——实测正序时「她」在两个角色间
    歧义，8b 会放弃改写；配合 _REWRITE_SYSTEM 的「默认第一个」规则消歧。
    """
    chars: list[str] = []
    slots: list[str] = []
    for m in reversed(history):
        for c in extract_characters(m.get("content", ""), known):
            if c not in chars:
                chars.append(c)
        for s in detect_slots(m.get("content", "")):
            if s not in slots:
                slots.append(s)
    if not chars and not slots:
        return ""
    parts = []
    if chars:
        parts.append(f"话题角色: {'、'.join(chars[:3])}")
    if slots:
        parts.append(f"聊过的方面: {'、'.join(slots[:4])}")
    return "；".join(parts)


_SUMMARY_SYSTEM = """你是对话压缩器。把「已有摘要」和「即将被遗忘的旧对话」合并压缩成一句不超过80字的会话摘要，只保留：聊过哪些《鸣潮》角色、涉及哪些方面（声骸/配队/突破/共鸣链等）、用户的偏好倾向。
- **角色名是最高优先级信息，必须逐字保留**，其次才是细节。宁可丢细节也不能丢名字。
- 只输出摘要正文，不要前缀、不要引号、不要解释。
- 旧摘要里的信息如果新对话没再提及，仍要保留（除非与新增内容冲突）。"""

_DEGRADED_MARK = "（摘要失败，话题未知）"


async def summarize_turns(evicted: list[dict], prev_summary: str) -> str:
    """B·滚动摘要：压缩将被滑出记忆窗口的轮次。

    模型选型实测：0.6b 合并多轮时会**丢角色名**（输出「讨论了声骸组合及相关话题」
    这类空话），而摘要的价值恰恰在保住名字，所以压缩器用 qwen3:8b（get_tool_llm）。
    实测首次压缩 12.9s、滚动合并 0.2s（输入短时有 KV 前缀缓存）。

    evicted 是被挤出窗口的旧轮次（调用方必须在截断 history 前取好——checkpointer
    里只有截断后的窗口，事后拿不到）。失败/超长时把 prev_summary 追加降级标记
    返回——标记的意义是让**下一轮重新压缩完整窗口**来修复（此时被压缩的原文还全在
    窗口内）；若原样吞掉，坏摘要会被一路继承、再也修不好。
    """
    s = get_settings()
    text = "\n".join(
        f"{'用户' if m.get('role') == 'user' else '助手'}: {(m.get('content') or '')[:200]}"
        for m in evicted
    )
    if not text:
        return prev_summary
    msgs = [
        SystemMessage(content=_SUMMARY_SYSTEM),
        HumanMessage(content=f"已有摘要：{prev_summary or '（无）'}\n\n旧对话：\n{text}\n\n合并摘要："),
    ]
    try:
        # tags 打标：ainvoke 内部同样产生 on_chat_model_stream 事件，且与 generate_node
        # 同属一个节点（langgraph_node='generate'），节点过滤挡不住它——实测摘要文本
        # 曾作为尾巴拼进流式答案。下游按 "wwa:summary" 标签丢弃（见 chain.ask_stream）。
        resp = await get_tool_llm().ainvoke(msgs, config={"tags": ["wwa:summary"]})
        out = (resp.content or "").strip().strip('"「」\'')
        out = out.splitlines()[0].strip() if out else ""
        if not out or len(out) > s.SUMMARY_MAX_CHARS:
            log.warning("摘要异常(%r)，标记降级待下轮修复", out[:60])
            return f"{prev_summary} {_DEGRADED_MARK}".strip()
        log.info("滚动摘要: %r（窗口外 %d 条）", out, len(evicted))
        return out
    except Exception as exc:  # noqa: BLE001 —— 摘要是增益，失败回落原摘要+降级标记，不得阻塞问答
        log.warning("摘要失败，标记降级待下轮修复: %s", exc)
        return f"{prev_summary} {_DEGRADED_MARK}".strip()


async def rewrite_query(
    question: str, history: list[dict], *, known: list[str] | None = None, summary: str = "",
) -> str:
    """追问改写：有上下文（历史或摘要）才调 LLM，无历史直接原句返回省一次调用。

    输入三路合并（A+B）：滚动摘要（窗口外的压缩记忆）+ 焦点锚点（结构化角色信号）
    + 最近 2 轮短原文。检索（意图/槽位/图谱/向量）都吃改写句——「那她配什么声骸」
    单拿去召回必落空，补出角色名后才能命中。任何失败（异常/空/过长）一律回落
    原句，改写是增益不是依赖，绝不能因它把问答弄挂。history 与展示仍用原句。
    """
    if not history and not summary:
        return question
    bits = []
    if summary:
        bits.append(f"更早对话摘要：{summary}")
    if history:
        if known:
            anchors = focus_anchors(history, known)
            if anchors:
                bits.append(anchors)
        turns = history[-4:]   # 最近两轮足够定位指代对象，也更省 token
        ctx = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content'][:120]}" for m in turns
        )
        bits.append(f"最近对话：\n{ctx}")
    msgs = [
        SystemMessage(content=_REWRITE_SYSTEM),
        HumanMessage(content="\n".join(bits) + f"\n\n原始问题：{question}\n改写为："),
    ]
    try:
        resp = await get_tool_llm().ainvoke(msgs)
        out = (resp.content or "").strip().strip('"「」\'').splitlines()[0].strip() if resp.content else ""
        if not out or len(out) > max(len(question) * 4, 60):
            log.warning("查询改写: 输出异常(%r)，用原句", out[:60])
            return question
        if out != question:
            log.info("查询改写: %r -> %r", question, out)
        return out
    except Exception as exc:  # noqa: BLE001 —— 改写是增益不是依赖，失败回落原句，不得阻塞问答
        log.warning("查询改写失败，用原句: %s", exc)
        return question
