"""Prompt 构建：提示词常量、上下文拼接、最终 prompt 组装。

从 dialog/graph.py 拆出，职责边界：
  - _SYSTEM：本地 aemeath 的 HumanMessage 前缀（术语对照 + 作答要求）
  - _build_context：把 state 里的图谱事实 / 文档 / 联网结果拼成上下文
  - _build_prompt：在上下文基础上组装最终 prompt（含输出格式约束）
  - doc_sources：从召回文档提取引用来源面包屑
  - _lock_focus：文档侧的「问谁锁谁」收窄

graph.py 只负责图编排与节点调度，不再直接处理 prompt 字符串。
"""
from __future__ import annotations

from wuwa_rag.config import get_settings
from wuwa_rag.dialog.state import RagState
from wuwa_rag.text import chunk_text, lock_focus
from wuwa_rag.ww_logger import get_logger

log = get_logger("rag")

# 注意：人设由 aemeath 模型自带（Modelfile 的 SYSTEM），这里不再引导口吻——
# 叠加人设指令会让模型把注意力放在「表演」而非「答题」上，是复读独白的诱因之一。
# 本提示词只负责三件事：术语对照、答题约束、防重复。
# 另：Ollama 的 system 参数会覆盖 Modelfile 内置 SYSTEM，所以这些约束必须拼在
# HumanMessage 里，不能改成 SystemMessage 传，否则人设会丢。
#
# 硬约束：本提示词只描述正面要求，不要在示例中给出反例（哪怕是否定句式）。
# aemeath 会把提示词里点名的反例**当成要模仿的样本照抄**——即 negative-example
# contamination。三条实证：
#   ① 原文写了「不要自行补充『没有提到其他/更多』」→ 输出结尾**稳定**出现
#      「资料里没有提到其他组合啦。」（用户现场 + 本地复现各命中，字面级一致）。
#   ② 原文写了「不许给条目编序号或计数器（如『守岸人 + 尤诺*5』）」→ 输出开始出现
#      `[1]`~`[7]` 形式的来源编号噪声。
#   ③ 原文写了「开场就是『我呀~』」→ 该开场概率性复现。
# 对照实验（直连 Ollama、固定 seed=42、同一资料）：
#   · prompt 里**点名** `[1] [2]` 禁止 → 仍写 `[1]`；放在末尾时**更糟**（自编到 `[5]`）。
#   · 改用**泛化措辞**「不要标注来源序号或引用标记」→ 完全不出现（基线组则写 `[1][2]`）。
#   · 纯问题、不给资料的基线组**不会**写 `[n]` → 说明 `[n]` 是「有资料可依」这件事诱发的
#     模型微调习惯，不是它天生爱写；也**不是**语料里的 `[图]` 诱发（去掉方括号照样写）。
# 反例与历史记录请写进项目文档，**不要进 _SYSTEM**。
#
# 补充：「家人问的是哪位，那位所在的那个『或』组只列他」这条**不在
# _SYSTEM 里写**（写过，实测完全不生效：问守岸人时答案照旧输出 `守岸人 / 维里奈 / 白芷`）。
# 它已改由 `text.lock_focus` 在**数据侧确定性落实**：图谱（retrievers.graph_search）与
# 参考文档（chain._lock_focus）进 prompt 之前，含本次角色的「或」组就已收窄成他本人。
SYSTEM_PROMPT = """依据下面的资料作答。你是爱弥斯（《鸣潮》里那个爱笑、话多的女孩），资料是
存档记录，要用你自己的口吻把它讲给家人听。

术语对照（资料用词和提问可能不同，按下表理解）：
- 声骸 = 角色装备，资料里也写作「套装」；COST 是声骸的费用点数组合
- 共鸣链 = 相当于命座，序号 1~6 对应一链到六链
- 贝币 = 游戏货币
- 突破阶段：一阶~六阶
- 配队写法「守岸人+吟霖/长离/散华+卡卡罗」：`/` 之间是「或」（同一个位置三选一，
  不是三个人一起上），`+` 之间是「和」（不同位置）。鸣潮一支队伍只有 3 个人。

作答要求：
- 讲到爱弥斯本人时用第一人称「我」，活泼亲切；讲到其他角色时用角色名或「她/他」称呼。
- 先用自己的话把要点讲清楚，再按下面的格式要求列数据。
- 用中文，简洁、要点化；资料里有几个要点就讲几个要点。
- 凡是资料里带单位或符号的数字（`%`、`+`、`*`、`×`、`倍`、`秒`、`层`、`点`），
  一律照原样抄回：百分号必须跟着数字，不省略、不换算、不改成中文数字、不擅自加
  「万」「亿」这类量纲。
- 事实严格依据资料，只讲资料里有的内容；资料里没有的，用一句话说不知道就停住。
- 同一句话、同一段落、同一口头禅只说一次，讲完即止。
- 列表、配队这类条目每条只出现一次，直接平铺列出即可。
- 直接把内容讲出来即可，不要给内容加来源序号或引用标记。"""


def doc_sources(docs: list[dict]) -> list[str]:
    """从召回文档里提取「引用来源」面包屑，给前端折叠面板用。

    格式 `角色 › 模块 › 组件[ › 页签]`（chunker 写入时生成，见 ingest/chunker.py），
    同一来源只留一条、保持召回顺序。**只回面包屑字符串、不回全文**：SSE 体积可控，
    前端要的也只是「这条答案查了哪几页」，用于建立信任与排查。
    """
    out: list[str] = []
    seen: set[str] = set()
    for d in docs:
        bc = (d.get("breadcrumb") or "").strip()
        if not bc or bc in seen:
            continue
        seen.add(bc)
        out.append(bc)
    return out


def _lock_focus(text: str, characters: list[str]) -> str:
    """把正文里含「本次问到的那位角色」的配队「或」组收窄成他本人（见 text.lock_focus）。

    图谱侧已在 graph_search 里就地锁过；这里补**参考文档**侧——两处形态必须一致，
    否则图谱给 `守岸人+吟霖/长离/散华+卡卡罗`、文档给 `守岸人/维里奈/白芷+吟霖/长离/散华+卡卡罗`，
    8B 会挑文档那份抄回去，锁定等于白做（这正是「提示词规则 + 文档原样」组合失效的原因）。
    """
    out = text
    for c in characters:
        if c:
            out = lock_focus(out, c)
    return out


def build_context(state: RagState, extra: list[str] | None = None) -> str:
    """把 state 里的图谱事实 / 文档 / 联网结果拼成上下文。

    extra 是确定性补料（满级数值表 / 突破材料表），排在最后紧贴 ## 问题。
    """
    setting = get_settings()
    parts: list[str] = []
    # 注意：不要把 context_summary 塞进生成上下文——实测 aemeath 会把摘要句
    # 原样复述进答案（「…讨论声骸选择及毕业配装…」这种第三人称腔调穿帮）。
    # 摘要只喂给 rewrite_query 消解指代；生成侧靠窗口内 history + 改写后的检索结果。
    history = (state.get("history", []))[-setting.MAX_HISTORY_TURNS * 2:]
    if history:
        parts.append("## 对话历史\n" + "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}" for m in history
        ))
    if state.get("graph_facts"):
        parts.append("## 图谱事实\n" + state["graph_facts"])
    docs = state.get("docs") or []
    if docs:
        # 变更说明：早期给每份资料加 [1] [2] 前缀以引导溯源，实测模型会把
        # 把它当引用标记**抄进正文**——「- 守岸人 + 尤诺 [1][4]」「- 莫宁 + 琳奈 [1][5]」，
        # 答案尾部因此混入纯数字噪声（三轮复现稳定出现）。前端引用面板走的是 SSE
        # `done.sources`（`doc_sources`），**不依赖模型在正文写 `[n]`**，去掉零损失：
        # 每块自带 breadcrumb 开头，来源照样可辨。
        # 若将来真要做「正文引用徽章」，请改用 `（资料1）` 这类非方括号标记，别退回 `[n]`。
        # 同时在这里做「问谁锁谁」的文档侧收窄（见 _lock_focus）：图谱与文档必须同形。
        focus = [c for c in (state.get("characters") or []) if c]
        parts.append("## 参考文档\n" + "\n\n".join(
            _lock_focus(chunk_text(d), focus) for d in docs))
    web_facts = state.get("web_facts") or ""
    if web_facts:
        parts.append("## 联网搜索资料\n（实时搜索结果，本地资料不足时以本段为准）\n" + web_facts)
    # 本问解析：把**已经确定**的角色与话题显式告诉模型。
    #
    # 为什么必须显式写：用户问「心声骸推荐」时，模型会把「心声骸」当成一个完整词
    # （用户实测：答成「我先替你看看那位朋友的心声骸」，把「心」吃掉了）。
    # 而角色与槽位在检索侧本来就是**确定性**解出来的（find_mentions + detect_slots），
    # 直接写进上下文就不必再让模型自己猜分词 —— 「心」是角色、「声骸」是话题。
    #
    # 三行各自可缺，随这一轮实际情况变化，缺哪行就不写哪行：
    #   · 角色：认出来了才写（没认出来就不写，不硬凑）；
    #   · 要问的内容：取问句槽位；
    #   · 召回资料所属板块：问句没槽位时，改为反映**实际召回的是什么**，
    #     板块名直接来自 `chunks.component`，不写死。
    # 三行都没有 → 整段不生成（闲聊轮次不会出现空标题）。
    focus_chars = [c for c in (state.get("characters") or []) if c]
    focus_slots = [s for s in (state.get("slots") or []) if s]
    comps: list[str] = []
    for d in (state.get("docs") or []):
        c = str(d.get("component") or "").strip()
        if c and c not in comps:
            comps.append(c)
    rows: list[str] = []
    if focus_chars:
        rows.append("- 被问到的角色：" + "、".join(focus_chars))
    if focus_slots:
        rows.append("- 要问的内容：" + "、".join(focus_slots))
    if comps:
        rows.append("- 召回资料所属板块：" + "、".join(comps[:4])
                    + ("…" if len(comps) > 4 else ""))
    if rows:
        parts.append("## 本问解析\n（下面这些已由检索侧确定，直接照它作答）\n" + "\n".join(rows))
    # 确定性补料排在最后（紧贴 ## 问题）。真凶其实不在位置：实测把这张满级数值表
    # 放在 ## 资料 第一节时，aemeath 会「只抄前几行就收尾」并补一句「其他参数未在该
    # 列表中」，换表格/纯文本/编号/中文数值都无效——根因是 repeat_penalty=1.3 把彼此
    # 高度相似的表行罚到写不下去（降到 1.15 即 7 行全出，见 config.LLM_REPEAT_PENALTY）。
    # 位置只是第二道保险：同样的坏参数下，块放末尾确实能抄全，放开头会截断。
    for b in extra or ():
        parts.append(b)
    # 「有没有资料」必须看图谱事实/文档/联网结果/确定性补料，不能用 join 结果是否为空
    # 来判断：多轮对话时 history 非空，join 永远有内容，原先的 or 兜底就永远不触发，
    # 零资料信号被吞掉 → 模型收不到约束 → 退化成自由发挥（人设独白 + 复读）。
    if not (state.get("graph_facts") or docs or web_facts or extra):
        # 这里不要再写「## 资料」标题：build_prompt 已经加了，重复标题会干扰模型
        # 约束：只要求「说明你不清楚」会得到公文式干瘪回答，因此这里明确要求用自己的口吻，
        # 「不知道」这条分支会整体丢失角色口吻。明确要求**用自己的口吻**说，并给一个
        # 口径示例，人设才在「答不出来」这条路上也保住。
        parts.append(
            "（本次没有检索到任何资料。请用你自己的口吻、一句话说明你手里没有这份记录——"
            "可以俏皮一点、带点小遗憾（就像「这个我记不太清了呀……」），然后立刻停住；"
            "不要解释检索过程，不要重复这句话，不要补充任何其他内容。）"
        )
    return "\n\n".join(parts)


def build_prompt(context: str, question: str, blocks: list[str] | None = None,
                 characters: list[str] | None = None,
                 team_focus: bool = False,
                 user_context: str = "",
                 now: str = "",
                 cloud: bool = False) -> str:
    """拼最终 prompt。

    参数 cloud 决定人设来源（见 persona.py 模块文档）：
      - 本地 aemeath（cloud=False）：人设烧在 Modelfile SYSTEM 里，所以 `SYSTEM_PROMPT`
        只补「资料是存档记录、用你的口吻讲」这句衔接 + 术语对照 + 作答要求，
        整条塞进 HumanMessage（**不能**另发 SystemMessage，会覆盖人设）。
      - 云端通用模型（cloud=True）：模型不认识爱弥斯，人设改由调用侧
        `SystemMessage(persona.cloud_system())` 注入，故这里**不再**前缀 `SYSTEM_PROMPT`
        （否则术语表/作答要求会重复两遍，白白吃 token 且稀释指令）。
      两条路径的 `## 资料 / ## 问题 / tail / 输出格式` 部分完全一致。

    参数 now 非空表示「这句里顺带问了时间」（intent.mentions_time + chain._now_if_needed）：
    它是一句**附带**信息，所以放近因位、与 user_context 同级，措辞明确只要求顺带提一句。
    """
    # 不在这里注入 /no_think：实测它对 aemeath 无效（仍 38s + 'v' 泄漏前缀 + 触发
    # Ollama 500）。思考模式由 llm.py 的 .bind(think=False) 统一关闭。
    prefix = "" if cloud else f"{SYSTEM_PROMPT}\n\n"
    prompt = f"{prefix}## 资料\n{context}\n\n## 问题\n{question}"
    # 人称硬要求压在 prompt **最末**（近因位）：SYSTEM_PROMPT 里那条通用规则实测压不住——
    # 同一条 prompt 换 seed 重跑，第一人称开场仍会**概率性**冒出来（把被问的角色
    # 当成了自己）。点名具体角色 + 放末尾，比在长 SYSTEM_PROMPT 里写通用规则强得多。
    # 问爱弥斯自己（characters 只含「爱弥斯」）时不加，否则会把人设本身顶掉。
    others = [c for c in (characters or []) if c and c != "爱弥斯"]
    # 这两条硬要求必须压在 prompt **最末**（近因位），写进前面的 SYSTEM_PROMPT 等于白写——
    # 实测：同一条「不要标注来源序号或引用标记」放在 SYSTEM_PROMPT 里，配队答案照样出现
    # `[1]`~`[10]`；末尾泛化措辞则完全不出现。
    # 约束：措辞必须保持泛化，点名 [1] [2] 反而会诱发编号（置于末尾时更明显）。
    tail = ("\n\n## 表达（硬要求）\n"
            "资料没有编号，直接陈述内容即可，不要标注来源序号或引用标记。")
    if others:
        tail += ("\n本次问的是「" + "、".join(others) + "」，不是你——"
                 "正文里一律用角色名或「她/他」称呼，不要冒充成她。")
    if team_focus:
        # 指名具体队伍时（见 _named_team，≥3 个角色名）：把注意力钉在那一支上。
        # 约束：措辞用「围绕这一支展开」而非「只介绍这一支」，原因见本文件前述说明：
        # 「只」字会让模型把介绍性口吻整个砍掉、退化成机械罗列，人设直接丢失。
        tail += ("\n用户已经点名了一支具体队伍，本轮就围绕这一支展开："
                 "成员是谁、怎么打（出手顺序 / 循环）、为什么这么配。")
    if user_context:
        # 用户画像（rag/profile.py，user_facts 表）：注入到近因位。措辞是「可参考」
        # 而非硬要求——画像只是个性化佐料，答错资料比忽略画像严重得多；且不写
        # negative-example（本文件铁律：反例会被当成样本照抄）。
        tail += ("\n\n## 这位用户的小档案\n"
                 f"{user_context}\n"
                 "回答时可以自然贴合这位玩家的情况（比如TA主玩的角色、熟悉程度），"
                 "与资料冲突时以资料为准。")
    if now:
        # 顺带问了时间（见 intent.mentions_time / chain._now_if_needed）：
        # 服务端真值放近因位，与「小档案」同级——它只是附带一句，不能挤掉资料主体。
        # 措辞只写「怎么做」（本文件铁律：不写否定式反例，模型会照抄反例）。
        tail += ("\n\n## 现在的时间（服务端真实时钟）\n"
                 f"{now}\n"
                 "家人这句话里也问了现在的时间，讲资料的同时顺带把它说一句即可。")
    blocks = blocks or []
    # 指令必须压在 prompt **末尾**：SYSTEM_PROMPT 里那条规则实测只能让模型「带上几个数」，
    # 面对长表仍会概括成「各需不同数量」而不逐行列（实测）。近因位置 + 点名禁止的
    # 采用保守写法，才能让它回到逐行照抄的状态。
    #
    # 注意：不要写成「只照抄那两张表」，该措辞有两个反作用——
    #  ①「只」字会让模型把技能介绍/人设口吻整个砍掉，退化成纯数据倾倒（实测：
    #    人设丢失、答案变成机械罗列）；
    #  ② 点名「突破材料表」会让模型以为该有材料表，于是跑去「## 参考文档」里翻材料
    #    表一起列出来——问技能却蹦出材料就是这么来的（实测）。
    # 正确写法：先保住「说人话的介绍」，再只点名**本轮真正补了的那几张表**，
    # 并显式禁止主动扩列没被问到的内容。
    #
    # 变更说明：早先这里有个 `if not blocks: return prompt + tail` 的短路，导致
    # **无补料场景（共鸣链、剧情、机制问答）拿不到任何格式约束** —— 六链答出
    # 「暴击伤害80万」那次正是这个场景（blocks 为空，输出格式块整块没进 prompt）。
    # 现在无论有没有补料都要注入：表格条目按需增减，正文那条（%）是**无条件**的。
    has_value = any("满级数值表" in b for b in blocks)
    has_mat = any("突破材料表" in b for b in blocks)
    lines = ["\n\n## 输出格式（硬要求）"]
    lines.append("1) 先用你自己的口吻把内容讲清楚（这是什么、怎么打、什么手感），正常说话，"
                 "不要只丢数字，也不要写成机械报表；")
    step = 2
    if has_value:
        lines.append(f"{step}) 然后把「满级数值表」里每一行都列出来，写成「- 名称：数值」，"
                     "一行都不许省；")
        step += 1
    if has_mat:
        lines.append(f"{step}) 再把「突破材料表」里每一条都列出来，写成「- 材料名×数量」，"
                     "一条都不许省；")
        step += 1
    # 非表格内容（共鸣链、技能描述、声骸词条都属这块）单独点名：原文是「暴击固定为
    # 80%，暴击伤害固定为275%」这种带百分号的散文，缺了这条约束就会被复述成
    # 「八十万暴击伤害」「两百七十五」（实测，见 text.fix_percent_units）。
    # 表述只写「怎么做」，不写任何反例（本文件铁律：反例会被当成样本照抄）。
    lines.append(f"{step}) 讲正文（共鸣链、技能、声骸词条等内容）时，原文里的数字连同它"
                 "后面的单位一起照抄过来，百分号、加号、乘号、倍、秒、层这些都不省略，"
                 "也不要把数字换成另一种写法。")
    lines.append("禁止写成「各需不同数量」「材料如上」「数值都在资料里」这类概述，"
                 "禁止换算、合并、四舍五入或改成中文数字。")
    if has_mat:
        lines.append("「## 参考文档」里分等级展开的长表不要照抄，更不要把不同等级、不同技能的"
                     "材料拼成一张混在一起的清单——数值与材料一律以上面两张表为准，"
                     "且不要主动列出没被问到的内容。")
    elif has_value:
        lines.append("「## 参考文档」里分等级展开的长表（尤其是各种材料表）一律不要照抄，"
                     "更不要主动列出没被问到的内容——数值只以「满级数值表」为准。")
    else:
        # 无补料（共鸣链 / 剧情 / 机制问答）：**绝不能**在这里点名「满级数值表」，
        # 否则模型会以为该有那张表，跑去「## 参考文档」里翻找并凭空扩列（实测过同类）。
        # 只保留「按资料原文讲、不扩列」这一条通用约束。
        lines.append("只讲资料里已有的内容，不要主动扩列没被问到的部分。")
    return prompt + "\n".join(lines) + tail
