"""两路检索：图谱（Cypher 模板）+ 向量/稀疏（Chroma + BM25，RRF 融合）。"""
from __future__ import annotations

import itertools
import re
from functools import lru_cache

import chromadb
from chromadb.config import Settings

from ..config import get_settings
from ..graph.neo4j_client import get_session
from ..retrieval.bm25 import BM25Index
from ..retrieval.embeddings import BgeM3Embeddings
from ..text import lock_focus
from ..ww_logger import get_logger
from .characters import normalize_character_name, normalize_team

log = get_logger("rag")

# 用模板而不是 Text2Cypher：固定领域下模板 100% 可控，
# qwen3:8b 生成 Cypher 幻觉率高，而且慢。
CYPHER: dict[str, str] = {
    "属性": """MATCH (c:Character {name:$n}) RETURN c.element AS 属性, c.weapon AS 武器,
        c.gender AS 性别, c.birthplace AS 出生, c.echo_main AS 首位声骸,
        c.echo_main_stats AS 主词条, c.echo_sub_stats AS 副词条""",
    "属性反查": """MATCH (c:Character) WHERE c.element = $e
        RETURN c.element AS 属性, c.name AS 角色, c.weapon AS 武器, c.gender AS 性别 ORDER BY 角色""",
    "技能": """MATCH (:Character {name:$n})-[:HAS_SKILL]->(s)
        RETURN s.kind AS 类型, s.name AS 名称 ORDER BY 名称""",
    "共鸣链": """MATCH (:Character {name:$n})-[:HAS_CHAIN]->(x)
        RETURN x.seq AS 序号, x.name AS 名称, x.effect AS 效果 ORDER BY 序号""",
    "突破材料": """MATCH (:Character {name:$n})-[r:NEEDS_MATERIAL]->(m)
        WHERE $stage = '' OR r.stage = $stage
        RETURN r.kind AS 类别, r.stage AS 阶段, m.name AS 材料, r.qty AS 数量
        ORDER BY 类别, 阶段, 材料""",
    "声骸": """MATCH (c:Character {name:$n})
        OPTIONAL MATCH (c)-[r:RECOMMENDS_ECHO]->(e:EchoSet)
        WITH c, collect(DISTINCT e.name) AS 推荐套装,
             collect(DISTINCT {stage:r.stage, cost:r.cost, pieces:r.pieces, set:e.name}) AS 配装方案原
        RETURN c.echo_main AS 首位声骸,
               c.echo_main_stats AS 主词条,
               c.echo_sub_stats AS 副词条,
               推荐套装,
               [x IN 配装方案原 WHERE x.cost IS NOT NULL] AS 配装方案""",
    "武器": """MATCH (:Character {name:$n})-[r:RECOMMENDS_WEAPON]->(w)
        RETURN r.rank AS 优先级, w.name AS 武器 ORDER BY 优先级""",
    # ⚠️ 2026-09-22：teams 里存的组合名来自 wiki 标题行（`#### 守岸人+尤诺`），**不含
    # 本角色**——因为整页都是本角色的，标题只写另外两个队友。原样返回会让模型把
    # 「守岸人+尤诺」读成「守岸人和尤诺是一对」，于是把「队友→队伍」反查表整张抄成
    # 配队清单，还会自己编编号（实测把 5 行抄成带 `1`/`*5` 的 9 项）。
    #
    # ⚠️⚠️ 2026-09-22 二次修，两个坑：
    #
    # ① **必须用无向关系** `-[r:SYNERGIZES_WITH]-`，不能写出边 `->`。实测守岸人：出边只有
    #    2 条（`主输出`/`副输出` —— 通用模板行「守岸人配队｜主输出+副输出」抽出的占位名），
    #    **入边 36 条**。抽取方向是「本角色 → 标题里提到的队友」，而守岸人大量出现在
    #    **别人的**页面里（秋水页、卡卡罗页、忌炎页…）。只查出边等于把守岸人最核心的配队
    #    数据整片漏掉——用户问「守岸人配队」时模型手里只有 2 行占位符，只能自己编，
    #    这正是「守岸人 / 维里奈 / 白芷」式幻觉的上游来源。队友关系本就对称，无向查询两边都拿。
    #
    # ② **补全要用「页面主人」，不是「提问角色」**。旧写法 `CASE WHEN x CONTAINS $n` 是
    #    错判据：`洛可可` 页的标题 `椿+守岸人` 里含「守岸人」，于是被判为「已完整」原样返回，
    #    真实队伍 `洛可可+椿+守岸人` 丢了一个人（实测守岸人一次返回 50 支，一半是 `椿+守岸人`
    #    这种缺主人的两人片段）。页面主人 = 出边的 `$n` / 入边的 `t`，所以用 UNION 分两路把
    #    方向带出来，补全与「或」组收窄都在 Python 侧做（Cypher 里写不进去）。
    "队友": """MATCH (c:Character {name:$n})-[r:SYNERGIZES_WITH]->(t:Character)
        RETURN DISTINCT t.name AS 队友, $n AS 页面主人,
               coalesce(r.teams, []) AS 原始队伍
        UNION
        MATCH (t:Character)-[r:SYNERGIZES_WITH]->(c:Character {name:$n})
        RETURN DISTINCT t.name AS 队友, t.name AS 页面主人,
               coalesce(r.teams, []) AS 原始队伍""",
}

# 队列表里的**占位槽位名**：wiki 的通用模板行「| 守岸人配队 | 主输出+副输出 |」没写真人，
# 抽出来就是这种纯槽位串。补全主人后变成 `守岸人+主输出+副输出`，会被模型当成一支真队伍念
# 给用户听。判据：去掉槽位词后剩下的**真人名 ≤ 1 个**即视为占位串丢掉（正常队伍至少 2 个真人）。
_ROLE_WORDS = ("其他输出", "主输出", "副输出", "主C", "副C", "奶辅", "奶位", "输出", "奶", "辅助", "治疗")


def _is_placeholder_team(team: str) -> bool:
    names = [
        part
        for group in team.split("+")
        for part in group.split("/")
        if part and not any(w in part for w in _ROLE_WORDS)
    ]
    return len(names) <= 1


# 鸣潮一支队伍只有 3 个位置 —— 拿到 3 段就说明这串已经是「完整队伍」，不能再补主人。
# 实测反例：`守岸人/维里奈/白芷+秧秧+漂泊者-衍射`（3 段）来自 `漂泊者-男-衍射` 的页面，
# 主人名和串里的写法不同（`漂泊者-男-衍射` vs `漂泊者-衍射`），`owner not in team` 判不出
# 「已含主人」，硬补会得到 4 个人的队伍（实测出现过 `漂泊者-男-衍射+守岸人+秧秧+漂泊者-衍射`）。
_TEAM_SLOTS = 3


def _split_teams(raw: str) -> list[str]:
    """把一条 teams 值拆成多支队伍串（有的 chunk 把两支队伍挤在同一格）。

    实测 折枝 页的值：`作为副输出：折枝+今汐/珂莱塔+守岸人/维里奈/白芷；作为主输出：折枝+散华/釉瑚+…`
    —— 整串当一支队伍念出来会变成 6 个人。
    """
    out: list[str] = []
    for piece in re.split(r"[；;\n]", raw or ""):
        piece = piece.split("：")[-1].strip()      # 去掉「作为副输出：」这类前缀
        if piece:
            out.append(piece)
    return out


def _team_key(team: str) -> tuple[str, ...]:
    """队伍指纹（忽略写法顺序）：`吟霖+灯灯+守岸人` 与 `灯灯+吟霖+守岸人` 是同一支队伍。

    同一支队伍会被**两个角色的页面各记一遍**（吟霖页写 `灯灯+守岸人`、灯灯页写 `吟霖+守岸人`），
    补全主人后就成了两队镜像。实测守岸人去重前 57 支里有 10 对是这种镜像。
    """
    return tuple(sorted(
        p.strip() for g in team.split("+") for p in g.split("/") if p.strip()
    ))


def _team_groups(team: str) -> list[frozenset[str]]:
    """把队伍拆成**位置组**：`守岸人+吟霖/长离+卡卡罗` -> [{守岸人}, {吟霖,长离}, {卡卡罗}]。

    用于「片段是否被完整队伍覆盖」的剪枝——按位置组比较，而不是把「或」的名字摊平
    （摊平会让 `吟霖/长离/散华` 看起来覆盖掉一整支三人队）。
    """
    return [frozenset(p.strip() for p in g.split("/") if p.strip())
            for g in team.split("+")]


def _covers(template: list[frozenset[str]], concrete: list[frozenset[str]]) -> bool:
    """含「或」的**模板**队伍是否逐位覆盖一支**具体**队伍（位置顺序可不同）。

    实测：`守岸人+吟霖/长离/散华+卡卡罗` 覆盖 `守岸人+吟霖+卡卡罗`（吟霖是中间那个
    位置的三选一）；`守岸人+吟霖/长离+卡卡罗` **不**覆盖 `吟霖+守岸人+散华`（散华不在
    模板的任何位置上）。二者语义完全不同，所以必须按位置组比、不能摊平名字集合。

    只在**位置数相同**时比较——「2 段片段 vs 3 段完整队」那种包含关系由片段剪枝负责
    （见 graph_search 队友分支①），不走这里。
    """
    if len(template) != len(concrete):
        return False
    return any(
        all(concrete[i] <= template[p[i]] for i in range(len(concrete)))
        for p in itertools.permutations(range(len(template)))
    )

# 图谱字段是 schema 名，用户说的是游戏术语，必须显式映射
SLOT_LABEL: dict[str, str] = {
    "配装":    "声骸配装（套装字段即声骸套装名，COST 是声骸费用组合）",
    "共鸣链":  "共鸣链（序号字段即第几链，相当于命座）",
    "突破材料": "突破材料（类别区分角色突破/技能突破）",
    "武器":    "武器推荐（优先级字段：1 为首选）",
    # 注意：这里的内容会原样出现在 prompt 的【】标题里，别写 markdown 记号（`**` 不会被渲染，
    # 只会当成字面字符喂给模型），也别写太长——8B 会被冗长标题带偏。
    "队友":    "队友推荐（队伍字段是包含本角色的完整组队，如「清宵+守岸人+尤诺」）",
    "声骸": "声骸配装（套装字段即声骸套装名，COST 是声骸费用组合，主/副词条为推荐词条）",
}


def _fmt_val(v):
    """graph_search 展示：把 list(推荐套装/配装方案) 与 dict 列表格式化为可读串。"""
    if isinstance(v, list):
        if not v:
            return None
        if isinstance(v[0], dict):
            seen, items = set(), []
            for x in v:
                key = (x.get("cost"), x.get("set"))
                if key in seen:
                    continue
                seen.add(key)
                items.append("/".join(
                    f"{k}={x[k]}" for k in ("stage", "cost", "set", "pieces") if x.get(k) is not None))
            return "; ".join(items) if items else None
        return "、".join(str(x) for x in v)
    return v


async def graph_search(
    characters: list[str], slots: list[str], element: str = "", stage: str = ""
) -> str:
    """属性反查问题提前解决。支持多角色：每个角色各查一遍，各自带小标题。"""
    if not slots:
        return ""
    # 配装语义 == 声骸：把"配装"槽位映射到"声骸"检索通道，
    # 避免只查 HAS_BUILD 漏掉节点属性(主/副词条)与推荐套装集合。
    _remap = []
    for sl in slots:
        target = "声骸" if sl == "配装" else sl
        if target not in _remap:
            _remap.append(target)
    slots = _remap

    if not characters and "属性反查" not in slots:
        return ""
    # 「指名模式」判定（详见队友分支）。两个粒度必须分开，实测踩过：
    #   · has_multi —— 点名了 **≥2 个**角色名。用于「只列同时含这些角色的队」+ 跨段去重。
    #   · is_exact  —— 点名了 **≥3 个**角色名。鸣潮一队就 3 个人，凑满 3 个名字才是
    #     「指明某一支具体队伍」，此时才保留被模板覆盖的具体队。
    #     反例：问「守岸人和吟霖配队」（2 个名字）不是指名某支队，仍要做覆盖剪枝，
    #     否则 `守岸人+吟霖+卡卡罗` 会和覆盖它的模板一起出现（同一件事说两遍）。
    named = [c for c in characters if c]
    has_multi = len(named) >= 2
    is_exact = len(named) >= 3
    named_set = set(named)
    named_shown: set[tuple[str, ...]] = set()
    blocks: list[str] = []
    async with get_session() as s:
        if element and "属性反查" in slots:
            rows = await (await s.run(CYPHER["属性反查"], e=element)).data()
            if rows:
                blocks.append("【属性反查】\n" + "\n".join(
                    "  " + " / ".join(f"{k}={v}" for k, v in r.items() if v not in (None, ""))
                    for r in rows
                ))
        for char in characters:
            lines: list[str] = []
            for slot in slots:
                if slot == "属性反查":
                    continue       
                cy = CYPHER.get(slot)
                if not cy:
                    continue
                rows = await (await s.run(cy, n=char , stage=stage)).data()
                if not rows:
                    continue
                # ⚠️ 2026-09-22：队友槽位**不能**用通用的「k=v」反查表——把「队友 → 它参与的
                # 所有队伍」逐行印出来，等于同时给模型一份「名字池」和一份「组合池」，它会
                # 做笛卡尔积混搭：实测吐出源数据里根本没有的「清宵+莫宁+尤诺」，并把同一支
                # 队伍列两遍（用户报「严重幻觉」）。改成**去重的完整队伍列表**，每行一支队伍，
                # 用正面措辞要求逐行照抄——没有名字索引，就没有混搭空间。
                #
                # 同时在这里做「问谁锁谁」（`lock_focus`）：用户问的这位角色，在队伍串里凡
                # 以「或」形式和他并列的（`守岸人/维里奈/白芷`），一律收窄成他本人
                # → `守岸人`。**放在数据侧而不是提示词侧**：同一条规则写 `_SYSTEM` 实测不生效
                # （模型照旧输出未锁定的三项），确定性变换则零概率残留（详见 text.lock_focus）。
                if slot == "队友":
                    # ⚠️ 2026-09-22 新增「指名模式」（判定见函数开头 named/has_multi/is_exact）。
                    # 两者要看的答案不同：
                    #   · 泛问 → 按「或」讲通用模板（`守岸人+吟霖/长离/散华+卡卡罗`），
                    #     被模板覆盖的具体队隐藏，否则同一件事说两遍；
                    #   · 指名 → 用户要的就是那一支，模板收窄成它、且必须出现在答案里。
                    # 锁角色范围：泛问只锁本次循环到的 `char`；点名多人时把点到的名字**全锁**——
                    # 模板 `守岸人+吟霖/长离/散华+卡卡罗` 因此被锁成用户点的那支
                    # `守岸人+吟霖+卡卡罗`，与数据里真实存在的同串（若有）去重合并，
                    # 答案就干净地落到那一支上，而不是「模板 + 具体队」各列一遍。
                    locks = named if has_multi else [char]
                    teams: list[str] = []
                    seen_keys: set[tuple[str, ...]] = set()
                    for r in rows:
                        owner = (r.get("页面主人") or "").strip()
                        owner = normalize_character_name(owner) or owner
                        for raw in (r.get("原始队伍") or []):
                            for t in _split_teams(raw):
                                # 名字归一（见 characters.normalize_team）：清掉 wiki 表格里的
                                # 别名与粘连 —— `漂泊者·湮灭`→`漂泊者-男-湮灭`、
                                # `渊武其他输出`→`渊武`、`维里奈（高熟练度）`→`维里奈`、
                                # `暗主`→`漂泊者-男-湮灭`；纯占位槽位（`主输出`）整格剔除，
                                # 队因此变短（这是刻意的：留着它模型会念成一个角色）。
                                # ⚠️ 必须排在**补主人与剪枝之前**——段数判断要按归一后的算，
                                # 否则 `守岸人/维里奈/白芷+秧秧+主输出` 会先被当成 3 段完整队，
                                # 既跳过补主人、又混进一个假队友。
                                # ⚠️ 归一也让 `_team_key` 的镜像去重真正生效：原来
                                # `漂泊者·气动` 与 `漂泊者-男-气动` 是两个名字，同一支队躲过去重
                                # （实测问「忌炎配队」同时吐出 `莫特斐+漂泊者·气动` 和
                                # `漂泊者-男-气动+莫特斐`）。
                                t = normalize_team(t)
                                if not t:
                                    continue
                                # 只有「不满 3 段（=不是完整队伍）」才补主人；见 _TEAM_SLOTS
                                if t.count("+") < _TEAM_SLOTS - 1 and owner and owner not in t:
                                    t = f"{owner}+{t}"
                                if _is_placeholder_team(t):
                                    continue
                                for c in locks:
                                    t = lock_focus(t, c)
                                key = _team_key(t)
                                if t and key not in seen_keys:
                                    seen_keys.add(key)
                                    teams.append(t)
                    parsed = [(_team_groups(t), t) for t in teams]
                    # ① 片段剪枝：只剩片段的两人串（`仇远+守岸人` ⊂ `仇远+嘉贝莉娜+守岸人`）
                    #    丢掉——它和完整队伍其实是同一支，重复出现只会让模型以为「还有一支」。
                    #    **只剪「不满 3 个位置段」的**。
                    full = [g for g, _ in parsed if len(g) >= _TEAM_SLOTS]
                    parsed = [
                        (g, t) for g, t in parsed
                        if len(g) >= _TEAM_SLOTS
                        or not any(all(any(x <= y for y in fy) for x in g) for fy in full)
                    ]
                    # ② 逐位覆盖剪枝（**只在泛问/多人时**）：一条队伍若被**另一条**逐位覆盖
                    #    （`守岸人+吟霖+卡卡罗` ⊂ `守岸人+吟霖/长离/散华+卡卡罗`），隐藏起来。
                    #    理由：覆盖者已经把「或」讲清楚了，被覆盖的那条再列一遍既是同一件事
                    #    说两遍，也会让人误以为「这是一支另外的队伍」。
                    #    ⚠️ 2026-09-22 扩围：原来**只剪不含 `/` 的具体队**、模板之间不互剪，
                    #    于是 `守岸人/维里奈/白芷+洛可可+漂泊者-男-湮灭` 明明被
                    #    `守岸人/维里奈/白芷+洛可可+椿/漂泊者-男-湮灭` 逐位完全覆盖，却仍被列出。
                    #    现在**模板之间也剪**（爸爸原话：「被前者逐位完全覆盖的覆盖掉」）。
                    #    ⚠️ 两条红线：
                    #      a) 不能拿**摊平的名字集合**判覆盖：`守岸人+吟霖+长离` 名字集合同样落在
                    #         模板里，但它压根不是合法队伍（吟霖与长离同位置），逐位比较才准。
                    #      b) 判据必须**严格**覆盖（对方覆盖我、我不覆盖对方）。等价的两条
                    #         （位置组一一对应）会**互相**覆盖，若按「被覆盖就剪」两边一起消失；
                    #         等价时只留排序键最优的一条——先按排序键排好，再把「位置组多重集合」
                    #         相同的丢掉后面那些。
                    #    指名时跳过本步——用户点的那支必须出现。
                    if not is_exact:
                        parsed.sort(key=lambda gt: (-gt[1].count("/"), -gt[1].count("+"), gt[1]))
                        kept: list[tuple[list[frozenset[str]], str]] = []
                        seen_groups: set[tuple[tuple[str, ...], ...]] = set()
                        for g, t in parsed:
                            canon = tuple(sorted(tuple(sorted(s)) for s in g))
                            if canon in seen_groups:
                                continue
                            if any(_covers(og, g) and not _covers(g, og) for og, _ in parsed):
                                continue
                            seen_groups.add(canon)
                            kept.append((g, t))
                        parsed = kept
                    teams = [t for _, t in parsed]
                    # ③ 点名多人：只保留「点到的名字**全在其中**」的队伍，且**跨角色段只列
                    #    一次**——点名时守岸人/吟霖/卡卡罗三个角色各查一遍图，同一支
                    #    `守岸人+吟霖+卡卡罗` 会在三段里各出现一遍（实测 3 段 25 支里重复 3 遍）。
                    #    给模型三份重复清单，它就会照列三遍、甚至混搭。
                    #    命中为空时回落全量：用户点的组合若数据里真没有，宁可给全景，
                    #    也别给空答案（生成侧还有向量检索 + 「一句话不知道」兜底）。
                    if has_multi:
                        hit = [t for t in teams if named_set <= set(_team_key(t))]
                        if hit:
                            teams = [t for t in hit if _team_key(t) not in named_shown]
                            named_shown.update(_team_key(t) for t in teams)
                    if teams:
                        # 排序：① **含「或」的模板优先**——`A+B/C+D` 这种形态来自 wiki 的
                        #    「配队推荐」表（`| 卡卡罗配队 | 守岸人/维里奈/白芷+吟霖/长离/散华+卡卡罗 |`），
                        #    一支模板覆盖多支具体队，信息量最大；不带 `/` 的三人串来自别人的
                        #    「主流队友」双人格（`| 队伍组成 | 卡卡罗+吟霖+守岸人 |`），低一档；
                        # ② 「或」候选越多越先（一个格子能包的位置越多，覆盖的队越多）；
                        # ③ `+` 段数多（位置定得越全）。
                        # ⚠️ 别用**长度**做次序：`守岸人+吟霖/长离/散华+卡卡罗`（16 字）会被
                        # 一堆 12 字的双人格串挤出上限之外，而它恰恰是最该出现的那一支。
                        teams.sort(key=lambda t: (-t.count("/"), -t.count("+"), t))
                        total = len(teams)
                        cap = get_settings().TEAM_MAX_SHOWN
                        if cap and total > cap:      # 0 = 不限
                            teams = teams[:cap]
                        lines.append("【可组队伍（每行是一支完整队伍，逐行照抄即可）】")
                        lines.extend("  " + t for t in teams)
                        log.info("图谱命中 %s/队友: %d 支队伍（指名=%s，去重/剪枝/锁定后展示 %d）",
                                 char, total, "精确" if is_exact else ("多人" if has_multi else "泛问"),
                                 len(teams))
                    continue
                lines.append(f"【{SLOT_LABEL.get(slot, slot)}】")
                for r in rows:
                    lines.append("  " + " / ".join(
                        f"{k}={_fmt_val(v)}" for k, v in r.items()
                        if _fmt_val(v) not in (None, "", [])
                    ))
                log.info("图谱命中 %s/%s: %d 行", char, slot, len(rows))
            if lines:
                blocks.append(f"## {char}\n" + "\n".join(lines))
    return "\n\n".join(blocks)


@lru_cache(maxsize=1)
def _collection():
    """返回chromadb连接"""
    s = get_settings()
    client = chromadb.PersistentClient(
        path=str(s.VECTOR_DIR / "chroma"),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(name=s.CHUNK_COLLECTION)


@lru_cache(maxsize=1)
def _bm25() -> BM25Index:
    return BM25Index.load()


@lru_cache(maxsize=1)
def _embedder() -> BgeM3Embeddings:
    """返回唯一BgeM3"""
    return BgeM3Embeddings()


def fetch_chunks(character: str, component: str | None = None) -> list[dict]:
    """按角色（可选再按组件）直取原始块，不过向量召回、不过重排。

    给「技能数值表」「突破材料表」这类结构固定的材料用：它们块小、与问句的语义
    相似度天然低于大段机制描述，在 topk=6 的精排里必然被挤掉（实测数值表 0 条进
    top6，且 top6 分数 0.9983~0.9994 完全无区分度）。但这两类材料的元数据
    （character + component）足以精确定位——绕开语义召回反而必中，且零 embedding、
    零重排开销，只多一次本地元数据查询。
    """
    col = _collection()
    where: dict = {"character": character}
    if component:
        where = {"$and": [{"character": character}, {"component": component}]}
    got = col.get(where=where)
    out = [
        {"chunk_id": cid, "text": doc, **(meta or {})}
        for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"])
    ]
    log.info("直取原始块 %s/%s -> %d 条", character, component or "*", len(out))
    return out


def _rrf(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    """RRF 融合：dense 余弦分和 BM25 分值量纲不同，不能直接加权。"""
    score: dict[str, float] = {}
    for ids in rank_lists:
        for rank, cid in enumerate(ids):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + rank + 1)                   # rrf核心，都有就叠加，根据排名打分
    return [cid for cid, _ in sorted(score.items(), key=lambda x: x[1], reverse=True)]


def vector_search(question: str, topk: int | None = None) -> list[dict]:
    s = get_settings()
    col = _collection()

    dense = col.query(
        query_embeddings=[_embedder().embed_query(question)],
        n_results=topk or s.TOPK_DENSE,
    )["ids"][0]
    sparse = [h.chunk_id for h in _bm25().search(question, s.TOPK_SPARSE)]

    fused = _rrf([dense, sparse])[: s.TOPK_RERANK_IN]
    if not fused:
        return []

    got = col.get(ids=fused)
    by_id = {
        i: (d, m or {})
        for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])
    }
    out = [{"chunk_id": c, "text": by_id[c][0], **by_id[c][1]} for c in fused if c in by_id]
    log.info("向量召回 dense=%d sparse=%d -> 融合 %d", len(dense), len(sparse), len(out))
    return out
