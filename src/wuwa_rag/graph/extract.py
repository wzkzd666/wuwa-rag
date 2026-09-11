"""从分块文本中规则化抽取结构化事实（喂给 Neo4j）。

为什么规则而不是 LLM：wiki 结构高度固定（技能 6 类 / 共鸣链 6 条 / 突破 6 阶），
规则确定性 100%；CPU 上 LLM 抽 102 块要几分钟，规则毫秒级，且结果可回归测试。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..config import get_settings
from ..ww_logger import get_logger

log=get_logger('rag')

# 技能名：原文可能是 **名** 或 ***名***（斜体+粗体），必须用 \*+ 吃掉连续星号
_RE_BOLD = re.compile(r"\*+([^*\s][^*]*?)\*+")
# 材料串：「基准镣环x2海蚀嵌合体226x2贝币x4500」多个材料粘在一格
_RE_MAT = re.compile(r"([\u4e00-\u9fa5][\u4e00-\u9fa5]{0,9}\d{0,3})x(\d+)")
_RE_ROW = re.compile(r"^\|\s*([^|\n]*?)\s*\|\s*([^|\n]*?)\s*\|.*$", re.M)
_RE_HEAD = re.compile(r"^#{2,6}\s*([^#\n]+?)\s*$", re.M)
_RE_ATTR = re.compile(r"^\s*[-*]\s*([^:\n]+?)\s*[:：]\s*\**([^*\n]+?)\**\s*$", re.M)

# 配装：「过渡配装 COST 44111彻空冥雷2不绝余音 2毕业配装 COST 43311彻空冥雷2彻空冥雷5」
_RE_BUILD_SPLIT = re.compile(r"(?=(?:过渡配装|毕业配装))")
_RE_BUILD_HEAD = re.compile(r"(过渡配装|毕业配装)\s*COST\s*(\d{5})\s*(.*)", re.S)
_RE_BUILD_SET = re.compile(r"([\u4e00-\u9fa5]{2,8}?)\s*(\d)")
_STAGE_NORM = {"毕业配装": "毕业", "过渡配装": "过渡"}

_SKILL_NOISE = ("分支强化", "属性加成", "伤害", "等级", "Lv", "效果")
_ECHO_NOISE = ("主流", "推荐", "武器", "词条", "分配", "声骸", "配装")
_RE_STAGE = re.compile(r"([一二三四五六]阶突破)")
_TEAM_SLOT_NOISE = ("奶&辅", "副输出", "主输出", "输出", "配队", "备注", "说明", "PS")
_RE_NAME = re.compile(r"[\u4e00-\u9fa5]{2,4}")


@dataclass
class CharacterFacts:
    name: str
    attrs: dict[str, str] = field(default_factory=dict)
    skills: list[dict] = field(default_factory=list)      # {name, kind}
    chains: list[dict] = field(default_factory=list)      # {name, seq, effect}
    materials: list[dict] = field(default_factory=list)   # {name, qty, stage, kind}
    weapons: list[str] = field(default_factory=list)
    echoes: list[str] = field(default_factory=list)
    teammates: list[dict] = field(default_factory=list)   # {name, team}
    echo_builds: list[dict] = field(default_factory=list)   # 声骸配装方案 {name, stage, cost, pieces}
    echo_main: str = ""            # 声骸首位推荐
    echo_main_stats: str = ""      # 主词条
    echo_sub_stats: str = ""       # 副词条


def _clean(v: str) -> str:
    return re.sub(r"\s+", " ", v.replace("**", "").replace("[图]", "")).strip(" *|-—")


def _fmt_cost_main(v: str) -> str:
    """新格式"""
    v = v.replace("\\", "/").replace("／", "/").replace("、", "/")
    return re.sub(r"(?<!^)COST", "；COST", v)


def _noisy(v: str, words: tuple[str, ...]) -> bool:
    return any(w in v for w in words)


def load_chunks(path: Path | None = None) -> list[dict]:
    p = path or get_settings().CHUNKS_JSONL
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def all_characters(chunks: list[dict]) -> list[str]:
    return sorted({d["character"] for d in chunks if d.get("character")})


def _extract_attrs(chunks: list[dict], name: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for d in chunks:
        if d["module"] == "基础资料" and d.get("component") == name:
            for k, v in _RE_ATTR.findall(d["text"]):
                k, v = k.strip(), _clean(v)
                if k and v and len(k) <= 6:
                    out[k] = v
    return out


def _extract_skills(chunks: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for d in chunks:
        if d.get("component") != "技能介绍" or d.get("has_table"):
            continue
        parts = d["breadcrumb"].split(" › ")
        if len(parts) < 4:
            continue
        kind = parts[3].strip()
        for m in _RE_BOLD.finditer(d["text"]):
            nm = _clean(m.group(1))
            if len(nm) < 2 or _noisy(nm, _SKILL_NOISE):
                continue
            seen.setdefault(kind, {"name": nm, "kind": kind})
            break
    return list(seen.values())


def _extract_chains(chunks: list[dict]) -> list[dict]:
    out, seq = [], 0
    for d in chunks:
        if d.get("component") != "共鸣链":
            continue
        for c1, c2 in _RE_ROW.findall(d["text"]):
            n, e = _clean(c1), _clean(c2)
            if (not n or n in ("名称", "列1", "列2") or set(n) <= set("-: ")
                    or len(n) > 12 or "效果" in n):
                continue
            seq += 1
            out.append({"name": n, "seq": seq, "effect": e})
    return out


def _extract_materials(chunks: list[dict]) -> list[dict]:
    """数量用 max 聚合，不用 sum：同一阶段材料在多个表格分片 chunk 里重复出现，
    sum 会把数量翻倍。"""
    agg: dict[tuple[str, str, str], int] = {}
    for d in chunks:
        comp = d.get("component")
        if comp not in ("角色突破材料", "技能突破材料"):
            continue
        kind = "角色突破" if comp == "角色突破材料" else "技能突破"
        if comp == "角色突破材料":
            # H3「角色突破材料」也含"突破"二字，必须用严格阶段格式匹配，
            # 否则会被 next() 优先命中，把 stage 写成 component 名
            m = _RE_STAGE.search(d["text"])
            stage = m.group(1) if m else "未知"
        else:
            # 技能突破材料：阶段信息在分块后不完整，用技能名做 stage
            parts = d["breadcrumb"].split(" › ")
            stage = parts[3].strip() if len(parts) > 3 else "未知"
                # 用 _row_cells 而非正则 findall：相邻单元格共享中间的 `|`，
        # findall 会被前一个匹配消耗掉，导致第二列永远抽不到
        for cells in _row_cells(d["text"]):
            for cell in cells:
                for nm, qty in _RE_MAT.findall(cell):
                    if "突破" in nm:
                        continue
                    key = (nm, stage, kind)
                    agg[key] = max(agg.get(key, 0), int(qty))
    return [{"name": n, "stage": s, "kind": k, "qty": q}
            for (n, s, k), q in agg.items()]


def _extract_weapons(chunks: list[dict]) -> list[str]:
    out: list[str] = []
    for d in chunks:
        # 兼容新旧版语料：新格式在「武器推荐」，旧格式可能在「角色养成推荐」
        if d.get("component") not in ("武器推荐", "角色搭配推荐", "角色养成推荐"):
            continue
        for m in re.finditer(r"\|\s*武器推荐\s*\|\s*([^|\n]+)\|", d["text"]):
            cell = m.group(1).replace("5阶", "")
            for raw in re.split(r"[＞>≈≥]+", cell):
                # 去掉注记：从"无特别"处截断，并删括号说明与句号
                w = re.split(r"无特别", raw)[0]
                w = re.sub(r"[（(][^）)]*[）)]", "", w)
                w = w.replace("。", "").strip()
                w = _clean(w)
                if w and w not in out:   # 角色内同名去重
                    out.append(w)
    return out


def _extract_echoes(chunks: list[dict]) -> list[str]:
    """来源有两个：Character Strategy 的「声骸套装推荐」，
    以及 角色养成推荐 的「声骸推荐」（breadcrumb 末段）。
    只扫前者会把「不绝余音」漏掉。"""
    out: list[str] = []
    for d in chunks:
        bc = d.get("breadcrumb", "")
        if not (d.get("component") == "声骸套装推荐" or bc.endswith("声骸推荐")):
            continue
        cands = list(_RE_HEAD.findall(d["text"]))
        cands += [c1 for c1, _ in _RE_ROW.findall(d["text"])]
        for c in cands:
            c = _clean(c)
            if not c or len(c) > 8 or _noisy(c, _ECHO_NOISE) or re.fullmatch(r"列\d+", c) or set(c) <= set("-: "):
                continue
            if c not in out:
                out.append(c)
    return out


def _row_cells(text: str) -> list[tuple[str, ...]]:
    """取整行所有非空单元格，支持任意列数。
    声骸推荐是 4 列表格（主词条横跨 3 列），_RE_ROW 只取前两列会丢数据。
    """
    out: list[tuple[str, ...]] = []
    for line in text.splitlines():
        line = line.strip()
        if not (line.startswith("|") and line.endswith("|")):
            continue
        cells = [_clean(c) for c in line.strip("|").split("|")]
        cells = [c for c in cells if c and not set(c) <= set("-: ")]
        if cells:
            out.append(tuple(cells))
    return out


def _extract_echo_plan(chunks: list[dict], char_name: str = "") -> dict:
    """抽配装方案 + 主/副词条。
    兼容两套 wiki 页面格式：
      - 旧格式：breadcrumb 以「声骸推荐」结尾，4 列表格，
                键 声骸配装推荐 / 声骸首位推荐 / 主词条 / 副词条
      - 新格式：breadcrumb 以「声骸套装推荐」结尾（component=声骸套装推荐），2 列表格，
                键 COST分配 / COST主词条 / 声骸词条，首位声骸在「主流声骸」首行
    COST 是鸣潮声骸核心概念（如 43311 = 五个声骸的 COST 组合，总和 12）。
    """
    builds: list[dict] = []
    main_echo = main_stats = sub_stats = ""
    for d in chunks:
        bc = d.get("breadcrumb", "")
        if not (bc.endswith("声骸推荐") or bc.endswith("声骸套装推荐")
                or d.get("component") == "声骸套装推荐"):
            continue
        for cells in _row_cells(d["text"]):
            k, rest = cells[0], list(cells[1:])
            if not rest:
                # 新格式：首位声骸名（如「长路启航之星 | [图]」）；新格式优先于旧格式占位值
                if (not _noisy(k, _ECHO_NOISE)
                        and re.fullmatch(r"[\u4e00-\u9fa5]{2,6}", k)):
                    main_echo = k
                continue
            v = " ".join(rest)
            if k == "声骸配装推荐":                 # 旧格式
                for part in _RE_BUILD_SPLIT.split(v):
                    m = _RE_BUILD_HEAD.match(part.strip())
                    if not m:
                        continue
                    stage_raw, cost, rest_b = m.groups()
                    stage = _STAGE_NORM.get(stage_raw, stage_raw)
                    raw = [(n, int(p)) for n, p in _RE_BUILD_SET.findall(rest_b)]
                    # 原文「彻空冥雷2彻空冥雷5」= 「2件套」+「5件套」粘在一起，
                    # 真实含义是彻空冥雷 5 件套，不是 2+5=7 件（COST 只有 5 个声骸位）
                    merged: dict[str, int] = {}
                    for n, p in raw:
                        merged[n] = max(merged.get(n, 0), p)
                    sets = list(merged.items())
                    total = sum(p for _, p in sets)
                    if len(raw) != len(sets) or total > 5:
                        log.warning(
                            "配装数据存疑 | %s %s COST=%s 原始=%s 合并后=%s 件数合计=%d",
                            char_name, stage, cost, raw, sets, total,
                        )
                    builds.append({"stage": stage, "cost": cost, "sets": sets})
            elif k == "声骸首位推荐":               # 旧格式
                if not main_echo:
                    main_echo = v
            elif k == "主词条":                      # 旧格式
                main_stats = "；".join(rest)
            elif k == "副词条":                      # 旧格式
                sub_stats = "；".join(rest).replace("副词条：", "").strip()
            elif k == "COST分配":                    # 新格式
                for m in re.finditer(r"(过渡|毕业)[:：](\d{5})", v):
                    stage = _STAGE_NORM.get(m.group(1), m.group(1))
                    builds.append({"stage": stage, "cost": m.group(2),
                                   "sets": [(main_echo, 5)] if main_echo else []})
            elif k == "COST主词条":                   # 新格式
                main_stats = _fmt_cost_main(v)
            elif k == "声骸词条":                     # 新格式
                sub_stats = v
    return {
        "builds": builds,
        "main_echo": main_echo,
        "main_stats": main_stats,
        "sub_stats": sub_stats,
    }


def _extract_team_effects(chunks: list[dict], self_name: str) -> dict[str, str]:
    """从「配队推荐」表格抽「队友 -> 推荐理由」。
    理由就是表格第二列的技能效果描述：「全伤害加深15%」「导电伤害加深20%」。
    原文形态：
        | 卡卡罗配队 | 守岸人/维里奈/白芷+吟霖/长离/散华+卡卡罗 |   <- 组合行，跳过
        | 奶&辅      | 声骸推荐套装为隐世回光                  |   <- 噪音行，跳过
        | 守岸人     | 双联合相…队伍中的角色全伤害加深15%       |   <- 要的
    """
    fx: dict[str, str] = {}
    for d in chunks:
        if "配队推荐" not in (d.get("breadcrumb") or ""):
            continue
        for cells in _row_cells(d["text"]):
            if len(cells) < 2:
                continue
            who, effect = _clean(cells[0]), _clean(cells[1])
            if not who or not effect or len(effect) < 12:
                continue
            if who == self_name or self_name in who or who in _TEAM_SLOT_NOISE:
                continue
            if not _RE_NAME.fullmatch(who):
                continue
            if len(effect) > len(fx.get(who, "")):      # 同人取最长描述
                fx[who] = effect
    return fx


def _extract_teammates(chunks: list[dict], self_name: str) -> list[dict]:
    """两个来源都要覆盖，否则会漏队友：
      1. 编队&队伍轴推荐 —— 队友写在【标题行】里：##### 相里要+守岸人/维里奈
      2. 角色养成推荐 › 配队推荐 —— 队友写在【表格首行第二列】：
         | 卡卡罗配队 | 守岸人/维里奈/白芷+吟霖/长离/散华+卡卡罗 |
    """
    fx = _extract_team_effects(chunks, self_name)
    out, seen = [], set()

    def _add(mate: str, team: str) -> None:
        mate = _clean(mate)
        if not mate or mate == self_name or not _RE_NAME.fullmatch(mate):
            return
        if (mate, team) in seen:
            return
        seen.add((mate, team))
        out.append({"name": mate, "team": team, "effect": fx.get(mate, "")})

    # 来源 1：编队&队伍轴推荐（标题行）
    for d in chunks:
        if d.get("component") != "编队&队伍轴推荐":
            continue
        for h in _RE_HEAD.findall(d["text"]):
            h = h.strip()
            if not re.search(r"[+＋]", h):
                continue
            for mate in re.split(r"[+＋/]", h):
                _add(mate, h)

    # 来源 2：配队推荐（表格首行第二列的组合串）
    for d in chunks:
        if "配队推荐" not in (d.get("breadcrumb") or ""):
            continue
        for cells in _row_cells(d["text"]):
            if len(cells) < 2:
                continue
            comb = _clean(cells[1])
            if not re.search(r"[+＋]", comb):
                continue
            for mate in re.split(r"[+＋/]", comb):
                _add(mate, comb)

    return out


def extract_character(chunks: list[dict], name: str) -> CharacterFacts:
    mine = [d for d in chunks if d.get("character") == name]
    plan = _extract_echo_plan(mine, name)
    echoes = _extract_echoes(mine)
    # 配装方案并入声骸：拆成「角色→套装」的 RECOMMENDS_ECHO（带 stage/cost/pieces），
    # 不再单独建 HAS_BUILD 关系。
    echo_builds_raw = [
        {"name": s, "stage": _STAGE_NORM.get(b["stage"], b["stage"]),
         "cost": b["cost"], "pieces": p}
        for b in plan["builds"] for (s, p) in b["sets"]
    ]
    # 按 (name, stage, cost) 去重：同源不同标签(毕业/毕业配装)或重复列出只留一条
    seen, echo_builds = set(), []
    for eb in echo_builds_raw:
        key = (eb["name"], eb["stage"], eb["cost"])
        if key in seen:
            continue
        seen.add(key)
        echo_builds.append(eb)
    # 纯声骸推荐若已被某配装方案覆盖，去掉以免重复建无 cost 的 RECOMMENDS_ECHO
    build_names = {eb["name"] for eb in echo_builds}
    echoes = [e for e in echoes if e not in build_names]

    return CharacterFacts(
        name=name,
        attrs=_extract_attrs(mine, name),
        skills=_extract_skills(mine),
        chains=_extract_chains(mine),
        materials=_extract_materials(mine),
        weapons=_extract_weapons(mine),
        echoes=echoes,
        echo_builds=echo_builds,
        echo_main=plan["main_echo"],
        echo_main_stats=plan["main_stats"],
        echo_sub_stats=plan["sub_stats"],
        teammates=_extract_teammates(mine, name),
    )

