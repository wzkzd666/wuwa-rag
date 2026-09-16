# scripts/gen_fact_set.py —— 零 LLM，从 Neo4j 图谱真值反查，生成 fact 类评测集
# 用法：uv run python scripts/gen_fact_set.py   （需 Neo4j 在运行，零显存、零 GPU）
import json
import asyncio

from wuwa_rag.rag.characters import CHARACTER_NAMES
from wuwa_rag.rag.retrievers import graph_search

# 每个 fact 槽位对应的问法模板；元素为 (问法, stage) 元组，stage="" 表示不限定阶段
# 槽位 key 必须 ∈ {"属性","技能","共鸣链","突破材料","声骸","武器","队友"}（即 CYPHER 的 key）
# "六阶突破" 会精确过滤突破阶段，让 gold 更干净
FACT_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "属性":     [("{n}是什么属性", ""), ("{n}的武器类型是什么", ""), ("{n}的稀有度是多少", "")],
    "技能":     [("{n}有哪些技能", ""), ("{n}的共鸣技能是什么", "")],
    "共鸣链":   [("{n}的共鸣链有哪些", ""), ("{n}满链效果是什么", "")],
    "突破材料": [("{n}角色突破材料有哪些", ""), ("{n}六阶突破要多少贝币", "六阶突破"), ("{n}技能突破要什么", "")],
    "声骸":     [("{n}推荐什么声骸", ""), ("{n}毕业配装是什么", "")],
    "武器":     [("{n}推荐武器有哪些", "")],
    "队友":     [("{n}和谁组队", ""), ("{n}队友推荐有哪些", "")],
}

# 只取纯名角色，避免「漂泊者-男-导电」这类分支怪问法（分支角色可单独跑）
PURE = lambda n: "-" not in n


async def main():
    names = sorted(n for n in CHARACTER_NAMES if PURE(n))
    out = []
    for name in names:
        gold_cache: dict[tuple[str, str], str] = {}   # (slot, stage) -> gold，避免重复打 Neo4j
        for slot, tmpls in FACT_TEMPLATES.items():
            for q_tmpl, stage in tmpls:
                key = (slot, stage)
                if key not in gold_cache:
                    gold_cache[key] = await graph_search([name], [slot], "", stage)
                gold = gold_cache[key]
                if not gold.strip():
                    continue
                out.append({
                    "q": q_tmpl.format(n=name), "type": "fact", "slots": [slot],
                    "element": "", "stage": stage, "character": name,
                    "gold_answer": gold,
                })
    with open("eval/data/eval_dataset.jsonl", "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"生成 {len(out)} 条 fact 评测集 -> eval/data/eval_dataset.jsonl")


if __name__ == "__main__":
    asyncio.run(main(), loop_factory=asyncio.SelectorEventLoop)
