# eval/run_answers.py —— 阶段A：批量调用 ask() 把答案落盘，供阶段B 离线评测
# 用法（阶段A 显存：停 VLM，amis 独占 GPU）：
#   ollama stop qwen3-vl:8b
#   uv run python eval/run_answers.py        # 进度条默认 tqdm（uv add tqdm）；缺依赖自动回退裸条
import asyncio
import json
import logging

# 屏蔽 httpx/httpcore 的 INFO 请求日志（amis 调用太频繁，刷屏挡住进度条）
for _n in ("httpx", "httpcore", "httpx2", "openai", "urllib3"):
    logging.getLogger(_n).setLevel(logging.WARNING)

from wuwa_rag.rag.chain import ask

try:                                # tqdm 更美观；装不上就回退零依赖进度条
    from tqdm import tqdm
    _HAVE_TQDM = True
except ImportError:
    _HAVE_TQDM = False

EVAL_SET = "eval/data/eval_dataset.jsonl"
ANSWERS = "eval/data/answers.jsonl"


async def run_one(i: int, row: dict) -> dict:
    try:
        r = await ask(row["q"], thread_id=f"eval-{i}")
        docs = r.get("docs") or []
        contexts = [d.get("text", "") for d in docs]
        if not contexts:                          # fact 类无向量召回，用拼好的 context
            contexts = [r.get("context") or ""]
        return {
            "q": row["q"], "type": row.get("type", ""),
            "gold_answer": row.get("gold_answer", ""),
            "answer": r.get("answer", ""),
            "context": r.get("context", ""),       # 完整上下文（图谱+文档），留底
            "contexts": contexts,                  # RAGAS 用的检索片段列表
            "intent": r.get("intent", ""),
            "slots": r.get("slots", []),
            "characters": r.get("characters", []),
            "docs": len(docs), "ok": True,
        }
    except Exception as exc:
        return {
            "q": row["q"], "type": row.get("type", ""),
            "gold_answer": row.get("gold_answer", ""),
            "answer": "", "context": "", "contexts": [],
            "intent": "", "slots": [], "characters": [],
            "docs": 0, "ok": False, "error": repr(exc),
        }


async def main():
    rows = [json.loads(l) for l in open(EVAL_SET, encoding="utf-8") if l.strip()]
    total = len(rows)
    print(f"载入 {total} 条评测问题，开始批量 ask()（阶段A，amis 生成）…")
    with open(ANSWERS, "w", encoding="utf-8") as f:
        if _HAVE_TQDM:
            for i, row in enumerate(tqdm(rows, desc="ask", unit="q")):
                rec = await run_one(i, row)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()                              # 逐条落盘，崩了也不丢
                if not rec["ok"]:
                    tqdm.write(f"FAIL  {rec['q']}  {rec.get('error', '')}")
        else:
            for i, row in enumerate(rows):
                rec = await run_one(i, row)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()                              # 逐条落盘，崩了也不丢
                pct = (i + 1) / total * 100
                bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
                print(f"\r[{bar}] {i + 1}/{total} ({pct:.0f}%) "
                      f"意图={rec['intent']} docs={rec['docs']}", end="", flush=True)
            print()                                     # 收尾换行
    print(f"完成：{total} 条 -> {ANSWERS}")


if __name__ == "__main__":
    asyncio.run(main(), loop_factory=asyncio.SelectorEventLoop)
