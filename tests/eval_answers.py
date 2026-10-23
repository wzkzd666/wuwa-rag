"""答案级效果回归（**手动跑的脚本，不进 pytest 收集**）。

跑一批固定问句过真链路，然后按两层判：
1. **确定性检查**（默认，零额外模型调用）：`dialog.quality.check_answer`
   —— 引用角标残留、百分比单位、重复列表行、昵称引号、情绪标签泄漏、配队追问缺失。
   这六条全是 AGENTS 里**已经踩过并修过**的形态，回归时最先复现的也是它们。
2. **忠实度（可选 `--judge`）**：把答案拆成原子陈述（1 次调用），再把陈述与本轮检索到的
   资料一次性做蕴含判定（1 次调用），报「无资料支持」的条数与占比。
   ⚠️ 判分用 qwen3:8b（`get_tool_llm`，temperature=0），和在线 verifier 同一个模型；
   这里的分数**只用于同一数据集上的前后对比**，不是绝对质量。

前置：`.\\dev.bat` 起服务；`--judge` 还需要 Ollama 的 qwen3:8b 与 aemeath 在位。
`--judge` 走的资料是从 `/debug/retrieval` 拿 chunk_id 再回 PG 取的——
与答案实际用的那几块**可能有出入**（多角色补召回等），所以脚本会把每问的块数打出来，
比对前后两次运行时先确认块数一致。

用法：
    uv run python tests/eval_answers.py                 # 只跑确定性检查（快）
    uv run python tests/eval_answers.py --judge         # 再加忠实度判定（每问 2 次 LLM）
    uv run python tests/eval_answers.py --json out.json # 落机器可读结果，便于前后对比
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import httpx
import psycopg

from wuwa_rag.config import get_settings
from wuwa_rag.dialog.quality import check_answer

API = "http://127.0.0.1:28000"

# 每条：问句 / 期望走的路由（intent 判错本身就是故障）/ 是否概括性配队（要追问句）
CASES = [
    {"q": "守岸人配队", "intent": "fact", "team_overview": True},
    {"q": "卡卡罗的共鸣链效果是什么", "intent": None, "team_overview": False},
    {"q": "忌炎六阶突破要多少贝币", "intent": None, "team_overview": False},
    {"q": "长离的声骸怎么配", "intent": None, "team_overview": False},
    {"q": "爱弥斯的角色故事讲给我听", "intent": None, "team_overview": False},
    {"q": "心配队", "intent": "fact", "team_overview": True},
    {"q": "现在几点了", "intent": "time", "team_overview": False},
    {"q": "你是谁", "intent": "chitchat", "team_overview": False},
]


def _login(api: str, user: str, pw: str) -> dict:
    httpx.post(f"{api}/auth/register", json={"username": user, "password": pw}, timeout=30)
    tok = httpx.post(f"{api}/auth/login", json={"username": user, "password": pw},
                     timeout=30).json()["token"]
    return {"Authorization": f"Bearer {tok}"}


def _json_array(raw: str) -> list | None:
    """从模型输出里抠出 JSON 数组；抠不出来返回 **None**。

    刻意不返回空列表：空数组是「全部有据 / 无需报告」这个**有效结论**，
    而解析失败是「这次判分不可用」—— 混在一起就会出现
    「模型输出被 markdown 包了一下，整轮评测静默变满分」这种最坏的假绿。

    ⚠️ 元素会被**拍平成字符串**：实测 8b 明明被要求输出字符串数组，却照样回
    `[{'statement': '…'}]`（它把示例理解成了 schema）。不拍平的话，判分层拿到的是
    `"{'statement': '共鸣解放可以为队伍回复生命'}"` 这种带壳文本，蕴含判定直接全程
    「支持」——自检就是这样抓到判分层是摆设的（见 selftest_judge）。
    """
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:  # noqa: BLE001 —— 判分器的输出畸形不该把整批评测带崩
        return None
    if not isinstance(data, list):
        return None
    out: list[str] = []
    for item in data:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):        # {'statement': '…'} / {'claim': '…'}
            vals = [str(v) for v in item.values() if isinstance(v, (str, int, float))]
            if vals:
                out.append("，".join(vals))
        elif item is not None:
            out.append(str(item))
    return out


def _claims(answer: str) -> list | None:
    from wuwa_rag.core.llm import get_tool_llm

    # 示例是必需的（本仓实测过两次：8b 光靠规则句不执行，给一条完整示例才会照做 ——
    # 追问改写器那条同一个教训）。没有示例时它对长答案直接回 `[]`。
    prompt = (
        "把下面这段话拆成**互相独立**的事实陈述，每句只说一件事；"
        "语气词、招呼、提问引导不要。只输出 JSON 数组。\n"
        "示例：\n"
        "输入：我的大招能回不少血呢~家人组队可以试试守岸人+吟霖+卡卡罗。\n"
        '输出：["守岸人的共鸣解放可以恢复生命", "守岸人可以与吟霖、卡卡罗组队"]\n'
        f"输入：{answer}\n输出："
    )
    return _json_array(str(get_tool_llm().invoke(prompt).content))


def _unsupported(claims: list[str], context: str) -> list | None:
    from wuwa_rag.core.llm import get_tool_llm

    # 示例是必需的：实测不给示例时，8b 对「资料里根本没有的那条」也一律回空数组
    # （= 全部支持），判分层就成了只会打满分的摆设 —— `--selftest` 就是这么抓到的。
    # 与追问改写器同一个经验：规则句它不执行，给一条完整示例才照做。
    prompt = (
        "下面给你【资料】和【陈述】列表。逐条核对：该陈述能否在资料里找到依据？"
        "只输出 JSON 数组，元素是**没被资料支持**的那几条原文（纯文本）；全部有依据就输出空数组。\n"
        "示例：\n"
        "【资料】守岸人的共鸣解放可以恢复队伍生命。\n"
        "【陈述】\n守岸人的共鸣解放可以恢复生命\n守岸人可以增加三百点攻击力\n"
        '输出：["守岸人可以增加三百点攻击力"]\n\n'
        f"【资料】\n{context}\n\n【陈述】\n" + "\n".join(claims) + "\n输出：")
    # 解析不出来返回 None（= 这次判分不可用），**不许**返回 []——[] 是有效结论「全部有据」，
    # 两者混起来就等于「判分器坏了 → 整轮评测满分」。
    return _json_array(str(get_tool_llm().invoke(prompt).content))


def selftest_judge() -> int:
    """判分层自己的门禁：**故意的幻觉必须被抓到**。

    没有这一步，`--judge` 完全可能一直在说「全部支持」（模型敷衍、资料太长被截、
    提示词没生效……），而报告看着就是满分 —— 一个从来不出错的检查器比没有更危险。
    """
    ctx = "守岸人的共鸣解放可以恢复队伍生命。"
    ans = "我的共鸣解放可以给队伍回血，而且我还能加三百点攻击。"
    claims = _claims(ans)
    if not claims:
        print(f"✗ 自检失败：拆分器没拆出陈述（claims={claims!r}）")
        return 1
    bad = _unsupported(claims, ctx)
    if bad is None:
        print("✗ 自检失败：蕴含判定输出不是 JSON 数组")
        return 1
    hit = any("三百" in c or "300" in c for c in bad)
    print("  拆分=", claims)
    print("  判为无支持=", bad)
    print("✓ 自检通过：资料里没有的那条被抓到了" if hit
          else "✗ 自检失败：造出来的幻觉没被抓到 —— 判分层现在是摆设，别用它下结论")
    return 0 if hit else 1


def _context_for(question: str, headers: dict) -> tuple[str, int]:
    """`/debug/retrieval` 拿精排留下的 chunk_id → 回 PG 取正文（判分用的资料）。"""
    d = httpx.get(f"{API}/debug/retrieval", params={"question": question},
                  headers=headers, timeout=180).json()
    ids = [x["chunk_id"] for x in d.get("reranked", [])]
    if not ids:
        return "", 0
    with psycopg.connect(get_settings().PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT breadcrumb, text FROM chunks WHERE chunk_id = ANY(%s)", (ids,))
            rows = cur.fetchall()
    return "\n\n".join(f"{bc}\n{t}" for bc, t in rows), len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true", help="额外跑忠实度判定（每问 2 次 LLM 调用）")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    ap.add_argument("--base", default=API)
    ap.add_argument("--selftest", action="store_true",
                    help="只跑判分层自检（造一条资料里没有的陈述，看抓不抓到），不跑数据集")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest_judge())

    # 判分要 /debug/retrieval（admin 专属），所以探针账号临时提权，跑完连数据一起删
    H = _login(args.base, "evalanswers_probe", "Eval-answers-2026")
    with psycopg.connect(get_settings().PG_DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET role='admin' WHERE username=%s", ("evalanswers_probe",))

    results, problems = [], 0
    for i, case in enumerate(CASES):
        q = case["q"]
        # thread_id 必须是 ASCII（判据在 `conversations.THREAD_ID_RE`），别拿问句前缀当 id ——
        # 中文会被入口校验拒成 400，而这脚本差点把 400 当成「答案有问题」报过去。
        # 中文会被入口校验拒成 400，而这脚本差点把 400 当成「答案有问题」报过去。
        r = httpx.post(f"{args.base}/ask", json={"question": q, "thread_id": f"evalanswers-{i}"},
                       headers=H, timeout=240)
        if r.status_code != 200:
            print(f"!! {q} → HTTP {r.status_code} {r.text[:100]}")
            problems += 1
            continue
        d = r.json()
        ans, intent = d.get("answer", ""), d.get("intent", "")
        issues = check_answer(ans, team_followup=case["team_overview"])
        if case["intent"] and intent != case["intent"]:
            issues.append(f"路由不符：期望 {case['intent']}，实得 {intent or '（空=走了不知道）'}")
        row = {"question": q, "intent": intent, "chars": len(ans), "issues": issues}
        if args.judge and intent not in ("fact", "semantic", "hybrid"):
            # 时间与闲聊的答案**不来自检索资料**，拿检索块去判它们必然是错的判据：
            # 实测「现在几点了」的 3 条陈述（报的是服务端时钟）被判「全部有据」，
            # 而资料里根本没有时间。这类问题只跑确定性检查。
            row["judge"] = f"跳过（intent={intent}，答案不由检索支撑）"
        elif args.judge:
            ctx, n = _context_for(q, H)
            claims = _claims(ans)
            bad = _unsupported(claims, ctx) if claims else []
            if claims == [] and len(ans) > 120:
                # 120 字以上的答案不可能「一条事实都没有」——这是**拆分器没干活**，
                # 不是「全部有据」。不分开就会得到最坏的读数：claims=0 无支持=0 看着满分。
                issues.append(f"陈述拆分为空（答案 {len(ans)} 字）：判分不可用")
            if claims is None or bad is None:
                # 判分器自己不可用 ≠ 答案没问题（这一条不许省，否则坏输出会被读成满分）
                issues.append("忠实度判分不可用（模型输出不是合法 JSON 数组）")
                claims, bad = [], []
            row.update({"context_chunks": n, "claims": len(claims), "unsupported": bad})
            if bad:
                issues.append(f"{len(bad)}/{len(claims)} 条陈述无资料支持")
        problems += 1 if row["issues"] else 0
        mark = "  " if not row["issues"] else "✗ "
        extra = ""
        if args.judge:
            extra = ("  " + row["judge"]) if "judge" in row else \
                f"  claims={row['claims']} 无支持={len(row['unsupported'])}（资料 {row['context_chunks']} 块）"
        print(f"{mark}{q:<22} intent={intent:<9} {len(ans):>4} 字{extra}")
        for i in row["issues"]:
            print("     ·", i)
        results.append(row)

    with psycopg.connect(get_settings().PG_DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE username=%s", ("evalanswers_probe",))
            uid = cur.fetchone()[0]
            cur.execute("SELECT session_id FROM sessions WHERE user_id=%s", (uid,))
            sids = [x[0] for x in cur.fetchall()]
            cur.execute("DELETE FROM messages WHERE session_id = ANY(%s)", (sids,))
            for tbl in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
                cur.execute(f"DELETE FROM lg.{tbl} WHERE thread_id = ANY(%s)", (sids,))
            cur.execute("DELETE FROM sessions WHERE user_id=%s", (uid,))
            cur.execute("DELETE FROM auth_tokens WHERE user_id=%s", (uid,))
            cur.execute("DELETE FROM user_facts WHERE user_id=%s", ("evalanswers_probe",))
            cur.execute("DELETE FROM llm_usage WHERE user_id=%s", (str(uid),))
            cur.execute("DELETE FROM users WHERE id=%s", (uid,))

    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已落盘：{args.json}")
    print(f"合计：{len(results)} 问，其中 {problems} 问有问题")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
