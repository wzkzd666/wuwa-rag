import json
rows = [json.loads(l) for l in open('data/chunks/chunks.jsonl', encoding='utf-8')]
assert len({r['hash'] for r in rows}) == len(rows), "存在重复内容块！"
print("总块数  :", len(rows), "  (预期 ~115)")
print("超500字 :", sum(r['char_count'] > 500 for r in rows), "  (预期 1)")
print("重复hash:", len(rows) - len({r['hash'] for r in rows}), "  (必须为 0)")
