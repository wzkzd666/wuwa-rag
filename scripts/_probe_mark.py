import pathlib
from langchain_text_splitters import MarkdownHeaderTextSplitter

md = pathlib.Path("data/raw/卡卡罗.md").read_text(encoding="utf-8")
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3"), ("####", "H4")],
    strip_headers=False,
)
docs = splitter.split_text(md)

print("块数:", len(docs))
print("超 1200 字需二次切:", sum(len(d.page_content) > 1200 for d in docs))
print("含表格:", sum("|" in d.page_content for d in docs))
for d in docs[:6]:
    print("---", d.metadata)
    print(d.page_content[:150])
