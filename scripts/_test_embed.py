from wuwa_rag.retrieval.embeddings import BgeM3Embeddings
e = BgeM3Embeddings()
v = e.embed_query('卡卡罗三阶突破材料')
print('维度:', len(v))