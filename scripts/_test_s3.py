from wuwa_rag.storage import s3
s3.ensure_bucket("wuwa-raw")
uri, h = s3.put_raw("卡卡罗", "# 卡卡罗\n测试".encode("utf-8"))
print(uri, h)
print("回读:", s3.get_bytes("wuwa-raw", uri.split("/", 3)[-1])[:20])