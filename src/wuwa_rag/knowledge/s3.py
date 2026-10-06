"""对象存储抽象层（当前后端 RustFS，S3 协议）。

为什么做抽象层而不是直接 boto3：
  MinIO 社区版 2026-04 已归档。
  业务代码只依赖本模块的函数签名，
  将来换 RustFS / SeaweedFS / COS / OSS / 本地 FS 只改这一层，业务零改动。
  抽象层比选对某个组件更重要。
"""
from __future__ import annotations

import hashlib
from functools import lru_cache

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from wuwa_rag.config import get_settings
from wuwa_rag.ww_logger import get_logger

log = get_logger("rag")


@lru_cache(maxsize=1)
def _client():
    s = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=s.S3_ENDPOINT,
        aws_access_key_id=s.S3_ACCESS_KEY,
        aws_secret_access_key=s.S3_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",      # RustFS/MinIO 必须给个 region，值随意
    )


# head_bucket / head_object 在「不存在」时给的错误码：不同后端（RustFS/MinIO/真 S3）
# 口径不完全一致，404 也可能以字符串形式出现在 Code 里，故一并收进来。
_NOT_FOUND_CODES = frozenset({"404", "NoSuchBucket", "NoSuchKey", "NotFound"})


def _is_not_found(exc: ClientError) -> bool:
    """这条 ClientError 是否只表示「桶/对象不存在」，而不是别的故障。"""
    return str(exc.response.get("Error", {}).get("Code", "")) in _NOT_FOUND_CODES


def ensure_bucket(bucket: str) -> None:
    """幂等建桶，已存在则跳过。

    只有「确认桶不存在」才去建。原来吞掉全部异常会导致：鉴权失败/网络不通时
    也去 `create_bucket`，真正的病因（凭据错、服务没起）被换成一句更莫名的建桶报错。
    """
    c = _client()
    try:
        c.head_bucket(Bucket=bucket)
    except ClientError as exc:
        if not _is_not_found(exc):
            log.error("检查桶 %s 失败（非「不存在」，不尝试建桶）: %s", bucket, exc)
            raise
        c.create_bucket(Bucket=bucket)
        log.info("桶 %s 不存在，已创建", bucket)


def exists(bucket: str, key: str) -> bool:
    """判断是否可以访问对象。

    仅「对象不存在」返回 False；鉴权/网络等故障**如实抛出**——原来一律 False 会让
    调用方（`put_raw`）误判成「没传过」而重传，并把真实故障掩盖成一次无意义的上传。
    """
    try:
        _client().head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        if _is_not_found(exc):
            return False
        log.error("检查对象 %s/%s 失败（非「不存在」）: %s", bucket, key, exc)
        raise


def put_bytes(
        bucket: str, 
        key: str, 
        data: bytes, 
        content_type: str = "application/octet-stream"
    ) -> str:
    """上传字节数据，返回 S3 URI"""
    _client().put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    return f"s3://{bucket}/{key}"


def get_bytes(bucket: str, key: str) -> bytes:
    """下载对象，返回 bytes"""
    return _client().get_object(Bucket=bucket, Key=key)["Body"].read()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def put_raw(character: str, data: bytes, ext: str = "md") -> tuple[str, str]:
    """上传原文，返回 (uri, sha256)。

    内容寻址：key 由内容 hash 派生 -> 同一份原文永远只存一份，天然去重。
    """
    h = sha256(data)
    bucket = get_settings().S3_BUCKET_RAW
    key = f"raw/{character}/{h[:16]}.{ext}"
    ensure_bucket(bucket)
    if not exists(bucket, key):       # 已存在就不重复传
        put_bytes(bucket, key, data, "text/markdown")
    return f"s3://{bucket}/{key}", h
