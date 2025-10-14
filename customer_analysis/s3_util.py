import gzip
import io
from urllib.parse import urlparse

import boto3
import matplotlib.pyplot as plt
import pandas as pd


def _parse_s3_uri(s3_uri: str):
    # s3_uri like "s3://bucket/prefix/..."
    u = urlparse(s3_uri)
    if u.scheme != "s3" or not u.netloc:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    bucket = u.netloc
    # Remove leading "/" from path
    key = u.path[1:] if u.path.startswith("/") else u.path
    return bucket, key


def _s3_client():
    # Picks up env vars, profiles, or IAM role automatically
    return boto3.client("s3")


def s3_upload_bytes(
    s3_uri: str, data: bytes, content_type: str = "application/octet-stream", sse: str | None = "AES256"
):
    bucket, key = _parse_s3_uri(s3_uri)
    extra = {"ContentType": content_type}
    if sse:
        extra["ServerSideEncryption"] = sse  # SSE-S3 by default; swap for "aws:kms" and add SSEKMSKeyId if you use KMS
    _s3_client().put_object(Bucket=bucket, Key=key, Body=data, **extra)


def s3_upload_dataframe_csv(df: pd.DataFrame, s3_uri: str, compression: str | None = None):
    buf = io.BytesIO()
    if compression == "gzip":
        gz = gzip.GzipFile(fileobj=buf, mode="wb")
        df.to_csv(io.TextIOWrapper(gz, encoding="utf-8", newline=""), index=False)
        gz.close()
        data = buf.getvalue()
        s3_upload_bytes(s3_uri, data, content_type="application/gzip")
    else:
        df.to_csv(buf, index=False)
        s3_upload_bytes(s3_uri, buf.getvalue(), content_type="text/csv")


def s3_upload_matplotlib_png(s3_uri: str, dpi: int = 150):
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close()
    s3_upload_bytes(s3_uri, buf.getvalue(), content_type="image/png")
