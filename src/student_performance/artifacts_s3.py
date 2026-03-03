# src/student_performance/artifacts_s3.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import boto3
from botocore.exceptions import ClientError


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI (must start with s3://): {uri}")
    rest = uri[5:]
    parts = rest.split("/", 1)
    bucket = parts[0]
    prefix = parts[1].strip("/") if len(parts) == 2 else ""
    return bucket, prefix


def _object_key(prefix: str, filename: str) -> str:
    return f"{prefix}/{filename}" if prefix else filename


def download_artifacts_from_s3(
    s3_uri: str,
    local_dir: Path,
    filenames: Iterable[str],
) -> None:
    """
    Downloads artifact files from:
      s3://bucket/prefix/<file>
    into local_dir/<file>.

    Skips files that already exist locally.
    """
    bucket_name, prefix = parse_s3_uri(s3_uri)

    local_dir.mkdir(parents=True, exist_ok=True)

    client = boto3.client("s3")

    for fname in filenames:
        dst = local_dir / fname
        if dst.exists():
            continue

        key = _object_key(prefix, fname)

        try:
            client.download_file(bucket_name, key, str(dst))
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                raise FileNotFoundError(
                    f"Missing artifact in S3: s3://{bucket_name}/{key}"
                ) from e
            raise
