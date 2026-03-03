from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import boto3
from botocore.exceptions import ClientError


@dataclass(frozen=True)
class S3Uri:
    bucket: str
    prefix: str  # can be empty


def parse_s3_uri(uri: str) -> S3Uri:
    """Parse s3://bucket/prefix into S3Uri."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an S3 uri: {uri}")
    rest = uri[5:]
    parts = rest.split("/", 1)
    bucket = parts[0].strip()
    prefix = ""
    if len(parts) == 2:
        prefix = parts[1].strip().rstrip("/")
    if not bucket:
        raise ValueError(f"Invalid S3 uri: {uri}")
    return S3Uri(bucket=bucket, prefix=prefix)


def _object_key(prefix: str, name: str) -> str:
    """Build a full S3 object key from prefix + filename."""
    if not prefix:
        return name
    return f"{prefix.rstrip('/')}/{name}"


def download_required_artifacts(
    registry_uri: str,
    local_dir: Path,
    required_files: Iterable[str],
    force: bool = False,
) -> None:
    """
    Downloads required artifact files from:
      s3://bucket/prefix/<file>
    into local_dir/<file>.

    Skips download if all files exist locally unless force=True.
    """
    local_dir.mkdir(parents=True, exist_ok=True)

    required_files = list(required_files)
    if not force and all((local_dir / f).exists() for f in required_files):
        return

    s3 = parse_s3_uri(registry_uri)
    client = boto3.client("s3")

    for fname in required_files:
        key = _object_key(s3.prefix, fname)
        dest = local_dir / fname
        try:
            client.download_file(s3.bucket, key, str(dest))
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                raise FileNotFoundError(
                    f"Missing in S3 registry: s3://{s3.bucket}/{key}"
                ) from e
            raise


def _upload_to_prefix(
    *,
    local_dir: Path,
    registry_base_uri: str,
    dest_prefix: str,
    files: Iterable[str],
) -> str:
    """
    Uploads artifacts to:
      s3://bucket/<base_prefix>/<dest_prefix>/<file>

    Returns the full prefix URI.
    """
    s3 = parse_s3_uri(registry_base_uri)
    client = boto3.client("s3")

    base_prefix = s3.prefix.rstrip("/")
    full_prefix = f"{base_prefix}/{dest_prefix}".strip("/")

    for fname in files:
        src = local_dir / fname
        if not src.exists():
            raise FileNotFoundError(f"Local artifact missing: {src}")

        key = _object_key(full_prefix, fname)
        client.upload_file(str(src), s3.bucket, key)

    return f"s3://{s3.bucket}/{full_prefix}"


def upload_run_index(
    *,
    local_dir: Path,
    registry_base_uri: str,
    run_id: str,
    files: Iterable[str],
) -> str:
    """
    Run index upload (immutable history):
      s3://bucket/<base_prefix>/latest/<run_id>/<file>
    """
    return _upload_to_prefix(
        local_dir=local_dir,
        registry_base_uri=registry_base_uri,
        dest_prefix=f"latest/{run_id}",
        files=files,
    )


def upload_release(
    *,
    local_dir: Path,
    registry_base_uri: str,
    release_tag: str,
    files: Iterable[str],
) -> str:
    """
    Serving release upload (versioned):
      s3://bucket/<base_prefix>/<release_tag>/<file>
    """
    release_tag = release_tag.strip().strip("/")
    if not release_tag:
        raise ValueError("release_tag must be non-empty")
    return _upload_to_prefix(
        local_dir=local_dir,
        registry_base_uri=registry_base_uri,
        dest_prefix=release_tag,
        files=files,
    )


def write_promoted_uri(registry_base_uri: str, run_uri: str) -> None:
    """
    Writes the promoted model URI pointer to:
      s3://bucket/<base_prefix>/promoted/latest_uri.txt
    (AWS equivalent of writing to GCS promoted pointer file)
    """
    s3 = parse_s3_uri(registry_base_uri)
    client = boto3.client("s3")
    key = _object_key(s3.prefix, "promoted/latest_uri.txt")
    client.put_object(Bucket=s3.bucket, Key=key, Body=run_uri.encode())


def read_promoted_uri(registry_base_uri: str) -> str:
    """
    Reads the promoted model URI from:
      s3://bucket/<base_prefix>/promoted/latest_uri.txt
    """
    s3 = parse_s3_uri(registry_base_uri)
    client = boto3.client("s3")
    key = _object_key(s3.prefix, "promoted/latest_uri.txt")
    try:
        response = client.get_object(Bucket=s3.bucket, Key=key)
        return response["Body"].read().decode().strip()
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            raise FileNotFoundError(
                f"No promoted model URI found at s3://{s3.bucket}/{key}. "
                "Run the retrain workflow first."
            ) from e
        raise