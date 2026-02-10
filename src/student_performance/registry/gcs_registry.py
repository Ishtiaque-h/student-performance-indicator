from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

from google.cloud import storage


@dataclass(frozen=True)
class GCSUri:
    bucket: str
    prefix: str  # can be empty


def parse_gs_uri(uri: str) -> GCSUri:
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS uri: {uri}")
    rest = uri[5:]
    parts = rest.split("/", 1)
    bucket = parts[0].strip()
    prefix = ""
    if len(parts) == 2:
        prefix = parts[1].strip().rstrip("/")
    if not bucket:
        raise ValueError(f"Invalid GCS uri: {uri}")
    return GCSUri(bucket=bucket, prefix=prefix)


def _blob_name(prefix: str, name: str) -> str:
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
      gs://bucket/prefix/<file>
    into local_dir/<file>.

    Skips download if all files exist locally unless force=True.
    """
    local_dir.mkdir(parents=True, exist_ok=True)

    required_files = list(required_files)
    if not force and all((local_dir / f).exists() for f in required_files):
        return

    gcs = parse_gs_uri(registry_uri)
    client = storage.Client()
    bucket = client.bucket(gcs.bucket)

    for fname in required_files:
        blob = bucket.blob(_blob_name(gcs.prefix, fname))
        if not blob.exists(client):
            raise FileNotFoundError(f"Missing in registry: gs://{gcs.bucket}/{_blob_name(gcs.prefix, fname)}")
        blob.download_to_filename(str(local_dir / fname))


def upload_artifacts(
    local_dir: Path,
    registry_base_uri: str,
    run_id: str,
    files: Iterable[str],
    also_update_latest: bool = True,
) -> Tuple[str, str]:
    """
    Uploads artifacts to:
      gs://bucket/base_prefix/runs/<run_id>/<file>

    Optionally also writes to:
      gs://bucket/base_prefix/latest/<file>

    Returns: (run_prefix_uri, latest_prefix_uri)
    """
    gcs = parse_gs_uri(registry_base_uri)
    client = storage.Client()
    bucket = client.bucket(gcs.bucket)

    base_prefix = gcs.prefix.rstrip("/")
    run_prefix = f"{base_prefix}/runs/{run_id}".strip("/")
    latest_prefix = f"{base_prefix}/latest".strip("/")

    for fname in files:
        src = local_dir / fname
        if not src.exists():
            raise FileNotFoundError(f"Local artifact missing: {src}")

        # versioned
        bucket.blob(_blob_name(run_prefix, fname)).upload_from_filename(str(src))

        # latest
        if also_update_latest:
            bucket.blob(_blob_name(latest_prefix, fname)).upload_from_filename(str(src))

    run_uri = f"gs://{gcs.bucket}/{run_prefix}"
    latest_uri = f"gs://{gcs.bucket}/{latest_prefix}"
    return run_uri, latest_uri
