from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
    print("GCS client project:", client.project)
    print("GCS client bucket:", gcs.bucket)

    for fname in required_files:
        blob = bucket.blob(_blob_name(gcs.prefix, fname))
        if not blob.exists(client):
            raise FileNotFoundError(f"Missing in registry: gs://{gcs.bucket}/{_blob_name(gcs.prefix, fname)}")
        blob.download_to_filename(str(local_dir / fname))


def _upload_to_prefix(
    *,
    local_dir: Path,
    registry_base_uri: str,
    dest_prefix: str,
    files: Iterable[str],
) -> str:
    """
    Uploads artifacts to:
      gs://bucket/<base_prefix>/<dest_prefix>/<file>

    Returns the full prefix URI.
    """
    gcs = parse_gs_uri(registry_base_uri)
    client = storage.Client()
    bucket = client.bucket(gcs.bucket)

    base_prefix = gcs.prefix.rstrip("/")
    full_prefix = f"{base_prefix}/{dest_prefix}".strip("/")

    for fname in files:
        src = local_dir / fname
        if not src.exists():
            raise FileNotFoundError(f"Local artifact missing: {src}")

        bucket.blob(_blob_name(full_prefix, fname)).upload_from_filename(str(src))

    return f"gs://{gcs.bucket}/{full_prefix}"


def upload_run_index(
    *,
    local_dir: Path,
    registry_base_uri: str,
    run_id: str,
    files: Iterable[str],
) -> str:
    """
    Run index upload (immutable history):
      gs://bucket/<base_prefix>/latest/<run_id>/<file>
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
      gs://bucket/<base_prefix>/<release_tag>/<file>
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
