# src/student_performance/artifacts_gcs.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

from google.cloud import storage


def parse_gcs_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI (must start with gs://): {uri}")
    rest = uri[5:]
    parts = rest.split("/", 1)
    bucket = parts[0]
    prefix = parts[1].strip("/") if len(parts) == 2 else ""
    return bucket, prefix


def _blob_name(prefix: str, filename: str) -> str:
    return f"{prefix}/{filename}" if prefix else filename


def download_artifacts_from_gcs(
    gcs_uri: str,
    local_dir: Path,
    filenames: Iterable[str],
) -> None:
    bucket_name, prefix = parse_gcs_uri(gcs_uri)

    local_dir.mkdir(parents=True, exist_ok=True)

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for fname in filenames:
        dst = local_dir / fname
        if dst.exists():
            continue

        blob = bucket.blob(_blob_name(prefix, fname))

        # This triggers a metadata lookup; good error message if missing.
        if not blob.exists():
            raise FileNotFoundError(
                f"Missing artifact in GCS: gs://{bucket_name}/{_blob_name(prefix, fname)}"
            )

        blob.download_to_filename(dst)
