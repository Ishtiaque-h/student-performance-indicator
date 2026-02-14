from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from student_performance.pipeline.train_pipeline import TrainPipeline
from student_performance.utils import find_project_root
from student_performance.registry.gcs_registry import upload_run_index, upload_release


def make_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = os.getenv("GITHUB_SHA", "").strip()[:7]
    return f"{ts}-{sha}" if sha else ts


def _normalize_registry_uri(uri: str) -> str:
    uri = uri.strip().rstrip("/")
    if not uri.startswith("gs://"):
        raise ValueError(f"--registry-uri must start with gs:// (got {uri})")
    return uri


def _assert_artifacts_exist(artifacts_dir: Path, files: List[str]) -> None:
    missing = [f for f in files if not (artifacts_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts in {artifacts_dir}: {missing}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--registry-uri",
        required=True,
        help="Base registry URI, e.g. gs://YOUR_BUCKET/student-performance",
    )
    p.add_argument(
        "--release-tag",
        default="",
        help="e.g. v2.1.1; if set, also publish to <registry>/<tag>/ (serving release)",
    )
    p.add_argument(
        "--index-latest",
        action="store_true",
        help="If set, upload run outputs to <registry>/latest/<run_id>/ (run index)",
    )
    p.add_argument(
        "--no-train",
        action="store_true",
        help="Skip training; only publish existing artifacts (debugging).",
    )
    args = p.parse_args()

    registry_uri = _normalize_registry_uri(args.registry_uri)
    release_tag = args.release_tag.strip()

    if not args.index_latest and not release_tag:
        raise ValueError("Nothing to do: set --index-latest and/or --release-tag")

    # Train locally in repo context
    if not args.no_train:
        TrainPipeline().run()

    repo_root = find_project_root()
    artifacts_dir = repo_root / "artifacts"

    files = [
        "model.pkl",
        "preprocessor.pkl",
        "model_report.json",
        "ingestion_meta.json",
    ]
    _assert_artifacts_exist(artifacts_dir, files)

    run_id = make_run_id()

    run_uri: Optional[str] = None
    rel_uri: Optional[str] = None

    if args.index_latest:
        run_uri = upload_run_index(
            local_dir=artifacts_dir,
            registry_base_uri=registry_uri,
            run_id=run_id,
            files=files,
        )

    if release_tag:
        # enforce your desired contract: only v* tags serve
        if not release_tag.startswith("v"):
            raise ValueError("--release-tag must start with 'v' (e.g. v2.1.1)")
        rel_uri = upload_release(
            local_dir=artifacts_dir,
            registry_base_uri=registry_uri,
            release_tag=release_tag,
            files=files,
        )

    # Emit parseable lines for GitHub Actions
    print(f"RUN_ID={run_id}")
    if run_uri:
        print(f"RUN_URI={run_uri}")
    if rel_uri:
        print(f"RELEASE_URI={rel_uri}")


if __name__ == "__main__":
    main()
