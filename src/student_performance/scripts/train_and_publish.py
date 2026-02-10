from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from student_performance.pipeline.train_pipeline import TrainPipeline
from student_performance.utils import find_project_root
from student_performance.registry.gcs_registry import upload_artifacts


def make_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = os.getenv("GITHUB_SHA", "")[:7]
    return f"{ts}-{sha}" if sha else ts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--registry-uri", required=True, help="e.g. gs://YOUR_BUCKET/student-performance")
    p.add_argument("--also-latest", action="store_true", help="also update latest/ prefix")
    args = p.parse_args()

    # Train locally in repo context
    _ = TrainPipeline().run()

    repo_root = find_project_root()
    artifacts_dir = repo_root / "artifacts"

    run_id = make_run_id()

    files = [
        "model.pkl",
        "preprocessor.pkl",
        "model_report.json",
        "ingestion_meta.json",
    ]

    run_uri, latest_uri = upload_artifacts(
        local_dir=artifacts_dir,
        registry_base_uri=args.registry_uri,
        run_id=run_id,
        files=files,
        also_update_latest=args.also_latest,
    )

    # Emit a parseable line for GitHub Actions
    print(f"RUN_ID={run_id}")
    print(f"RUN_URI={run_uri}")
    if args.also_latest:
        print(f"LATEST_URI={latest_uri}")


if __name__ == "__main__":
    # Cloud Run injects the PORT environment variable (default 8080)
    # You MUST listen on 0.0.0.0 to be accessible outside the container
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
