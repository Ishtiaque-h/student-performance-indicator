from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

def _mlflow_available() -> bool:
    try:
        import mlflow  # noqa
        return True
    except Exception:
        return False

def log_training_run(
    *,
    best_model_name: str,
    report: Dict[str, Any],
    artifacts_dir: Path,
    run_name: Optional[str] = None,
) -> None:
    """
    Logs params/metrics/artifacts to MLflow if MLFLOW_TRACKING_URI is set and mlflow is installed.
    Safe no-op otherwise.
    """
    if not _mlflow_available():
        return

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return

    import mlflow

    mlflow.set_tracking_uri(tracking_uri)

    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "student-performance")
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("best_model", best_model_name)

        best = report.get("best_model", {})
        mlflow.log_metric("selection_score", float(best.get("selection_score", 0.0)))
        mlflow.log_metric("test_r2", float(best.get("test_r2", 0.0)))

        # Optional: log more metrics if present
        models = report.get("models", {})
        bm = models.get(best_model_name, {})
        test = bm.get("test", {})
        if "rmse" in test:
            mlflow.log_metric("test_rmse", float(test["rmse"]))
        if "mae" in test:
            mlflow.log_metric("test_mae", float(test["mae"]))

        # Log artifacts you already create
        for fname in ["model_report.json", "preprocessor.pkl", "model.pkl", "ingestion_meta.json"]:
            fpath = artifacts_dir / fname
            if fpath.exists():
                mlflow.log_artifact(str(fpath))
