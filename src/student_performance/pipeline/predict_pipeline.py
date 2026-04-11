import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union, Tuple
from threading import Lock

import numpy as np
import pandas as pd

from student_performance.exception import CustomException
from student_performance.logger import logging
from student_performance.utils import find_project_root, load_object
from student_performance.components.config import CONFIG
from student_performance.artifacts_gcs import download_artifacts_from_gcs


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class PredictPipelineConfig:
    artifacts_dir: Path
    preprocessor_path: Path
    model_path: Path
    pipeline_path: Path
    report_path: Path
    ingestion_meta_path: Path


class PredictPipeline:
    _lock = Lock()  # protects download + first load

    def __init__(self):
        root = find_project_root()
        self._loaded_from_uri: str | None = None

        default_artifacts_dir = CONFIG.artifacts.artifacts_dir(root)
        artifacts_dir = (
            Path(os.getenv("ARTIFACTS_DIR", str(default_artifacts_dir)))
            .expanduser()
            .resolve()
        )

        self.config = PredictPipelineConfig(
            artifacts_dir=artifacts_dir,
            preprocessor_path=artifacts_dir / CONFIG.artifacts.preprocessor_name,
            model_path=artifacts_dir / CONFIG.artifacts.model_name,
            pipeline_path=artifacts_dir / CONFIG.artifacts.pipeline_name,
            report_path=artifacts_dir / "model_report.json",
            ingestion_meta_path=artifacts_dir / "ingestion_meta.json",
        )

        # in-memory cache
        self._preprocessor: Any = None
        self._model: Any = None
        self._pipeline: Any = None

    def _read_test_mae(self) -> float:
        if not self.config.report_path.exists():
            return 8.0
        try:
            report = json.loads(self.config.report_path.read_text())
            return float(report.get("best_model", {}).get("test_mae", 8.0))
        except Exception:
            return 8.0

    def _risk_probability(self, score_prediction: float) -> float:
        threshold = float(CONFIG.product.risk_threshold_score)
        scale = max(float(CONFIG.product.risk_probability_scale), 1e-6)
        # Higher probability when predicted score is below threshold.
        z = (threshold - float(score_prediction)) / scale
        return float(1.0 / (1.0 + np.exp(-z)))

    def _risk_tier(self, risk_probability: float) -> str:
        if risk_probability >= float(CONFIG.product.risk_tier_high_min):
            return "high"
        if risk_probability >= float(CONFIG.product.risk_tier_medium_min):
            return "medium"
        return "low"

    def _performance_band(self, score_prediction: float) -> str:
        if score_prediction < float(CONFIG.product.performance_band_low_max):
            return "low"
        if score_prediction < float(CONFIG.product.performance_band_medium_max):
            return "medium"
        return "high"

    def _score_range(self, score_prediction: float) -> Tuple[float, float]:
        half_width = max(self._read_test_mae(), 1.0)
        low = max(0.0, float(score_prediction) - half_width)
        high = min(100.0, float(score_prediction) + half_width)
        return (low, high)

    def _ensure_artifacts(self) -> None:
        force = _env_flag("FORCE_MODEL_DOWNLOAD", "0")

        core_required = [
            "model_report.json",
            "ingestion_meta.json",
        ]
        pipeline_required = [CONFIG.artifacts.pipeline_name]

        core_ok = all((self.config.artifacts_dir / f).exists() for f in core_required)
        local_pipeline_ok = self.config.pipeline_path.exists()
        local_ok = core_ok and local_pipeline_ok

        # If artifacts exist locally and we are not forcing a download, do nothing.
        if local_ok and not force:
            return

        # Resolve GCS URI (new preferred var first, then legacy fallback)
        gcs_uri = (
            os.getenv("GCS_ARTIFACTS_URI", "").strip()
            or os.getenv("MODEL_REGISTRY_URI", "").strip()
        )

        # If no GCS URI is provided:
        # - If local artifacts exist, fall back to local (keeps tests/dev working)
        # - If local artifacts don't exist, error
        if not gcs_uri:
            if local_ok:
                logging.warning(
                    "FORCE_MODEL_DOWNLOAD is set but no GCS URI provided; using local artifacts."
                )
                return
            raise FileNotFoundError(
                f"Artifacts not found locally in {self.config.artifacts_dir} and GCS_ARTIFACTS_URI is not set."
            )

        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)

        logging.info(
            f"Downloading required artifacts from {gcs_uri} -> {self.config.artifacts_dir} (force={force})"
        )

        if force:
            for f in core_required + pipeline_required:
                p = self.config.artifacts_dir / f
                if p.exists():
                    p.unlink()

        download_artifacts_from_gcs(
            gcs_uri=gcs_uri,
            local_dir=self.config.artifacts_dir,
            filenames=core_required,
        )

        download_artifacts_from_gcs(
            gcs_uri=gcs_uri,
            local_dir=self.config.artifacts_dir,
            filenames=pipeline_required,
        )

        missing_core = [
            f for f in core_required if not (self.config.artifacts_dir / f).exists()
        ]
        has_pipeline = (
            self.config.artifacts_dir / CONFIG.artifacts.pipeline_name
        ).exists()
        if missing_core or not has_pipeline:
            raise FileNotFoundError(
                "Downloaded artifacts are incomplete. "
                f"Missing core files: {missing_core}. "
                "Expected pipeline.pkl."
            )

    def _load_artifacts(self) -> Tuple[Any, Any]:
        # Fast path: already loaded in memory
        if self._preprocessor is not None and self._model is not None:
            return self._preprocessor, self._model

        # Slow path: ensure + load once (thread-safe)
        with self._lock:
            if self._preprocessor is not None and self._model is not None:
                return self._preprocessor, self._model

            self._ensure_artifacts()

            if not self.config.pipeline_path.exists():
                raise FileNotFoundError(
                    f"Missing required inference artifact: {self.config.pipeline_path}"
                )

            self._pipeline = load_object(str(self.config.pipeline_path))
            named_steps = getattr(self._pipeline, "named_steps", None)
            if (
                named_steps is not None
                and "preprocessor" in named_steps
                and "model" in named_steps
            ):
                # Extract steps so schema endpoint can inspect preprocessor.
                self._preprocessor = named_steps["preprocessor"]
                self._model = named_steps["model"]
            else:
                raise ValueError(
                    "pipeline.pkl exists but does not contain expected steps "
                    "['preprocessor', 'model']."
                )

            return self._preprocessor, self._model

    def _to_dataframe(
        self, X: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]
    ) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, dict):
            return pd.DataFrame([X])
        if isinstance(X, list) and all(isinstance(r, dict) for r in X):
            return pd.DataFrame(X)
        raise TypeError("X must be a pandas DataFrame, a dict, or a list of dicts.")

    def _align_to_training_schema(
        self, df: pd.DataFrame, preprocessor: Any
    ) -> pd.DataFrame:
        required = getattr(preprocessor, "feature_names_in_", None)
        if required is None:
            return df

        required_cols = list(required)
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for prediction: {missing}")

        return df.reindex(columns=required_cols)

    def predict(
        self, X: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]
    ) -> np.ndarray:
        logging.info("Prediction started")
        try:
            preprocessor, model = self._load_artifacts()
            df = self._to_dataframe(X)

            # Drop target if accidentally included
            target = CONFIG.dataset.target_col
            if target in df.columns:
                df = df.drop(columns=[target])

            # Drop configured leak columns if present
            drop_cols = CONFIG.dataset.drop_cols or []
            cols_to_drop = [c for c in drop_cols if c in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)

            df = self._align_to_training_schema(df, preprocessor)

            # If we loaded a combined inference pipeline, use it directly so
            # the preprocessor and model are always called as a single unit.
            if self._pipeline is not None:
                preds = self._pipeline.predict(df)
            else:
                X_transformed = preprocessor.transform(df)
                preds = model.predict(X_transformed)

            preds = np.asarray(preds).ravel()
            logging.info(f"Prediction completed. Output shape: {preds.shape}")
            return preds

        except ValueError as e:
            # user-input/schema error: keep it as ValueError
            logging.exception(f"Bad prediction input: {e}")
            raise  # Re-raise the ValueError directly

        except Exception as e:
            logging.exception("Prediction failed")
            raise CustomException(e, sys)

    def predict_with_assessment(
        self, X: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        preds = self.predict(X)
        output: List[Dict[str, Any]] = []
        for raw_pred in preds:
            pred = float(raw_pred)
            risk_prob = self._risk_probability(pred)
            output.append(
                {
                    "score_prediction": pred,
                    "score_range": list(self._score_range(pred)),
                    "performance_band": self._performance_band(pred),
                    "risk_probability": risk_prob,
                    "risk_tier": self._risk_tier(risk_prob),
                }
            )
        return output
