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

def _download_artifacts(uri: str, local_dir: Path, filenames: list) -> None:
    """Dispatch artifact download to the correct backend based on URI scheme."""
    if uri.startswith("gs://"):
        from student_performance.artifacts_gcs import download_artifacts_from_gcs
        download_artifacts_from_gcs(gcs_uri=uri, local_dir=local_dir, filenames=filenames)
    elif uri.startswith("s3://"):
        from student_performance.artifacts_s3 import download_artifacts_from_s3
        download_artifacts_from_s3(s3_uri=uri, local_dir=local_dir, filenames=filenames)
    else:
        raise ValueError(f"Unsupported artifact URI scheme: {uri}")

def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class PredictPipelineConfig:
    artifacts_dir: Path
    preprocessor_path: Path
    model_path: Path
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
            report_path=artifacts_dir / "model_report.json",
            ingestion_meta_path=artifacts_dir / "ingestion_meta.json",
        )

        # in-memory cache
        self._preprocessor: Any = None
        self._model: Any = None

    def _ensure_artifacts(self) -> None:
        force = _env_flag("FORCE_MODEL_DOWNLOAD", "0")

        required = [
            CONFIG.artifacts.preprocessor_name,
            CONFIG.artifacts.model_name,
            "model_report.json",
            "ingestion_meta.json",
        ]

        local_ok = all((self.config.artifacts_dir / f).exists() for f in required)

        if local_ok and not force:
            return

        # Resolve artifact URI — support both GCS and S3 env vars
        artifact_uri = (
            os.getenv("GCS_ARTIFACTS_URI", "").strip()   # GCP Cloud Run sets this
            or os.getenv("S3_ARTIFACTS_URI", "").strip()  # AWS ECS sets this
            or os.getenv("MODEL_REGISTRY_URI", "").strip() # legacy fallback
        )

        if not artifact_uri:
            if local_ok:
                logging.warning(
                    "FORCE_MODEL_DOWNLOAD is set but no artifact URI provided; using local artifacts."
                )
                return
            raise FileNotFoundError(
                f"Artifacts not found locally in {self.config.artifacts_dir} "
                "and neither GCS_ARTIFACTS_URI nor S3_ARTIFACTS_URI is set."
            )

        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)

        logging.info(
            f"Downloading required artifacts from {artifact_uri} -> "
            f"{self.config.artifacts_dir} (force={force})"
        )

        if force:
            for f in required:
                p = self.config.artifacts_dir / f
                if p.exists():
                    p.unlink()

        _download_artifacts(artifact_uri, self.config.artifacts_dir, required)

        missing = [f for f in required if not (self.config.artifacts_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Downloaded from {artifact_uri} but missing required files: {missing}"
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

            self._preprocessor = load_object(str(self.config.preprocessor_path))
            self._model = load_object(str(self.config.model_path))
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
