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
    report_path: Path
    ingestion_meta_path: Path


class PredictPipeline:
    _lock = Lock()  # protects download + first load

    def __init__(self):
        root = find_project_root()
        self._loaded_from_uri: str | None = None

        default_artifacts_dir = CONFIG.artifacts.artifacts_dir(root)
        artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", str(default_artifacts_dir))).expanduser().resolve()

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

        gcs_uri = (
            os.getenv("GCS_ARTIFACTS_URI", "").strip()
            or os.getenv("MODEL_REGISTRY_URI", "").strip()  # legacy fallback
        )
        if not gcs_uri:
            raise FileNotFoundError(
                f"Artifacts not found locally in {self.config.artifacts_dir} and GCS_ARTIFACTS_URI is not set."
            )

        # If we already loaded from a different URI in this process, re-download
        uri_changed = self._loaded_from_uri is not None and self._loaded_from_uri != gcs_uri

        # If artifacts exist and we are not forcing and URI didn't change, we're done
        if (not force) and (not uri_changed) and self.config.preprocessor_path.exists() and self.config.model_path.exists():
            return

        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)

        logging.info(
            f"Downloading artifacts from {gcs_uri} -> {self.config.artifacts_dir} "
            f"(force={force}, uri_changed={uri_changed})"
        )

        # If we are going to re-download, drop in-memory cache
        self._preprocessor = None
        self._model = None

        download_artifacts_from_gcs(
            gcs_uri=gcs_uri,
            local_dir=self.config.artifacts_dir,
            filenames=[
                CONFIG.artifacts.preprocessor_name,
                CONFIG.artifacts.model_name,
                # include these only if you want them locally too
                "model_report.json",
                "ingestion_meta.json",
            ],
        )

        if not (self.config.preprocessor_path.exists() and self.config.model_path.exists()):
            raise FileNotFoundError(
                f"Downloaded from {gcs_uri} but expected files not found: "
                f"{self.config.preprocessor_path.name}, {self.config.model_path.name}"
            )

        self._loaded_from_uri = gcs_uri

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

    def _to_dataframe(self, X: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, dict):
            return pd.DataFrame([X])
        if isinstance(X, list) and all(isinstance(r, dict) for r in X):
            return pd.DataFrame(X)
        raise TypeError("X must be a pandas DataFrame, a dict, or a list of dicts.")

    def _align_to_training_schema(self, df: pd.DataFrame, preprocessor: Any) -> pd.DataFrame:
        required = getattr(preprocessor, "feature_names_in_", None)
        if required is None:
            return df

        required_cols = list(required)
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for prediction: {missing}")

        return df.reindex(columns=required_cols)

    def predict(self, X: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> np.ndarray:
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
            logging.error(f"Bad prediction input: {e}")
            raise

        except Exception as e:
            logging.error("Prediction failed")
            raise CustomException(e, sys)
