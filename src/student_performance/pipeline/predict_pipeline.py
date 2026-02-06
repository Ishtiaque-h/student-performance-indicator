import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from student_performance.exception import CustomException
from student_performance.logger import logging
from student_performance.utils import find_project_root, load_object
from student_performance.components.config import CONFIG


@dataclass
class PredictPipelineConfig:
    artifacts_dir: Path = Path("artifacts")
    preprocessor_path: Path = artifacts_dir / "preprocessor.pkl"
    model_path: Path = artifacts_dir / "model.pkl"


class PredictPipeline:
    def __init__(self):
        self.config = PredictPipelineConfig()

        repo_root = find_project_root()
        self.config.artifacts_dir = CONFIG.artifacts.artifacts_dir(repo_root)
        self.config.preprocessor_path = CONFIG.artifacts.preprocessor_path(repo_root)
        self.config.model_path = CONFIG.artifacts.model_path(repo_root)

    def _load_artifacts(self):
        if not self.config.preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found at: {self.config.preprocessor_path}")
        if not self.config.model_path.exists():
            raise FileNotFoundError(f"Model not found at: {self.config.model_path}")

        preprocessor = load_object(str(self.config.preprocessor_path))
        model = load_object(str(self.config.model_path))
        return preprocessor, model

    def _to_dataframe(self, X: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, dict):
            return pd.DataFrame([X])
        if isinstance(X, list) and all(isinstance(r, dict) for r in X):
            return pd.DataFrame(X)
        raise TypeError("X must be a pandas DataFrame, a dict, or a list of dicts.")

    def _align_to_training_schema(self, df: pd.DataFrame, preprocessor: Any) -> pd.DataFrame:
        """
        Align inference input to exactly the columns the preprocessor was fit on.
        - validates missing required columns
        - drops extra columns by reindexing
        """
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

            # Drop target if someone accidentally includes it
            target = CONFIG.dataset.target_col
            if target in df.columns:
                df = df.drop(columns=[target])

            # Drop any configured drop_cols (keeps inference consistent with training)
            drop_cols = CONFIG.dataset.drop_cols or []
            cols_to_drop = [c for c in drop_cols if c in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)

            # Enforce training schema
            df = self._align_to_training_schema(df, preprocessor)

            logging.info(f"Input shape (before transform): {df.shape}")

            X_transformed = preprocessor.transform(df)
            preds = model.predict(X_transformed)

            preds = np.asarray(preds).ravel()
            logging.info(f"Prediction completed. Output shape: {preds.shape}")
            return preds

        except Exception as e:
            logging.error("Prediction failed")
            raise CustomException(e, sys)


if __name__ == "__main__":
    sample = {
        "gender": "female",
        "race_ethnicity": "group B",
        "parental_level_of_education": "bachelor's degree",
        "lunch": "standard",
        "test_preparation_course": "none",
    }

    preds = PredictPipeline().predict(sample)
    print(f"Predicted {CONFIG.dataset.target_col}: {preds[0]:.2f}")
