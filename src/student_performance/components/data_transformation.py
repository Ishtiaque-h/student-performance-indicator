import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from student_performance.exception import CustomException
from student_performance.logger import logging
from student_performance.utils import save_object, find_project_root
from student_performance.components.config import CONFIG


@dataclass
class DataTransformationConfig:
    artifacts_dir: Path = Path("artifacts")
    preprocessor_obj_file_path: Path = artifacts_dir / "preprocessor.pkl"


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, X: pd.DataFrame) -> ColumnTransformer:
        try:
            num_features = X.select_dtypes(exclude=["object", "category", "string"]).columns.to_list()
            cat_features = X.select_dtypes(include=["object", "category", "string"]).columns.to_list()

            transformers = []

            if cat_features:
                cat_pipeline = Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ])
                transformers.append(("categorical", cat_pipeline, cat_features))

            if num_features:
                num_pipeline = Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ])
                transformers.append(("numeric", num_pipeline, num_features))

            if not transformers:
                raise ValueError(
                    "No numeric or categorical features found for preprocessing. "
                    "Check target/drop settings or input data."
                )

            return ColumnTransformer(transformers=transformers)

        except Exception as e:
            logging.error("Error in get_data_transformer_object")
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self, train_path: str, test_path: str
    ) -> Tuple[Any, Any, Any, Any, str]:
        try:
            train_df = pd.read_parquet(train_path)
            test_df = pd.read_parquet(test_path)
            logging.info(f"Read train/test completed | train={train_df.shape}, test={test_df.shape}")

            target = CONFIG.dataset.target_col
            if target not in train_df.columns or target not in test_df.columns:
                raise KeyError(f"Target column '{target}' not found in train/test data.")

            drop_cols = CONFIG.dataset.drop_cols or []
            cols_to_drop = [c for c in drop_cols if c in train_df.columns]

            X_train_df = train_df.drop(columns=cols_to_drop + [target])
            y_train = train_df[target]

            X_test_df = test_df.drop(columns=cols_to_drop + [target])
            y_test = test_df[target]

            if X_train_df.shape[1] == 0:
                raise ValueError(
                    "No feature columns left after dropping target/drop_cols. "
                    "Update CONFIG.dataset.drop_cols or dataset columns."
                )

            logging.info("Building preprocessing object from X_train")
            preprocessor = self.get_data_transformer_object(X_train_df)

            logging.info("Applying preprocessor on train/test")
            X_train_arr = preprocessor.fit_transform(X_train_df)
            X_test_arr = preprocessor.transform(X_test_df)

            repo_root = find_project_root()
            artifacts_dir = CONFIG.artifacts.artifacts_dir(repo_root)
            preprocessor_path = CONFIG.artifacts.preprocessor_path(repo_root)
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            save_object(file_path=str(preprocessor_path), obj=preprocessor)
            logging.info(f"Saved preprocessor at {preprocessor_path}")

            logging.info("Data transformation completed")
            return (
                X_train_arr, y_train.to_numpy(),
                X_test_arr, y_test.to_numpy(),
                str(preprocessor_path),
            )

        except Exception as e:
            logging.exception("Error occurred during data transformation")
            raise CustomException(e, sys)
