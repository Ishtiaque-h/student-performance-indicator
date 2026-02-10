import sys, os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from student_performance.exception import CustomException
from student_performance.logger import logging
from student_performance.utils import find_project_root

from student_performance.components.config import CONFIG
from student_performance.components.data_ingestion import DataIngestion
from student_performance.components.data_transformation import DataTransformation
from student_performance.components.model_trainer import ModelTrainer
from student_performance.mlops.mlflow_logger import log_training_run


@dataclass
class TrainPipelineConfig:
    artifacts_dir: Path = Path("artifacts")


class TrainPipeline:
    def __init__(self):
        self.config = TrainPipelineConfig()

    def run(self) -> Tuple[str, Dict[str, Any]]:
        """
        Orchestrates: Ingestion -> Transformation -> Training
        Returns: best_model_name, full_report_dict
        """
        logging.info("===== TRAIN PIPELINE STARTED =====")

        try:
            repo_root = find_project_root()

            # Centralized artifacts directory (single source of truth)
            self.config.artifacts_dir = CONFIG.artifacts.artifacts_dir(repo_root)
            self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)

            logging.info(f"Repo root: {repo_root}")
            logging.info(f"Artifacts dir: {self.config.artifacts_dir}")
            logging.info(f"Dataset path: {repo_root / CONFIG.dataset.data_rel_path}")
            logging.info(f"Target column: {CONFIG.dataset.target_col}")
            logging.info(f"Drop cols: {CONFIG.dataset.drop_cols}")

            # 1) Ingestion
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()
            logging.info(f"Ingestion output -> train: {train_path} | test: {test_path}")

            # 2) Transformation
            transformer = DataTransformation()
            X_train, y_train, X_test, y_test, preprocessor_path = transformer.initiate_data_transformation(
                train_path, test_path
            )
            logging.info(f"Transformation output -> preprocessor: {preprocessor_path}")
            logging.info(f"X_train shape: {getattr(X_train, 'shape', None)} | y_train shape: {y_train.shape}")
            logging.info(f"X_test shape: {getattr(X_test, 'shape', None)} | y_test shape: {y_test.shape}")

            # 3) Training
            trainer = ModelTrainer()
            best_model_name, report = trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)

            log_training_run(
                best_model_name=best_model_name,
                report=report,
                artifacts_dir=self.config.artifacts_dir,
                run_name=os.getenv("GITHUB_REF_NAME"),
            )

            logging.info(f"===== TRAIN PIPELINE COMPLETED: best_model={best_model_name} =====")
            return best_model_name, report

        except Exception as e:
            logging.exception("TRAIN PIPELINE FAILED")
            raise CustomException(e, sys)

