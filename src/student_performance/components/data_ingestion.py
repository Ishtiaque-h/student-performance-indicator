import sys
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from student_performance.exception import CustomException
from student_performance.logger import logging
from student_performance.utils import find_project_root
from student_performance.components.config import CONFIG


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class DataIngestion:
    def initiate_data_ingestion(self) -> Tuple[str, str]:
        logging.info("Data Ingestion started")
        try:
            repo_root = find_project_root()

            artifacts_dir = CONFIG.artifacts.artifacts_dir(repo_root)
            raw_data_path = CONFIG.artifacts.raw_data_path(repo_root)
            train_data_path = CONFIG.artifacts.train_path(repo_root)
            test_data_path = CONFIG.artifacts.test_path(repo_root)
            meta_path = CONFIG.artifacts.ingestion_meta_path(repo_root)

            artifacts_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Artifacts directory: {artifacts_dir}")

            data_path = repo_root / CONFIG.dataset.data_rel_path
            if not data_path.exists():
                raise FileNotFoundError(f"Dataset not found at: {data_path}")

            df = pd.read_csv(data_path)
            logging.info(
                f"Dataset read successfully from: {data_path} | shape={df.shape}"
            )

            # immutable raw snapshot (don’t overwrite)
            if raw_data_path.exists():
                logging.info(
                    f"Raw data already exists at {raw_data_path}; not overwriting."
                )
            else:
                df.to_csv(raw_data_path, index=False)
                logging.info(f"Raw data saved at {raw_data_path}")

            # split
            train_set, test_set = train_test_split(
                df,
                test_size=CONFIG.split.test_size,
                random_state=CONFIG.split.random_state,
                shuffle=CONFIG.split.shuffle,
            )

            train_set.to_parquet(train_data_path, index=False)
            test_set.to_parquet(test_data_path, index=False)
            logging.info(
                f"Train data saved at {train_data_path} | shape={train_set.shape}"
            )
            logging.info(
                f"Test data saved at {test_data_path} | shape={test_set.shape}"
            )

            meta = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "source_path": str(data_path),
                "source_md5": file_md5(data_path),
                "n_rows": int(df.shape[0]),
                "n_cols": int(df.shape[1]),
                "split": {
                    "test_size": CONFIG.split.test_size,
                    "random_state": CONFIG.split.random_state,
                    "shuffle": CONFIG.split.shuffle,
                },
                "artifacts": {
                    "raw_data_path": str(raw_data_path),
                    "train_path": str(train_data_path),
                    "test_path": str(test_data_path),
                },
                "rows": {
                    "train_rows": int(train_set.shape[0]),
                    "test_rows": int(test_set.shape[0]),
                },
            }
            meta_path.write_text(json.dumps(meta, indent=2))
            logging.info(f"Saved ingestion metadata at {meta_path}")

            logging.info("Data Ingestion completed")
            return str(train_data_path), str(test_data_path)

        except Exception as e:
            logging.exception("Error occurred during data ingestion")
            raise CustomException(e, sys)
