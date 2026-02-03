import sys, json
import pandas as pd
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from student_performance.exception import CustomException
from student_performance.logger import logging


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return start

def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

@dataclass
class DataIngestionConfig:
    artifacts_dir: Path = Path("artifacts")
    raw_data_path: Path = artifacts_dir / "raw_data.csv"
    train_data_path: Path = artifacts_dir / "train.parquet"
    test_data_path: Path = artifacts_dir / "test.parquet"
    meta_path: Path = artifacts_dir / "ingestion_meta.json"
    test_size: float = 0.2
    random_state: int = 42


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            project_root = find_project_root()

            # anchor artifacts inside repo
            cfg = self.ingestion_config
            cfg.artifacts_dir = project_root / "artifacts"
            cfg.raw_data_path = cfg.artifacts_dir / "raw_data.csv"
            cfg.train_data_path = cfg.artifacts_dir / "train.parquet"
            cfg.test_data_path = cfg.artifacts_dir / "test.parquet"
            cfg.meta_path = cfg.artifacts_dir / "ingestion_meta.json"

            logging.info(f"Artifacts directory: {cfg.artifacts_dir}")

            data_path = project_root / "data" / "raw" / "stud.csv"
            if not data_path.exists():
                raise FileNotFoundError(f"Dataset not found at: {data_path}")

            df = pd.read_csv(data_path)
            logging.info(f"Dataset read successfully from: {data_path}")

            cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

            # immutable raw
            if cfg.raw_data_path.exists():
                logging.info(f"Raw data already exists at {cfg.raw_data_path}; not overwriting.")
            else:
                df.to_csv(cfg.raw_data_path, index=False)
                logging.info(f"Raw data saved at {cfg.raw_data_path}")

            # split
            train_set, test_set = train_test_split(df, test_size= cfg.test_size, random_state= cfg.random_state, shuffle=True)

            # save splits
            train_set.to_parquet(cfg.train_data_path, index=False)
            test_set.to_parquet(cfg.test_data_path, index=False)
            logging.info(f"Train data saved at {cfg.train_data_path}")
            logging.info(f"Test data saved at {cfg.test_data_path}")

            # metadata
            meta = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "source_path": str(data_path),
                "n_rows": int(df.shape[0]),
                "n_cols": int(df.shape[1]),
                "test_size": cfg.test_size,
                "random_state": cfg.random_state,
                "train_rows": int(train_set.shape[0]),
                "test_rows": int(test_set.shape[0]),
                "train_path": str(cfg.train_data_path),
                "test_path": str(cfg.test_data_path),
                "source_md5": file_md5(data_path),
            }
            cfg.meta_path.write_text(json.dumps(meta, indent=2))
            logging.info(f"Saved ingestion metadata at {cfg.meta_path}")

            logging.info("Data Ingestion completed")

            return str(cfg.train_data_path), str(cfg.test_data_path)

        except Exception as e:
            logging.error("Error occurred during data ingestion")
            raise CustomException(e, sys)


if __name__ == "__main__":
    DataIngestion().initiate_data_ingestion()
