from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# ----------------------------
# Core schema / dataset config
# ----------------------------


@dataclass
class DatasetConfig:
    """
    Central place to define:
      - where raw data lives
      - what the target column is
      - which columns are excluded from features (drop_cols)
    """

    data_rel_path: Path = Path("data/raw/stud.csv")

    target_col: str = "math_score"

    # Columns you want to exclude from training/prediction features.
    # (Example: if you treat reading/writing as leakage or you intentionally
    # want to predict math without them.)
    drop_cols: List[str] = field(
        default_factory=lambda: ["reading_score", "writing_score"]
    )


# ----------------------------
# Train-test split config
# ----------------------------


@dataclass
class SplitConfig:
    test_size: float = 0.2
    random_state: int = 42
    shuffle: bool = True


# ----------------------------
# Artifacts config
# ----------------------------


@dataclass
class ArtifactsConfig:
    """
    All artifact outputs are anchored under repo_root/artifacts by using find_project_root().
    Your components should set artifacts_dir = repo_root / artifacts_dir_name.
    """

    artifacts_dir_name: str = "artifacts"

    raw_csv_name: str = "raw_data.csv"
    train_parquet_name: str = "train.parquet"
    test_parquet_name: str = "test.parquet"
    ingestion_meta_name: str = "ingestion_meta.json"

    preprocessor_name: str = "preprocessor.pkl"
    model_name: str = "model.pkl"
    model_report_name: str = "model_report.json"

    def artifacts_dir(self, repo_root: Path) -> Path:
        return repo_root / self.artifacts_dir_name

    def raw_data_path(self, repo_root: Path) -> Path:
        return self.artifacts_dir(repo_root) / self.raw_csv_name

    def train_path(self, repo_root: Path) -> Path:
        return self.artifacts_dir(repo_root) / self.train_parquet_name

    def test_path(self, repo_root: Path) -> Path:
        return self.artifacts_dir(repo_root) / self.test_parquet_name

    def ingestion_meta_path(self, repo_root: Path) -> Path:
        return self.artifacts_dir(repo_root) / self.ingestion_meta_name

    def preprocessor_path(self, repo_root: Path) -> Path:
        return self.artifacts_dir(repo_root) / self.preprocessor_name

    def model_path(self, repo_root: Path) -> Path:
        return self.artifacts_dir(repo_root) / self.model_name

    def model_report_path(self, repo_root: Path) -> Path:
        return self.artifacts_dir(repo_root) / self.model_report_name


# ----------------------------
# Training / tuning config
# ----------------------------


@dataclass
class TuningConfig:
    """
    Tuning knobs used by evaluate_models (RandomizedSearch -> refined GridSearch).
    """

    scoring: str = "r2"
    cv: int = 5
    n_jobs: int = -1
    verbose: int = 0

    random_n_iter: int = 25
    random_seed: int = 42

    prefer_cv_for_selection: bool = True


@dataclass
class DenseSafetyConfig:
    """
    Safety limits when densifying sparse matrices for models that require dense.
    """

    dense_feature_threshold: int = 5000
    dense_cell_threshold: int = 5_000_000  # rows * features


# ----------------------------
# Full pipeline config bundle
# ----------------------------


@dataclass
class PipelineConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    dense_safety: DenseSafetyConfig = field(default_factory=DenseSafetyConfig)


# Single shared instance (optional, but convenient)
CONFIG = PipelineConfig()
