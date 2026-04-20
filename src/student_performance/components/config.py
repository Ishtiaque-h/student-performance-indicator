from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

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

    Deployment scenario
    -------------------
    This model is used at the **point of enrolment** — before any exams
    have been taken.  Only demographic and administrative information is
    available at that time (gender, race/ethnicity, parental education,
    lunch programme, test-prep enrolment).

    reading_score and writing_score would be perfect predictors of
    math_score (they are all measured on the same sitting), but they are
    NOT available at inference time in the target deployment scenario —
    including them would constitute target leakage.  They are therefore
    listed in drop_cols so they are excluded from both training features
    and prediction inputs.
    """

    data_rel_path: Path = Path("data/raw/stud.csv")

    target_col: str = "math_score"

    # Columns excluded from training features because they are not
    # available at prediction time (see deployment scenario above).
    drop_cols: List[str] = field(
        default_factory=lambda: ["reading_score", "writing_score"]
    )

    # Allowed [min, max] range for numeric input features at inference time.
    # Keys must match column names used as prediction features (not drop_cols).
    # Out-of-range values are rejected with HTTP 422 before they reach the
    # model, preventing silent nonsense predictions.
    # Example: if you add a numeric feature "age", set {"age": (5, 25)}.
    numeric_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)


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
    # Combined preprocessor+model sklearn Pipeline written after training.
    # predict_pipeline.py uses this single artifact so the same fitted
    # ColumnTransformer that was used during training is always paired with
    # the model — eliminating any risk of train/serve skew.
    pipeline_name: str = "pipeline.pkl"
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

    def pipeline_path(self, repo_root: Path) -> Path:
        return self.artifacts_dir(repo_root) / self.pipeline_name

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
# Product output config
# ----------------------------


@dataclass
class ProductConfig:
    """
    Product-facing prediction settings.
    The primary output is risk + banding, with score estimate as secondary.
    """

    # Student is considered at-risk if expected score trends below this threshold.
    risk_threshold_score: float = 50.0
    # Controls risk-probability smoothness around risk_threshold_score.
    risk_probability_scale: float = 10.0
    # Probability cutoffs for operational risk tiers.
    risk_tier_medium_min: float = 0.40
    risk_tier_high_min: float = 0.70
    # Performance bands based on expected score.
    performance_band_low_max: float = 50.0
    performance_band_medium_max: float = 70.0


# ----------------------------
# Monitoring SLO/SLA config
# ----------------------------


@dataclass
class MonitoringSLOConfig:
    """
    SLOs and guardrails for monitoring the model in production.
     - Service SLOs ensure the prediction service is performant and reliable.
     - Model freshness SLO ensures the model is retrained before it becomes stale.
     - Online quality guardrails set expected performance thresholds on recent predictions to catch data drift or other issues.
     Adjust these values based on your specific requirements and risk tolerance.
        Note: these are just example values for demonstration purposes.  In a real deployment, you would set these based on your specific requirements and risk tolerance.
    """
    # Service SLOs
    max_prediction_latency_ms_p95: float = 250.0    # 95th percentile latency in milliseconds
    min_api_availability: float = 0.995             # Minimum acceptable availability (e.g. 99.5%)
    max_error_rate: float = 0.02                    # Maximum acceptable error rate (e.g. 2% of requests resulting in errors)

    # Model freshness SLOs
    max_model_age_hours: float = 24.0 * 30          # (e.g. retrain if model is older than 30 days)

    # Online quality guardrails (e.g. monitored on a rolling window of recent predictions)
    min_r2: float = 0.05
    max_mae: float = 15.0
    max_rmse: float = 20.0


# ----------------------------
# Monitoring alerting config
# ----------------------------


@dataclass
class MonitoringAlertThresholdConfig:
    """
    Thresholds for triggering alerts when SLOs or guardrails are violated.
    Adjust these values based on your specific requirements and risk tolerance.
        Note: these are just example values for demonstration purposes.  In a real deployment, you would set these based on your specific requirements and risk tolerance.
    """
    # Drift thresholds
    categorical_drift_warning: float = 0.15         # e.g. 15% of predictions in a category have changed distribution compared to training data
    categorical_drift_critical: float = 0.25
    prediction_drift_warning: float = 0.10          # e.g. overall distribution of predictions has changed by 10% compared to training data
    prediction_drift_critical: float = 0.20
    segment_drift_warning: float = 0.15             # e.g. performance on a specific segment (e.g. demographic group) has degraded by 15% compared to training data
    segment_drift_critical: float = 0.25

    # Performance degradation thresholds ( relative to baseline)
    r2_drop_warning: float = 0.10                   # e.g. R2 has dropped by 10% compared to baseline performance on training data
    r2_drop_critical: float = 0.20
    mae_increase_warning: float = 0.10              # e.g. MAE has increased by 10% compared to baseline performance on training data
    mae_increase_critical: float = 0.20
    rmse_increase_warning: float = 0.10             # e.g. RMSE has increased by 10% compared to baseline performance on training data
    rmse_increase_critical: float = 0.20

    # Service degradation thresholds
    latency_warning_multiplier: float = 1.0         # e.g. 100% increase in latency compared to baseline latency during training
    latency_critical_multiplier: float = 1.4        
    error_rate_warning_multiplier: float = 1.0      # e.g. 100% increase in error rate compared to baseline error rate during training
    error_rate_critical_multiplier: float = 2.0     


# ----------------------------
# Combined monitoring config
# ----------------------------
@dataclass
class MonitoringConfig:
    """
    Overall monitoring configuration, combining SLOs and alert thresholds, as well as controls for how monitoring is implemented.
    """

    enabled: bool = True
    # Default online monitoring segments for this project
    segment_columns: Tuple[str, ...] = ("gender", "lunch", "race_ethnicity")
    # Rolling windows for delayed-label quality evaluation
    rolling_windows: Tuple[int, ...] = (50, 200)
    # Automation controls for retraining and alerting
    sustained_breach_windows_for_retrain: int = 3   # Number of consecutive windows breaching thresholds to trigger retraining
    min_canary_sample_size: int = 50                # Minimum number of predictions in a canary test to consider the results statistically meaningful
    # Governance controls for monitoring access and alerting
    fairness_segment_max_mae_gap: float = 3.0       # Maximum acceptable gap in MAE between any two demographic segments before triggering a fairness alert
    promote_requires_segment_improvement: bool = True # Whether to require improvement in all segments before promoting a new model, or allow promotion if overall metrics improve even if some segments degrade

    slo: MonitoringSLOConfig = field(default_factory=MonitoringSLOConfig)
    alert_thresholds: MonitoringAlertThresholdConfig = field(default_factory=MonitoringAlertThresholdConfig)


# ----------------------------
# Full pipeline config bundle
# ----------------------------


@dataclass
class PipelineConfig:
    """
    Centralized configuration for the entire pipeline, combining all sub-configs.
    This makes it easy to pass around a single config object to all components, while still keeping related settings organized in logical groups.
    """
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    dense_safety: DenseSafetyConfig = field(default_factory=DenseSafetyConfig)
    product: ProductConfig = field(default_factory=ProductConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)


# Single shared instance (optional, but convenient)
CONFIG = PipelineConfig()
