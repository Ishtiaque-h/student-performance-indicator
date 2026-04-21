from __future__ import annotations
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
import numpy as np
from student_performance.components.config import CONFIG

DRIFT_BASELINE_FILENAME = "monitoring_baseline.json"
INFERENCE_LOG_FILENAME = "inference_events.jsonl"


def utc_now_iso() -> str:
    """Get the current UTC time as an ISO 8601 formatted string."""
    return datetime.now(timezone.utc).isoformat()


def hash_value(value: Any) -> str:
    """Hash a value using SHA-256 and return the hexadecimal digest."""
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Compute a safe ratio, returning 0.0 if the denominator is zero."""
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _normalize_baseline_metric(value: Any) -> float | None:
    """Normalize a baseline metric value, ensuring it's a positive float or None if invalid."""
    if value is None:
        return None
    numeric = float(value)
    if numeric <= 0.0:
        return None
    return max(numeric, 1e-8)


def _to_distribution(values: Sequence[Any]) -> Dict[str, float]:
    """Convert a sequence of values into a normalized distribution dictionary."""
    if not values:
        return {}
    counts: Dict[str, int] = {}
    for value in values:
        key = str(value).strip().lower()
        counts[key] = counts.get(key, 0) + 1
    total = float(sum(counts.values()))
    return {key: count / total for key, count in counts.items()}


def total_variation_distance(
    baseline_distribution: Dict[str, float], current_distribution: Dict[str, float]
) -> float:
    """
    Calculate the Total Variation Distance between two distributions.
    TVD is defined as 0.5 * sum of absolute differences in probabilities for all keys.
    """
    keys = set(baseline_distribution) | set(current_distribution)
    tvd = 0.0
    for key in keys:
        tvd += abs(
            float(baseline_distribution.get(key, 0.0))
            - float(current_distribution.get(key, 0.0))
        )
    return float(0.5 * tvd)


def psi_score(
    baseline_values: Sequence[float], current_values: Sequence[float]
) -> float:
    """
    Calculate the Population Stability Index (PSI) between two sets of values.
    PSI is a metric that quantifies the difference between two probability distributions,
    often used to detect distribution drift in model monitoring. It ranges from 0 (identical distributions)
    to 1 (completely different distributions).
    Interpretation guidelines (commonly used but can vary by context):
    - PSI < 0.1: No significant drift
    - 0.1 <= PSI < 0.25: Moderate drift (monitor closely)
    - PSI >= 0.25: Significant drift (investigate and consider retraining)
    """
    if not baseline_values or not current_values:
        return 0.0
    baseline = np.asarray(baseline_values, dtype=float)
    current = np.asarray(current_values, dtype=float)
    bins = np.histogram_bin_edges(baseline, bins=10)
    bins[0] = min(bins[0], float(np.min(current)))
    bins[-1] = max(bins[-1], float(np.max(current)))
    baseline_hist, _ = np.histogram(baseline, bins=bins)
    current_hist, _ = np.histogram(current, bins=bins)
    baseline_pct = baseline_hist / max(float(np.sum(baseline_hist)), 1.0)
    current_pct = current_hist / max(float(np.sum(current_hist)), 1.0)
    epsilon = 1e-8
    baseline_pct = np.maximum(baseline_pct, epsilon)
    current_pct = np.maximum(current_pct, epsilon)
    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return float(max(psi, 0.0))


# ---------------------------------------------
# Data classes for structured representation of alerts,
# automation actions, and champion/challenger decisions
# ---------------------------------------------


@dataclass
class Alert:
    name: str
    severity: str
    metric: str
    current: float
    threshold: float
    message: str
    action: str


@dataclass
class AutomationAction:
    action_type: str
    reason: str
    details: Dict[str, Any]


@dataclass
class ChampionChallengerDecision:
    promote: bool
    summary: str
    checks: Dict[str, bool]
    metrics: Dict[str, float]


class InferenceLogger:
    def __init__(self, log_path: Path, hash_feature_values: bool = True):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.hash_feature_values = hash_feature_values

    def _sanitize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        for key, value in features.items():
            if self.hash_feature_values:
                sanitized[key] = hash_value(value)
            else:
                sanitized[key] = value
        return sanitized

    def log_event(
        self,
        *,
        request_id: str,
        model_version: str,
        endpoint: str,
        status_code: int,
        latency_ms: float,
        features: Dict[str, Any],
        prediction: float | None,
    ) -> Dict[str, Any]:
        """Log an inference event with all relevant details, including hashed features and prediction."""
        event = {
            "request_id": request_id,
            "timestamp": utc_now_iso(),
            "model_version": model_version,
            "endpoint": endpoint,
            "status_code": int(status_code),
            "latency_ms": float(latency_ms),
            "endpoint_status": "ok" if 200 <= int(status_code) < 400 else "error",
            "features": self._sanitize_features(features),
            "prediction": None if prediction is None else float(prediction),
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")
        return event


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSON Lines file and return a list of dictionaries."""
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_training_baseline(
    *,
    train_rows: List[Dict[str, Any]],
    train_predictions: Sequence[float],
    segment_columns: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Build a baseline dictionary from training data and predictions, including distributions for categorical features and segments."""
    segment_columns = segment_columns or CONFIG.monitoring.segment_columns
    categorical_drift_baseline: Dict[str, Dict[str, float]] = {}
    if train_rows:
        feature_names = list(train_rows[0].keys())
        for name in feature_names:
            categorical_drift_baseline[name] = _to_distribution(
                [row.get(name) for row in train_rows]
            )
    segment_baseline: Dict[str, Dict[str, float]] = {}
    for segment_col in segment_columns:
        segment_baseline[segment_col] = _to_distribution(
            [row.get(segment_col, "unknown") for row in train_rows]
        )
    baseline = {
        "generated_at": utc_now_iso(),
        "categorical_features": categorical_drift_baseline,
        "prediction_distribution_reference": [float(v) for v in train_predictions],
        "segments": segment_baseline,
    }
    return baseline


def save_training_baseline(path: Path, baseline: Dict[str, Any]) -> None:
    """Save the training baseline dictionary to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")


def load_training_baseline(path: Path) -> Dict[str, Any]:
    """Load the training baseline from a JSON file, returning an empty dictionary if the file does not exist."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def compute_online_drift(
    *,
    live_events: Sequence[Dict[str, Any]],
    baseline: Dict[str, Any],
    segment_columns: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """
    Compute drift metrics for live inference events compared to the training baseline, including
    categorical feature drift, prediction distribution drift, and segment distribution drift.
    """
    segment_columns = segment_columns or CONFIG.monitoring.segment_columns
    live_features = [event.get("features", {}) for event in live_events]
    live_predictions = [
        float(event["prediction"])
        for event in live_events
        if event.get("prediction") is not None
    ]
    categorical: Dict[str, float] = {}
    baseline_categorical = baseline.get("categorical_features", {})
    for feature_name, base_dist in baseline_categorical.items():
        curr_dist = _to_distribution(
            [row.get(feature_name, "unknown") for row in live_features]
        )
        categorical[feature_name] = total_variation_distance(base_dist, curr_dist)
    prediction_drift = psi_score(
        baseline.get("prediction_distribution_reference", []), live_predictions
    )
    segment_drift: Dict[str, float] = {}
    baseline_segments = baseline.get("segments", {})
    for segment_col in segment_columns:
        base_segment_dist = baseline_segments.get(segment_col, {})
        curr_segment_dist = _to_distribution(
            [row.get(segment_col, "unknown") for row in live_features]
        )
        segment_drift[segment_col] = total_variation_distance(
            base_segment_dist, curr_segment_dist
        )
    return {
        "categorical_drift": categorical,
        "prediction_distribution_drift": prediction_drift,
        "segment_drift": segment_drift,
        "sample_size": len(live_events),
    }


def _regression_metrics(
    y_true: Sequence[float], y_pred: Sequence[float]
) -> Dict[str, float]:
    """Compute regression performance metrics (R², MAE, RMSE) given true and predicted values."""
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)
    if len(y_true_arr) == 0:
        return {"r2": 0.0, "mae": 0.0, "rmse": 0.0, "count": 0.0}
    errors = y_true_arr - y_pred_arr
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    y_mean = float(np.mean(y_true_arr))
    ss_tot = float(np.sum(np.square(y_true_arr - y_mean)))
    ss_res = float(np.sum(np.square(errors)))
    r2 = 1.0 if ss_tot == 0 else float(1 - (ss_res / ss_tot))
    return {"r2": r2, "mae": mae, "rmse": rmse, "count": float(len(y_true_arr))}


def join_inference_with_labels(
    *,
    inference_events: Sequence[Dict[str, Any]],
    label_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Join inference events with their corresponding labels based on request_id, returning a list
    of dictionaries containing request_id, timestamp, prediction, label, and features.
    """
    labels_by_request: Dict[str, Dict[str, Any]] = {}
    for row in label_rows:
        key = str(row.get("request_id", "")).strip()
        if key:
            labels_by_request[key] = row
    joined: List[Dict[str, Any]] = []
    for event in inference_events:
        request_id = str(event.get("request_id", "")).strip()
        label_row = labels_by_request.get(request_id)
        if not label_row:
            continue
        if event.get("prediction") is None:
            continue
        joined.append(
            {
                "request_id": request_id,
                "timestamp": event.get("timestamp"),
                "prediction": float(event["prediction"]),
                "label": float(label_row["label"]),
                "features": event.get("features", {}),
            }
        )
    return joined


def compute_performance_monitoring(
    *,
    joined_rows: Sequence[Dict[str, Any]],
    rolling_windows: Sequence[int] | None = None,
    segment_columns: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """
    Compute performance monitoring metrics, including global regression metrics, rolling window metrics, and segment-based metrics.
    """
    rolling_windows = rolling_windows or CONFIG.monitoring.rolling_windows
    segment_columns = segment_columns or CONFIG.monitoring.segment_columns
    sorted_rows = sorted(joined_rows, key=lambda row: str(row.get("timestamp", "")))
    y_true = [row["label"] for row in sorted_rows]
    y_pred = [row["prediction"] for row in sorted_rows]
    global_metrics = _regression_metrics(y_true, y_pred)
    rolling: Dict[str, Dict[str, float]] = {}
    for window in rolling_windows:
        subset = sorted_rows[-int(window) :]
        rolling[str(window)] = _regression_metrics(
            [row["label"] for row in subset],
            [row["prediction"] for row in subset],
        )
    by_segment: Dict[str, Dict[str, Dict[str, float]]] = {}
    for segment_col in segment_columns:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for row in sorted_rows:
            segment_value = str(row.get("features", {}).get(segment_col, "unknown"))
            groups.setdefault(segment_value, []).append(row)
        by_segment[segment_col] = {
            segment_value: _regression_metrics(
                [item["label"] for item in rows], [item["prediction"] for item in rows]
            )
            for segment_value, rows in groups.items()
        }
    return {
        "global": global_metrics,
        "rolling": rolling,
        "by_segment": by_segment,
        "label_joined_count": len(sorted_rows),
    }


def compute_degradation(
    *,
    baseline_metrics: Dict[str, float],
    current_metrics: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute degradation metrics comparing current performance to baseline, including R² drop and percentage increases in MAE and RMSE.
    """
    baseline_r2 = float(baseline_metrics.get("r2", 0.0))
    baseline_mae = _normalize_baseline_metric(baseline_metrics.get("mae"))
    baseline_rmse = _normalize_baseline_metric(baseline_metrics.get("rmse"))
    current_r2 = float(current_metrics.get("r2", 0.0))
    current_mae = float(current_metrics.get("mae", 0.0))
    current_rmse = float(current_metrics.get("rmse", 0.0))
    mae_increase_ratio = (
        max(0.0, (current_mae - baseline_mae) / baseline_mae)
        if baseline_mae is not None
        else 0.0
    )
    rmse_increase_ratio = (
        max(0.0, (current_rmse - baseline_rmse) / baseline_rmse)
        if baseline_rmse is not None
        else 0.0
    )
    return {
        "r2_drop": max(0.0, baseline_r2 - current_r2),
        "mae_increase_ratio": mae_increase_ratio,
        "rmse_increase_ratio": rmse_increase_ratio,
    }


def _build_alert(
    *,
    name: str,
    metric: str,
    current: float,
    threshold: float,
    severity: str,
    reason: str,
    action: str,
) -> Alert:
    """Helper function to build an Alert object with a standardized message format."""
    return Alert(
        name=name,
        severity=severity,
        metric=metric,
        current=float(current),
        threshold=float(threshold),
        message=(
            f"[{severity.upper()}] {name}: {reason}. "
            f"Current={current:.4f}, threshold={threshold:.4f}. {action}"
        ),
        action=action,
    )


def evaluate_alerts(
    *,
    drift_metrics: Dict[str, Any],
    service_metrics: Dict[str, float],
    performance_metrics: Dict[str, Any],
    degradation_metrics: Dict[str, float],
) -> List[Alert]:
    """Evaluate all monitoring metrics against configured thresholds and SLOs to generate a list of alerts for any detected issues."""
    thresholds = CONFIG.monitoring.alert_thresholds
    slo = CONFIG.monitoring.slo
    alerts: List[Alert] = []
    for feature, drift_value in drift_metrics.get("categorical_drift", {}).items():
        if drift_value >= thresholds.categorical_drift_critical:
            alerts.append(
                _build_alert(
                    name=f"categorical_drift_{feature}",
                    metric="categorical_drift",
                    current=drift_value,
                    threshold=thresholds.categorical_drift_critical,
                    severity="critical",
                    reason=f"feature {feature} drift is high",
                    action="Investigate feature pipeline and trigger retrain candidate.",
                )
            )
        elif drift_value >= thresholds.categorical_drift_warning:
            alerts.append(
                _build_alert(
                    name=f"categorical_drift_{feature}",
                    metric="categorical_drift",
                    current=drift_value,
                    threshold=thresholds.categorical_drift_warning,
                    severity="warning",
                    reason=f"feature {feature} drift started",
                    action="Track trend and prepare retrain candidate if persistent.",
                )
            )
    prediction_drift = float(drift_metrics.get("prediction_distribution_drift", 0.0))
    if prediction_drift >= thresholds.prediction_drift_critical:
        alerts.append(
            _build_alert(
                name="prediction_distribution_drift",
                metric="prediction_distribution_drift",
                current=prediction_drift,
                threshold=thresholds.prediction_drift_critical,
                severity="critical",
                reason="prediction distribution changed sharply",
                action="Enable investigation, inspect canary cohort, and prepare rollback.",
            )
        )
    elif prediction_drift >= thresholds.prediction_drift_warning:
        alerts.append(
            _build_alert(
                name="prediction_distribution_drift",
                metric="prediction_distribution_drift",
                current=prediction_drift,
                threshold=thresholds.prediction_drift_warning,
                severity="warning",
                reason="prediction distribution drift increased",
                action="Increase monitoring frequency and watch service segments.",
            )
        )
    for segment_col, drift_value in drift_metrics.get("segment_drift", {}).items():
        if drift_value >= thresholds.segment_drift_critical:
            alerts.append(
                _build_alert(
                    name=f"segment_drift_{segment_col}",
                    metric="segment_drift",
                    current=drift_value,
                    threshold=thresholds.segment_drift_critical,
                    severity="critical",
                    reason=f"segment allocation changed for {segment_col}",
                    action="Run fairness/segment checks and prepare retraining.",
                )
            )
        elif drift_value >= thresholds.segment_drift_warning:
            alerts.append(
                _build_alert(
                    name=f"segment_drift_{segment_col}",
                    metric="segment_drift",
                    current=drift_value,
                    threshold=thresholds.segment_drift_warning,
                    severity="warning",
                    reason=f"segment allocation drift for {segment_col}",
                    action="Review segment funnel and continue trend monitoring.",
                )
            )
    availability = float(service_metrics.get("availability", 1.0))
    latency_p95 = float(service_metrics.get("latency_p95_ms", 0.0))
    error_rate = float(service_metrics.get("error_rate", 0.0))
    if availability < slo.min_api_availability:
        alerts.append(
            _build_alert(
                name="api_availability",
                metric="availability",
                current=availability,
                threshold=slo.min_api_availability,
                severity="critical",
                reason="API availability is below SLO",
                action="Page on-call and start incident triage immediately.",
            )
        )
    if (
        latency_p95
        > slo.max_prediction_latency_ms_p95 * thresholds.latency_critical_multiplier
    ):
        alerts.append(
            _build_alert(
                name="latency_p95",
                metric="latency_p95_ms",
                current=latency_p95,
                threshold=slo.max_prediction_latency_ms_p95
                * thresholds.latency_critical_multiplier,
                severity="critical",
                reason="prediction latency breached critical threshold",
                action="Scale service and rollback latest deployment if unresolved.",
            )
        )
    elif (
        latency_p95
        > slo.max_prediction_latency_ms_p95 * thresholds.latency_warning_multiplier
    ):
        alerts.append(
            _build_alert(
                name="latency_p95",
                metric="latency_p95_ms",
                current=latency_p95,
                threshold=slo.max_prediction_latency_ms_p95
                * thresholds.latency_warning_multiplier,
                severity="warning",
                reason="prediction latency above target",
                action="Investigate traffic and pod/resource pressure.",
            )
        )
    if error_rate > slo.max_error_rate * thresholds.error_rate_critical_multiplier:
        alerts.append(
            _build_alert(
                name="error_rate",
                metric="error_rate",
                current=error_rate,
                threshold=slo.max_error_rate
                * thresholds.error_rate_critical_multiplier,
                severity="critical",
                reason="error rate critically elevated",
                action="Page on-call and activate rollback plan.",
            )
        )
    elif error_rate > slo.max_error_rate * thresholds.error_rate_warning_multiplier:
        alerts.append(
            _build_alert(
                name="error_rate",
                metric="error_rate",
                current=error_rate,
                threshold=slo.max_error_rate * thresholds.error_rate_warning_multiplier,
                severity="warning",
                reason="error rate above SLO",
                action="Inspect recent release and endpoint error traces.",
            )
        )
    current_global = performance_metrics.get("global", {})
    r2_now = float(current_global.get("r2", 0.0))
    mae_now = float(current_global.get("mae", 0.0))
    rmse_now = float(current_global.get("rmse", 0.0))
    if r2_now < slo.min_r2:
        alerts.append(
            _build_alert(
                name="online_r2",
                metric="r2",
                current=r2_now,
                threshold=slo.min_r2,
                severity="critical",
                reason="online R² dropped below minimum guardrail",
                action="Pause promotion and create retrain candidate.",
            )
        )
    if mae_now > slo.max_mae:
        alerts.append(
            _build_alert(
                name="online_mae",
                metric="mae",
                current=mae_now,
                threshold=slo.max_mae,
                severity="critical",
                reason="online MAE exceeded guardrail",
                action="Run post-deploy analysis and retrain candidate.",
            )
        )
    if rmse_now > slo.max_rmse:
        alerts.append(
            _build_alert(
                name="online_rmse",
                metric="rmse",
                current=rmse_now,
                threshold=slo.max_rmse,
                severity="critical",
                reason="online RMSE exceeded guardrail",
                action="Escalate to model on-call and assess rollback.",
            )
        )
    r2_drop = float(degradation_metrics.get("r2_drop", 0.0))
    if r2_drop >= thresholds.r2_drop_critical:
        alerts.append(
            _build_alert(
                name="r2_drop",
                metric="r2_drop",
                current=r2_drop,
                threshold=thresholds.r2_drop_critical,
                severity="critical",
                reason="R² degradation reached critical level",
                action="Create retrain candidate and evaluate rollback.",
            )
        )
    elif r2_drop >= thresholds.r2_drop_warning:
        alerts.append(
            _build_alert(
                name="r2_drop",
                metric="r2_drop",
                current=r2_drop,
                threshold=thresholds.r2_drop_warning,
                severity="warning",
                reason="R² degradation detected",
                action="Track trend for sustained windows.",
            )
        )
    mae_ratio = float(degradation_metrics.get("mae_increase_ratio", 0.0))
    if mae_ratio >= thresholds.mae_increase_critical:
        alerts.append(
            _build_alert(
                name="mae_increase",
                metric="mae_increase_ratio",
                current=mae_ratio,
                threshold=thresholds.mae_increase_critical,
                severity="critical",
                reason="MAE increase is critical",
                action="Create retrain candidate and check latest deployment.",
            )
        )
    elif mae_ratio >= thresholds.mae_increase_warning:
        alerts.append(
            _build_alert(
                name="mae_increase",
                metric="mae_increase_ratio",
                current=mae_ratio,
                threshold=thresholds.mae_increase_warning,
                severity="warning",
                reason="MAE increase detected",
                action="Inspect recent data segments and monitor trend.",
            )
        )
    rmse_ratio = float(degradation_metrics.get("rmse_increase_ratio", 0.0))
    if rmse_ratio >= thresholds.rmse_increase_critical:
        alerts.append(
            _build_alert(
                name="rmse_increase",
                metric="rmse_increase_ratio",
                current=rmse_ratio,
                threshold=thresholds.rmse_increase_critical,
                severity="critical",
                reason="RMSE increase is critical",
                action="Begin rollback decision workflow.",
            )
        )
    elif rmse_ratio >= thresholds.rmse_increase_warning:
        alerts.append(
            _build_alert(
                name="rmse_increase",
                metric="rmse_increase_ratio",
                current=rmse_ratio,
                threshold=thresholds.rmse_increase_warning,
                severity="warning",
                reason="RMSE increase detected",
                action="Watch next window and prepare retraining action.",
            )
        )
    return alerts


def derive_service_metrics(events: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Derive service-level metrics such as availability, error rate, and latency percentiles from raw inference events."""
    if not events:
        return {"availability": 1.0, "error_rate": 0.0, "latency_p95_ms": 0.0}
    statuses = [int(event.get("status_code", 500)) for event in events]
    latencies = [float(event.get("latency_ms", 0.0)) for event in events]
    success_count = sum(1 for s in statuses if 200 <= s < 400)
    total = len(statuses)
    server_errors = sum(1 for s in statuses if s >= 500)
    availability = _safe_ratio(success_count, total)
    error_rate = _safe_ratio(server_errors, total)
    latency_p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
    return {
        "availability": availability,
        "error_rate": error_rate,
        "latency_p95_ms": latency_p95,
    }


def plan_automation_actions(
    *,
    alerts: Sequence[Alert],
    critical_window_streak: int,
    required_sustained_windows: int,
    has_post_deploy_regression: bool,
    canary_metrics: Dict[str, float] | None = None,
) -> List[AutomationAction]:
    """Derive a list of automation actions based on the current alerts, whether breaches are sustained, and canary validation results."""
    actions: List[AutomationAction] = []
    critical_alerts = [alert for alert in alerts if alert.severity == "critical"]
    if critical_window_streak >= required_sustained_windows and critical_alerts:
        actions.append(
            AutomationAction(
                action_type="create_retrain_candidate",
                reason="Sustained critical drift/performance breaches detected.",
                details={
                    "critical_alert_names": [alert.name for alert in critical_alerts],
                    "critical_window_streak": critical_window_streak,
                    "required_sustained_windows": required_sustained_windows,
                },
            )
        )
    if has_post_deploy_regression:
        actions.append(
            AutomationAction(
                action_type="rollback_deployment",
                reason="Post-deploy regression detected in canary/online metrics.",
                details={"source": "post_deploy_regression"},
            )
        )
    if canary_metrics:
        canary_r2 = float(canary_metrics.get("r2", 0.0))
        canary_error_rate = float(canary_metrics.get("error_rate", 0.0))
        if (
            canary_r2 < CONFIG.monitoring.slo.min_r2
            or canary_error_rate > CONFIG.monitoring.slo.max_error_rate
        ):
            actions.append(
                AutomationAction(
                    action_type="rollback_deployment",
                    reason="Canary validation failed.",
                    details=canary_metrics,
                )
            )
        else:
            actions.append(
                AutomationAction(
                    action_type="promote_canary",
                    reason="Canary checks passed.",
                    details=canary_metrics,
                )
            )
    return actions


def champion_challenger_decision(
    *,
    champion_metrics: Dict[str, float],
    challenger_metrics: Dict[str, float],
    champion_segment_mae: Dict[str, float],
    challenger_segment_mae: Dict[str, float],
) -> ChampionChallengerDecision:
    """Evaluate champion vs challenger performance and fairness metrics to make a promotion decision."""
    champion_r2 = champion_metrics.get("r2")
    challenger_r2 = challenger_metrics.get("r2")
    champion_mae = champion_metrics.get("mae")
    challenger_mae = challenger_metrics.get("mae")
    champion_rmse = champion_metrics.get("rmse")
    challenger_rmse = challenger_metrics.get("rmse")
    metrics_available = all(
        value is not None
        for value in [
            champion_r2,
            challenger_r2,
            champion_mae,
            challenger_mae,
            champion_rmse,
            challenger_rmse,
        ]
    )
    checks: Dict[str, bool] = {}
    checks["metrics_present"] = metrics_available
    checks["r2_improved"] = metrics_available and float(challenger_r2) >= float(
        champion_r2
    )
    checks["mae_not_worse"] = metrics_available and float(challenger_mae) <= float(
        champion_mae
    )
    checks["rmse_not_worse"] = metrics_available and float(challenger_rmse) <= float(
        champion_rmse
    )
    max_champion_gap = (
        max(champion_segment_mae.values()) - min(champion_segment_mae.values())
        if champion_segment_mae
        else 0.0
    )
    max_challenger_gap = (
        max(challenger_segment_mae.values()) - min(challenger_segment_mae.values())
        if challenger_segment_mae
        else 0.0
    )
    checks["fairness_gap_within_limit"] = (
        max_challenger_gap <= CONFIG.monitoring.fairness_segment_max_mae_gap
    )
    checks["fairness_not_regressed"] = max_challenger_gap <= max_champion_gap
    if CONFIG.monitoring.promote_requires_segment_improvement:
        checks["segment_mae_improved"] = all(
            challenger_segment_mae.get(segment, np.inf)
            <= champion_segment_mae.get(segment, np.inf)
            for segment in champion_segment_mae
        )
    else:
        checks["segment_mae_improved"] = True
    promote = all(checks.values())
    summary = (
        "Promote challenger model."
        if promote
        else "Keep champion model; challenger failed promotion checks."
    )
    return ChampionChallengerDecision(
        promote=promote,
        summary=summary,
        checks=checks,
        metrics={
            "champion_r2": float(champion_r2 or 0.0),
            "challenger_r2": float(challenger_r2 or 0.0),
            "champion_mae": float(champion_mae or 0.0),
            "challenger_mae": float(challenger_mae or 0.0),
            "champion_rmse": float(champion_rmse or 0.0),
            "challenger_rmse": float(challenger_rmse or 0.0),
            "champion_segment_mae_gap": float(max_champion_gap),
            "challenger_segment_mae_gap": float(max_challenger_gap),
        },
    )


def alerts_to_dicts(alerts: Iterable[Alert]) -> List[Dict[str, Any]]:
    """Convert a list of Alert dataclass instances to a list of dictionaries for easier serialization."""
    return [asdict(alert) for alert in alerts]


def actions_to_dicts(actions: Iterable[AutomationAction]) -> List[Dict[str, Any]]:
    """Convert a list of AutomationAction dataclass instances to a list of dictionaries for easier serialization."""
    return [asdict(action) for action in actions]
