from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
from student_performance.components.config import CONFIG

from student_performance.mlops.monitoring import (
    DRIFT_BASELINE_FILENAME,
    INFERENCE_LOG_FILENAME,
    actions_to_dicts,
    alerts_to_dicts,
    compute_degradation,
    compute_online_drift,
    compute_performance_monitoring,
    derive_service_metrics,
    evaluate_alerts,
    join_inference_with_labels,
    load_jsonl,
    load_training_baseline,
    plan_automation_actions,
)

from student_performance.pipeline.predict_pipeline import PredictPipeline


def _load_json(path: Path) -> Dict[str, Any]:
    """Utility to load JSON file if it exists, otherwise return empty dict."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _take_last(events: List[Dict[str, Any]], lookback: int) -> List[Dict[str, Any]]:
    """Utility to take the last N events from a list, where N is the lookback parameter."""
    if lookback <= 0:
        return events
    return events[-lookback:]


def main() -> None:
    """
    Main function to run online monitoring. This will be called by an external scheduler (e.g. cron, Airflow) on a regular cadence (e.g. hourly, daily).
    """
    parser = argparse.ArgumentParser(
        description="Run online drift/performance monitoring."
    )
    parser.add_argument(
        "--artifacts-dir",
        default="",
        help="Artifacts directory. Defaults to pipeline artifacts dir.",
    )
    parser.add_argument(
        "--labels-jsonl",
        default="",
        help="Optional delayed labels JSONL with fields {request_id,label}.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=5000,
        help="Number of latest inference events to evaluate.",
    )
    parser.add_argument(
        "--history-json",
        default="",
        help="Optional monitoring history JSON used for sustained-breach automation.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional output JSON path for metrics/alerts/actions.",
    )
    args = parser.parse_args()
    pipeline = PredictPipeline()
    artifacts_dir = (
        Path(args.artifacts_dir).expanduser().resolve()
        if args.artifacts_dir
        else pipeline.config.artifacts_dir
    )
    baseline_path = artifacts_dir / DRIFT_BASELINE_FILENAME
    inference_path = artifacts_dir / INFERENCE_LOG_FILENAME
    model_report_path = artifacts_dir / "model_report.json"
    monitoring_history_path = (
        Path(args.history_json).expanduser().resolve()
        if args.history_json
        else artifacts_dir / "monitoring_history.json"
    )
    baseline = load_training_baseline(baseline_path)
    events = _take_last(load_jsonl(inference_path), args.lookback)
    labels = (
        load_jsonl(Path(args.labels_jsonl).expanduser().resolve())
        if args.labels_jsonl
        else []
    )
    joined = join_inference_with_labels(inference_events=events, label_rows=labels)
    drift_metrics = compute_online_drift(
        live_events=events,
        baseline=baseline,
        segment_columns=CONFIG.monitoring.segment_columns,
    )
    service_metrics = derive_service_metrics(events)
    performance_metrics = compute_performance_monitoring(
        joined_rows=joined,
        rolling_windows=CONFIG.monitoring.rolling_windows,
        segment_columns=CONFIG.monitoring.segment_columns,
    )
    model_report = _load_json(model_report_path)
    baseline_metrics = {
        "r2": float(model_report.get("best_model", {}).get("test_r2", 0.0)),
        "mae": float(model_report.get("best_model", {}).get("test_mae", 0.0)),
        "rmse": float(model_report.get("best_model", {}).get("test_rmse", 0.0)),
    }
    degradation_metrics = compute_degradation(
        baseline_metrics=baseline_metrics,
        current_metrics=performance_metrics.get("global", {}),
    )
    alerts = evaluate_alerts(
        drift_metrics=drift_metrics,
        service_metrics=service_metrics,
        performance_metrics=performance_metrics,
        degradation_metrics=degradation_metrics,
    )
    history = _load_json(monitoring_history_path)
    prev_critical_windows = int(history.get("critical_windows", 0))
    critical_count = len([a for a in alerts if a.severity == "critical"])
    current_critical_windows = prev_critical_windows + 1 if critical_count > 0 else 0
    has_post_deploy_regression = critical_count > 0 and (
        degradation_metrics["r2_drop"]
        >= CONFIG.monitoring.alert_thresholds.r2_drop_critical
    )
    actions = plan_automation_actions(
        alerts=alerts,
        critical_window_streak=current_critical_windows,
        required_sustained_windows=CONFIG.monitoring.sustained_breach_windows_for_retrain,
        has_post_deploy_regression=has_post_deploy_regression,
    )
    history_update = {"critical_windows": current_critical_windows}
    monitoring_history_path.write_text(
        json.dumps(history_update, indent=2), encoding="utf-8"
    )
    report = {
        "drift_metrics": drift_metrics,
        "service_metrics": service_metrics,
        "performance_metrics": performance_metrics,
        "degradation_metrics": degradation_metrics,
        "alerts": alerts_to_dicts(alerts),
        "actions": actions_to_dicts(actions),
    }
    if args.output_json:
        Path(args.output_json).expanduser().resolve().write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
