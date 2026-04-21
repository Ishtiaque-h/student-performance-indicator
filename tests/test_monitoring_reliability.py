from __future__ import annotations
from student_performance.mlops.monitoring import (
    champion_challenger_decision,
    compute_degradation,
    compute_online_drift,
    compute_performance_monitoring,
    derive_service_metrics,
    evaluate_alerts,
    join_inference_with_labels,
    plan_automation_actions,
)

def _event(
    request_id: str,
    *,
    gender: str,
    lunch: str,
    race: str,
    prediction: float,
    status_code: int = 200,
    latency_ms: float = 50.0,
):
    """Utility to create a synthetic inference event with specified characteristics."""
    return {
        "request_id": request_id,
        "timestamp": f"2026-01-01T00:00:{int(request_id.split('-')[-1]):02d}Z",
        "features": {"gender": gender, "lunch": lunch, "race_ethnicity": race},
        "prediction": prediction,
        "status_code": status_code,
        "latency_ms": latency_ms,
    }

def test_drift_and_alerting_detects_synthetic_failures():
    """Test that the monitoring functions can detect synthetic drift and performance issues."""
    baseline = {
        "categorical_features": {
            "gender": {"female": 0.5, "male": 0.5},
            "lunch": {"standard": 0.7, "free/reduced": 0.3},
            "race_ethnicity": {"group a": 0.5, "group b": 0.5},
        },
        "prediction_distribution_reference": [50 + (i % 5) for i in range(100)],
        "segments": {
            "gender": {"female": 0.5, "male": 0.5},
            "lunch": {"standard": 0.7, "free/reduced": 0.3},
            "race_ethnicity": {"group a": 0.5, "group b": 0.5},
        },
    }
    live_events = [
        _event(
            f"req-{i}",
            gender="female",
            lunch="free/reduced",
            race="group b",
            prediction=80 + (i % 3),
            status_code=500 if i % 4 == 0 else 200,
            latency_ms=450.0,
        )
        for i in range(80)
    ]
    drift = compute_online_drift(live_events=live_events, baseline=baseline)
    service = derive_service_metrics(live_events)
    joined = join_inference_with_labels(
        inference_events=live_events,
        label_rows=[{"request_id": f"req-{i}", "label": 40.0} for i in range(80)],
    )
    perf = compute_performance_monitoring(joined_rows=joined)
    degradation = compute_degradation(
        baseline_metrics={"r2": 0.3, "mae": 10.0, "rmse": 12.0},
        current_metrics=perf["global"],
    )
    alerts = evaluate_alerts(
        drift_metrics=drift,
        service_metrics=service,
        performance_metrics=perf,
        degradation_metrics=degradation,
    )
    critical_names = {alert.name for alert in alerts if alert.severity == "critical"}
    assert any(name.startswith("categorical_drift_") for name in critical_names)
    assert "prediction_distribution_drift" in critical_names
    assert "latency_p95" in critical_names
    assert "error_rate" in critical_names

def test_automation_creates_retrain_and_rollback_on_sustained_regression():
    """Test that the automation planning function creates appropriate actions when there are sustained critical alerts and post-deploy regression."""
    class _Alert:
        def __init__(self, name: str, severity: str):
            self.name = name
            self.severity = severity
    actions = plan_automation_actions(
        alerts=[_Alert("r2_drop", "critical"), _Alert("mae_increase", "critical")],
        critical_window_streak=3,
        required_sustained_windows=3,
        has_post_deploy_regression=True,
        canary_metrics={"r2": 0.01, "error_rate": 0.1},
    )
    action_types = [action.action_type for action in actions]
    assert "create_retrain_candidate" in action_types
    assert "rollback_deployment" in action_types

def test_champion_challenger_gate_blocks_unfair_or_weaker_model():
    """Test that the champion-challenger decision function correctly identifies when a challenger model should not be promoted due to overall weaker performance or fairness regression."""
    decision = champion_challenger_decision(
        champion_metrics={"r2": 0.20, "mae": 11.0, "rmse": 14.0},
        challenger_metrics={"r2": 0.22, "mae": 10.8, "rmse": 13.8},
        champion_segment_mae={"female": 10.0, "male": 12.0},
        challenger_segment_mae={"female": 8.5, "male": 13.5},
    )
    assert decision.promote is False
    assert decision.checks["fairness_not_regressed"] is False