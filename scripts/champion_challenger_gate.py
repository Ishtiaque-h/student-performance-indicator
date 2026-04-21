from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict
from student_performance.mlops.monitoring import champion_challenger_decision

def _load_json(path: Path) -> Dict[str, Any]:
    """Utility to load JSON file if it exists, otherwise return empty dict."""
    return json.loads(path.read_text(encoding="utf-8"))

def main() -> None:
    """
    Main function to run champion/challenger gate. This will be called by an external scheduler (e.g. cron, Airflow) after challenger model has been evaluated on the defined metrics and fairness checks.
    The champion and challenger metrics and fairness check results should be provided as JSON files, and the output will be a JSON with the promotion decision, summary, and details of the checks.
    """
    parser = argparse.ArgumentParser(
        description="Promotion gate using champion/challenger metrics and fairness checks."
    )
    parser.add_argument("--champion-metrics-json", required=True)
    parser.add_argument("--challenger-metrics-json", required=True)
    parser.add_argument("--champion-segment-mae-json", required=True)
    parser.add_argument("--challenger-segment-mae-json", required=True)
    args = parser.parse_args()
    champion_metrics = _load_json(Path(args.champion_metrics_json))
    challenger_metrics = _load_json(Path(args.challenger_metrics_json))
    champion_segment_mae = _load_json(Path(args.champion_segment_mae_json))
    challenger_segment_mae = _load_json(Path(args.challenger_segment_mae_json))
    decision = champion_challenger_decision(
        champion_metrics=champion_metrics,
        challenger_metrics=challenger_metrics,
        champion_segment_mae=champion_segment_mae,
        challenger_segment_mae=challenger_segment_mae,
    )
    output = {
        "promote": decision.promote,
        "summary": decision.summary,
        "checks": decision.checks,
        "metrics": decision.metrics,
    }
    print(json.dumps(output, indent=2))
    if not decision.promote:
        raise SystemExit(1)
    
    
if __name__ == "__main__":
    main()