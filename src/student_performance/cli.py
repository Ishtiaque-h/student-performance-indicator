from __future__ import annotations

from student_performance.pipeline.train_pipeline import TrainPipeline


def train_main() -> None:
    best_model_name, report = TrainPipeline().run()
    print(f"\nBest model: {best_model_name}")
    print(f"Best test R2: {report.get('best_model', {}).get('test_r2')}")
