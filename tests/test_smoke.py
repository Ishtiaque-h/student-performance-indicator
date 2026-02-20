# tests/test_smoke.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from student_performance.pipeline.train_pipeline import TrainPipeline
from student_performance.pipeline.predict_pipeline import PredictPipeline
from student_performance.components.config import CONFIG


def _write_minimal_repo_layout(repo_root: Path) -> None:
    """
    Create a minimal fake dataset based on CONFIG.
    Fully dataset-agnostic - works with any tabular dataset!
    
    Strategy:
    1. Try to read real data from multiple locations
    2. If found, sample it for fast tests
    3. If not found, generate synthetic data with learnable patterns
    """
    # Make root markers for find_project_root()
    (repo_root / "pyproject.toml").write_text(
        "[project]\nname='student-performance-smoke'\nversion='0.0.0'\n"
    )

    # Create data directory matching CONFIG
    data_dir = repo_root / CONFIG.dataset.data_rel_path.parent
    data_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = repo_root / CONFIG.dataset.data_rel_path

    # STRATEGY 1: Try to read and sample real data
    real_data_found = False
    
    # List of potential data locations
    search_paths = [
        Path(__file__).parent.parent / CONFIG.dataset.data_rel_path,
        repo_root.parent.parent / CONFIG.dataset.data_rel_path,
        repo_root.parent / CONFIG.dataset.data_rel_path,
    ]
    
    # Also check up to 4 parent directories from test file location
    for depth in range(5):
        search_paths.append(
            Path(__file__).parents[depth] / CONFIG.dataset.data_rel_path
        )
    
    for potential_path in search_paths:
        if potential_path.exists():
            try:
                real_df = pd.read_csv(potential_path)
                
                # Validate it has required columns
                required_cols = {CONFIG.dataset.target_col}
                if CONFIG.dataset.drop_cols:
                    required_cols.update(CONFIG.dataset.drop_cols)
                
                if required_cols.issubset(set(real_df.columns)):
                    # Sample for fast tests
                    n_sample = min(40, len(real_df))
                    sample_df = real_df.sample(n=n_sample, random_state=42)
                    sample_df.to_csv(output_path, index=False)
                    real_data_found = True
                    break
            except Exception:
                continue
    
    if real_data_found:
        return
    
    # STRATEGY 2: Generate synthetic data WITH LEARNABLE PATTERNS
    # This ensures Ridge regression can achieve positive R²
    
    rng = np.random.default_rng(42)
    n = 40  # Small dataset for fast tests
    
    synthetic_data = {}
    
    # Generate categorical features with encoded values for correlation
    num_categorical_features = 5
    feature_encodings = []
    
    for i in range(num_categorical_features):
        feature_name = f"feature_{i}"
        num_categories = rng.integers(2, 6)
        categories = [f"cat_{j}" for j in range(num_categories)]
        
        # Generate categorical values
        cat_values = rng.choice(categories, size=n)
        synthetic_data[feature_name] = cat_values
        
        # Store numeric encoding for target generation
        encoding_map = {cat: idx for idx, cat in enumerate(categories)}
        feature_encodings.append([encoding_map[val] for val in cat_values])
    
    # Generate target with LINEAR RELATIONSHIP to features
    # This simulates a real regression problem
    base_value = 50.0
    target_values = np.full(n, base_value, dtype=float)
    
    # Each feature contributes to the target
    for i, encoded_feature in enumerate(feature_encodings):
        weight = rng.uniform(3, 8)  # Random weight for each feature
        target_values += np.array(encoded_feature) * weight
    
    # Add small random noise (but keep signal-to-noise ratio high)
    noise = rng.normal(0, 3, size=n)  # Small noise
    target_values += noise
    
    # Clip to reasonable range and convert to int
    target_values = np.clip(target_values, 40, 100).astype(int)
    synthetic_data[CONFIG.dataset.target_col] = target_values
    
    # Add drop columns (correlated with target, like reading/writing scores)
    if CONFIG.dataset.drop_cols:
        for col in CONFIG.dataset.drop_cols:
            # Highly correlated with target + small noise
            correlated = target_values + rng.normal(0, 2, size=n)
            correlated = np.clip(correlated, 40, 100).astype(int)
            synthetic_data[col] = correlated
    
    df = pd.DataFrame(synthetic_data)
    df.to_csv(output_path, index=False)


@pytest.fixture()
def fake_repo(tmp_path: Path) -> Path:
    """
    Create a temporary repository with minimal test data.
    Uses CONFIG to determine data schema and location.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    _write_minimal_repo_layout(repo_root)
    return repo_root


def _patch_ultra_fast_trainer(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Make smoke tests finish in seconds:
    - Fit a single fast model (Ridge)
    - No RandomizedSearchCV / GridSearchCV
    - Still exercises: ingestion -> transformation -> model save -> predict load
    """
    import student_performance.components.model_trainer as mt
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from sklearn.base import clone

    def fast_evaluate_models(
        *,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        models: Dict[str, Any],
        random_param=None,
        refine_spec=None,
        scoring="r2",
        cv=5,
        n_jobs=-1,
        verbose=0,
        random_n_iter=25,
        random_seed=42,
        prepare_X=None,
        prefer_cv_for_selection=True,
    ):
        """
        Fast model evaluation for smoke tests.
        Picks Ridge if available, otherwise uses first model.
        """
        # Pick Ridge if present; else fallback to first model
        if "Ridge" in models:
            name = "Ridge"
            estimator = clone(models["Ridge"])
        else:
            name, estimator = next(iter(models.items()))
            estimator = clone(estimator)

        Xtr, Xte = X_train, X_test
        if prepare_X is not None:
            Xtr, Xte = prepare_X(Xtr, Xte, name)

        estimator.fit(Xtr, y_train)
        
        train_pred = estimator.predict(Xtr)
        test_pred = estimator.predict(Xte)

        # Calculate comprehensive metrics
        train_r2 = float(r2_score(y_train, train_pred))
        train_mae = float(mean_absolute_error(y_train, train_pred))
        train_rmse = float(np.sqrt(mean_squared_error(y_train, train_pred)))
        
        test_r2 = float(r2_score(y_test, test_pred))
        test_mae = float(mean_absolute_error(y_test, test_pred))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))

        report = {
            "models": {
                name: {
                    "random_search": {
                        "best_params": {},
                        "cv_best_score": None,
                        "n_iter": 0,
                    },
                    "grid_search": {
                        "best_params": {},
                        "cv_best_score": None,
                    },
                    "train": {
                        "r2": train_r2,
                        "mae": train_mae,
                        "rmse": train_rmse,
                    },
                    "test": {
                        "r2": test_r2,
                        "mae": test_mae,
                        "rmse": test_rmse,
                    },
                }
            },
            "best_model": {
                "name": name,
                "selection_score": test_r2,
                "test_r2": test_r2,
                "test_mae": test_mae,
                "test_rmse": test_rmse,
            },
        }
        
        return report, name, estimator

    monkeypatch.setattr(mt, "evaluate_models", fast_evaluate_models)


@pytest.mark.smoke
def test_end_to_end_smoke(fake_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    End-to-end smoke test - fully dataset-agnostic.
    
    PURPOSE: Validate the pipeline runs end-to-end without crashing.
    NOT focused on model quality (that's for integration tests).
    """
    monkeypatch.chdir(fake_repo)
    _patch_ultra_fast_trainer(monkeypatch)

    # === TRAIN PIPELINE ===
    best_model_name, report = TrainPipeline().run()
    
    # Validate outputs exist (not quality)
    assert best_model_name, "No model name returned"
    assert best_model_name in report["models"]
    assert "best_model" in report
    assert "test_r2" in report["best_model"]
    
    # R² can be negative for poorly correlated data - that's OK for smoke test
    # We just want to ensure it's a valid number
    assert isinstance(report["best_model"]["test_r2"], (int, float))
    assert -10.0 <= report["best_model"]["test_r2"] <= 1.0, \
        f"R² out of reasonable range: {report['best_model']['test_r2']}"
    
    # Check artifacts were created
    artifacts_dir = fake_repo / "artifacts"
    model_path = artifacts_dir / "model.pkl"
    preprocessor_path = artifacts_dir / "preprocessor.pkl"
    model_report_path = artifacts_dir / "model_report.json"
    assert model_path.exists(), "model.pkl not created"
    assert preprocessor_path.exists(), "preprocessor.pkl not created"
    assert model_report_path.exists(), "model_report.json not created"

    # Ensure artifacts aren't empty (catches serialization failures)
    assert model_path.stat().st_size > 100, "model.pkl suspiciously small"
    assert preprocessor_path.stat().st_size > 100, "preprocessor.pkl suspiciously small"
    assert model_report_path.stat().st_size > 100, "model_report.json suspiciously small"
    
    # === PREDICT PIPELINE ===
    predict_pipeline = PredictPipeline()
    
    df = pd.read_csv(fake_repo / CONFIG.dataset.data_rel_path)
    
    feature_cols = [
        col for col in df.columns
        if col != CONFIG.dataset.target_col
        and col not in (CONFIG.dataset.drop_cols or [])
    ]
    
    sample_row = df[feature_cols].iloc[0].to_dict()
    
    # Test single prediction
    predictions = predict_pipeline.predict(sample_row)
    assert len(predictions) == 1
    assert isinstance(predictions[0], (int, float, np.number))
    
    # Test batch prediction
    batch_predictions = predict_pipeline.predict([sample_row, sample_row])
    assert len(batch_predictions) == 2
    assert predictions[0] == batch_predictions[0]

    # For student scores, predictions should be somewhat reasonable
    # (even if model is bad, it shouldn't predict -1000 or 999999)
    assert -100 <= predictions[0] <= 200, \
        f"Prediction out of reasonable range: {predictions[0]}"   
    
    print(f"\n✅ Smoke test passed!")
    print(f"   Model: {best_model_name}")
    print(f"   Test R²: {report['best_model']['test_r2']:.4f}")
    print(f"   Sample prediction: {predictions[0]:.2f}")