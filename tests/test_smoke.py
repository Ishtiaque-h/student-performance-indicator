# tests/test_smoke.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from student_performance.pipeline.train_pipeline import TrainPipeline
from student_performance.pipeline.predict_pipeline import PredictPipeline


def _write_minimal_repo_layout(repo_root: Path) -> None:
    # Make root markers for find_project_root()
    (repo_root / "pyproject.toml").write_text(
        "[project]\nname='student-performance-smoke'\nversion='0.0.0'\n"
    )

    data_dir = repo_root / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    n = 40  # small but stable

    df = pd.DataFrame(
        {
            "gender": rng.choice(["female", "male"], size=n),
            "race_ethnicity": rng.choice(["group A", "group B", "group C"], size=n),
            "parental_level_of_education": rng.choice(
                ["some college", "bachelor's degree", "high school"], size=n
            ),
            "lunch": rng.choice(["standard", "free/reduced"], size=n),
            "test_preparation_course": rng.choice(["none", "completed"], size=n),
            "reading_score": rng.integers(40, 100, size=n),
            "writing_score": rng.integers(40, 100, size=n),
            "math_score": rng.integers(40, 100, size=n),
        }
    )

    df.to_csv(data_dir / "stud.csv", index=False)


@pytest.fixture()
def fake_repo(tmp_path: Path) -> Path:
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
    from sklearn.metrics import r2_score
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
        test_pred = estimator.predict(Xte)

        test_r2 = float(r2_score(y_test, test_pred))
        report = {
            "models": {
                name: {
                    "random_search": {
                        "best_params": None,
                        "cv_best_score": None,
                        "n_iter": 0,
                    },
                    "grid_search": {"best_params": None, "cv_best_score": None},
                    "train": {},  # not needed for smoke
                    "test": {"r2": test_r2},
                }
            },
            "best_model": {
                "name": name,
                "selection_score": test_r2,
                "selection_source": "test_r2_smoke",
                "test_r2": test_r2,
            },
        }
        return report, name, estimator

    # Patch the symbol used by model_trainer.py (important!)
    monkeypatch.setattr(mt, "evaluate_models", fast_evaluate_models)


def _clear_registry_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure tests don't try to pull from GCS (keeps tests hermetic)
    monkeypatch.delenv("MODEL_REGISTRY_URI", raising=False)
    monkeypatch.delenv("FORCE_MODEL_DOWNLOAD", raising=False)
    monkeypatch.delenv("ARTIFACTS_DIR", raising=False)
    monkeypatch.delenv("TARGET_COL", raising=False)


@pytest.mark.smoke
def test_end_to_end_train_and_predict(
    fake_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(fake_repo)
    _clear_registry_env(monkeypatch)
    _patch_ultra_fast_trainer(monkeypatch)

    best_name, report = TrainPipeline().run()
    assert isinstance(best_name, str)
    assert "best_model" in report

    artifacts = fake_repo / "artifacts"
    assert (artifacts / "train.parquet").exists()
    assert (artifacts / "test.parquet").exists()
    assert (artifacts / "preprocessor.pkl").exists()
    assert (artifacts / "model.pkl").exists()
    assert (artifacts / "model_report.json").exists()
    assert (artifacts / "ingestion_meta.json").exists()

    sample = {
        "gender": "female",
        "race_ethnicity": "group B",
        "parental_level_of_education": "bachelor's degree",
        "lunch": "standard",
        "test_preparation_course": "none",
    }

    preds = PredictPipeline().predict(sample)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (1,)
    assert np.isfinite(preds[0])


@pytest.mark.smoke
def test_prediction_schema_validation(
    fake_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(fake_repo)
    _clear_registry_env(monkeypatch)
    _patch_ultra_fast_trainer(monkeypatch)

    TrainPipeline().run()

    bad_sample = {
        "gender": "female",
        "race_ethnicity": "group B",
        "parental_level_of_education": "bachelor's degree",
        # "lunch" missing on purpose
        "test_preparation_course": "none",
    }

    with pytest.raises(ValueError, match="Missing required columns"):
        PredictPipeline().predict(bad_sample)
