from __future__ import annotations

from typing import Any, Callable, Dict, Tuple, Optional
import numpy as np

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _make_refined_grid(
    best_params: Dict[str, Any],
    refine_specs: Dict[str, Dict[str, Any]],
) -> Dict[str, list]:
    grid: Dict[str, list] = {}

    for k, spec in refine_specs.items():
        if k not in best_params:
            continue

        v = best_params[k]
        t = spec.get("type")

        # If best value is None, numeric neighborhood doesn't make sense
        if v is None and t in {"float_log", "float_window", "int_window"}:
            continue

        if t == "float_log":
            factors = spec.get("factors", [0.5, 1.0, 2.0])
            vals = sorted({float(v) * float(f) for f in factors if float(v) * float(f) > 0})
            if vals:
                grid[k] = vals

        elif t == "float_window":
            deltas = spec.get("deltas", [-0.05, 0.0, 0.05])
            vals = sorted({float(v) + float(d) for d in deltas if float(v) + float(d) > 0})
            if vals:
                grid[k] = vals

        elif t == "int_window":
            deltas = spec.get("deltas", [-1, 0, 1])
            mn = spec.get("min", None)
            mx = spec.get("max", None)
            vals = []
            for d in deltas:
                nv = int(v) + int(d)
                if mn is not None:
                    nv = max(int(mn), nv)
                if mx is not None:
                    nv = min(int(mx), nv)
                vals.append(nv)
            vals = sorted(set(vals))
            if vals:
                grid[k] = vals

        elif t == "categorical":
            cats = spec.get("values", [])
            if v not in cats:
                cats = [v] + cats
            cats = list(dict.fromkeys(cats))
            if cats:
                grid[k] = cats

    return grid


def evaluate_models(
    X_train: Any,
    y_train: np.ndarray,
    X_test: Any,
    y_test: np.ndarray,
    models: Dict[str, Any],
    random_param: Optional[Dict[str, Dict[str, Any]]] = None,
    refine_spec: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    scoring: str = "r2",
    cv: int = 5,
    n_jobs: int = -1,
    verbose: int = 0,
    random_n_iter: int = 25,
    random_seed: int = 42,
    prepare_X: Optional[Callable[[Any, Any, str], Tuple[Any, Any]]] = None,
    prefer_cv_for_selection: bool = True,
) -> Tuple[Dict[str, Any], str, Any]:
    """
    Two-stage tuning:
      1) RandomizedSearchCV (if random_param grid provided for model)
      2) Refined GridSearchCV around best random params (if refine_spec provided)

    Returns:
      report, best_model_name, best_estimator
    """
    y_train = np.asarray(y_train).ravel()
    y_test = np.asarray(y_test).ravel()

    report: Dict[str, Any] = {"models": {}}
    best_name: Optional[str] = None
    best_estimator: Any = None
    best_sel_score = -float("inf")
    best_sel_source: str = "unset"  # <- safe default

    random_param = random_param or {}
    refine_spec = refine_spec or {}

    # ✅ shuffled CV reduces sensitivity to row ordering (esp. small datasets)
    cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_seed)

    for name, model in models.items():
        try:
            base = clone(model)

            Xtr, Xte = X_train, X_test
            if prepare_X is not None:
                Xtr, Xte = prepare_X(Xtr, Xte, name)

            rand_grid = random_param.get(name)
            model_refine_spec = refine_spec.get(name)

            best_params_random = None
            best_random_score = None
            best_params_grid = None
            best_grid_score = None

            estimator = base

            # --- Stage 1: RandomizedSearch ---
            if rand_grid:
                rs = RandomizedSearchCV(
                    estimator=base,
                    param_distributions=rand_grid,
                    n_iter=random_n_iter,
                    scoring=scoring,
                    cv=cv_splitter,
                    n_jobs=n_jobs,
                    refit=True,
                    random_state=random_seed,
                    verbose=verbose,
                )
                rs.fit(Xtr, y_train)
                estimator = rs.best_estimator_
                best_params_random = rs.best_params_
                best_random_score = float(rs.best_score_)

            # --- Stage 2: Refined GridSearch ---
            if best_params_random and model_refine_spec:
                refined_grid = _make_refined_grid(best_params_random, model_refine_spec)
                if refined_grid:
                    gs = GridSearchCV(
                        estimator=clone(model).set_params(**best_params_random),
                        param_grid=refined_grid,
                        scoring=scoring,
                        cv=cv_splitter,
                        n_jobs=n_jobs,
                        refit=True,
                        verbose=verbose,
                    )
                    gs.fit(Xtr, y_train)
                    estimator = gs.best_estimator_
                    best_params_grid = gs.best_params_
                    best_grid_score = float(gs.best_score_)

            # Evaluate final estimator on train/test
            train_pred = estimator.predict(Xtr)
            test_pred = estimator.predict(Xte)
            train_metrics = regression_metrics(y_train, train_pred)
            test_metrics = regression_metrics(y_test, test_pred)

            report["models"][name] = {
                "random_search": {
                    "best_params": best_params_random,
                    "cv_best_score": best_random_score,
                    "n_iter": random_n_iter if rand_grid else 0,
                },
                "grid_search": {
                    "best_params": best_params_grid,
                    "cv_best_score": best_grid_score,
                },
                "train": train_metrics,
                "test": test_metrics,
            }

            # Selection: prefer CV score when available, else fallback to test_r2
            if prefer_cv_for_selection:
                if best_grid_score is not None:
                    sel = best_grid_score
                    sel_source = "grid_cv"
                elif best_random_score is not None:
                    sel = best_random_score
                    sel_source = "random_cv"
                else:
                    sel = test_metrics["r2"]
                    sel_source = "test_r2_fallback"
            else:
                sel = test_metrics["r2"]
                sel_source = "test_r2"

            if float(sel) > best_sel_score:
                best_sel_score = float(sel)
                best_name = name
                best_estimator = estimator
                best_sel_source = sel_source

        except Exception as e:
            report["models"][name] = {"error": str(e)}
            continue

    if best_name is None or best_estimator is None:
        raise RuntimeError("No model succeeded in evaluate_models().")

    report["best_model"] = {
        "name": best_name,
        "selection_score": float(best_sel_score),
        "selection_source": best_sel_source,
        "test_r2": float(report["models"][best_name]["test"]["r2"]),
    }
    return report, best_name, best_estimator
