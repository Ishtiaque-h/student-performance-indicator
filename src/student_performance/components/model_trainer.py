import json
import sys
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

from student_performance.exception import CustomException
from student_performance.logger import logging
from student_performance.utils import find_project_root, save_object
from student_performance.modeling import evaluate_models
from student_performance.components.config import CONFIG


# Optional models
try:
    from xgboost import XGBRegressor  # type: ignore
    from catboost import CatBoostRegressor  # type: ignore
    _HAS_XGBOOST = True
    _HAS_CATBOOST = True

except Exception:
    _HAS_XGBOOST = False
    _HAS_CATBOOST = False


class DensePredictWrapper:
    """
    Ensures .fit()/.predict() work even if X is a scipy sparse matrix.
    Pickle-safe (avoids recursion during dill/pickle loading).
    """
    def __init__(self, estimator: Any):
        self.estimator = estimator

    # ---- pickle / dill safety ----
    def __getstate__(self):
        # Keep state minimal and explicit
        return {"estimator": self.estimator}

    def __setstate__(self, state):
        # Restore without touching __getattr__
        self.estimator = state["estimator"]

    # ---- sklearn-like API ----
    def fit(self, X: Any, y: Any = None):
        try:
            from scipy import sparse
            if sparse.issparse(X):
                X = X.toarray()
        except Exception:
            pass
        self.estimator.fit(X, y)
        return self

    def predict(self, X: Any):
        try:
            from scipy import sparse
            if sparse.issparse(X):
                X = X.toarray()
        except Exception:
            pass
        return self.estimator.predict(X)

    def __getattr__(self, name: str):
        """
        Proxy everything else to underlying estimator.
        IMPORTANT: use object.__getattribute__ so we never recurse on 'estimator'.
        """
        est = object.__getattribute__(self, "estimator")
        return getattr(est, name)
    

@dataclass
class ModelTrainerConfig:
    artifacts_dir: Path = Path("artifacts")
    model_file_path: Path = artifacts_dir / "model.pkl"
    report_file_path: Path = artifacts_dir / "model_report.json"


class ModelTrainer:
    def __init__(self):
        self.trainer_config = ModelTrainerConfig()

    @staticmethod
    def _is_sparse(X: Any) -> bool:
        try:
            from scipy import sparse
            return sparse.issparse(X)
        except Exception:
            return False

    def _maybe_to_dense(self, X: Any, model_name: str) -> Any:
        if not self._is_sparse(X):
            return X

        n_rows, n_features = X.shape
        if n_features > CONFIG.dense_safety.dense_feature_threshold:
            raise ValueError(
                f"{model_name}: X is sparse with {n_features} features; densifying may be too large. "
                f"Increase CONFIG.dense_safety.dense_feature_threshold or skip this model."
            )

        cells = int(n_rows) * int(n_features)
        if cells > CONFIG.dense_safety.dense_cell_threshold:
            raise ValueError(
                f"{model_name}: X is sparse with {cells} total cells; densifying may be too large. "
                f"Increase CONFIG.dense_safety.dense_cell_threshold or skip this model."
            )

        logging.info(f"{model_name}: densifying X (rows={n_rows}, features={n_features}).")
        return X.toarray()

    def initiate_model_trainer(
        self,
        X_train: Any,
        y_train: np.ndarray,
        X_test: Any,
        y_test: np.ndarray,
    ) -> Tuple[str, Dict[str, Any]]:
        try:
            repo_root = find_project_root()

            artifacts_dir = CONFIG.artifacts.artifacts_dir(repo_root)
            model_path = CONFIG.artifacts.model_path(repo_root)
            report_path = CONFIG.artifacts.model_report_path(repo_root)

            artifacts_dir.mkdir(parents=True, exist_ok=True)

            y_train = np.asarray(y_train).ravel()
            y_test = np.asarray(y_test).ravel()

            models: Dict[str, Any] = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(random_state=42),
                "Lasso": Lasso(random_state=42, max_iter=10000),
                "KNN": KNeighborsRegressor(),
                "DecisionTree": DecisionTreeRegressor(random_state=42),
                "RandomForest": RandomForestRegressor(random_state=42),
                "AdaBoost": AdaBoostRegressor(random_state=42),
                "GradientBoosting": GradientBoostingRegressor(random_state=42),
            }

            if _HAS_XGBOOST:
                models["XGBoost"] = XGBRegressor(random_state=42, objective="reg:squarederror")
            else:
                logging.info("XGBoost not installed; skipping XGBoost model.")

            if _HAS_CATBOOST:
                models["CatBoost"] = CatBoostRegressor(random_seed=42, verbose=0, loss_function='RMSE')
            else:
                logging.info("CatBoost not installed; skipping CatBoost model.")

            random_params = {
                "Ridge": {"alpha": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]},
                "Lasso": {"alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0]},
                "KNN": {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"], "p": [1, 2]},
                "DecisionTree": {"max_depth": [None, 5, 10, 20],
                                 "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"]},
                "RandomForest": {"n_estimators": [50, 100, 200, 400],
                                 "max_depth": [None, 5, 10, 20, 30],
                                 "max_features": ["sqrt", None],
                                 "min_samples_split": [2, 5, 10]},
                "AdaBoost": {"n_estimators": [8, 16, 32, 64, 128, 256],
                             "learning_rate": [0.001, 0.01, 0.1, 0.5]},
                "GradientBoosting": {"n_estimators": [50, 100, 200, 400],
                                     "learning_rate": [0.01, 0.05, 0.1, 0.2],
                                     "subsample": [0.7, 0.85, 1.0],
                                     "max_depth": [2, 3, 4]},
                "XGBoost": {"n_estimators": [100, 300, 600],
                            "learning_rate": [0.01, 0.05, 0.1],
                            "max_depth": [3, 5, 7],
                            "subsample": [0.7, 0.85, 1.0],
                            "colsample_bytree": [0.7, 0.85, 1.0]},
                "CatBoost": {"iterations": [200, 500, 800],
                             "learning_rate": [0.01, 0.03, 0.05, 0.1],
                             "depth": [4, 6, 8, 10],
                             "l2_leaf_reg": [1, 3, 5, 9]},
            }

            refine_spec = {
                "Ridge": {"alpha": {"type": "float_log", "factors": [0.3, 1.0, 3.0]}},
                "Lasso": {"alpha": {"type": "float_log", "factors": [0.3, 1.0, 3.0]}},
                "RandomForest": {
                    "n_estimators": {"type": "int_window", "deltas": [-50, 0, 50], "min": 20, "max": 2000},
                    "max_depth": {"type": "int_window", "deltas": [-5, 0, 5], "min": 1, "max": 50},
                    "max_features": {"type": "categorical", "values": ["sqrt", None]},
                },
                "GradientBoosting": {
                    "n_estimators": {"type": "int_window", "deltas": [-50, 0, 50], "min": 20, "max": 2000},
                    "learning_rate": {"type": "float_log", "factors": [0.5, 1.0, 2.0]},
                    "subsample": {"type": "float_window", "deltas": [-0.1, 0.0, 0.1]},
                },
                "XGBoost": {
                    "n_estimators": {"type": "int_window", "deltas": [-100, 0, 100], "min": 50, "max": 5000},
                    "learning_rate": {"type": "float_log", "factors": [0.5, 1.0, 2.0]},
                    "max_depth": {"type": "int_window", "deltas": [-1, 0, 1], "min": 2, "max": 12},
                    "subsample": {"type": "float_window", "deltas": [-0.1, 0.0, 0.1]},
                    "colsample_bytree": {"type": "float_window", "deltas": [-0.1, 0.0, 0.1]},
                },
                "CatBoost": {
                    "iterations": {"type": "int_window", "deltas": [-100, 0, 100], "min": 100, "max": 5000},
                    "learning_rate": {"type": "float_log", "factors": [0.5, 1.0, 2.0]},
                    "depth": {"type": "int_window", "deltas": [-1, 0, 1], "min": 2, "max": 12},
                    "l2_leaf_reg": {"type": "int_window", "deltas": [-100, 0, 100], "min": 100, "max": 5000},
                },
            }

            needs_dense = {"KNN", "DecisionTree", "RandomForest", "AdaBoost", "GradientBoosting", "CatBoost"}

            def prepare_X(Xtr: Any, Xte: Any, model_name: str) -> Tuple[Any, Any]:
                if model_name in needs_dense:
                    Xtr = self._maybe_to_dense(Xtr, model_name)
                    Xte = self._maybe_to_dense(Xte, model_name)
                return Xtr, Xte

            model_report, best_name, best_model = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                random_param=random_params,
                refine_spec=refine_spec,
                scoring=CONFIG.tuning.scoring,
                cv=CONFIG.tuning.cv,
                n_jobs=CONFIG.tuning.n_jobs,
                verbose=CONFIG.tuning.verbose,
                random_n_iter=CONFIG.tuning.random_n_iter,
                random_seed=CONFIG.tuning.random_seed,
                prepare_X=prepare_X,
                prefer_cv_for_selection=CONFIG.tuning.prefer_cv_for_selection,
            )

            model_report["run_config"] = {
                "tuning": {
                    "scoring": CONFIG.tuning.scoring,
                    "cv": CONFIG.tuning.cv,
                    "n_jobs": CONFIG.tuning.n_jobs,
                    "verbose": CONFIG.tuning.verbose,
                    "random_n_iter": CONFIG.tuning.random_n_iter,
                    "random_seed": CONFIG.tuning.random_seed,
                    "prefer_cv_for_selection": CONFIG.tuning.prefer_cv_for_selection,
                },
                "dense_safety": {
                    "dense_feature_threshold": CONFIG.dense_safety.dense_feature_threshold,
                    "dense_cell_threshold": CONFIG.dense_safety.dense_cell_threshold,
                },
            }

            model_report["artifacts"] = {
                "artifacts_dir": str(artifacts_dir),
                "model_path": str(model_path),
                "report_path": str(report_path),
            }

            if best_name in needs_dense:
                best_model = DensePredictWrapper(best_model)
                model_report["best_model"]["requires_dense_at_inference"] = True
            else:
                model_report["best_model"]["requires_dense_at_inference"] = False

            save_object(str(model_path), best_model)
            model_report["best_model"]["model_path"] = str(model_path)

            report_path.write_text(json.dumps(model_report, indent=2))

            logging.info(f"Saved model report at {report_path}")
            logging.info(
                f"Best model: {best_name} | selection_score={model_report['best_model']['selection_score']:.4f} "
                f"| test_r2={model_report['best_model']['test_r2']:.4f}"
            )
            logging.info(f"Saved best model at {model_path}")

            return best_name, model_report

        except Exception as e:
            logging.exception("Error occurred during model training")
            raise CustomException(e, sys)
