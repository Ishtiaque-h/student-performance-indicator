from __future__ import annotations
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, model_validator

from student_performance import __version__
from student_performance.exception import CustomException
from student_performance.logger import get_logger
from student_performance.pipeline.predict_pipeline import PredictPipeline

APP_TITLE = "Student Performance Predictor"

logger = get_logger(__name__)


# ---- Pydantic request models ----
class StudentFeatures(BaseModel):
    """
    Generic prediction request model.
    Validates against categories learned during training.
    Works with ANY dataset - no hard-coded values!
    """

    # Define fields as generic strings (no Literal types)
    model_config = {"extra": "forbid"}  # Reject unexpected fields

    # Dynamic fields - populated based on preprocessor
    # These will be validated against the trained categories

    @model_validator(mode="before")
    @classmethod
    def normalize_and_validate(cls, data: Any) -> Any:
        """
        1. Normalize inputs (strip whitespace, lowercase)
        2. Will be validated against preprocessor categories later
        """
        if isinstance(data, dict):
            return {
                key: value.strip().lower() if isinstance(value, str) else value
                for key, value in data.items()
            }
        return data


# ---- Lifespan (preferred over on_event startup) ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline = PredictPipeline()
    pipeline._load_artifacts()  # pre-load artifacts at startup; subsequent requests will use in-memory cache
    app.state.pipeline = pipeline
    yield
    # nothing to cleanup


app = FastAPI(title=APP_TITLE, version=__version__, lifespan=lifespan)


# ---- Middleware ----
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid4())
    request.state.request_id = request_id

    # Add to response headers
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ---- UI wiring ----
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# ---- Validation helper ----
def _validate_and_normalize(
    item: Dict[str, Any],
    expected_features: List[str],
    preprocessor: Any,
    item_prefix: str = "",
) -> Dict[str, Any]:
    """
    Validate and normalize a single prediction payload dict.
    Raises ValueError with an optional item_prefix for batch error messages.
    """
    prefix = f"{item_prefix}: " if item_prefix else ""

    missing = [f for f in expected_features if f not in item]
    if missing:
        raise ValueError(f"{prefix}Missing required fields: {missing}")

    extra = [f for f in item.keys() if f not in expected_features]
    if extra:
        raise ValueError(f"{prefix}Unexpected fields: {extra}")

    normalized = {
        key: (
            value.strip().lower() if isinstance(value, str) and value.strip() else value
        )
        for key, value in item.items()
    }

    empty = [k for k, v in normalized.items() if v == "" or v is None]
    if empty:
        raise ValueError(f"{prefix}Empty values not allowed for fields: {empty}")

    if hasattr(preprocessor, "transformers_"):
        for transformer_name, transformer, feature_cols in preprocessor.transformers_:
            if transformer_name == "categorical" and hasattr(
                transformer.named_steps["onehot"], "categories_"
            ):
                ohe = transformer.named_steps["onehot"]
                categories = ohe.categories_

                for i, col in enumerate(feature_cols):
                    if col in normalized:
                        valid_cats = [str(c).lower() for c in categories[i]]
                        user_value = str(normalized[col]).lower()

                        if user_value not in valid_cats:
                            raise ValueError(
                                f"{prefix}Invalid value '{item[col]}' for field '{col}'. "
                                f"Expected one of: {', '.join(sorted(set(str(c) for c in categories[i])))}"
                            )

    return normalized


# ---- API endpoints ----
@app.get("/health")
def health(request: Request) -> dict:
    loaded = (
        hasattr(request.app.state, "pipeline")
        and request.app.state.pipeline is not None
    )
    return {"status": "ok", "pipeline_loaded": loaded}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "app_title": APP_TITLE}
    )


@app.get("/schema")
def schema(request: Request) -> Dict[str, Any]:
    """
    Generic schema endpoint (helps UI + external clients).
    Uses the trained preprocessor to expose required input columns.
    """
    pipeline: PredictPipeline = request.app.state.pipeline
    preprocessor, _ = pipeline._load_artifacts()  # uses your existing artifact loader

    cols = getattr(preprocessor, "feature_names_in_", None)
    if cols is None:
        return {"features": [], "note": "preprocessor has no feature_names_in_"}

    # Minimal schema info; UI can render text inputs for everything.
    return {"features": [{"name": c, "type": "text"} for c in list(cols)]}


@app.get("/meta")
def meta(request: Request) -> dict:
    """Expose useful metadata about the model and artifacts for debugging and UI display."""
    pipeline: PredictPipeline = request.app.state.pipeline

    return {
        "artifacts_dir": str(pipeline.config.artifacts_dir),
        "model_path": str(pipeline.config.model_path),
        "preprocessor_path": str(pipeline.config.preprocessor_path),
        "GCS_ARTIFACTS_URI": os.getenv("GCS_ARTIFACTS_URI"),
        "MODEL_REGISTRY_URI": os.getenv("MODEL_REGISTRY_URI"),
        "FORCE_MODEL_DOWNLOAD": os.getenv("FORCE_MODEL_DOWNLOAD"),
    }


@app.get("/model_info")
def model_info(request: Request) -> dict:
    """Get information about the currently loaded model."""
    pipeline: PredictPipeline = request.app.state.pipeline

    try:
        report_path = pipeline.config.report_path
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)

            return {
                "best_model": report.get("best_model", {}),
                "trained_at": report.get("best_model", {}).get("timestamp", "unknown"),
                "test_r2": report.get("best_model", {}).get("test_r2"),
            }
    except Exception:
        pass

    return {"error": "Model report not available"}


@app.post("/predict")
def predict_one(payload: Dict[str, Any], request: Request) -> dict:
    """
    Predict single instance with dynamic validation.
    Validations:
    - Required fields are present (based on preprocessor)
    - No extra fields (forbids typos)
    - String values are normalized (strip whitespace, lowercase)
    - Categorical values are validated against preprocessor categories (case-insensitive)
    - Empty values are not allowed (after normalization)
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger.info(f"Received prediction request with ID: {request_id}")

    try:
        pipeline: PredictPipeline = request.app.state.pipeline
        preprocessor, model = pipeline._load_artifacts()

        expected_features = list(preprocessor.feature_names_in_)
        normalized_payload = _validate_and_normalize(
            payload, expected_features, preprocessor
        )

        pred = pipeline.predict(normalized_payload)[0]
        return {"prediction": float(pred)}

    except ValueError as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=str(e))

    except CustomException:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")

    except Exception as e:
        logger.exception(f"Prediction failed with error: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/predict_batch")
def predict_batch(payload: List[Dict[str, Any]], request: Request) -> dict:
    """
    Batch prediction with dynamic validation.
    """
    try:
        pipeline: PredictPipeline = request.app.state.pipeline
        preprocessor, model = pipeline._load_artifacts()

        expected_features = list(preprocessor.feature_names_in_)

        normalized_items = [
            _validate_and_normalize(
                item, expected_features, preprocessor, item_prefix=f"Item {idx}"
            )
            for idx, item in enumerate(payload)
        ]

        preds = pipeline.predict(normalized_items)
        preds = np.asarray(preds).ravel()
        return {"prediction": [float(x) for x in preds]}

    except ValueError as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=str(e))

    except CustomException:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")

    except Exception as e:
        logger.exception(f"Prediction failed with error: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")
