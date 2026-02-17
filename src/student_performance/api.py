from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from student_performance.exception import CustomException
from student_performance.pipeline.predict_pipeline import PredictPipeline

APP_TITLE = "Student Performance Predictor"


# ---- Pydantic request models (keep these; they're good practice) ----
class StudentFeatures(BaseModel):
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str


# ---- Lifespan (preferred over on_event startup) ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline = PredictPipeline()
    pipeline._load_artifacts()  # pre-load artifacts at startup; subsequent requests will use in-memory cache
    app.state.pipeline = pipeline
    yield
    # nothing to cleanup


app = FastAPI(title=APP_TITLE, version="1.0", lifespan=lifespan)

# ---- UI wiring (optional, but great for portfolio) ----
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


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
    pipeline: PredictPipeline = request.app.state.pipeline

    import os

    return {
        "artifacts_dir": str(pipeline.config.artifacts_dir),
        "model_path": str(pipeline.config.model_path),
        "preprocessor_path": str(pipeline.config.preprocessor_path),
        "GCS_ARTIFACTS_URI": os.getenv("GCS_ARTIFACTS_URI"),
        "MODEL_REGISTRY_URI": os.getenv("MODEL_REGISTRY_URI"),
        "FORCE_MODEL_DOWNLOAD": os.getenv("FORCE_MODEL_DOWNLOAD"),
    }


logger = logging.getLogger("uvicorn.error")


@app.post("/predict")
def predict_one(payload: StudentFeatures, request: Request) -> dict:
    try:
        pipeline: PredictPipeline = request.app.state.pipeline
        pred = pipeline.predict(payload.model_dump())[0]
        return {"prediction": float(pred)}
    except ValueError as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except CustomException as e:
        # Server error - system issue
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")
    except Exception:
        logger.exception(
            f"Prediction failed with error: {type(e).__name__}: {str(e)}"
        )  # <-- this prints stack trace to Cloud Run logs
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/predict_batch")
def predict_batch(payload: List[StudentFeatures], request: Request) -> dict:
    try:
        pipeline: PredictPipeline = request.app.state.pipeline
        rows = [p.model_dump() for p in payload]
        preds = pipeline.predict(rows)
        preds = np.asarray(preds).ravel()
        return {"prediction": [float(x) for x in preds]}
    except ValueError as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception(
            f"Prediction failed with error: {type(e).__name__}: {str(e)}"
        )  # <-- this prints stack trace to Cloud Run logs
        raise HTTPException(status_code=500, detail="Prediction failed")
