from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Any, Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, model_validator

from student_performance.exception import CustomException
from student_performance.pipeline.predict_pipeline import PredictPipeline

APP_TITLE = "Student Performance Predictor"


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
    
    @model_validator(mode='before')
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


app = FastAPI(title=APP_TITLE, version="1.0", lifespan=lifespan)

# ---- UI wiring ----
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

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
    try:
        pipeline: PredictPipeline = request.app.state.pipeline
        preprocessor, model = pipeline._load_artifacts()
        
        # Get expected features from preprocessor
        expected_features = list(preprocessor.feature_names_in_)
        
        # Validate: check all required fields are present
        missing = [f for f in expected_features if f not in payload]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        
        # Validate: check no extra fields
        extra = [f for f in payload.keys() if f not in expected_features]
        if extra:
            raise ValueError(f"Unexpected fields: {extra}")
        
        # Normalize string values
        normalized_payload = {
            key: value.strip().lower() if isinstance(value, str) and value.strip() else value
            for key, value in payload.items()
        }
        
        # Check for empty values
        empty = [k for k, v in normalized_payload.items() if v == "" or v is None]
        if empty:
            raise ValueError(f"Empty values not allowed for fields: {empty}")
        
        # Validate categories against preprocessor (if categorical)
        if hasattr(preprocessor, 'transformers_'):
            for transformer_name, transformer, feature_cols in preprocessor.transformers_:
                if transformer_name == "categorical" and hasattr(transformer.named_steps['onehot'], 'categories_'):
                    ohe = transformer.named_steps['onehot']
                    categories = ohe.categories_
                    
                    for i, col in enumerate(feature_cols):
                        if col in normalized_payload:
                            valid_cats = [str(c).lower() for c in categories[i]]
                            user_value = str(normalized_payload[col]).lower()
                            
                            if user_value not in valid_cats:
                                raise ValueError(
                                    f"Invalid value '{payload[col]}' for field '{col}'. "
                                    f"Expected one of: {', '.join(sorted(set(str(c) for c in categories[i])))}"
                                )
        
        # Make prediction
        pred = pipeline.predict(normalized_payload)[0]
        return {"prediction": float(pred)}
    
    except ValueError as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=str(e))
    
    except CustomException:
        logging.exception("Prediction failed")
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
        
        # Get expected features
        expected_features = list(preprocessor.feature_names_in_)
        
        # Validate and normalize each item
        normalized_items = []
        for idx, item in enumerate(payload):
            # Check required fields
            missing = [f for f in expected_features if f not in item]
            if missing:
                raise ValueError(f"Item {idx}: Missing required fields: {missing}")
            
            # Check extra fields
            extra = [f for f in item.keys() if f not in expected_features]
            if extra:
                raise ValueError(f"Item {idx}: Unexpected fields: {extra}")
            
            # Normalize
            normalized = {
                key: value.strip().lower() if isinstance(value, str) and value.strip() else value
                for key, value in item.items()
            }
            
            # Check empty
            empty = [k for k, v in normalized.items() if v == "" or v is None]
            if empty:
                raise ValueError(f"Item {idx}: Empty values not allowed for fields: {empty}")
            
            # Validate categories
            if hasattr(preprocessor, 'transformers_'):
                for transformer_name, transformer, feature_cols in preprocessor.transformers_:
                    if transformer_name == "categorical" and hasattr(transformer.named_steps['onehot'], 'categories_'):
                        ohe = transformer.named_steps['onehot']
                        categories = ohe.categories_
                        
                        for i, col in enumerate(feature_cols):
                            if col in normalized:
                                valid_cats = [str(c).lower() for c in categories[i]]
                                user_value = str(normalized[col]).lower()
                                
                                if user_value not in valid_cats:
                                    raise ValueError(
                                        f"Item {idx}: Invalid value '{item[col]}' for field '{col}'. "
                                        f"Expected one of: {', '.join(sorted(set(str(c) for c in categories[i])))}"
                                    )
            
            normalized_items.append(normalized)
        
        # Make predictions
        preds = pipeline.predict(normalized_items)
        preds = np.asarray(preds).ravel()
        return {"prediction": [float(x) for x in preds]}
    
    except ValueError as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=str(e))
    
    except CustomException:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")
    
    except Exception as e:
        logger.exception(f"Prediction failed with error: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")