# 🎓 Student Performance Predictor — Practical End-to-End MLOps

[![CI](https://github.com/Ishtiaque-h/student-performance-indicator/actions/workflows/ci.yml/badge.svg)](https://github.com/Ishtiaque-h/student-performance-indicator/actions/workflows/ci.yml)
[![Staging Deploy](https://github.com/Ishtiaque-h/student-performance-indicator/actions/workflows/deploy.yml/badge.svg)](https://github.com/Ishtiaque-h/student-performance-indicator/actions/workflows/deploy.yml)
[![CD Production](https://github.com/Ishtiaque-h/student-performance-indicator/actions/workflows/cd-cloudrun.yml/badge.svg?event=push)](https://github.com/Ishtiaque-h/student-performance-indicator/actions/workflows/cd-cloudrun.yml)
[![Retrain](https://github.com/Ishtiaque-h/student-performance-indicator/actions/workflows/retrain.yml/badge.svg)](https://github.com/Ishtiaque-h/student-performance-indicator/actions/workflows/retrain.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A production-oriented ML system that predicts student **math score** from enrollment-time attributes, with reproducible training, staged deployment, and artifact promotion.

🔗 **Live API**: https://student-performance-api-654581958038.us-central1.run.app

---

## 1) ML framing and intuition

### Problem
Predict math score (0–100) using only features available **before exams**:
- gender
- race/ethnicity
- parental level of education
- lunch
- test preparation course

Dataset: [Students Performance in Exams (Kaggle)](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

### Why this feature set is intentional
`reading_score` and `writing_score` are excluded from training and serving because they are measured in the same sitting as math score. They are highly predictive but unavailable at real decision time, so including them would create target leakage and unrealistic performance.

### Practical expectation for model quality
With only 5 categorical enrollment-time features, predictive signal is real but limited. Moderate R² is expected. This project uses a quality gate (`test_r2 >= 0.10`) to reject very weak models while staying realistic about the information available at inference time.

---

## 2) What is production-relevant here

- **Promote, don’t retrain** for production: production serves the exact artifact validated in staging.
- **Pipeline-only serving contract**: inference requires `pipeline.pkl` + metadata files.
- **Dynamic validation**: API validates request fields/categories against the trained preprocessor schema.
- **Artifact versioning in GCS**: immutable run-indexed artifacts + promoted pointer.
- **Automated MLOps path**: CI → staging deploy/retrain → manual tag-triggered production deploy.

---

## 3) End-to-end lifecycle

1. Train candidate model(s) and save artifacts.
2. Evaluate with CV + holdout metrics; select best model.
3. Publish run artifacts to GCS run index (`latest/<run_id>/...`).
4. Gate candidate in retrain workflow (`test_r2` threshold).
5. Write promoted pointer (`promoted/latest_uri.txt`).
6. Production CD reads pointer and deploys the exact promoted artifact set.

This avoids training-time randomness between staging and production.

---

## 4) Model development approach

### Models evaluated
- Dummy (mean baseline)
- LinearRegression, Ridge, Lasso
- KNN, DecisionTree, RandomForest
- AdaBoost, GradientBoosting
- XGBoost (optional), CatBoost (optional)

### Selection strategy
- 5-fold CV scoring (`r2`)
- Two-stage tuning: RandomizedSearchCV → refined GridSearchCV
- Best model selected by CV-preferred criterion (not direct test-set overfitting)
- Reported metrics: R², MAE, RMSE

### Preprocessing
- Categorical: most-frequent imputation + one-hot encoding (`handle_unknown='ignore'`)
- Numeric (if present): median imputation + standard scaling

---

## 5) Artifact contract (important)

### Training outputs
Training currently writes:
- `pipeline.pkl` (required for serving)
- `model_report.json`
- `ingestion_meta.json`
- `model.pkl` and `preprocessor.pkl` (kept as auxiliary artifacts)

### Serving requirement
Serving is pipeline-first and expects:
- `pipeline.pkl`
- `model_report.json`
- `ingestion_meta.json`

No fallback to `preprocessor.pkl + model.pkl` is used in inference serving path.

---

## 6) API behavior

### Core endpoints
- `GET /health` — service health + pipeline state
- `GET /schema` — learned input schema/categories
- `GET /meta` — artifact/source metadata
- `GET /model_info` — selected model metrics from report
- `POST /predict` — single prediction
- `POST /predict_batch` — batch prediction

### Validation behavior
- Required fields enforced
- Unexpected fields rejected
- Strings normalized (`strip + lowercase`)
- Whitespace-only / null values rejected
- Category values checked against trained categories
- Optional numeric range guards via config

---

## 7) Workflows

- `ci.yml`: lint + format + smoke tests on PR/push
- `deploy.yml`: train + publish + deploy to staging after CI on `main`
- `retrain.yml`: scheduled/manual retrain, gate, promote pointer, rollout to staging
- `cd-cloudrun.yml`: tag-triggered production deploy from promoted pointer

### Required cloud contracts
These workflows require correct GitHub variables/secrets and GCP setup (WIF, service accounts, project/region/service/repository, GCS URIs). Local checks can be green while cloud workflows still fail if env/secrets are missing or mismatched.

---

## 8) Quickstart

### Local setup
```bash
git clone https://github.com/Ishtiaque-h/student-performance-indicator.git
cd student-performance-indicator
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[all]"
```

### Train locally
```bash
python scripts/train_and_publish.py \
  --registry-uri gs://YOUR-BUCKET/student-performance \
  --index-latest
```

### Run API locally
```bash
uvicorn student_performance.api:app --reload --port 8000
curl http://localhost:8000/health
```

### Example prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "female",
    "race_ethnicity": "group B",
    "parental_level_of_education": "bachelor'\''s degree",
    "lunch": "standard",
    "test_preparation_course": "none"
  }'
```

---

## 9) Validation commands

From repo root:

```bash
ruff check src tests scripts
black --check src tests scripts
pytest -m smoke -q
python -m pytest -q
```

---

## 10) Project structure (current)

```text
student-performance-indicator/
├── .github/workflows/
├── data/
├── notebooks/
├── scripts/
├── src/student_performance/
│   ├── api.py
│   ├── modeling.py
│   ├── artifacts_gcs.py
│   ├── components/
│   ├── pipeline/
│   ├── registry/
│   ├── mlops/
│   ├── templates/
│   └── static/
├── tests/
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## 11) Practical caveats

- This is a strong MLOps learning system and practical template, but not a causal model of student outcomes.
- Current monitoring is deployment-health oriented; online drift/performance monitoring can be added later.
- Performance interpretation should always be tied to the deployment-time feature availability constraint.

---

## License
MIT — see [LICENSE](LICENSE)

## Author
**Md Ishtiaque Hossain**
- GitHub: [@Ishtiaque-h](https://github.com/Ishtiaque-h)
- LinkedIn: [@ishtiaque-h](https://linkedin.com/in/ishtiaque-h)
