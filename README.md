# 🎓 Student Performance Predictor — End-to-End MLOps Pipeline

[![CI](https://github.com/Ishtiaque-h/student-performance-indicator/actions/workflows/ci.yml/badge.svg)](https://github.com/Ishtiaque-h/student-performance-indicator/actions/workflows/ci.yml)
[![Staging Deploy](https://github.com/Ishtiaque-h/student-performance-indicator/actions/workflows/deploy.yml/badge.svg)](https://github.com/Ishtiaque-h/student-performance-indicator/actions/workflows/deploy.yml)
[![CD Production](https://github.com/Ishtiaque-h/student-performance-indicator/actions/workflows/cd-cloudrun.yml/badge.svg?event=push)](https://github.com/Ishtiaque-h/student-performance-indicator/actions/workflows/cd-cloudrun.yml)
[![Retrain](https://github.com/Ishtiaque-h/student-performance-indicator/actions/workflows/retrain.yml/badge.svg)](https://github.com/Ishtiaque-h/student-performance-indicator/actions/workflows/retrain.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A **production-grade machine learning system** that predicts student math scores based on demographic and educational factors. This project demonstrates end-to-end MLOps practices including automated retraining, quality gates, staged deployments, and cloud-native architecture.

🔗 **Live Demo**: [https://student-performance-api-654581958038.us-central1.run.app](https://student-performance-api-654581958038.us-central1.run.app)

---

## 🎯 Project Highlights

Focus isn't just a model — it's a **complete ML production system**:

- ✅ **10+ ML models** with hyperparameter tuning (RandomizedSearch → GridSearch)
- ✅ **Automated CI/CD** with GitHub Actions (4 workflows)
- ✅ **Staging + Production environments** with manual promotion gates
- ✅ **Weekly automated retraining** with quality thresholds
- ✅ **"Promote, don't retrain"** — production always gets the exact staged model
- ✅ **FastAPI REST service** with dynamic schema validation
- ✅ **Cloud-native deployment** on Google Cloud Run
- ✅ **MLflow integration** for experiment tracking
- ✅ **Docker containerization** with multi-stage builds
- ✅ **Workload Identity Federation** — no long-lived GCP credentials
- ✅ **Dataset-agnostic design** — change one file to use any dataset

---

## 🌟 Key Differentiators

### **1. "Promote, Don't Retrain" Model Deployment**
Production deployments never retrain from scratch. The model artifact is:
1. Trained once in a controlled environment
2. Validated against a quality gate (R²)
3. Deployed to staging for human review
4. Promoted to production **as the exact same binary** — no variance, no surprises

### **2. Dataset-Agnostic Design**
- Change **one file** (`config.py`) to use a different dataset
- API validation, preprocessing, and training adapt automatically
- Validation logic reads from **trained preprocessor** (single source of truth)

### **3. Production-Grade Testing**
- Smoke tests focus on "does the whole pipeline run?"
- Dynamic schema validation prevents invalid inputs
- Post-deploy API smoke tests validate every production release

### **4. MLOps Best Practices**
- Staged deployments (staging → manual review → production)
- Quality gates (R² thresholds, artifact existence checks)
- GCS model registry with versioned run index + promoted pointer
- Automated retraining with human oversight before production

---

## 📊 Problem Statement

**Question**: How do demographic and educational factors affect student academic performance?

**Dataset**: [Kaggle Student Performance Dataset](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- 1,000 students
- 8 features: gender, race/ethnicity, parental education, lunch type, test prep course, scores
- **Target**: Math score (regression task)
- **Features used**: 5 categorical features (excluded reading/writing scores to prevent data leakage)

---

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    Developer Workflow                       │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│               GitHub Actions (4 Workflows)                  │
├─────────────────────────────────────────────────────────────┤
│  ci.yml        │ Lint, Format, Smoke Tests (every PR/push)  │
│  deploy.yml    │ Staging deploy (after CI passes on main)   │
│  retrain.yml   │ Weekly retraining + staging rollout        │
│  cd-cloudrun   │ Production deploy (on v* tag)              │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────┬──────────────────────┐
│     Google Cloud Storage (GCS)       │   Artifact Registry  │
├──────────────────────────────────────┼──────────────────────┤
│ • Model artifacts (model.pkl)        │ • Docker images      │
│ • Preprocessor (preprocessor.pkl)    │ • Tagged by SHA/ver  │
│ • Run index (latest/<run_id>/)       │                      │
│ • Promoted pointer (promoted/        │                      │
│     latest_uri.txt)                  │                      │
│ • MLflow experiment runs             │                      │
└───────────────────────────┬──────────┴──────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│            Google Cloud Run (Serverless)                    │
├─────────────────────────────────────────────────────────────┤
│  Staging:    student-performance-api-staging                │
│  Production: student-performance-api                        │
│  • Auto-scaling (0→N instances)                             │
│  • HTTPS endpoints                                          │
│  • Model loaded from GCS at startup (FORCE_MODEL_DOWNLOAD)  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 MLOps Pipeline

### **Workflow 1 — CI (`ci.yml`)**

Triggered on every pull request and every push to `main`.

```
PR / push to main
    ↓
Lint (ruff) + Format check (black)
    ↓
Smoke tests (pytest -m smoke)
    ↓
✅ Pass → unblocks staging deploy
❌ Fail → blocks merge + staging deploy
```

### **Workflow 2 — Staging Deploy (`deploy.yml`)**

Triggered only after CI passes on `main` (via `workflow_run`).

```
CI passes on main
    ↓
Train model (all 10 models + hyperparameter tuning)
    ↓
Build & push Docker image (tagged with commit SHA)
    ↓
Deploy to Cloud Run STAGING
```

> This deploy is for **code changes**. The model is retrained fresh to validate the new code works end-to-end with a real model.

### **Workflow 3 — Automated Retraining (`retrain.yml`)**

Triggered every Monday at 07:00 UTC (or manually).

```
Schedule: Every Monday 07:00 UTC
    ↓
Train all models with latest data
    ↓
Sync MLflow run to GCS
    ↓
Quality Gate: test_r2 >= 0.10
    ├─ FAIL → workflow stops (model discarded)
    └─ PASS ↓
Write promoted URI → GCS: promoted/latest_uri.txt
    ↓
Update Cloud Run STAGING with new model (hot-reload)
    ↓
[Human reviews staging predictions & MLflow metrics]
```

### **Workflow 4 — Production Deploy (`cd-cloudrun.yml`)**

Triggered by pushing a `v*` tag (e.g. `v3.4.0`).

```
git tag v3.4.0 && git push origin v3.4.0
    ↓
Smoke tests (code quality gate)
    ↓
Read promoted/latest_uri.txt from GCS  ← exact model from staging
    ↓
Build & push Docker image (tagged with version)
    ↓
Verify model.pkl + preprocessor.pkl exist in GCS
    ↓
Deploy to Cloud Run PRODUCTION with promoted model URI
    ↓
Post-deploy smoke tests:
    ├─ GET /health → must return 200
    ├─ POST /predict (valid input) → must return 200 + prediction key
    └─ POST /predict (missing field) → must return 422
```

> **Key principle**: No retraining happens here. Production gets the **exact same `.pkl` binary** that was validated on staging. The tag is a human approval gate, not a training trigger.

---

## 🔑 The "Promote, Don't Retrain" Pattern

This is the most important MLOps design decision in this project:

```
❌ Naive approach (what most tutorials show):
   push tag → retrain → deploy to prod
   Problem: production model was never validated; training is non-deterministic

✅ This project:
   retrain → gate → staging → human review → promote same artifact → prod
   Result: production gets the exact model you reviewed on staging
```

The hand-off is a simple GCS pointer:
```
gs://bucket/student-performance/promoted/latest_uri.txt
→ contents: gs://bucket/student-performance/latest/20260222T070000Z-abc1234/
```

`retrain.yml` writes it. `cd-cloudrun.yml` reads it. One line. Fully auditable.

---

## 📊 Exploratory Data Analysis

The `notebooks/` directory contains Jupyter notebooks documenting the data exploration process:

- **EDA_student_performance.ipynb**:
  - Dataset overview and statistics
  - Feature distributions and correlations
  - Missing value analysis
  - Insights that informed feature engineering decisions

This analysis informed key decisions:
- ✅ Dropping reading/writing scores to prevent data leakage
- ✅ Using only categorical features (no numerical scaling needed)
- ✅ Setting R² threshold at 0.10 (conservative — reflects limited predictive signal from 5 categorical features alone)

[View EDA Notebook →](./notebooks/EDA_student_performance.ipynb)

---

## 🧠 Machine Learning

### **Models Evaluated**

| Model | Description | Hyperparams Tuned |
|-------|-------------|-------------------|
| **Linear Regression** | Baseline | — |
| **Ridge** | L2 regularization | `alpha` |
| **Lasso** | L1 regularization | `alpha` |
| **KNN** | K-nearest neighbors | `n_neighbors`, `weights`, `p` |
| **Decision Tree** | Single tree | `max_depth`, `criterion` |
| **Random Forest** | Ensemble of trees | `n_estimators`, `max_depth`, `max_features` |
| **AdaBoost** | Boosting ensemble | `n_estimators`, `learning_rate` |
| **Gradient Boosting** | GB ensemble | `n_estimators`, `learning_rate`, `max_depth` |
| **XGBoost** | Advanced GB | `n_estimators`, `learning_rate`, `max_depth`, `subsample` |
| **CatBoost** | Cat-optimized GB | `iterations`, `learning_rate`, `depth` |

### **Hyperparameter Tuning Strategy**

**Two-stage approach**:

1. **Broad Search** (RandomizedSearchCV, 25 iterations):
   - Wide parameter ranges
   - Identifies promising regions quickly

2. **Refined Search** (GridSearchCV):
   - Narrows around best params from stage 1
   - Custom refinement strategies:
     - `float_log`: Multiplicative factors (e.g., 0.5×, 1×, 2×)
     - `int_window`: Additive deltas (e.g., −50, 0, +50)
     - `categorical`: Discrete choices

### **Model Selection**

- **Scoring metric**: R² (5-fold CV)
- **Selection criterion**: Prefers CV score over test R² (prevents overfitting to test set)
- **Quality gate**: Test R² ≥ 0.10 required for staging promotion

### **Preprocessing**

```python
Numerical Features:
  ├─ Imputation: SimpleImputer(strategy='median')
  └─ Standardization: StandardScaler()

Categorical Features:
  ├─ Imputation: SimpleImputer(strategy='most_frequent')
  └─ Encoding: OneHotEncoder(handle_unknown='ignore')
```

Output: Sparse matrix (memory-efficient) with automatic densification for models that require dense input (KNN, Decision Trees, Boosting).

---

## 🚀 Quick Start

### **Prerequisites**

- Python 3.11+
- Docker (optional, for local testing)
- GCP account (for deployment)

### **Installation**

```bash
# Clone the repository
git clone https://github.com/Ishtiaque-h/student-performance-indicator.git
cd student-performance-indicator

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev,api,ml,mlops]"
```

### **Train Model Locally**

```bash
# Train all models with hyperparameter tuning
python scripts/train_and_publish.py \
  --registry-uri gs://YOUR-BUCKET/student-performance \
  --index-latest

# Artifacts saved to: artifacts/
# - model.pkl
# - preprocessor.pkl
# - model_report.json
# - ingestion_meta.json
```

### **Run API Locally**

```bash
# Start FastAPI server
uvicorn student_performance.api:app --reload --port 8000

# Test health endpoint
curl http://localhost:8000/health

# Make prediction
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

### **Run Tests**

```bash
# Smoke tests (fast, ~3 seconds)
pytest -m smoke -v

# All tests
pytest -v

# With coverage
pytest --cov=student_performance --cov-report=html
```

---

## 📁 Project Structure

```
student-performance-indicator/
├── .github/workflows/
│   ├── ci.yml                  # Lint, format, smoke tests (every PR + main push)
│   ├── deploy.yml              # Staging deploy (after CI passes)
│   ├── cd-cloudrun.yml         # Production deploy (on v* tag, promote-don't-retrain)
│   └── retrain.yml             # Weekly retraining + staging rollout
├── data/
│   └── raw/stud.csv            # Source dataset
├── src/student_performance/
│   ├── components/
│   │   ├── config.py           # Centralized configuration (change dataset here)
│   │   ├── data_ingestion.py   # Data loading + splitting
│   │   ├── data_transformation.py  # Preprocessing
│   │   └── model_trainer.py    # Model training + tuning
│   ├── pipeline/
│   │   ├── train_pipeline.py   # Training orchestration
│   │   └── predict_pipeline.py # Inference pipeline
│   ├── mlops/
│   │   └── mlflow_logger.py    # MLflow integration
│   ├── registry/
│   │   └── gcs_registry.py     # GCS artifact management
│   ├── api.py                  # FastAPI application
│   ├── modeling.py             # Model evaluation
│   ├── artifacts_gcs.py        # GCS artifact download
│   ├── utils.py                # Utility functions
│   ├── logger.py               # Logging setup
│   └── exception.py            # Custom error handling
├── static/
│   └── app.js                  # Frontend JavaScript
├── templates/
│   └── index.html              # Web UI template
├── scripts/
│   └── train_and_publish.py    # CLI: train + publish to GCS registry
├── tests/
│   ├── test_smoke.py           # Dataset-agnostic end-to-end tests
│   └── test_api_validation.py  # Dynamic schema validation tests
├── notebooks/
│   ├── EDA_student_performance.ipynb
│   └── model_training.ipynb
├── Dockerfile                  # Multi-stage build
├── pyproject.toml              # Package + extras configuration
└── README.md
```

---

## 🔧 Configuration

All configuration is centralized in `src/student_performance/components/config.py`:

```python
CONFIG = PipelineConfig(
    dataset=DatasetConfig(
        data_rel_path="data/raw/stud.csv",
        target_col="math_score",
        drop_cols=["reading_score", "writing_score"]
    ),
    split=SplitConfig(test_size=0.2, random_state=42, shuffle=True),
    tuning=TuningConfig(
        cv=5,
        scoring="r2",
        random_n_iter=25,
        random_seed=42,
        prefer_cv_for_selection=True
    ),
    dense_safety=DenseSafetyConfig(
        dense_feature_threshold=5000,
        dense_cell_threshold=5_000_000
    )
)
```

**To use a different dataset**: Update `data_rel_path`, `target_col`, and `drop_cols`. Everything else adapts automatically.

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check + artifact status |
| `GET` | `/` | Web UI (form-based prediction) |
| `GET` | `/schema` | Feature schema (from trained preprocessor) ✨ |
| `GET` | `/meta` | Artifact metadata (paths, versions) |
| `POST` | `/predict` | Single prediction (with dynamic validation) ✨ |
| `POST` | `/predict_batch` | Batch predictions (with dynamic validation) ✨ |

✨ = **Dynamic validation** — inputs validated against categories learned during training

### **Example Request**

```bash
curl -X POST https://student-performance-api-654581958038.us-central1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "male",
    "race_ethnicity": "group C",
    "parental_level_of_education": "some college",
    "lunch": "free/reduced",
    "test_preparation_course": "completed"
  }'
```

**Response**:
```json
{ "prediction": 68.4 }
```

---

## 🔄 Dynamic Schema Validation

This API validates incoming requests against the **categories learned during training**, not hard-coded values. This makes the system fully dataset-agnostic.

### How It Works

1. **Training**: `OneHotEncoder` learns valid categories from data
2. **API startup**: Loads preprocessor, extracts `encoder.categories_`
3. **Request time**: Validates user input against learned categories

### Example Error Responses

```json
{ "detail": "Empty values not allowed for fields: ['gender']" }
{ "detail": "Invalid value 'non-binary' for field 'gender'. Expected one of: female, male" }
{ "detail": "Missing required fields: ['test_preparation_course']" }
```

---

## 🔒 Production Deployment

### **Infrastructure**

| Component | Technology |
|-----------|------------|
| Serving platform | Google Cloud Run (serverless) |
| Authentication | Workload Identity Federation (no SA keys) |
| Artifact storage | Google Cloud Storage |
| Container registry | Google Artifact Registry |
| Secrets | GitHub Secrets + Environments |
| Experiment tracking | MLflow → GCS |

### **Deployment Process**

```bash
# 1. Retraining runs automatically every Monday
#    → staging is updated automatically

# 2. Review staging
open https://student-performance-api-staging-....run.app

# 3. Promote to production (triggers cd-cloudrun.yml)
git tag v3.4.0
git push origin v3.4.0
```

### **Quality Gates**

| Gate | Where | What |
|------|-------|------|
| Lint + format | CI | Code quality before any deploy |
| Smoke tests (code) | CI + CD | Pipeline runs end-to-end |
| R² ≥ 0.10 | `retrain.yml` | Model quality before staging |
| Artifact existence | `cd-cloudrun.yml` | `model.pkl` + `preprocessor.pkl` in GCS |
| Post-deploy smoke | `cd-cloudrun.yml` | Live `/health` + `/predict` + 422 check |

---

## 📈 Monitoring & Observability

### **Current**

- ✅ Post-deploy smoke tests (health + prediction validation)
- ✅ MLflow experiment tracking (synced to GCS per run)
- ✅ Artifact versioning (GCS run index + promoted pointer)
- ✅ Docker image versioning (SHA for staging, semver for production)
- ✅ Deployment logs (Cloud Run)

### **Future Enhancements**

- [ ] Data drift detection
- [ ] Model performance monitoring (production predictions vs. ground truth)
- [ ] Cloud Monitoring alerts
- [ ] A/B testing / canary deployments
- [ ] Slack/email notifications on retrain gate failure

---

## 🧪 Testing Strategy

| Test Type | Coverage | Trigger | Purpose |
|-----------|----------|---------|---------|
| **Smoke tests** | End-to-end pipeline | Every PR + push | Does the pipeline run? |
| **API validation** | Dynamic schema | Every commit | Are inputs validated? |
| **Linting (ruff)** | Code quality | Every commit | Style + correctness |
| **Formatting (black)** | Code style | Every commit | Consistent formatting |
| **Post-deploy smoke** | Live API | After production deploy | Is the deployment healthy? |

### **Running Tests**

```bash
pytest -m smoke -v                              # fast smoke tests
pytest -v                                       # all tests
pytest --cov=student_performance --cov-report=html  # with coverage
pytest tests/test_api_validation.py -v          # schema validation only
```

---

## 🤝 Contributing

This is a portfolio project, but feedback is welcome!

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit: `git commit -m 'Add my feature'`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📝 License

MIT License — see [LICENSE](LICENSE) file.

---

## 👤 Author

**Md Ishtiaque Hossain**
- GitHub: [@Ishtiaque-h](https://github.com/Ishtiaque-h)
- LinkedIn: [@ishtiaque-h](https://linkedin.com/in/ishtiaque-h)

---

## 🙏 Acknowledgments

- Dataset: [Kaggle — Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- Inspiration: Production ML systems at companies like Netflix, Uber, Airbnb
- Tools: FastAPI, scikit-learn, XGBoost, CatBoost, MLflow, GitHub Actions, Google Cloud

---

**Related Projects**: [Boston House Price Prediction](https://github.com/Ishtiaque-h/boston-house-pricing.git)

---

**⭐ If you find this project helpful, please consider giving it a star!**