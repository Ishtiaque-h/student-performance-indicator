# 🎓 Student Performance Predictor - End-to-End MLOps Pipeline

[![CI](https://github.com/Ishtiaque-h/student-performance-indicator/actions/workflows/ci.yml/badge.svg)](https://github.com/Ishtiaque-h/student-performance-indicator/actions/workflows/ci.yml)
[![CD - Deploy](https://github.com/Ishtiaque-h/student-performance-indicator/actions/workflows/deploy.yml/badge.svg)](https://github.com/Ishtiaque-h/student-performance-indicator/actions/workflows/deploy.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A **production-grade machine learning system** that predicts student math scores based on demographic and educational factors. This project demonstrates end-to-end MLOps practices including automated retraining, quality gates, staged deployments, and cloud-native architecture.

🔗 **Live Demo**: [https://student-performance-api-654581958038.us-central1.run.app](https://student-performance-api-654581958038.us-central1.run.app)

---

## 🎯 Project Highlights

This isn't just a model - it's a **complete ML production system**:

- ✅ **10+ ML models** with hyperparameter tuning (RandomizedSearch → GridSearch)
- ✅ **Automated CI/CD** with GitHub Actions
- ✅ **Staging + Production environments** with manual promotion gates
- ✅ **Weekly automated retraining** with quality thresholds
- ✅ **FastAPI REST service** with dynamic schema validation
- ✅ **Cloud-native deployment** on Google Cloud Run
- ✅ **MLflow integration** for experiment tracking
- ✅ **Docker containerization** with multi-stage builds
- ✅ **Dataset-agnostic design** - change one file to use any dataset

---

## 🌟 Key Differentiators

What makes this project stand out:

### **1. Dataset-Agnostic Design**
- Change **one file** (`config.py`) to use a different dataset
- API validation, preprocessing, and training adapt automatically
- Validation logic reads from **trained preprocessor** (single source of truth)

### **2. Production-Grade Testing**
- Smoke tests focus on "does the whole pipeline run?"
- Dynamic schema validation prevents invalid inputs
- Tests work with **any dataset** (real data → synthetic fallback)

### **3. Sophisticated Hyperparameter Tuning**
- Two-stage approach: RandomizedSearch (broad) → GridSearch (refined)
- Custom refinement strategies (window-based, log-scale)
- Smart densification for sparse matrices

### **4. MLOps Best Practices**
- Staged deployments (staging → manual review → production)
- Quality gates (R² thresholds, health checks)
- Artifact versioning (GCS + Git tags)
- Automated retraining with human oversight

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
│               GitHub Actions (CI/CD)                        │
├─────────────────────────────────────────────────────────────┤
│  • CI: Lint, Format, Smoke Tests                            │
│  • Staging: Auto-deploy on push to main                     │
│  • Production: Tag-based deployment (manual)                │
│  • Retrain: Weekly scheduled (Monday 7 AM UTC)              │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────┬──────────────────────┐
│     Google Cloud Storage (GCS)       │   Artifact Registry  │
├──────────────────────────────────────┼──────────────────────┤
│ • Model artifacts (model.pkl)        │ • Docker images      │
│ • Preprocessor (preprocessor.pkl)    │ • Tagged versions    │
│ • MLflow experiments                 │                      │
└───────────────────────────┬──────────┴──────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│            Google Cloud Run (Serverless)                    │
├─────────────────────────────────────────────────────────────┤
│  Staging: student-performance-api-staging                   │
│  Production: student-performance-api                        │
│  • Auto-scaling (0→N instances)                             │
│  • HTTPS endpoints                                          │
│  • Health monitoring                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 MLOps Pipeline

### **Deployment Flow**

```
Developer Push → main branch
    ↓
CI: Lint + Format + Smoke Tests (GitHub Actions)
    ↓
Train Model (10 models + hyperparameter tuning)
    ↓
Build Docker Image (with artifacts)
    ↓
Deploy to STAGING (auto)
    ↓
[Manual Testing & Review]
    ↓
Create Release Tag (e.g., v3.3.0)
    ↓
Deploy to PRODUCTION (auto-triggered by tag)
    ↓
Post-Deploy Smoke Tests (health + predictions)
```

### **Automated Retraining Flow**

```
Schedule: Every Monday 7 AM UTC
    ↓
Retrain all models with latest data
    ↓
Quality Gate: test_r2 >= 0.10
    ├─ PASS → Deploy to Staging
    └─ FAIL → Stop (notification sent)
    ↓
Manual Review: Check MLflow metrics
    ↓
Manual Promotion: Create release tag if satisfied
    ↓
Production Deployment
```

**Note on Scheduled Retraining:**
> ❓ **Does retraining automatically update production?**
> 
> **No** - scheduled retraining follows a **safe deployment pattern**:
> 1. ✅ Retraining happens automatically (every Monday)
> 2. ✅ Quality gates ensure model meets minimum R² threshold
> 3. ✅ New model is deployed to **staging** automatically
> 4. ⏸️ **Manual review required** before production promotion
> 5. ✅ Create release tag (e.g., `v3.3.1`) to deploy to production
>
> This prevents untested models from reaching production and allows you to:
> - Review MLflow metrics
> - Test staging deployment
> - Validate predictions look reasonable
> - Control production release timing

---

## 📊 Exploratory Data Analysis

The `notebooks/` directory contains Jupyter notebooks documenting the data exploration process:

- **EDA.ipynb**: 
  - Dataset overview and statistics
  - Feature distributions and correlations
  - Missing value analysis
  - Insights that informed feature engineering decisions

This analysis informed key decisions:
- ✅ Dropping reading/writing scores to prevent data leakage
- ✅ Using only categorical features (no numerical scaling needed)
- ✅ Setting R² threshold at 0.10 (reflecting data characteristics)

[View EDA Notebook →](./notebooks/EDA.ipynb)

---

## 🧠 Machine Learning

### **Models Evaluated**

| Model | Description | Hyperparams Tuned |
|-------|-------------|-------------------|
| **Linear Regression** | Baseline | - |
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

1. **Broad Search** (RandomizedSearchCV):
   - 20 iterations per model
   - Wide parameter ranges
   - Identifies promising regions

2. **Refined Search** (GridSearch):
   - Narrows around best params from stage 1
   - Custom refinement strategies:
     - `float_log`: Multiplicative factors (e.g., 0.5x, 1x, 2x)
     - `int_window`: Additive deltas (e.g., -50, 0, +50)
     - `categorical`: Discrete choices

### **Model Selection**

- **Scoring metric**: Negative MSE (5-fold CV)
- **Selection criterion**: Prefers CV score over test R² (prevents overfitting)
- **Quality gate**: Test R² ≥ 0.10 required for production promotion

### **Preprocessing**

```python
Numerical Features:
  ├─ Imputation: SimpleImputer(strategy='median')
  └─ Standardization: StandardScaler()

Categorical Features:
  ├─ Imputation: SimpleImputer(strategy='most_frequent')
  └─ Encoding: OneHotEncoder(handle_unknown='ignore')
```

**Output**: Sparse matrix (memory-efficient) with automatic densification for models that require it (KNN, Decision Trees, Boosting models).

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
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

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
# Smoke tests
pytest -m smoke -v

# All tests
pytest -v

# With coverage
pytest --cov=student_performance
```

---

## 📁 Project Structure

```
student-performance-indicator/
├── .github/workflows/          # CI/CD pipelines
│   ├── ci.yml                  # Lint, format, tests
│   ├── deploy.yml              # Staging deployment
│   ├── cd-cloudrun.yml         # Production deployment
│   └── retrain.yml             # Scheduled retraining
├── data/
│   └── raw/stud.csv            # Source dataset
├── src/student_performance/
│   ├── components/             # ML pipeline components
│   │   ├── config.py           # Centralized configuration
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
├── static/
│   └── app.js                  # Frontend JavaScript (UI)
├── templates/
│   └── index.html              # HTML templates (UI)
│   ├── api.py                  # FastAPI application
│   ├── modeling.py             # evaluate_models implementation
│   └── utils.py                # Utility functions
│   └── logger.py               # Logging setup
│   └── exception.py            # Custome error handling
│   └── artifacts_gcs.py        # GCS downloads
├── scripts/
│   └── train_and_publish.py    # CLI for training + publishing
├── tests/
│   ├── test_smoke.py           # Dataset-agnostic smoke tests
│   └── test_api_validation.py  # Dynamic schema validation tests
├── notebooks/
│   ├── EDA_student_performance.ipynb   # Model EDA prep
│   └── model_training.ipynb    # Model training prep
├── Dockerfile                  # Multi-stage Docker build
├── pyproject.toml              # Package configuration
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
    split=SplitConfig(
        test_size=0.2,
        random_state=42,
        shuffle=True
    ),
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

**To use a different dataset**: Update `CONFIG.dataset.data_rel_path`, `target_col`, and `drop_cols`.

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

✨ = **Dynamic validation** - inputs validated against categories learned during training

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
{
  "prediction": 68.4
}
```

---

## 🔄 Dynamic Schema Validation

### Design Philosophy

This API uses **dynamic schema validation** - it validates incoming requests against the categories learned during training, not hard-coded values. This makes the entire pipeline **dataset-agnostic**.

### How It Works

1. **Training Phase**:
   ```python
   # Preprocessor learns valid categories from data
   OneHotEncoder().fit(["male", "female", ...])
   ```

2. **API Startup**:
   ```python
   # API loads preprocessor and extracts learned categories
   preprocessor = load_object("artifacts/preprocessor.pkl")
   valid_categories = preprocessor.categories_
   ```

3. **Request Validation**:
   ```python
   # Validates user input against learned categories
   if user_value.lower() not in valid_categories:
       raise ValueError(f"Invalid value. Expected one of: {valid_categories}")
   ```

### Benefits

- ✅ **Single source of truth**: Validation logic comes from training data
- ✅ **Dataset-agnostic**: Works with any tabular dataset
- ✅ **Always in sync**: Impossible for API validation to diverge from model expectations
- ✅ **User-friendly**: Case-insensitive + whitespace trimming
- ✅ **Production-safe**: Rejects invalid inputs before they reach the model

### Example Error Messages

**Empty field:**
```json
{
  "detail": "Empty values not allowed for fields: ['gender']"
}
```

**Invalid category:**
```json
{
  "detail": "Invalid value 'non-binary' for field 'gender'. Expected one of: female, male"
}
```

**Missing field:**
```json
{
  "detail": "Missing required fields: ['test_preparation_course']"
}
```

### Using with a New Dataset

Simply update `config.py` and retrain - the API adapts automatically:

```python
# config.py
CONFIG = PipelineConfig(
    dataset=DatasetConfig(
        data_rel_path="data/raw/new_data.csv",
        target_col="target_column",
        drop_cols=["id", "timestamp"]
    )
)
```

**No API code changes needed!** 🎉

---

## 🔒 Production Deployment

### **Infrastructure**

- **Platform**: Google Cloud Run (serverless)
- **Authentication**: Workload Identity Federation (no service account keys!)
- **Artifact Storage**: Google Cloud Storage
- **Container Registry**: Google Artifact Registry
- **Secrets Management**: GitHub Secrets

### **Deployment Process**

1. **Manual Promotion** (current):
   ```bash
   # After reviewing staging deployment
   git tag v3.3.0
   git push origin v3.3.0
   ```

2. **Automated CI/CD**:
   - Tag push triggers `cd-cloudrun.yml`
   - Trains model with tagged code
   - Builds Docker image
   - Deploys to Cloud Run (production)
   - Runs post-deploy smoke tests

3. **Quality Gates**:
   - Pre-deployment: R² ≥ 0.10
   - Post-deployment: Health check + prediction validation

---

## 📈 Monitoring & Observability

### **Current**

- ✅ Post-deploy smoke tests (health + prediction validation)
- ✅ MLflow experiment tracking
- ✅ Artifact versioning (GCS + tags)
- ✅ Deployment logs (Cloud Run)

### **Future Enhancements**

- [ ] Data drift detection
- [ ] Model performance monitoring
- [ ] Cloud Monitoring alerts
- [ ] A/B testing framework
- [ ] Gradual rollouts (canary deployments)

---

## 🧪 Testing Strategy

### **Smoke Tests** 

Our smoke tests are **fully dataset-agnostic**:

```bash
pytest -m smoke -v
```

**Design Philosophy:**
- **Purpose**: Validate pipeline runs end-to-end, NOT model quality
- **Focus**: Structural correctness (artifacts created, predictions deterministic)
- **Data strategy**: Prefer real data sampling → fall back to synthetic with learnable patterns
- **Speed**: Complete in ~3 seconds using fast Ridge model
- **R² threshold**: `-10.0 to 1.0` (validates it's a number, not model accuracy)

**Why lenient R² thresholds?**
> With the student dataset, R² can be **legitimately negative** when we drop correlated features (reading/writing scores) to prevent data leakage. Smoke tests validate the **pipeline works**, not that the **model is good**.

| Test Type | Coverage | When | Purpose |
|-----------|----------|------|---------|
| **Smoke Tests** | End-to-end pipeline | Every PR + deployment | Does it run? |
| **API Validation Tests** | Dynamic schema validation | Every commit | Are inputs validated? |
| **Linting** | Code quality (ruff) | Every commit | Code style |
| **Formatting** | Code style (black) | Every commit | Consistent formatting |
| **Post-Deploy** | Live API validation | After production deploy | Is deployment healthy? |

### **Running Tests**

```bash
# Run all tests
pytest -v

# Run only smoke tests (fast)
pytest -m smoke -v

# Run with coverage report
pytest --cov=student_performance --cov-report=html

# Run API validation tests
pytest tests/test_api_validation.py -v
```

---

## 🤝 Contributing

This is a portfolio project, but feedback is welcome!

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file.

---

## 👤 Author

**Md Ishtiaque Hossain**
- GitHub: [@Ishtiaque-h](https://github.com/Ishtiaque-h)
- LinkedIn: [@ishtiaque-h](https://linkedin.com/in/ishtiaque-h)

---

## 🙏 Acknowledgments

- Dataset: [Kaggle - Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- Inspiration: Production ML systems at companies like Netflix, Uber, Airbnb
- Tools: FastAPI, scikit-learn, XGBoost, CatBoost, MLflow, GitHub Actions, Google Cloud

---

**Related Projects**:
- [Boston House Price Prediction](https://github.com/Ishtiaque-h/boston-house-pricing.git)

---

**⭐ If you find this project helpful, please consider giving it a star!**