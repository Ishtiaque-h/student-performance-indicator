# 🎓 Student Performance Predictor - End-to-End MLOps Pipeline

[![CI](https://github.com/Ishtiaque-h/student-performance-indicator/actions/workflows/ci.yml/badge.svg)](https://github.com/Ishtiaque-h/student-performance-indicator/actions/workflows/ci.yml)
[![Deploy Staging](https://github.com/Ishtiaque-h/student-performance-indicator/actions/workflows/deploy.yml/badge.svg)](https://github.com/Ishtiaque-h/student-performance-indicator/actions/workflows/deploy.yml)
[![Deploy Production](https://github.com/Ishtiaque-h/student-performance-indicator/actions/workflows/cd-cloudrun.yml/badge.svg)](https://github.com/Ishtiaque-h/student-performance-indicator/actions/workflows/cd-cloudrun.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A **production-grade machine learning system** that predicts student math scores based on demographic and educational factors. This project demonstrates end-to-end MLOps practices including automated retraining, quality gates, staged deployments, and cloud-native architecture.

🔗 **Live Demo**: [https://student-performance-api-[YOUR-DOMAIN].run.app](https://student-performance-api-654581958038.us-central1.run.app)

---

## 🎯 Project Highlights

Focus isn't just a model - it's a **complete ML production system**:

- ✅ **10+ ML models** with hyperparameter tuning (RandomizedSearch → GridSearch)
- ✅ **Automated CI/CD** with GitHub Actions
- ✅ **Staging + Production environments** with manual promotion gates
- ✅ **Weekly automated retraining** with quality thresholds
- ✅ **FastAPI REST service** with health checks and schema introspection
- ✅ **Cloud-native deployment** on Google Cloud Run
- ✅ **MLflow integration** for experiment tracking
- ✅ **Docker containerization** with multi-stage builds
- ✅ **Comprehensive testing** (smoke tests, dynamic schema validation, post-deploy validation)

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
Create Release Tag (e.g., v3.2.6)
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
Numerical Features: 2 (Here we used none to prevent data leakage)
  ├─ Imputation: SimpleImputer(strategy='median')
  └─ Standardization: StandardScaler()

Categorical Features: 5
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
│   │   ├── data_ingestion.py  # Data loading + splitting
│   │   ├── data_transformation.py  # Preprocessing
│   │   └── model_trainer.py   # Model training + tuning
│   ├── pipeline/
│   │   ├── train_pipeline.py  # Training orchestration
│   │   └── predict_pipeline.py  # Inference pipeline
│   ├── mlops/
│   │   └── mlflow_logger.py   # MLflow integration
│   ├── registry/
│   │   └── gcs_registry.py    # GCS artifact management
│   ├── api.py                  # FastAPI application
│   ├── modeling.py             # evaluate_models implementation
│   └── utils.py                # Utility functions
├── scripts/
│   └── train_and_publish.py   # CLI for training + publishing
├── tests/
│   └── test_smoke.py           # Smoke tests
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
        random_state=42
    ),
    tuning=TuningConfig(
        cv=5,
        scoring="r2",
        random_n_iter=25,
        prefer_cv_for_selection=True
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
| `GET` | `/schema` | Feature schema (from trained preprocessor) |
| `GET` | `/meta` | Artifact metadata (paths, versions) |
| `POST` | `/predict` | Single prediction |
| `POST` | `/predict_batch` | Batch predictions |

### **Example Request**

```bash
curl -X POST https://student-performance-api-[YOUR-DOMAIN].run.app/predict \
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

| Test Type | Coverage | When |
|-----------|----------|------|
| **Smoke Tests** | End-to-end pipeline | Every PR + deployment |
| **Linting** | Code quality (ruff) | Every commit |
| **Formatting** | Code style (black) | Every commit |
| **Post-Deploy** | Live API validation | After production deploy |

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

**Ishtiaque Hossain**
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