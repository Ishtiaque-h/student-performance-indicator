# Service + Model Health Dashboard
Use a single dashboard with two panels:
## 1) Service health panel
- API availability (target: >= 99.5%)
- Error rate (target: <= 2.0%)
- Prediction latency p95 (target: <= 250ms)
- Request volume by endpoint
- 4xx/5xx trend split by endpoint
## 2) Model health panel
- Categorical drift score by feature (`gender`, `lunch`, `race_ethnicity`, etc.)
- Prediction distribution drift (PSI)
- Segment drift score by `gender`, `lunch`, `race_ethnicity`
- Rolling online quality with delayed labels:
  - global R² / MAE / RMSE
  - segment R² / MAE / RMSE
- Degradation trend:
  - R² drop vs baseline
  - MAE increase ratio vs baseline
  - RMSE increase ratio vs baseline
## Alert overlay
- Warning/critical markers on timeline
- Alert message includes:
  - what breached
  - by how much
  - next action (triage, rollback, retrain)
## Suggested refresh
- Service health: 1 minute
- Drift metrics: 5 minutes
- Delayed-label performance: every label-join run