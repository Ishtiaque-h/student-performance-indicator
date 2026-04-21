# Monitoring Incident Runbook
## 1. Triage
1. Confirm whether alert is warning or critical.
2. Check dashboard service health panel (availability/error/latency).
3. Check model health panel (drift + online performance + segment metrics).
4. Verify blast radius (single endpoint vs all endpoints, single segment vs all segments).
## 2. Immediate response
- **Critical service alert:** page on-call and start incident bridge.
- **Critical model alert:** freeze promotions and inspect canary/production comparison.
- **Sustained drift/performance criticals:** auto-create retrain candidate.
## 3. Rollback decision
Rollback immediately if either is true:
- post-deploy regression is confirmed, or
- canary fails R²/error-rate guardrails.
Rollback actions:
1. Switch traffic back to champion deployment.
2. Confirm service SLO recovery.
3. Continue monitoring for 30 minutes.
## 4. Retrain workflow
1. Trigger retrain candidate pipeline.
2. Run champion/challenger gate:
   - challenger must beat champion on defined metrics
   - challenger must pass segment/fairness checks
3. Promote only if gate passes.
## 5. Incident closure
1. Document timeline, root cause, and impact.
2. Record which threshold fired and whether threshold tuning is needed.
3. Add prevention item (feature validation, data contract, rollback automation update).
4. Close incident after 24h stable monitoring window.