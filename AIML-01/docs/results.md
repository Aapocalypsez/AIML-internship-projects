# Measured project results

All values below are read from saved run artifacts.

## Model ranking

| PR-AUC (average precision) | Validation | Test |
|---|---:|---:|
| Original logged baseline | 0.520484 | 0.473686 |
| Engineered XGBoost | 0.523149 | 0.474301 |
| Calibrated GNN alone | 0.051242 | 0.041858 |
| Selected calibrated combination | 0.523149 | 0.474301 |

Selected XGBoost weight: **1.0**. GNN did not improve ranking.
Graph account coverage: validation 44.2%, test 32.5%.
Validation ablation includes calibration rows; the second validation half selected weights. Test was held out.

## Live delivery and review

- 1,000 unique inputs -> 1,000 unique scored outputs -> 1,000 database rows.
- 1 duplicate output carried the cached prediction; 1 invalid input reached the DLQ.
- Database label persistence: True; dashboard case/label button proof: True / True.
- Kafka integration run: `da90ea115d`; model versions: 2.
- Actual persisted-scorer parity: 6,000 transactions, 18 features; max error 5.684e-13.

## Latency

| Measurement | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|
| Initial Kafka scoring timer | 93.42 | 300.14 | 542.07 |
| Initial HTTP round trip | 80.16 | 137.21 | 252.95 |
| Optimized scorer including Redis commit | 11.81 | 25.73 | 52.71 |
| Optimized HTTP round trip | 18.09 | 38.33 | 72.33 |

Environment: Minikube internal Service HTTP; 2 CPU / 4 GiB node; builds/transfers finished.
Latest HTTP benchmark: 1,000 requests. Scoring p99 <= 100 ms: **True**.
The initial timer includes SHAP and excludes the final Redis write; the optimized scorer moves SHAP to review and includes that write. These runs also had different background loads. They are not a controlled production throughput comparison.

## Drift and challenger

- Unchanged input drifted share: 0.00; tenfold amount-shift share: 0.25.
- Decision: **rejected**; incumbent validation PR-AUC 0.445927, challenger 0.444985.
- Required improvement: 0.001; eligible feedback applied: 0.
This was a controlled demonstration, not observed production drift. No synthetic fraud labels were used.

## Tests and deployment

- Tests: 41; failures: 0; errors: 0; skipped: 0.
- Kubernetes verification completed: **True**.
- Cluster: fraud-demo; ready deployments: 9.
- Kafka replay stored in cluster: 1000.

## Exact sources

- Original baseline reference: `fcb25615e762440292d6fa9a5b49f04b`. It was imported by the earlier logging script and is not a fresh baseline reproduction.
- Engineered run: `312571adee16448887f2999d349340aa`.
- Graph/blend run: `cb9476c2289b45f087a0e557692e6485`.
- Challenger run: `8e15a567de7e40d1873bd1bc0bbcfe55`.
- Local JSON, prediction Parquet, model files and XML are under `data/processed/` (ignored by Git). Model metrics and verification evidence are attached to MLflow.
- Core versions at report time: xgboost 2.1.4, scikit-learn 1.6.1, feast 0.67.0, mlflow 2.22.5, SQLAlchemy 2.0.43, evidently 0.7.23, torch 2.15.0, torch-geometric 2.8.1.

## Limits

This is a single-node local prototype, not a production bank deployment. Pseudo-account uncertainty, frozen graph coverage, no demonstrated GNN ranking gain, fixed training/validation feedback eligibility, ordered-event requirements and any unmet latency target remain explicit limitations. Production authentication, TLS, HA and a remote backup policy are outside this build.
