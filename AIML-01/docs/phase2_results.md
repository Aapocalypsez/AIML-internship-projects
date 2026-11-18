# Phase 2 measured results - 2026-10-29

Historical Phase 2 report. Later deployment fixes and measurements are recorded in
[current results](results.md), including persistent storage added after this phase.

MLflow run: [312571adee16448887f2999d349340aa](http://localhost:5000/#/experiments/2/runs/312571adee16448887f2999d349340aa) (FINISHED).
Baseline reference: `fcb25615e762440292d6fa9a5b49f04b`. This is the pre-existing logged baseline, not a newly reproduced fit.

| PR-AUC (average precision) | Baseline | Engineered | Absolute delta |
|---|---:|---:|---:|
| val | 0.5204838048 | 0.5231492972 | +0.0026654924 |
| test | 0.4736862235 | 0.4743013992 | +0.0006151756 |

This is a small PR-AUC gain. ROC-AUC declined slightly relative to the supplied baseline; no significance claim or model promotion is made.

| Engineered model metric | Validation | Test |
|---|---:|---:|
| roc_auc | 0.8974002233 | 0.8739740254 |
| precision_at_top_1% | 0.8914285714 | 0.8843212237 |
| recall_at_top_1% | 0.2288508557 | 0.2596855699 |
| random_baseline_pr_auc | 0.0389197625 | 0.0340414958 |

## Evidence

- 32 tests passed with Redis enabled; two existing pandas FutureWarnings.
- 590,540 historical Feast rows verified against batch before fitting.
- 590,540 snapshots materialized; 5,000 sampled Redis rows matched exactly.
- 6,000 real transactions replayed across 18 features; max absolute difference 5.684341886080801e-13 (rtol=atol=1e-8).
- 418 model features; best iteration 164; device cuda.
- Split rows: train 380,815, validation 105,088, test 104,637.
- Uploaded model reloaded and matched the local model on 5 stored example rows.
- PR-AUC independently recomputed from both saved prediction files and matched local JSON and MLflow metrics.
- New/changed Phase 2 modules pass Ruff; six existing lint findings remain in baseline/data-loader/test-data files.
- Core pinned environment passes `python -m pip check`.

## Source artifacts

All numbers above are generated from `data/processed/features/results.json`, `parity.json`, `materialization.json`, `model_verification.json` and `tests.xml`. The result JSON records the raw input hashes, split ID hashes, engine hash, dependency versions, feature order and model parameters. Predictions and model are saved in the same folder. Raw data/generated outputs remain ignored by Git.

## Reproduction commands

```powershell
$env:FRAUD_TEST_REDIS = '1'
.\.venv\Scripts\python.exe -m pytest tests/ -q --junitxml=data/processed/features/tests.xml
.\.venv\Scripts\python.exe -m src.training.train_with_features --device cuda
.\.venv\Scripts\python.exe -m scripts.materialize_features
.\.venv\Scripts\python.exe -m scripts.verify_feature_parity
```

## Limits and next step

Incremental account state is in memory and expects ordered delivery. Durable state, duplicate handling, late events and Kafka/FastAPI serving are Phase 3. MLflow container artifacts have no persistent volume; host model/results are retained. The baseline was logged by the earlier script and is still RUNNING; its status/history was preserved. CUDA differs numerically from the baseline CPU path. Phase 2 stops here; no champion alias was set.

Feature rules, implementation decisions and service port changes are documented in [features.md](features.md) and [decisions.md](decisions.md).
