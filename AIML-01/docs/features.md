# Phase 2: features and Feast

## Reproduce on Windows (Python 3.11)

From the repository root, with the virtual environment installed and Docker running:

```powershell
docker compose up -d
.\.venv\Scripts\python.exe scripts/health_check.py
$env:FRAUD_TEST_REDIS = '1'
.\.venv\Scripts\python.exe -m pytest tests/ -q
.\.venv\Scripts\python.exe -m src.training.train_with_features --device cuda
.\.venv\Scripts\python.exe -m scripts.materialize_features
.\.venv\Scripts\python.exe -m scripts.verify_feature_parity
```

The existing baseline run is the default comparison source; use `--baseline-run-id`
to specify another existing run. Training requires MLflow and the raw IEEE-CIS files.
Use `--device cpu` (the default) on machines without a CUDA-enabled XGBoost build.
The default test suite requires neither the raw dataset nor Docker; the Redis
integration test is enabled by `FRAUD_TEST_REDIS=1` and fails if Redis is unavailable.

## Feature contract

`src/features/engine.py` contains both the batch calculations and the incremental
`OnlineFeatureEngine.process(transaction)` path intended for Phase 3.
There are **18** engineered features (the handoff's count of 19 was a counting error).

- History is per `pseudo_user_id`, strictly earlier than the current event time.
- Velocity windows include the start: `[time - window, time)`.
- Equal-time rows cannot see one another, including for amount, recency and changes.
  Once time advances, their arrival order determines the last device/email observed.
  Preserve that same order in replay and serving.
- Z-scores use sample standard deviation (`ddof=1`), zero when fewer than two past
  observations exist or historical variance is zero. Means/maxima use all past rows.
- New accounts have zero counts/sums/changes/z-score, amount ratios of 1, and
  `seconds_since_last_txn=-1`. Missing device/email values do not count as changes.
- Cyclical time features use the loader's calendar fields, or derive them from
  `TransactionDT` and the existing reference date `2017-12-01` when absent.
- The current amount is allowed in its own ratios/log/z-score; it is excluded
  from the history used to calculate their denominators.
- Incremental processing validates amounts/timestamps and rejects per-account
  out-of-order events. State is in memory; durable recovery, deduplication and
  late-event policy belong to Phase 3. Feed unlabeled transactions only; labels
  are never consulted by this engine.

Batch windows use binary searches and prefix sums instead of scanning all earlier
rows per transaction. Expanding variance uses Welford's numerically stable update.
Tests compare batch and incremental calculations with each other and with
brute-force expected history at exact boundaries.

## Feast's role

Feast stores named feature values with entity IDs and event timestamps. A
point-in-time join returns a historical snapshot available at the requested time.
It does not make a feature calculation leakage-safe automatically: our engine
must still exclude future and simultaneous observations.

`transaction_features` is keyed by integer `TransactionID`, not only by account.
For example, `log_amount` and `amount_zscore` depend on the transaction being
scored; two transactions for the same account at the same timestamp can have
different values. An account-only snapshot would overwrite one of them.
Account history remains keyed by `pseudo_user_id` inside the engine. This view
does not replace the future account-level GNN score view or durable rolling state.

The offline store is a Parquet file (`feature_repo/data/transaction_features.parquet`).
The local registry is `feature_repo/data/registry.db`. The online store is the
project's Docker Redis 7.2 on **localhost:6380**; port 6379 already serves a separate
Windows Redis 3.0 instance on this machine. Both generated Feast files are ignored
by Git. Training exports snapshots and actually retrieves the engineered columns
through Feast before fitting; it verifies that every retrieved row matches batch.

Materialization copies snapshots using the dataset's real 2017–2018 dates, not
today's date, then verifies 5,000 sampled Redis lookups. This is a historical replay
store: never use it to look up a new transaction as if it were live account history.
The live scorer must compute fresh values before updating state and may write a
new immutable snapshot afterward. Feast's offline TTL is 365 days; Redis key expiry
is not enabled so historical demonstration lookups remain reproducible.
Writes use 1,000-row batches: Feast's default single batch caused memory pressure
when materializing the full dataset alongside training on the 16 GB laptop.
Bulk Redis verification reads use 500-row batches to avoid intermittent socket
write timeouts on large pipelines through Docker Desktop. Each row is still checked.

## Model experiment and evidence

The trainer adds the 18 columns to the original 400 numeric columns, keeps the
original time split and baseline model settings, and limits CPU threads to four.
The measured run uses CUDA on the RTX 3050 Ti after the initial CPU fit proved
slow. CUDA and CPU histograms may differ numerically; the device is logged with
the experiment. Raw 400-column frames are released before fitting to reduce memory.
Historical features carry earlier unlabeled activity across split boundaries.
Early stopping observes validation only. Test predictions are evaluated after
fitting; the code neither tunes on test nor promotes a model alias.

PR-AUC here means scikit-learn average precision: it measures how well fraud
cases rank while balancing precision and recall. The reference score for random
ranking is the fraud rate, making it more informative here than plain accuracy.

Outputs live in `data/processed/features/`:

- `results.json`: measured metrics and deltas, baseline/new run IDs, split ID hashes,
  raw input hashes, feature engine hash, versions, parameters and feature order.
- `model.ubj`, validation/test prediction Parquet files, precision-recall plots.
- `materialization.json`: actual materialization dates and Redis readback evidence.
- `parity.json`: 6,000 real transactions replayed through the incremental path,
  compared to saved batch snapshots and Redis values (`rtol=atol=1e-8`).

MLflow receives the actual model, signature, metrics, plots and result JSON.
The new experiment uses HTTP artifact serving (`mlflow-artifacts:/...`) because
the old baseline experiment points at a container-local `/mlflow/artifacts` path.
The existing baseline run was populated by the previous metrics logging script
and remains marked RUNNING; this phase reads it without rewriting its history.
Its metrics are a comparison reference, not a newly reproduced baseline model.

The model and prediction files are retained locally even if MLflow logging fails.
The original Phase 2 MLflow container had no artifact volume. The later deployment
work backed up and migrated those artifacts and Redis state into persistent Docker
volumes; Kubernetes uses its own PVCs. Host-side model/results are retained too.
The CLI configures UTF-8 stdout because MLflow's Unicode run links otherwise
raise an encoding error on Windows redirected output.

## References

- [Feast point-in-time joins](https://docs.feast.dev/getting-started/concepts/point-in-time-joins)
- [Feast Redis configuration](https://docs.feast.dev/reference/online-stores/redis)
- [MLflow XGBoost integration](https://mlflow.org/docs/latest/ml/traditional-ml/xgboost)

Implementation APIs were also checked against the installed Feast 0.67.0 and
MLflow 2.22.5 packages. Core experiment dependencies are pinned in `pyproject.toml`.
scikit-learn is pinned to 1.6.1 because the previously installed 1.9.1 made
XGBoost 2.1.4 fail during model saving. A fitted-model save/load test protects
this boundary; see the [XGBoost 2.1.4 compatibility notes](https://xgboost.readthedocs.io/en/release_3.0.0/changes/v2.1.0.html).
