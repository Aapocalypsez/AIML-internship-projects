# Decisions Log

This file tracks architectural and implementation decisions for the fraud detection project.
Use it to explain *why* during the presentation.

## 2026-10-26

### D001: Python 3.11 (not 3.14)
- **Context**: Machine has Python 3.14.5, but Feast, PyTorch Geometric, and other ML libraries don't support 3.14 yet.
- **Decision**: Install Python 3.11 as a separate version and use it for this project.
- **Alternatives**: Use 3.12 (some libs have limited support), wait for 3.14 support (not feasible with deadline).

### D002: Docker Compose first, Minikube later
- **Context**: Need Kafka, Redis, Postgres, MLflow running locally.
- **Decision**: Docker Compose for development (Phase 0-6), Minikube only in Phase 7.
- **Rationale**: Simpler iteration loop. Minikube adds complexity we don't need until the system works.

### D003: Single Kafka broker
- **Context**: Production would use 3+ brokers. We're on a laptop with 16GB RAM.
- **Decision**: Single broker, replication factor 1. Sufficient for demonstrating the streaming pattern.

### D004: Confluent Kafka images (v7.5.3)
- **Context**: Multiple Kafka Docker images exist (bitnami, wurstmeister, confluent).
- **Decision**: Confluent (cp-kafka) — most widely used, well-documented, stable health checks.

### D005: MLflow with Postgres backend
- **Context**: MLflow can use SQLite or Postgres for its backend store.
- **Decision**: Postgres — already running for the app, avoids file locking issues, closer to production.

## 2026-10-29 — Phase 2

### D006: Strict event-time history and sample variance
- **Problem**: The initial amount implementation used input order and population
  standard deviation; amount/change/recency calculations could see equal-time rows.
- **Decision**: Every historical input must have `past_time < current_time`.
  Include exact left window boundaries. Use sample standard deviation (`ddof=1`),
  matching the existing z-score test. Hold equal-time online observations pending.
- **Proof**: Original tests, boundary/oracle tests, future-prefix invariance,
  independent batch/incremental parity over 6,000 events, and actual Feast retrieval.

### D007: Transaction-specific Feast feature snapshots
- **Problem**: Amount ratios, z-score and log amount are transaction-specific.
  Account-only keys would collide for simultaneous transactions with different amounts.
- **Decision**: Store this view by `TransactionID`, with all 18 columns defined by
  the shared engine's feature list. Keep incremental history keyed by pseudo-user.
  Add an account-level ring-risk view in Phase 4. No live account-state claim is
  made for historical transaction snapshots.
- **Storage**: Parquet offline, local registry, Redis online. Train via real Feast
  historical retrieval; materialize original dataset dates without rewriting time.

### D008: Port 6380 for the project's Redis
- **Evidence**: Localhost port 6379 reports Redis 3.0.504 on Windows; Docker reports
  Redis 7.2.17. Feast's multi-field HSET fails against the Windows instance.
- **Decision**: Publish Docker Redis on 6380 and update project clients/checks.
  Keep the separate Windows service untouched.

### D009: One measured experiment and explicit baseline provenance
- **Decision**: Use the baseline's fixed XGBoost settings with 18 new features,
  validation-only early stopping, float32 matrices, and four CPU threads for the
  16 GB laptop. Do not tune or promote based on test scores.
- **Evidence**: Hash input files and split IDs; persist predictions/model/metrics;
  compare PR-AUC to the existing baseline MLflow run rather than typing numbers.
  That baseline was logged from prior saved results and is still RUNNING; preserve
  it as received. New runs use HTTP artifact serving instead of container-local URIs.
- **Resource adjustment**: Materialize in 1,000-row batches rather than Feast's
  whole-dataset default. Release raw feature frames before fitting. Use CUDA for
  this measured run on the installed RTX 3050 Ti; keep CPU as the portable default
  and record the device to make its numerical differences explicit.
- **Compatibility fix**: Pin scikit-learn 1.6.1, supported by XGBoost 2.1.4.
  The installed 1.9.1 removed `_estimator_type`, causing model saving to fail after
  fitting. Add a small fitted-model save/load regression test.

## 2026-10-29 — Phases 3–8

### D010: Durable score-before-update state and retry safety
- Redis WATCH commits account history and the TransactionID result cache together.
  Same ID/different payload is rejected; out-of-order account events are rejected.
- Kafka transactions bind scored/alert output to consumed offsets. The database
  sink commits first, acknowledges second, and uses unique transaction IDs.
- Single broker/partition/worker remains an explicit local-development constraint.

### D011: Evidence decides the graph weight
- Train-only heterogeneous entity graph, hub cap 20, shared-weight two-layer
  GraphSAGE with bounded neighbor sampling and weighted binary loss.
- Calibrate on validation's first half and choose the blend on the second half.
  Validation selected XGBoost weight 1.0; forcing 70/30 would reduce PR-AUC.
- Keep graph links in case review. Scheduled inference uses the evaluated
  checkpoint and verifies frozen scores, without resetting a newer champion.

### D012: Explanations during case review
- Per-transaction SHAP missed the latency target. Compute explanations lazily
  using the exact case model version. Keep the actual initial and later benchmark
  reports; quote their different scopes and background loads explicitly.

### D013: Drift triggers evaluation, not automatic deployment
- Evidently monitors four numeric columns. A drifted share >= 0.25 triggers a
  challenger, with a per-window fingerprint/cooldown.
- Require validation PR-AUC gain > 0.001 and protect promotion with a Redis lock
  plus incumbent-version check. Demo feedback is excluded from training.
- The real controlled-drift challenger was rejected; no fake improvement or
  synthetic fraud label was introduced to make the demonstration succeed.

### D014: Preserve state through service upgrades
- Backed up Redis data, MLflow artifacts and Postgres before migration.
- Redis uses append-only persistence and a named volume; MLflow artifacts have
  a named volume; Postgres keeps its existing volume. Kubernetes uses separate PVCs.
- MLflow server is 2.22.5. SQLAlchemy 2.1 removed an API it needs, so pin 2.0.43.
  The old MLflow database was upgraded with its supported migration command.
- Force host Redis to 127.0.0.1 after localhost/IPv6 connections stalled through
  Docker Desktop. Container clients use the service DNS name instead.

### D015: Minikube compatibility and measured scope
- An isolated `fraud-demo` profile uses Docker, two CPUs and a 4 GiB limit.
- Disable automatic service environment variables: Kubernetes' KAFKA_PORT
  otherwise conflicts with Confluent's startup script.
- Stream seed data with tar through docker exec; docker cp could not resolve the
  node's mounted data path. Keep deployment and bootstrap operations reproducible.
- Report internal Service HTTP latency separately from Windows port-forward
  measurements. The single-client local benchmark is not a throughput/HA claim.

## 2026-11-01 — Resume and resource recovery

### D016: Wait for dependencies and separate full training from the demo node
- The resumed monitoring Job exhausted its retry limit before its dependencies
  were ready. Add an init container that waits for Postgres, Redis, and the
  registered champion. Regression tests exercise recovery from each dependency
  being unavailable on the first attempt.
- Rerunning full CPU training pushed the 4 GiB node to 3.972 GiB and made the
  control plane/dashboard unresponsive. Stop that attempt, restart the node with
  its persistent data, and suspend `drift-resume-20261101`.
- Keep the hourly schedule active for drift reports, omitting `--retrain` on this
  node. Full training remains available separately with the existing validation
  gate. This is an explicit resource limit, not a completed challenger result.
- The fresh report-only Job completed. Current startup checks and counts are in
  `data/processed/integration/resume_20261101.json` and `restart_tests.xml`.

### D017: Preserve unique replay columns for all-missing categorical fields
- A one-row CSV slice inferred missing device/email columns as numeric, so the
  producer selected them twice and failed JSON serialization. Append categorical
  columns only if they were not already selected. A small source-CSV regression
  test covers null device/email values and confirms the label is excluded.
