# Real-time financial fraud detection

A local, measured prototype: Kafka transaction replay, XGBoost scoring, a training-only
GraphSAGE account graph, Feast/Redis, MLflow versioning, a Streamlit analyst desk,
and Evidently-triggered challenger evaluation. See [results](docs/results.md) for
actual measurements and limitations; this is not a production banking service.

## Windows setup

Use Python 3.11 and Docker Desktop in Linux-container mode. Download the IEEE-CIS
`train_transaction.csv` and `train_identity.csv` from Kaggle into `data/raw/`.
Data, generated models, feature stores and reports are ignored by Git.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r docker/requirements-runtime.txt
.\.venv\Scripts\python.exe -m pip install pytest==8.4.2 ruff==0.17.3
# CPU graph dependencies; the graph Docker image uses these versions too.
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install torch-geometric==2.5.3

docker compose up -d
.\.venv\Scripts\python.exe scripts/health_check.py
```

The measured host graph run used the versions in its evidence report; repeated
training on a different PyTorch version may vary slightly. Scoring uses the
saved XGBoost bundle and frozen account scores, so it does not require PyTorch.
The current tested runtime versions are pinned in `docker/requirements-runtime.txt`.

Host Redis is **6380**, Postgres **5433**, Kafka **9092**, MLflow **5000**.
These host ports avoid existing Windows services; internal container ports remain
6379 and 5432. Development credentials are in config.yaml and can be overridden
by environment variables with the same names. Run modules from the project root.

## First model on a fresh clone

The historical baseline MLflow run is specific to the original local database.
On a fresh database, reproduce it and log the actual results rather than using
the earlier script containing copied baseline metrics.

```powershell
# Optional SHAP analysis for the original baseline trainer:
.\.venv\Scripts\python.exe -m pip install shap
.\.venv\Scripts\python.exe -X utf8 -m src.training.train_baseline
# Obtain that actual baseline run ID from MLflow, then pass it here:
.\.venv\Scripts\python.exe -X utf8 -m src.training.train_with_features --baseline-run-id YOUR_RUN_ID --device cuda
.\.venv\Scripts\python.exe -X utf8 -m scripts.materialize_features
.\.venv\Scripts\python.exe -X utf8 -m scripts.verify_feature_parity
.\.venv\Scripts\python.exe -X utf8 -m src.training.matrices
.\.venv\Scripts\python.exe -X utf8 -m src.graph.train
```

Use `--device cpu` if CUDA-enabled XGBoost is unavailable. CPU full-data training
can be slow on this laptop. Only validation is used for early stopping/selection.
The graph job registers its calibrated, evaluated bundle as `fraud-detector@champion`
after its feature snapshots are materialized. Validation may select zero GNN weight.

If continuing this existing checkout, the evaluated model, snapshots and champion
already exist: do not redo the baseline or training to start the application.
`src.training.registry` can bootstrap an XGBoost-only champion from the saved Phase 2
model; `scripts.bootstrap_runtime` can initialize a fresh deployment from saved artifacts.

## Run the application

```powershell
# All application services, with persistent Redis/Postgres/MLflow volumes:
docker compose --profile app up -d
# Send transactions through the full workflow:
.\.venv\Scripts\python.exe -X utf8 -m src.streaming.producer --limit 1000 --speed 1000
```

Open the dashboard at http://127.0.0.1:8501 and API docs at http://127.0.0.1:8000/docs.
The API's direct `/score` route returns predictions; only the Kafka path publishes
results to the analyst workflow. `--speed 0` replays as fast as possible.
`--drift` multiplies transaction amounts by ten for a controlled shift. Replaying
modified payloads under an existing TransactionID is deliberately rejected; use a
fresh demo namespace/topic set or the verification scripts for isolated demonstrations.

For host development, start the API, worker, sink and dashboard in separate terminals:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m uvicorn src.serving.app:app --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe -X utf8 -m src.streaming.worker
.\.venv\Scripts\python.exe -X utf8 -m src.streaming.sink
.\.venv\Scripts\python.exe -X utf8 -m streamlit run src/dashboard/app.py --server.address 127.0.0.1
```

Do not run host and container API/dashboard processes on the same ports together.

## Verify and demonstrate

```powershell
$env:FRAUD_TEST_REDIS='1'
.\.venv\Scripts\python.exe -X utf8 -m pytest tests/ -q --junitxml=data/processed/integration/tests.xml
.\.venv\Scripts\python.exe -m ruff check src/ scripts/ tests/ feature_repo
.\.venv\Scripts\python.exe -X utf8 -m scripts.verify_pipeline --count 1000
.\.venv\Scripts\python.exe -X utf8 -m scripts.load_test --count 1000
.\.venv\Scripts\python.exe -X utf8 -m scripts.verify_online_parity
.\.venv\Scripts\python.exe -X utf8 -m scripts.demo_drift --device cuda
.\.venv\Scripts\python.exe -X utf8 -m src.monitoring.run --retrain --device cuda
```

The integration script uses isolated topics and reserved IDs, verifies one duplicate
and one invalid message, and records database/feedback/latency evidence. The drift
demo is injected data, not a claim that production drift occurred. It trains a real
challenger and preserves the champion unless the validation gate passes. Demo labels
are excluded from training. Detailed commands and expected screens: [demo guide](docs/demo.md).

## Minikube

Install Minikube from its official site, or use the verified project-local
`data/tools/minikube.exe` already downloaded in this checkout.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/k8s_up.ps1
# Equivalent when GNU Make is available: make k8s-up
kubectl --context fraud-demo -n fraud get pods,jobs,cronjobs
kubectl --context fraud-demo -n fraud port-forward service/api 8001:8000
# In another terminal:
kubectl --context fraud-demo -n fraud port-forward service/dashboard 8502:8501
```

The script builds/loads images, copies the local dataset and evaluated artifacts into
an isolated Docker-driver node, applies the manifests, then waits for bootstrap and
application readiness. It does not retrain the baseline during deployment. The first
image load can be slow. Stop unused host/Compose app processes on a 16 GB machine.
Persistent volumes are local to the `fraud-demo` node, not a remote backup.
Kafka/ZooKeeper storage is ephemeral in this demo; broker replacement requires
recreating topics and replaying source transactions. Database, Redis and model
artifacts have persistent volumes.
The hourly `drift-retraining` CronJob generates drift reports and waits for its
dependencies on startup. Full CPU retraining exhausted the 4 GiB node during the
November 1 resume, so this scheduled command omits `--retrain`. Run challenger
training separately in a configured host/GPU environment; the validation promotion
gate remains in place. Do not resume the suspended `drift-resume-20261101` Job
beside the interactive services on this node.
To stop while retaining data, use `minikube stop -p fraud-demo` (or the project-local
executable). Deleting the namespace or PVCs can delete the cluster database and
artifacts; the host-side source artifacts are retained separately.

## Documentation

- [Current demonstration: your next steps](docs/next_steps.md)
- [Using the application and handing it over](docs/use_and_handover.md)
- [Architecture and guarantees](docs/architecture.md)
- [Measured results and source artifacts](docs/results.md)
- [Feature rules and Feast reproduction](docs/features.md)
- [Decisions](docs/decisions.md)
- [Demo guide](docs/demo.md)
- [Ten-minute presentation outline](docs/presentation.md)
- [Dataset dictionary](docs/data_dictionary.md)

Key folders: `src/serving`, `src/streaming`, `src/graph`, `src/dashboard`,
`src/monitoring`, `src/training`, `feature_repo`, `k8s`, `tests`, `scripts`.
