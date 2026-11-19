#  Real-Time Financial Fraud Detection Platform

A local, end-to-end ML platform that ingests financial transactions through Kafka, engineers leakage-safe features, scores them in real time with XGBoost, and surfaces predictions through an interactive Streamlit dashboard. Includes Feast feature serving, MLflow model versioning, Evidently drift monitoring, a training-only GraphSAGE experiment, and Kubernetes deployment on Minikube.

> See [results](docs/results.md) for actual measurements and limitations. This is a research prototype, not a production banking service.

---

##  Key Results

| Metric | Value |
|--------|-------|
| **Test PR-AUC** | 0.4743 (13.4× lift over random) |
| **Scoring Latency (p99)** | 72 ms |
| **Online/Offline Feature Parity** | max error 5.68e-13 |
| **Tests** | 44 passed, 2 skipped (Redis-dependent) |
| **Dataset** | IEEE-CIS Fraud Detection (590K transactions, 3.5% fraud rate) |

The GraphSAGE GNN experiment showed no ranking gain over XGBoost alone. This is documented honestly in the results rather than hidden.

---

##  Architecture

```
Kafka Producer → Kafka Broker → Streaming Worker → FastAPI Scorer → Postgres
                                       ↓                 ↓
                                  Feast (Redis)      Streamlit Dashboard
                                       ↓
                                  XGBoost + GNN → MLflow Registry
                                       ↓
                                  Evidently Drift Monitor
```

**Services:** Kafka, ZooKeeper, Redis, PostgreSQL, MLflow, FastAPI, Streamlit, Evidently — orchestrated via Docker Compose or Minikube (9 services).

---

##  Tech Stack

`Python 3.11` · `Apache Kafka` · `Redis` · `PostgreSQL` · `XGBoost` · `PyTorch Geometric (GraphSAGE)` · `Feast` · `MLflow` · `FastAPI` · `Streamlit` · `Evidently AI` · `Docker Compose` · `Minikube`

---

##  Quick Start (Windows)

**Prerequisites:** Python 3.11, Docker Desktop (Linux-container mode), NVIDIA GPU recommended.

1. **Download the dataset** — get IEEE-CIS `train_transaction.csv` and `train_identity.csv` from Kaggle and place them in `data/raw/`.

2. **Set up the environment:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r docker/requirements-runtime.txt
.\.venv\Scripts\python.exe -m pip install pytest==8.4.2 ruff==0.17.3

# CPU graph dependencies (optional, for GNN experiment)
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install torch-geometric==2.5.3
```

3. **Start infrastructure:**

```powershell
docker compose up -d
.\.venv\Scripts\python.exe scripts/health_check.py
```

Host ports: Redis **6380**, Postgres **5433**, Kafka **9092**, MLflow **5000**. These avoid conflicts with existing Windows services; container-internal ports remain 6379/5432.

---

##  Training Pipeline

```powershell
# Train baseline XGBoost
.\.venv\Scripts\python.exe -X utf8 -m src.training.train_baseline

# Train with Feast features (pass the baseline MLflow run ID)
.\.venv\Scripts\python.exe -X utf8 -m src.training.train_with_features --baseline-run-id YOUR_RUN_ID --device cuda

# Materialize features and validate parity
.\.venv\Scripts\python.exe -X utf8 -m scripts.materialize_features
.\.venv\Scripts\python.exe -X utf8 -m scripts.verify_feature_parity
.\.venv\Scripts\python.exe -X utf8 -m src.training.matrices

# Graph experiment (optional)
.\.venv\Scripts\python.exe -X utf8 -m src.graph.train
```

Use `--device cpu` if CUDA is unavailable. The graph job registers its calibrated bundle as `fraud-detector@champion`. Validation may select zero GNN weight (this happened in practice). If continuing an existing checkout, the trained model and snapshots already exist.

---

##  Running the Application

```powershell
# Start all application services
docker compose --profile app up -d

# Replay transactions through the pipeline
.\.venv\Scripts\python.exe -X utf8 -m src.streaming.producer --limit 1000 --speed 1000
```

- **Dashboard:** http://127.0.0.1:8501
- **API Docs:** http://127.0.0.1:8000/docs
- **MLflow UI:** http://127.0.0.1:5000

The `--speed 0` flag replays as fast as possible. Use `--drift` to multiply amounts by 10× for a controlled distribution shift.

For host-mode development, run services in separate terminals:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m uvicorn src.serving.app:app --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe -X utf8 -m src.streaming.worker
.\.venv\Scripts\python.exe -X utf8 -m src.streaming.sink
.\.venv\Scripts\python.exe -X utf8 -m streamlit run src/dashboard/app.py --server.address 127.0.0.1
```

---

##  Testing & Verification

```powershell
# Full test suite (set FRAUD_TEST_REDIS=1 for Redis integration tests)
$env:FRAUD_TEST_REDIS='1'
.\.venv\Scripts\python.exe -X utf8 -m pytest tests/ -q --junitxml=data/processed/integration/tests.xml
.\.venv\Scripts\python.exe -m ruff check src/ scripts/ tests/ feature_repo

# End-to-end pipeline verification
.\.venv\Scripts\python.exe -X utf8 -m scripts.verify_pipeline --count 1000
.\.venv\Scripts\python.exe -X utf8 -m scripts.load_test --count 1000
.\.venv\Scripts\python.exe -X utf8 -m scripts.verify_online_parity

# Drift detection demo
.\.venv\Scripts\python.exe -X utf8 -m scripts.demo_drift --device cuda
.\.venv\Scripts\python.exe -X utf8 -m src.monitoring.run --retrain --device cuda
```

The integration script uses isolated topics and reserved IDs, verifies duplicate rejection and invalid message handling, and records latency evidence. The drift demo injects modified data — it trains a real challenger model and preserves the champion unless the validation gate passes.

---

##  Minikube Deployment

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/k8s_up.ps1

kubectl --context fraud-demo -n fraud get pods,jobs,cronjobs
kubectl --context fraud-demo -n fraud port-forward service/api 8001:8000
kubectl --context fraud-demo -n fraud port-forward service/dashboard 8502:8501
```

The script builds and loads images, copies the local dataset and model artifacts into an isolated node, and applies the manifests. Stop unused host/Compose processes on a 16 GB machine.

---

##  Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | System design, components, and data flow |
| [Results](docs/results.md) | Measured metrics, artifacts, and limitations |
| [Features](docs/features.md) | Feature engineering rules and Feast setup |
| [Decisions](docs/decisions.md) | Design choices and trade-offs |
| [Demo Guide](docs/demo.md) | Step-by-step demonstration walkthrough |
| [Reflection](docs/reflection.md) | Personal reflections on the project |
| [Presentation](docs/presentation.md) | Ten-minute presentation outline |
| [Data Dictionary](docs/data_dictionary.md) | Dataset column descriptions |

---

##  Project Structure

```
├── src/
│   ├── serving/       # FastAPI scoring endpoint
│   ├── streaming/     # Kafka producer, worker, sink
│   ├── graph/         # GraphSAGE GNN experiment
│   ├── dashboard/     # Streamlit analyst interface
│   ├── monitoring/    # Evidently drift detection
│   └── training/      # XGBoost + feature engineering
├── feature_repo/      # Feast feature definitions
├── k8s/               # Kubernetes manifests
├── tests/             # Pytest test suite
├── scripts/           # Verification and utility scripts
├── docker/            # Dockerfiles and requirements
├── docs/              # Architecture, results, guides
└── notebooks/         # EDA and analysis notebooks
```
