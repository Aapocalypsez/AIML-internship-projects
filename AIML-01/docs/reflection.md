# Individual Reflection — Atul Kumar

## What I Built
I built a **Real-Time Financial Fraud Detection Platform** — an end-to-end ML
system that ingests financial transactions via Kafka, engineers leakage-safe
features, scores them with an XGBoost + GNN ensemble, and surfaces predictions
through a Streamlit dashboard, all orchestrated with Docker Compose and
deployable on Minikube.

## What I Owned
- Designed the overall 8-phase build plan and project architecture
- Implemented the repository skeleton, Docker Compose infra, data
  loading, EDA notebook, and baseline XGBoost trainer (PR-AUC 0.52)
- Debugged dependency conflicts (numpy/feast/evidently), MLflow container
  crashes, and Windows-specific issues (PowerShell syntax, port collisions)
- Built the feature engineering pipeline with Feast, Kafka streaming pipeline,
  FastAPI scorer, GraphSAGE GNN experiment, Streamlit dashboard, Evidently
  drift monitoring, and Minikube deployment
- Authored documentation: README, architecture diagram, decision log, results,
  and operating guide
- Ran and verified all 44 tests (2 skipped — Redis-dependent)

## Biggest Challenge
Managing the complexity of so many moving parts (Kafka, Redis, Postgres,
MLflow, Feast, FastAPI, Streamlit) on a single 16 GB Windows laptop was the
hardest part. Debugging container health-check failures and dependency version
conflicts taught me that infrastructure work is often harder than the ML itself.

## What I Learned
- How to build a production-style ML pipeline, not just a Jupyter notebook
- Feature stores (Feast) ensure training/serving parity and prevent data leakage
- Honest evaluation matters — the GNN showed no ranking gain, and I documented
  that instead of cherry-picking numbers
- Docker Compose and Kubernetes orchestration for multi-service applications
- MLflow for experiment tracking and model versioning

## What I Would Do Differently
- Start with a simpler streaming setup and add Kafka later, to reduce early
  debugging time
- Invest more time in graph feature engineering before concluding the GNN
  doesn't help
- Add CI/CD (GitHub Actions) from Day 1 instead of running tests manually
- Use a cloud VM for heavier workloads instead of squeezing everything onto a
  laptop
