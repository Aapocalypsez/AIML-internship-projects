# AIML Internship Projects — Atul Kumar
## AIML-01: Real-Time Financial Fraud Detection Platform
End-to-end ML platform using Kafka streaming, XGBoost (PR-AUC 0.47 on IEEE-CIS),
Feast feature store, FastAPI scorer (p99 ~72ms), Streamlit dashboard, Evidently
drift monitoring, and Minikube deployment.
**Stack**: Python, Kafka, Redis, Postgres, XGBoost, PyTorch Geometric, Feast,
MLflow, FastAPI, Streamlit, Docker Compose, Minikube
**Key Results**: 44 tests passing, 13.4× lift over random, leakage-safe features
→ See [AIML-01/README.md](AIML-01/README.md) for setup and details.
---
## AIML-02: Lumen — AI-Assisted Medical Imaging and Diagnosis System
Full-stack application for chest X-ray classification (DenseNet-121, mean AUROC
0.892) and CT spleen segmentation (MONAI U-Net, Dice 0.716). Features
authenticated reviewer workflow, Grad-CAM explanations, DICOM reconstruction,
and Docker deployment with Orthanc.
**Stack**: Python, React/TypeScript, FastAPI, PyTorch, MONAI, DenseNet-121,
U-Net, SQLite, Docker, Orthanc DICOM
**Key Results**: 49 tests passing, AUROC 0.831–0.936 across 5 findings, Dice
improved from 0.063 to 0.716
→ See [AIML-02/README.md](AIML-02/README.md) for setup and details.
