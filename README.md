# 🎓 AIML Internship Projects — Atul Kumar

> Two end-to-end AI/ML projects built during my internship: a **real-time fraud detection platform** and an **AI-assisted medical imaging system**.

---

## 📁 Repository Structure

```
├── AIML-01/    → Real-Time Financial Fraud Detection Platform
└── AIML-02/    → Lumen: Medical Imaging and Diagnosis System
```

---

## 🔍 AIML-01: Real-Time Financial Fraud Detection Platform

An end-to-end ML platform that ingests financial transactions via **Kafka**, engineers leakage-safe features, scores them in real-time, and surfaces predictions through an interactive **Streamlit** dashboard.

### Architecture

```
Kafka Producer → Kafka Broker → Streaming Worker → FastAPI Scorer → Postgres
                                       ↓                 ↓
                                  Feast (Redis)      Streamlit Dashboard
                                       ↓
                                  XGBoost + GNN → MLflow Registry
                                       ↓
                                  Evidently Drift Monitor
```

### Key Results

| Metric | Value |
|--------|-------|
| **Test PR-AUC** | 0.4743 (13.4× lift over random) |
| **Scoring Latency (p99)** | 72 ms |
| **Tests** | 44 passed, 2 skipped |
| **Dataset** | IEEE-CIS Fraud Detection (590K transactions) |

### Tech Stack

`Python` · `Apache Kafka` · `Redis` · `PostgreSQL` · `XGBoost` · `PyTorch Geometric (GraphSAGE)` · `Feast` · `MLflow` · `FastAPI` · `Streamlit` · `Evidently AI` · `Docker Compose` · `Minikube`

### Features

- ⚡ Real-time transaction scoring via Kafka streaming pipeline
- 🏪 Feast feature store with online/offline parity (max error: 5.68e-13)
- 📊 Streamlit dashboard with live fraud alerts and analyst queue
- 🔄 Automated drift detection and model retraining pipeline
- 🧪 GraphSAGE GNN experiment (honestly documented — no ranking gain)
- ☸️ Kubernetes deployment on Minikube with 9 services

→ **[Full Setup & Documentation](AIML-01/README.md)**

---

## 🏥 AIML-02: Lumen — Medical Imaging and Diagnosis System

A full-stack application for **chest X-ray classification** and **CT spleen segmentation** with an authenticated reviewer workflow, Grad-CAM explanations, and DICOM support.

### Architecture

```
React Viewer → FastAPI Backend → SQLite (Studies + Reviews)
                    ↓
            Background Inference
              ↙           ↘
  DenseNet-121          MONAI U-Net
  (Chest X-ray)        (CT Spleen)
  + Grad-CAM           + 3D Masks
                    ↓
            Orthanc DICOM Server
```

### Key Results

| Finding | AUROC | 95% Bootstrap CI |
|---------|-------|------------------|
| Atelectasis | 0.831 | 0.609 – 0.983 |
| Cardiomegaly | 0.891 | 0.852 – 0.929 |
| Consolidation | **0.930** | 0.904 – 0.955 |
| Edema | 0.872 | 0.838 – 0.901 |
| Pleural Effusion | **0.936** | 0.919 – 0.951 |

| CT Segmentation | Value |
|-----------------|-------|
| **Mean Dice** | 0.716 (improved from 0.063 baseline) |
| **95% CI** | 0.628 – 0.794 |
| **Tests** | 49 passed |

### Tech Stack

`Python` · `React` · `TypeScript` · `FastAPI` · `PyTorch` · `MONAI` · `DenseNet-121` · `U-Net` · `Grad-CAM` · `SQLite` · `Docker` · `Orthanc DICOM` · `DVC`

### Features

- 🩻 Multi-modal support: PNG, JPEG, DICOM (DX/CR), NIfTI CT, DICOM CT series
- 🔬 Per-finding Grad-CAM heatmap explanations
- 📐 Validation-fitted Platt calibration for probability estimates
- 👤 Authenticated reviewer workflow with append-only audit trail
- 📊 Subgroup analysis by recorded sex and age
- 🐳 Pinned CPU Docker containers with Orthanc integration

> ⚠️ **For education only, not clinical use.**

→ **[Full Setup & Documentation](AIML-02/README.md)**

---

## 🚀 Quick Start

Both projects require **Docker Desktop** and local infrastructure. See each project's README for detailed setup:

```bash
# AIML-01: Fraud Detection
cd AIML-01
docker compose --profile app up -d
# Dashboard: http://localhost:8501 | API: http://localhost:8000/docs

# AIML-02: Medical Imaging
cd AIML-02
.\scripts\start-local.ps1    # Terminal 1
cd frontend && npm run dev    # Terminal 2
# Viewer: http://localhost:5173 | API: http://localhost:8000/docs
```

## 📋 Documentation

| Document | AIML-01 | AIML-02 |
|----------|---------|---------|
| Architecture | [architecture.md](AIML-01/docs/architecture.md) | [ARCHITECTURE.md](AIML-02/docs/ARCHITECTURE.md) |
| Results | [results.md](AIML-01/docs/results.md) | [RESULTS.md](AIML-02/docs/RESULTS.md) |
| Model Details | [decisions.md](AIML-01/docs/decisions.md) | [MODEL_CARD.md](AIML-02/docs/MODEL_CARD.md) |
| Reflection | [reflection.md](AIML-01/docs/reflection.md) | [REFLECTION_DRAFT.md](AIML-02/docs/REFLECTION_DRAFT.md) |

## 🛠️ Development Environment

- **OS**: Windows 11
- **GPU**: NVIDIA RTX 3050 Ti (4 GB VRAM)
- **RAM**: 16 GB
- **Python**: 3.11 (AIML-01) / 3.12 (AIML-02)

## 📄 License

Educational project — not for production or clinical use.
