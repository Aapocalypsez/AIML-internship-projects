#  Lumen — AI-Assisted Medical Imaging and Diagnosis System

A full-stack research application for **chest X-ray classification** and **CT spleen segmentation** with an authenticated reviewer workflow, Grad-CAM explanations, and DICOM support via Orthanc.

>  **For education only, not clinical use.** See [MODEL_CARD.md](docs/MODEL_CARD.md) for intended use and known limitations.

---

##  Key Results

### Chest X-ray Classification (DenseNet-121)

| Finding | AUROC | 95% Bootstrap CI |
|---------|-------|------------------|
| Atelectasis | 0.831 | 0.609 – 0.983 |
| Cardiomegaly | 0.891 | 0.852 – 0.929 |
| Consolidation | **0.930** | 0.904 – 0.955 |
| Edema | 0.872 | 0.838 – 0.901 |
| Pleural Effusion | **0.936** | 0.919 – 0.951 |

**Mean AUROC:** 0.892 across 5 findings, with validation-fitted Platt calibration.

### CT Spleen Segmentation (MONAI U-Net)

| Metric | Value |
|--------|-------|
| **Mean Dice** | 0.716 (improved from 0.063 baseline) |
| **95% CI** | 0.628 – 0.794 |
| **Tests** | 49 passed |

Full measured evidence is in [RESULTS.md](docs/RESULTS.md). No accuracy figures are invented.

---

##  Architecture

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

---

##  Tech Stack

`Python 3.12` · `React` · `TypeScript` · `FastAPI` · `PyTorch` · `MONAI` · `DenseNet-121` · `U-Net` · `Grad-CAM` · `SQLite` · `Docker` · `Orthanc DICOM` · `DVC`

---

##  Features

-  **Multi-modal input:** PNG, JPEG, single-frame DICOM (DX/CR), NIfTI CT volumes, and complete classic CT DICOM ZIP series
-  **Per-finding Grad-CAM** heatmap explanations with adjustable opacity
-  **Platt calibration** — validation-fitted probability estimates with bootstrap confidence intervals
-  **Authenticated reviewer workflow** with append-only audit trail and decision history
-  **Subgroup analysis** by recorded sex and age
-  **Content deduplication** — repeated uploads are detected and linked, not reprocessed
-  **Pinned Docker containers** with Orthanc integration for DICOM management
-  **Synthetic demo mode** — clearly labeled synthetic illustrations for exercising the workflow without model weights

---

##  Quick Start

**Prerequisites:** Python 3.12, Node.js, Docker Desktop (for Orthanc), NVIDIA GPU recommended for segmentation.

### Option 1: Local Development

Open two terminals in the project folder:

```powershell
# Terminal 1 — Backend
.\scripts\start-local.ps1
```

```powershell
# Terminal 2 — Frontend
cd frontend
npm.cmd run dev
```

- **Viewer:** http://127.0.0.1:5173
- **API Docs:** http://127.0.0.1:8000/docs

Sign in with the credentials created on first launch (stored locally in `data/auth/LOCAL_ACCESS.txt`). Choose **Load synthetic demo** for interface fixtures, or upload a de-identified scan for real inference.

### Option 2: Fresh Setup

```powershell
# Create Python environment
py -3.12 -m venv .venv-ml
python -m pip --python .venv-ml/Scripts/python.exe install --extra-index-url https://download.pytorch.org/whl/cu128 -r ml/requirements-lock.txt

# Install frontend dependencies
cd frontend
npm.cmd ci
```

**Dependency versions:** PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, MONAI 1.5.2 — pinned in `ml/requirements-lock.txt`.

---

##  Training & Model Setup

See [DATA_AND_MODELS.md](docs/DATA_AND_MODELS.md) for dataset preparation, training procedures, and evaluation policy. Use only authorized, de-identified data. Keep data and weights outside Git.

After training, configure checkpoints before starting the backend:

```powershell
$env:CLASSIFIER_CHECKPOINT='models/active/classifier.pt'
$env:SEGMENTATION_CHECKPOINT='models/active/spleen.pt'
$env:REVIEW_USERS_FILE='data/auth/users.json'
.\.venv-ml\Scripts\python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

X-ray serving uses CPU; segmentation uses CUDA when available. Set `INFERENCE_DEVICE=cpu` to avoid GPU contention during training.

---

##  Testing

```powershell
New-Item -ItemType Directory -Force data | Out-Null
.\.venv-ml\Scripts\python -m pytest -q --basetemp=data/test-run-local

cd frontend
npm.cmd run build
```

Pytest clears its `--basetemp` directory on each run — never point it at a directory containing files you want to keep.

---

##  Docker / Orthanc Deployment

Start Docker Desktop, then prepare credentials and launch the model-enabled stack:

```powershell
.\.venv-ml\Scripts\python -m tools.setup_inference
docker compose -f compose.yaml -f compose.ml.yaml up --build -d
```

- **Viewer:** http://127.0.0.1:8080
- **Orthanc:** http://127.0.0.1:8042 (user `research`, password in `.env`)

Base-image digests and dependency versions are pinned. The lightweight base Compose configuration is available for demo/import checks without ML serving.

---

##  Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | Components, API, and data storage design |
| [Data & Models](docs/DATA_AND_MODELS.md) | Dataset preparation, training, and evaluation |
| [Model Card](docs/MODEL_CARD.md) | Intended use and known limitations |
| [Results](docs/RESULTS.md) | Measured checks with actual evidence |
| [Code Walkthrough](docs/CODE_WALKTHROUGH.md) | Detailed implementation guide |
| [Reflection](docs/REFLECTION_DRAFT.md) | Personal project reflection |

---

##  Project Structure

```
├── backend/           # FastAPI application and inference engine
├── frontend/          # React/TypeScript viewer with study queue
├── ml/                # Training scripts, configs, and requirements
├── scripts/           # Launcher and utility scripts
├── tests/             # Pytest test suite (49 tests)
├── tools/             # Setup and packaging utilities
├── docs/              # Architecture, results, model card
├── videos/            # Demo recordings
├── compose.yaml       # Base Docker Compose (no ML)
├── compose.ml.yaml    # ML-enabled Docker Compose overlay
├── dvc.yaml           # DVC pipeline definitions
└── pytest.ini         # Test configuration
```

---

##  Security Notes

The application uses account sessions, roles, origin checks, and sign-in rate limits. Account identity does not establish medical qualifications. For remote shared use, additional configuration is required (HTTPS with `COOKIE_SECURE=true`, explicit allowed origins, and further security review). DICOM/NIfTI text metadata is discarded by the API, but burned-in identifiers are not removed.
