# Lumen — AI-Assisted Medical Imaging and Diagnosis System

**For education only, not clinical use.** A local research application for chest X-ray review and spleen CT segmentation, with authenticated review and complete classic CT DICOM reconstruction. Read `docs/CODE_WALKTHROUGH.md` for the detailed implementation guide.

## Working milestone

- React/TypeScript viewer: upload, study queue, original/overlay view, per-finding selection, opacity, reviewer comments and decision history.
- Python/FastAPI: PNG/JPEG and single-frame monochrome DX/CR DICOM ingestion, content deduplication, persisted SQLite records, append-only review events, prediction-version checks and failed-job recovery.
- Clearly labeled synthetic illustration, scores and overlays for exercising the workflow without a model.
- Real uploads wait for a checkpoint; they never receive synthetic predictions.

Both supplied datasets are extracted, versioned and cached locally. Selected models provide real predictions, with validation-fitted classifier calibration where supported. CT NIfTI and complete classic DICOM ZIP upload, acquired-slice viewing and geometry-preserving masks are implemented. Measured evidence is in `reports/FINAL_REPORT.md`. MRI, enhanced multi-frame CT and clinical validation remain unsupported. Data, credentials and weights are excluded from Git; selected weights are supplied separately from source.

## Run on this Windows computer

Use the GPU-capable `.venv-ml` environment and installed frontend dependencies. Open two PowerShell terminals in this project folder:

```powershell
# Terminal 1
.\scripts\start-local.ps1
```

```powershell
# Terminal 2
cd frontend
npm.cmd run dev
```

Open http://127.0.0.1:5173 and sign in. Initial credentials are saved locally in `data/auth/LOCAL_ACCESS.txt`; never commit or share this file. The launcher creates the account if missing and prefers `models/active` checkpoints. Choose **Load synthetic demo** for explicitly invented interface fixtures, or upload a permitted de-identified scan for real inference. Select a finding, adjust overlays or CT slices, then record a review. Authenticated identity overrides submitted names. API documentation: http://127.0.0.1:8000/docs.

For a fresh Windows checkout, use an installed Python 3.12 interpreter:

```powershell
py -3.12 -m venv .venv-ml
python -m pip --python .venv-ml/Scripts/python.exe install --extra-index-url https://download.pytorch.org/whl/cu128 -r ml/requirements-lock.txt
cd frontend
npm.cmd ci
```

`ml/requirements-lock.txt` records the tested Windows GPU environment: PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128 and MONAI 1.5.2. The command above supplies its required CUDA wheel index. `ml/requirements-serving-lock.txt` records the tested Linux CPU inference image; `backend/requirements-lock.txt` is its lightweight API subset. These are separate environments. On this computer the working `.venv-ml` already exists; the older `.venv` is separate. Restore the separate model bundle into `models/active` for inference, or train using the documented data pipeline.

## Checks

```powershell
New-Item -ItemType Directory -Force data | Out-Null
.\.venv-ml\Scripts\python -m pytest -q --basetemp=data/test-run-local
cd frontend
npm.cmd run build
```

Pytest owns and clears its `--basetemp` directory. Never point it at a dataset or a directory containing files you want to retain. On this host restricted execution blocked local sockets/temp access; tests passed with normal local access.

## Training preparation

See `docs/DATA_AND_MODELS.md`. Use only authorized, de-identified data. Keep data and weights outside Git. After training, set the checkpoint before starting the backend:

```powershell
$env:CLASSIFIER_CHECKPOINT='models/active/classifier.pt'
$env:SEGMENTATION_CHECKPOINT='models/active/spleen.pt'
$env:REVIEW_USERS_FILE='data/auth/users.json'
.\.venv-ml\Scripts\python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

An invalid checkpoint produces a failed job. Use **Retry analysis** after fixing configuration. X-ray serving uses CPU; segmentation uses CUDA when available. Set `INFERENCE_DEVICE=cpu` while training uses the GPU. A single inference lock bounds model execution in one process; there is no durable multi-worker queue. The baseline and selected-source checkpoints remain preserved in separate experiment directories.

## Docker / Orthanc

Start Docker Desktop and stop the native API before using its port8000. Prepare private accounts/service credentials and active model paths, then start the model-enabled stack:

```powershell
.\.venv-ml\Scripts\python -m tools.setup_inference
docker compose -f compose.yaml -f compose.ml.yaml up --build -d
```

Viewer: http://127.0.0.1:8080. Orthanc: http://127.0.0.1:8042 (user `research`, private password in `.env`). Accounts and checkpoints are read-only mounts; runtime storage is separate. Keep `.env` and account files private. The lightweight base Compose configuration remains available for isolated demo/import checks without ML serving.

API/viewer/importer builds, authenticated CPU model inference, overlays, reviews and complete CT import have been exercised. Base-image digests and Python/Node dependency versions are pinned. Orthanc CT import waits for stable, Complete status and a known matching expected count; unknown completeness is deferred. Direct CT ZIP requires a root `manifest.json` such as `{"expected_slices":55}` plus the paired single-frame `.dcm` files. Geometry and intensity rescaling are validated before reconstruction. See `docs/RESULTS.md` for measured commands and limits.

## Documentation

- `docs/ARCHITECTURE.md`: components, API and data storage.
- `docs/DATA_AND_MODELS.md`: dataset preparation, training and evaluation policy.
- `docs/MODEL_CARD.md`: intended use and known limitations.
- `docs/RESULTS.md`: actual checks; no invented accuracy figures.
- `docs/SUBMISSION.md`: remaining deadline deliverables.

The normal launcher and model-enabled containers use account sessions, roles, origin checks and sign-in rate limits. Account identity does not establish medical qualifications. Remote shared use needs HTTPS (`COOKIE_SECURE=true`), explicit allowed origins, account administration and further security/operational review. SQLite triggers do not prevent a database owner from tampering. DICOM/NIfTI text metadata is discarded by the API; burned-in identifiers are not removed, and Orthanc retains originals.

## Submission materials

`outputs/submission` contains the presentation, speaker notes, measured report, captioned real-capture demonstration, code workbook/source archive, separate model bundle and reflection draft. The reflection must be reviewed by the project owner. No GitHub/Drive destination or clinical approval has been fabricated. `docs/SUBMISSION.md` records remaining external actions.
