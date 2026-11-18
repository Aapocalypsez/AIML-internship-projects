# Data and training plan

Both user-supplied archives are now extracted locally. `reports/archive_verification.json` records their SHA-256 hashes, and `reports/data_audit.json` records split counts and geometry checks. Official sources:

- CheXpert: https://aimi.stanford.edu/datasets/chexpert-chest-x-rays
- CheXpert methods/labels: https://stanfordmlgroup.github.io/competitions/chexpert/
- Medical Segmentation Decathlon, Task09_Spleen: https://medicaldecathlon.com/dataaws/

The user must review and accept any dataset access agreement. Do not redistribute dataset files or put them in the submission folder. The supplied lower-resolution CheXpert archive is preserved under `data/raw/chexpert/`, with CSVs and images together under `data/raw/chexpert/extracted/`. The spleen archive is under `data/raw/spleen/`, and its extracted dataset is `data/raw/spleen/extracted/Task09_Spleen/`.

## Prepare manifests

```powershell
.\.venv-ml\Scripts\python -m ml.prepare_data chexpert --csv data/raw/chexpert/extracted/train.csv --root data/raw/chexpert/extracted --output data/splits/classifier.csv
.\.venv-ml\Scripts\python -m ml.prepare_data spleen --root data/raw/spleen/extracted/Task09_Spleen --output data/splits/spleen.json
```

Adjust paths to the actual download. The splitter uses seed 42, sorted unique patient IDs and 70/15/15 development partitions. CheXpert includes frontal views only. This internal test split is not the official CheXpert benchmark test set. Spleen case names are treated as patient identifiers; confirm that assumption against dataset metadata before final experiments. Tiny manifests (<10 patients) are rejected. Paths are absolute for local execution; regenerate manifests when moving hosts and record the new DVC version.

Classification CSV columns: `path,patient_id,split,Atelectasis,Cardiomegaly,Consolidation,Edema,Pleural Effusion`. Values: 0 negative, 1 positive, -1 uncertain, blank missing. Both uncertain and missing labels are masked out of the baseline loss and validation metrics. This is an explicit baseline policy, not the paper's best-performing per-label strategy.

Spleen JSON is a list of `{image,label,patient_id,split}` records, with paired NIfTI files. Never put patients in multiple partitions. Image and mask transforms share spatial operations; mask interpolation is nearest-neighbor.

## Execute after ML environment setup

```powershell
.\.venv-ml\Scripts\python -m ml.train_classifier --manifest data/splits/classifier.csv --epochs 5 --batch-size 4
.\.venv-ml\Scripts\python -m ml.train_segmentation --manifest data/splits/spleen.json --epochs 50
```

DenseNet121 starts with ImageNet weights and uses five sigmoid outputs, masked BCE and AdamW. Validation mean AUROC selects `best.pt`; histories include per-label AUROC and average precision. No final test evaluation is performed during training. Seed 42 is recorded; full GPU determinism is not guaranteed. Serving uses the same resize/normalization and produces Grad-CAM from final convolutional features. Outputs are uncalibrated. JPEG-trained performance on windowed DICOM images requires separate domain-shift validation.

MONAI uses a 3D residual U-Net, CT window -57 to 164, RAS orientation, 1.5 × 1.5 × 2 mm spacing, 64-cube training patches and sliding-window full-volume validation. Full validation volumes remain on CPU while inference patches use the device. Training validation Dice is measured in transformed space. Offline evaluation restores native geometry and checks mask shape/affine before computing test Dice. CT viewer integration accepts canonicalized NIfTI volumes; complete CT DICOM-series ingestion is still outstanding.

Verified local hardware: NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4096 MiB VRAM and 15.4 GiB system RAM. Available RAM was about 2.9 GiB during this session, so large in-memory CT caching was avoided. Initial runs and timings are recorded under `models/*-development` and `reports`. Development-cohort metrics are not full CheXpert benchmark results.

## DVC

Local Git and DVC are initialized. Original archives and split manifests were initially tracked without caching, then committed to the local DVC cache. The enlarged cohort was recovered into separate storage using `dvc get`; SHA-256 matched the original frozen manifest. Both archive cache objects are present with matching sizes. Evidence: `reports/reproduction_check.json`. No remote is configured and nothing is uploaded. Preserve the originals and cache; off-machine disaster recovery still requires an authorized backup. `dvc.yaml` declares full-data training stages; measured experiments were executed directly and are not described as completed `dvc repro` stages.

The initial classifier run used `data/splits/classifier_development.csv`; its baseline metrics remain in `reports/classifier_evaluation.json`. The larger experiment uses `classifier_improved.csv` with8192/2048/2048 images, excluding all478 exposed baseline test patients from the fresh test. The baseline and candidate were compared on the same enlarged validation cohort before final evaluation. Results: `reports/classifier_final_evaluation.json`, including confidence intervals, calibration, confusion counts and subgroup support. CT results: `reports/segmentation_improved_evaluation.json`. All evaluation reports contain checkpoint/manifest fingerprints.

The current classifier fresh test has now been evaluated; future model development needs another untouched patient-disjoint final cohort. CT re-evaluation uses the seven previously examined cases and is explicitly developmental. `scripts/train-baselines.ps1` continues historical baselines and requires their `last.pt` files; it requests50 total spleen epochs. Use new output directories and preserve evaluated checkpoints. Never run GPU training and serving concurrently on this4GB GPU.
