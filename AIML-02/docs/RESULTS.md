# Verified results — 8 November 2026

For education only, not clinical use. Full measured tables, intervals, calibration and failures: reports/FINAL_REPORT.md (also delivered as outputs/submission/FINAL_REPORT.md). JSON evidence remains under reports; no invented radiologist validation or clinical approval.

## Selected models

Classifier: five improved training epochs on 8,192 training images; 2,048 validation and 2,048 fresh patient-test images. Candidate mean AUROC 0.879829 versus baseline 0.871035 on the same enlarged validation cohort. Test AUROC: Atelectasis 0.830671, Cardiomegaly 0.891392, Consolidation 0.9305, Edema 0.872157, Pleural Effusion 0.936246. Counts and 500 patient-bootstrap intervals are in classifier_final_evaluation.json. Seven known Atelectasis negatives make its interval wide. Validation-only calibration applies to four findings, not Atelectasis. Calibration does not improve every metric. Recorded-sex and age support is explicitly reported, with sparse groups withheld.

CT: eight-epoch baseline mean native Dice 0.062899; 35-epoch 64-cube warm start followed by 40-epoch 96-cube context experiment. Validation selects epoch 38. Native mean Dice 0.715528, case-bootstrap 95% interval [0.628165,0.794051], seven cases, all original shapes/affines matched. These cases were exposed during baseline development. No further optimization used these results. Per-case failures, volume error and precision/recall are retained. Grad-CAM occlusion does not consistently outperform random occlusion; cardiomegaly is an explicit counterexample.

Commands actually executed (project .venv-ml Python):

- python -m ml.improved_cohort (frozen enlarged cohort)
- python -m ml.train_classifier --manifest data/splits/classifier_improved.csv --epochs 5 --batch-size 4 --output models/classifier-improved (balanced known-label weighting enabled in the recorded run)
- python -m ml.select_classifier (same-cohort validation comparison; exact run configuration in reports/classifier_selection.json)
- python -m ml.strong_evaluation (validation calibration, fresh patient test and bootstrap)
- python -m ml.evaluate_segmentation (context checkpoint; source-space seven-case evaluation)
- python -m ml.explanation_check; python -m tools.promote_models

Use each module --help and recorded run.json/evaluation JSON for full arguments; these lines identify actual executed modules, not a claim that every optional default reproduces the saved experiment. Evaluated baseline checkpoints remain under models/releases/20261107.

## Software and integration verification

Final: CUDA_VISIBLE_DEVICES empty; .venv-ml/Scripts/python -m pytest -q --basetemp=data/test-final49 --tb=short: **49 passed**, four non-failing warnings, 20.95 seconds. Evidence reports/test-results-final.txt. Frontend npm run build passed, including authenticated controls and backend calibration notes. Tests cover role enforcement, spoofed identities, Origin checks, rate limits, expiry/logout, ZIP bounds, CT ordering/geometry/HU values, incomplete series, evaluation helpers and existing workflows.

Pinned model-enabled Docker images were rebuilt. python -m tools.check_model_containers passed with isolated DATA_DIR=/app/data/verification-final and the actual active fingerprints: classifier f503f1b02d13 and CT 469a4bae39c6. Real radiograph inference took approximately 1.072 seconds CPU; CT 16.135 seconds CPU. Overlay returned 200; versioned review 201. Stable complete synthetic Orthanc CT automatically imported and inferred; incomplete ZIP returned 422. Nginx configuration passed. These are individual local timings, not performance guarantees or a security audit. reports/model_container_results.json records details; baseline container evidence remains separately preserved. Project containers are stopped with volumes retained, while native authenticated API and viewer are available locally.

DVC cache now holds original archives and manifests. dvc get . data/splits/classifier_improved.csv --out data/reproduction-check/classifier_improved.csv restored an identical SHA-256 file; archive cache object sizes match pointers. reports/reproduction_check.json records this. No remote backup, remote publication or full dvc repro run is claimed.

## Deliverables and review

The editable 12-slide presentation has ten-minute speaker notes, measured figures and actual validation images. Structural/import/font/layout checks and visual review were performed; native PowerPoint playback was not tested. The full-source workbook is regenerated and checked against inventory text. Source and model bundles are separate and exclude credentials/runtime databases/raw datasets.

Lumen_Live_Review.mp4 records 60 actual sequential browser captures at approximately two captures per second with original timing (35.233-second H.264 output). It shows finding selection, overlays, CT navigation and an explicitly labeled software-test rejection. Lumen_Demo.mp4 is a separate 84-second captioned still-capture walkthrough, not a continuous recording. The code guide, model card and reflection are delivered.

Remaining limits: no clinical validity, independent radiologist review, production remote deployment, MRI/enhanced CT support, or off-machine DVC backup.

## Final artifact checks

Presentation finalization passed package integrity, 12-slide geometry, fonts and artifact import; all 12 slide previews were visually reviewed. Scan images are embedded as PNG bytes, and the editable chart has an embedded workbook. Final walkthrough render completed: 84 seconds, 1920 × 1080, H.264, 30 fps, no audio; encoded frames at 6, 30, 54 and 78 seconds were sampled. The 35.233-second live recording also passed duration/codec checks. Phone-sized fine UI text is not intended for detailed inspection. Neither video supplies narration; the presentation notes supply the suggested ten-minute talk.

python -m tools.code_inventory records 81 source/config files and 136 functions/classes. The final workbook contains six sheets and 5,045 source chunks. python -m tools.verify_workbook compares every chunk to the actual inventory. python -m tools.package_source and python -m tools.package_submission assemble explicit files; source/model/submission ZIP integrity checks pass. The final package contains 29 allowlisted files including integrity hashes; no credentials, dataset archives or runtime databases are included. The source ZIP contains 108 code/documentation/lock/DVC files, with model weights only in the separate bundle.
