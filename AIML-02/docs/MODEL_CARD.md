# Model card — selected educational models, 8 November 2026

For education only, not clinical use. No radiologist validation, diagnostic safety or medical approval has been established.

## Active models

DenseNet121 accepts frontal chest radiographs resized to 224 × 224 and produces five finding scores plus Grad-CAM. MONAI 3D U-Net produces a spleen anatomy mask restored to CT source geometry. Active files are models/active/classifier.pt and models/active/spleen.pt. reports/active_models.json records promotion provenance and full SHA-256 fingerprints. Baseline checkpoints and raw selection checkpoints are preserved separately.

## Data and selection

User-supplied CheXpert-small and Decathlon Task09 archives are preserved, fingerprinted and cached locally with DVC. Raw datasets are excluded from delivery archives; this project does not establish redistribution permission.

Improved classifier cohort: 8,192 / 2,048 / 2,048 train/validation/test images and 6,783 / 1,690 / 1,691 patients. Patient partitions are disjoint; all 478 previously exposed baseline test patients are excluded from the new test cohort. Missing and uncertain labels are masked. Training adds known negatives and uses training-only class weights; validation/test retain natural sampling within their original partitions. Five epochs fine-tune the full network. Epoch three wins validation selection. On the same enlarged validation cohort, candidate mean AUROC 0.87983 exceeds baseline 0.87103. Fresh test assessment follows selection.

Spleen case split is 28 / 6 / 7; independent case-to-person linkage is unavailable. After eight baseline epochs, 35 warm-start epochs use 64-cube patches and 40 use 96-cube patches. Validation selects context epoch 38. Shared orientation, spacing and CT intensity preprocessing is inverted after sliding-window inference. The seven CT assessment cases were already examined in baseline development: this is development re-evaluation, not an untouched clinical assessment.

## Measured results

| Finding | Known positive / negative | Fresh-test AUROC |
|---|---:|---:|
| Atelectasis | 313 / 7 | 0.8307 |
| Cardiomegaly | 223 / 82 | 0.8914 |
| Consolidation | 124 / 251 | 0.9305 |
| Edema | 480 / 148 | 0.8722 |
| Pleural Effusion | 722 / 297 | 0.9362 |

These condition on known labels and are not population diagnostic accuracy. Patient-bootstrap intervals use 500 replicates. Atelectasis has only seven negatives and a wide AUROC interval, 0.6088–0.9831. Its validation set has only five negatives, so calibration is skipped. Four other findings use positive-slope Platt scaling fitted on validation only. Calibration does not improve every metric: consolidation ECE worsens slightly. Full intervals, confusion matrices, thresholds, AP, Brier/ECE and recorded-sex/age subgroup counts are in reports/classifier_final_evaluation.json and reports/FINAL_REPORT.md. Sparse subgroup AUROCs remain unavailable; AP is affected by positive-heavy known labels.

Spleen mean Dice is 0.71553 versus baseline 0.06290. The 2,000-replicate case-bootstrap 95% interval is 0.62817–0.79405. Seven case Dice values range from 0.50663 to 0.86337. Every exported mask matches source shape/affine. Per-case precision, recall and volume error remain in reports/segmentation_improved_evaluation.json.

Grad-CAM hottest-20%-region occlusion does not consistently outperform random occlusion on eight positive validation examples per finding. Cardiomegaly is a clear failure example. Occlusion creates distribution shift; this is descriptive evidence, not certified explanation validity. No disease-boundary ground truth validates these heatmaps.

## Scope and limitations

PNG/JPEG and single-frame monochrome DX/CR DICOM feed the classifier. Training uses JPEGs, so DICOM acquisition is an unevaluated domain shift. CT accepts compressed NIfTI and complete classic single-frame DICOM stacks with consistent patient-coordinate geometry. Enhanced multi-frame CT, irregular/tilted stacks, MRI, arbitrary anatomy and disease severity are unsupported. Acquired slices retain oblique geometry; general multiplanar reconstruction is not promised.

Authentication supplies server-controlled account identity and viewer/reviewer/admin roles, not verified medical qualifications. Append-only reviews retain prediction fingerprints; database-file owners can still tamper. Local authenticated CPU containers and GPU inference work. Remote use requires HTTPS, secure cookies and operational review; public deployment and security certification are not established.

Application derivatives remove source filenames/free-text metadata, but burned-in pixel identifiers remain possible and Orthanc retains source DICOM. Use authorized de-identified data. Synthetic fixtures remain visibly separate and are never substituted for real predictions.
