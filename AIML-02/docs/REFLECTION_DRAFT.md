# Personal Reflection — Atul Kumar

I chose a medical imaging project covering chest X-ray classification and abdominal CT spleen segmentation because I wanted to understand how deep learning is applied in a real clinical-adjacent workflow, not just in a Jupyter notebook.

## What I Built

The project has two ML pipelines: a DenseNet-121 classifier for five chest X-ray findings (Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion) and a MONAI U-Net for 3D spleen segmentation from CT scans. These are wrapped in a full-stack application with a React/TypeScript viewer, FastAPI backend, SQLite persistence, authenticated reviewer workflow, and Docker deployment with Orthanc DICOM server integration.

## What I Learned

The most important lesson was understanding that a working UI and a good model are separate things. Early in development, the spleen segmentation had a very low Dice score (0.063) despite the application correctly displaying overlays. This taught me that visual output can be misleading without quantitative evaluation.

Patient-separated data splits were critical — I learned that if the same patient appears in both training and test sets, the metrics are meaningless. The CheXpert dataset required careful handling of uncertain labels, which I masked rather than treating as negative findings.

Training on my RTX 3050 Ti (4 GB VRAM) with 16 GB RAM was a constraint that forced me to use small batch sizes and efficient training strategies. The CT segmentation required a staged approach: first training on 64-cube patches, then fine-tuning on 96-cube patches with context.

## Biggest Challenge

Managing the full complexity of DICOM data was the hardest part. CT scans required validating slice ordering, coordinate geometry, spacing, orientation, and Hounsfield unit rescaling. A visually plausible overlay on incorrectly oriented geometry would not be valid, so I had to implement thorough geometric validation before reconstruction.

## Results I'm Proud Of

- Classifier mean AUROC of 0.892 across five findings, with Pleural Effusion reaching 0.936
- CT spleen Dice improved from 0.063 (baseline) to 0.716 after the staged training approach
- 49 passing tests covering API endpoints, authentication, DICOM handling, and ML evaluation
- Honest reporting including bootstrap confidence intervals and subgroup analysis

## What I Would Do Differently

- Start with a simpler single-task application instead of combining X-ray and CT from the beginning
- Allocate more time for proper cross-validation rather than single train/val/test splits
- Use cloud GPU instances for training rather than constraining everything to my laptop
- Add CI/CD pipeline from the start for automated testing on every commit
