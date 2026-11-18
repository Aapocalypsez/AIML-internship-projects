# Local demonstration guide

This is a recording plan, not evidence that a recording or presentation has been completed. Use the current results report rather than memorizing a promised accuracy number.

## Before recording

Start the API with `scripts/start-local.ps1` and run `npm.cmd run dev` in `frontend`. Open `http://127.0.0.1:5173`. Both development checkpoints should be present. Finish GPU training before running CT inference. Use only authorized de-identified examples. Do not show source filenames, patient metadata, credentials or unrelated desktop windows in the recording.

## Ten-minute presentation structure

| Time | Topic | What to explain or show |
|---|---|---|
| 0:00–1:00 | Problem and scope | Educational chest X-ray classification and spleen CT segmentation; human review remains necessary. |
| 1:00–2:00 | Data | Two separate datasets; images and masks are training material, not downloaded trained models. Explain patient-separated partitions and masked uncertain labels. |
| 2:00–4:00 | Architecture | React viewer → FastAPI → shared model inference → stored results → append-only review history. Show the architecture diagram. Distinguish NIfTI CT support from unfinished CT DICOM-series assembly. |
| 4:00–6:00 | Models and explanations | DenseNet transfer learning, five independent scores, Grad-CAM limits; 3D U-Net patch training, sliding-window inference and geometry inversion. Explain validation-selected checkpoints and development-scale experiments. |
| 6:00–9:00 | Demo and evidence | Show an X-ray upload, its five outputs and heatmaps; show CT slice navigation and an imperfect mask. Record a simulated review. Show reload persistence and measured results, including failures. |
| 9:00–10:00 | Limits and next steps | Explain poor/unstable metrics, uncalibrated scores, lack of authenticated radiologist validation and remaining integration work. |

## Suggested viewer sequence

1. Show the educational notice and supported formats.
2. Use **Load synthetic demo** to explain controls. Explicitly say its image, scores and overlays are invented UI fixtures.
3. Select different findings; turn overlays on/off and change opacity. Enter a test reviewer name and an explicitly simulated comment. Reject the fixture, reload, and show that the event remains in history.
4. Upload a permitted real validation-set radiograph. The saved classifier produces real model outputs; they are not proven diagnoses. Show the model fingerprint and switch explanation overlays.
5. Upload a permitted real validation CT `.nii.gz`. Wait for completion, move through slices and compare the image with the predicted spleen mask. Explain that a poor mask is a model failure, not something to hide.
6. Show `reports/EXPERIMENT_REPORT.md`, the test summary and the all-case segmentation visualization. Do not call development metrics an official challenge result.
7. Open the code workbook's File guide and Source code sheets. Explain how someone can inspect and modify the implementation using the original files.

## Reflection prompts for the project owner

Write these in your own words after reviewing and running the project: which parts you personally implemented or modified; which parts were AI-assisted; how you learned the difference between data and checkpoints; how patient separation and geometry affect valid evaluation; one failure you investigated; and what you would change next. Do not claim clinical review, model ownership or independent implementation that did not occur.
