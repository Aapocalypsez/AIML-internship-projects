"""Assemble an explicit local delivery allowlist; exclude runtime and private files."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/submission'


def digest(file):
    with file.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def archive(destination, sources):
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for source, name in sources:
            assert source.is_file(), str(source)
            assert not any(part in {'node_modules', '.env', 'auth', 'runtime'} for part in Path(name).parts)
            bundle.write(source, name)
    with zipfile.ZipFile(destination) as bundle:
        assert bundle.testzip() is None
        assert set(bundle.namelist()) == {name for _, name in sources}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    copies = {
        'outputs/code-review-20261107/Lumen_Code_Reference.xlsx': 'Lumen_Code_Reference.xlsx',
        'outputs/code-review-20261107/Lumen_Source_Code.zip': 'Lumen_Source_Code.zip',
        'docs/REFLECTION_DRAFT.md': 'REFLECTION_DRAFT.md',
        'docs/MODEL_CARD.md': 'MODEL_CARD.md',
        'docs/CODE_WALKTHROUGH.md': 'CODE_WALKTHROUGH.md',
        'reports/classifier_roc.png': 'Classifier_ROC.png',
        'reports/classifier_reliability.png': 'Classifier_Calibration.png',
        'reports/final_learning_curves.png': 'Learning_Curves.png',
        'reports/segmentation_improved_predictions.png': 'CT_Case_Review.png',
    }
    for source, target in copies.items():
        shutil.copyfile(ROOT / source, OUT / target)
    evidence = [
        'final_summary.json', 'active_models.json', 'classifier_selection.json',
        'classifier_final_evaluation.json', 'segmentation_improved_evaluation.json',
        'explanation_check.json', 'model_container_results.json',
        'reproduction_check.json', 'live_recording_result.json',
        'demo_recording_result.json', 'test-results-final.txt',
    ]
    (OUT / 'evidence').mkdir(exist_ok=True)
    for name in evidence:
        shutil.copyfile(ROOT / 'reports' / name, OUT / 'evidence' / name)
    archive(OUT / 'Lumen_Model_Checkpoints.zip', [
        (ROOT / 'models/active/classifier.pt', 'models/active/classifier.pt'),
        (ROOT / 'models/active/spleen.pt', 'models/active/spleen.pt'),
        (ROOT / 'docs/MODEL_CARD.md', 'MODEL_CARD.md'),
        (ROOT / 'reports/active_models.json', 'active_models.json'),
    ])
    (OUT / 'README_SUBMISSION.md').write_text('''# Lumen educational submission

Start with FINAL_REPORT.md, Lumen_Final_Presentation.pptx and Presentation_Speaker_Notes.md (ten-minute suggested talk).
Lumen_Live_Review.mp4 contains actual sequential UI captures, approximately two source frames per second, preserving capture timing. Lumen_Demo.mp4 is a separate silent captioned still-capture walkthrough. Neither represents clinical review.

Inspect CODE_WALKTHROUGH.md and the six-sheet Lumen_Code_Reference.xlsx. Source is in Lumen_Source_Code.zip. Extract source to a working folder, then extract the separate model bundle into that same folder so weights land in models/active. Follow README.md for Windows GPU or Docker CPU setup. Create local accounts using the launcher; credentials from this computer are deliberately excluded. Dataset archives are not bundled. Training additionally requires the two authorized datasets and the documented manifests.

Evidence contains actual measured JSON reports and the final 49-test log. SHA256SUMS.json checks delivery-file integrity. Raw datasets, dependency folders, runtime databases and credentials are excluded. Video capture assets are excluded from the source ZIP; finished videos are supplied separately. The source workbook is a snapshot, not executable code synchronization: edit source files and regenerate it.

Review REFLECTION_DRAFT.md in your own words. It discloses extensive AI implementation assistance and does not certify independent personal learning. Clinical validity, production security, external publication and off-machine backup are not established. No repository or Drive destination was supplied; this is a local submission package.
''', encoding='utf-8')
    names = list(copies.values()) + [
        'Lumen_Final_Presentation.pptx', 'Presentation_Speaker_Notes.md',
        'FINAL_REPORT.md', 'Measured_Results.html', 'Lumen_Live_Review.mp4',
        'Lumen_Demo.mp4', 'Lumen_Model_Checkpoints.zip', 'README_SUBMISSION.md',
    ] + ['evidence/' + name for name in evidence]
    sources = [(OUT / name, name) for name in sorted(names)]
    assert len(names) == len(set(names))
    checksums = {name: {'sha256': digest(file), 'bytes': file.stat().st_size} for file, name in sources}
    (OUT / 'SHA256SUMS.json').write_text(json.dumps(checksums, indent=2), encoding='utf-8')
    sources.append((OUT / 'SHA256SUMS.json', 'SHA256SUMS.json'))
    archive(ROOT / 'outputs/Lumen_Submission_Package.zip', sources)
    print(f'Validated {len(sources)} allowlisted deliverables; package CRC and file-set checks passed.')


if __name__ == '__main__':
    main()
