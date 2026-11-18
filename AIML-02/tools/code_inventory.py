"""Build a source snapshot and explanations for the user's editable review workbook."""
import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTIONS = {
    'backend/auth.py': ('Account security','Verifies salted password hashes, manages expiring sessions and enforces account roles/origins.','Private account-file path, secure cookies and allowed origins. Never embed credentials.'),
    'backend/dicom_series.py': ('CT DICOM assembly','Validates complete classic CT stacks, sorts patient coordinates and reconstructs a geometry-preserving volume.','Supported geometry and explicit limits. Preserve count, identity and coordinate checks.'),
    'ml/strong_evaluation.py': ('Final classifier evaluation','Fits calibration on validation only, scores fresh test patients and estimates patient-bootstrap uncertainty.','Evaluation policy before examining final test results. Never fit calibration on test labels.'),
    'ml/select_classifier.py': ('Model selection','Compares baseline and candidate using the same enlarged validation cohort.','Candidate paths. Preserve validation-only selection.'),
    'ml/explanation_check.py': ('Explanation check','Measures score changes after hottest-region versus random-region occlusion on positive validation examples.','Occluded fraction and sample count before measuring; report distribution-shift limits.'),
    'ml/improved_cohort.py': ('Larger cohort','Freezes an enlarged classifier experiment with prior exposed test patients excluded from the new test cohort.','Training inclusion policy before experiments. Preserve existing patient partition membership.'),
    'compose.ml.yaml': ('Model containers','Enables CPU model dependencies, read-only weights/accounts and authenticated importer.','Model paths and local verification DATA_DIR; never add plaintext credentials.'),
    'backend/app.py': ('Application API', 'Validates uploads, persists study state, invokes models, serves images and records versioned reviews.', 'Routes, upload limits and local storage location. Preserve SQL parameterization and review guards.'),
    'backend/volume_io.py': ('CT image handling', 'Bounds decompression, validates NIfTI geometry, canonicalizes CT volumes and draws axial images/masks.', 'Display window and upload bounds. Keep image/mask display transforms identical.'),
    'backend/orthanc_import.py': ('DICOM integration', 'Polls Orthanc and forwards DX/CR or explicitly complete stable CT series with persistent import state.', 'Polling interval, server URL and retry policy. Preserve expected-count and geometry checks.'),
    'ml/classifier.py': ('X-ray architecture', 'Defines the five label outputs, DenseNet121 and shared resize/normalization.', 'Architecture and transforms. Changing label order requires retraining and compatible checkpoints.'),
    'ml/train_classifier.py': ('X-ray training', 'Checks patient separation, masks unknown labels, trains and saves best/last checkpoints using validation AUROC.', 'Epochs, batch size, frozen backbone and resume. Never select models using test scores.'),
    'ml/inference.py': ('X-ray inference', 'Loads trained weights, computes sigmoid scores and one Grad-CAM overlay per finding.', 'Heatmap colors and serving device. Keep preprocessing consistent with training.'),
    'ml/prepare_data.py': ('Dataset preparation', 'Reads original CSV/NIfTI layout and creates deterministic patient-separated split manifests.', 'Paths and split proportions before experiments. Do not change splits after evaluating test data.'),
    'ml/development_subset.py': ('Development cohort', 'Selects reproducible image subsets inside already assigned patient partitions.', 'Development image counts. A small development run is not a full-dataset benchmark.'),
    'ml/extract_data.py': ('Archive extraction', 'Checks archive paths, preserves originals, skips resource forks and records extraction checksums.', 'Destination folders. Do not allow traversal, overwrite original archives or accept links from TAR.'),
    'ml/audit_data.py': ('Data verification', 'Reports split counts, label distributions, sample image validity and CT-mask geometry.', 'Sample verification count. Geometry and patient overlap failures must remain visible.'),
    'ml/evaluate_classifier.py': ('Held-out evaluation', 'Chooses thresholds on validation then calculates test AUROC, AP, Brier and confusion counts.', 'Report output location and threshold policy before final evaluation.'),
    'ml/segmentation.py': ('CT architecture/inference', 'Shares MONAI preprocessing, runs sliding-window prediction and restores mask geometry.', 'Patch and network sizes with matching training. Preserve inverse spatial transforms.'),
    'ml/train_segmentation.py': ('CT training', 'Trains 3D U-Net on patches and evaluates full validation volumes; saves best and resumable checkpoints.', 'Epochs and optimizer. Resumed augmentation restarts at seed 42 and is not bit-identical.'),
    'frontend/src/main.tsx': ('React interface', 'Controls study selection, upload, X-ray explanations, CT slices, reviews and polling.', 'Display text, layout and controls. Keep synthetic labels and educational notice visible.'),
    'frontend/src/style.css': ('Visual design', 'Defines desktop/mobile layout, scan canvas, typography, controls and focus states.', 'Colors, spacing and responsive breakpoints. Preserve keyboard focus visibility.'),
    'frontend/vite.config.ts': ('Development proxy', 'Forwards browser /api requests to localhost:8000.', 'Backend port when changing local services.'),
    'compose.yaml': ('Container services', 'Defines local API, viewer, Orthanc and importer containers.', 'Port mapping, image tags and model mounts. Keep host ports bound to loopback.'),
    'dvc.yaml': ('Reproducible stages', 'Declares data/code dependencies and model outputs for both training stages.', 'Paths and commands when experiment layout changes. DVC initialization is separate.'),
}
FUNCTIONS = {
    'now': 'Return the current timestamp in UTC using an ISO-8601 string.',
    'lifespan': 'Initialize and migrate database storage when the FastAPI process starts.',
    'health': 'Report local service health and whether each model checkpoint is configured.',
    'studies': 'Return study details in reverse creation order for the review queue.',
    'get_study': 'Expose the study lookup through the detail API route.',
    'read_image': 'Decode raster or supported DICOM input, apply intensity conversion and validate pixel dimensions.',
    'Review': 'Validate reviewer, approve/reject decision, bounded comment and expected model version.',
    'volume_slice': 'Check CT type and slice bounds, then serve an image or completed-mask slice.',
    '_analyze': 'Execute the selected trained model and persist either a result or a recoverable failure.',
    'transform': 'Compose shared 224-pixel resizing and ImageNet normalization, with training-only rotation.',
    'Images': 'PyTorch dataset for loading selected images, target values and known-label masks.',
    '__init__': 'Filter records to one partition and choose training or evaluation preprocessing.',
    '__len__': 'Return the number of selected image records.',
    '__getitem__': 'Read one image and return its tensor, targets and mask for known labels.',
    'client': 'Provide an isolated test API with temporary storage and no classifier checkpoint.',
    'png': 'Create a synthetic PNG in memory for upload tests.',
    'volume_bytes': 'Create a compressed synthetic NIfTI with known shape and orientation for tests.',
    'safe_target': 'Reject absolute paths, drive names and parent traversal before joining archive destinations.',
    'extract': 'Hash archive, check free space, copy regular files safely, then write completion metadata.',
    'normalize_label': 'Convert numeric CSV labels such as 1.0 to 1 while preserving missing/uncertain states.',
    'partition': 'Shuffle sorted unique patients with seed 42 and assign disjoint train/val/test groups.',
    'chexpert': 'Keep frontal radiographs, resolve original paths and write normalized labels and split membership.',
    'spleen': 'Pair each CT with its mask and assign case-based partitions.',
    'read_manifest': 'Reject missing patient IDs, invalid labels and patients appearing in multiple partitions.',
    'evaluate': 'Collect predictions without gradients and compute label-specific validation AUROC/AP.',
    'classify': 'Compute DenseNet scores and gradients of each target logit with respect to convolution features.',
    'segment': 'Run tiled 3D inference, invert preprocessing and save a binary mask using source geometry.',
    'preprocessing': 'Load volumes, set channel dimension, orient RAS, resample and scale CT intensities.',
    'validate_volume': 'Bound gzip expansion, check shape/affine/pixels and return canonical data without text metadata.',
    'slice_png': 'Create a CT or mask axial slice using the same orientation and physical aspect ratio.',
    'initialize': 'Create/migrate SQLite tables, add immutable audit triggers and recover interrupted jobs.',
    'db': 'Open a foreign-key-enabled database transaction and close the connection afterward.',
    'study': 'Read one study, decode prediction JSON and attach its chronological review events.',
    'upload': 'Read a bounded image upload, validate, deduplicate and schedule inference only when configured.',
    'upload_volume': 'Validate a CT upload, deduplicate, save canonical volume and schedule segmentation.',
    'review': 'Require a completed output and matching model version before appending a reviewer event.',
    'analyze': 'Choose the classifier or segmenter, persist success, and record recoverable failure.',
    'retry': 'Atomically transition failed/waiting studies to processing when the correct checkpoint exists.',
    'demo': 'Create a clearly marked illustration and invented scores solely for interface testing.',
    'asset': 'Serve only allowlisted files belonging to an existing study.',
    'predict': 'Collect scores, targets and known-label masks for one partition with gradients disabled.',
    'subset': 'Sample within original splits without moving patients between partitions.',
}


def main():
    paths=[]
    for folder in ('backend','ml','frontend/src','tests','tools'):
        for directory, dirs, names in os.walk(ROOT/folder):
            dirs[:] = [d for d in dirs if d not in ('node_modules','__pycache__')]
            for name in names:
                p=Path(directory)/name
                if p.suffix in ('.py','.tsx','.css','.mjs','.cjs'):
                    paths.append(p)
    for name in ('compose.yaml','dvc.yaml','pytest.ini','scripts/start-local.ps1','.gitignore','.dockerignore','.env.example','frontend/package.json','frontend/vite.config.ts','frontend/tsconfig.json','frontend/index.html','frontend/nginx.conf','frontend/Dockerfile','backend/Dockerfile','backend/requirements.txt','ml/requirements.txt'):
        paths.append(ROOT/name)
    paths.extend((ROOT/'scripts').glob('*.ps1'))
    paths.append(ROOT/'compose.ml.yaml')
    video=ROOT/'videos/lumen-demo'
    if video.exists():
        paths.extend(video.glob('*.html'))
        paths.extend((video/'compositions').glob('*.html'))
        paths.extend(video/name for name in ('package.json','hyperframes.json'))
    files,functions,source=[],[],[]
    for p in sorted(set(paths)):
        relative=p.relative_to(ROOT).as_posix(); text=p.read_text(encoding='utf-8-sig'); lines=text.splitlines()
        category,purpose,change=DESCRIPTIONS.get(relative,('Tests' if relative.startswith('tests/') else 'Project support',
            'Executable regression coverage.' if relative.startswith('tests/') else 'Configuration or reproducible project support.',
            'Edit the original source file and rerun the relevant verification.'))
        files.append([relative,category,len(lines),purpose,change,hashlib.sha256(p.read_bytes()).hexdigest(),''])
        for number,line in enumerate(lines,1):
            # Split exceptionally long source lines into numbered chunks without dropping characters.
            chunks=[line[i:i+180] for i in range(0,len(line),180)] or ['']
            for chunk,content in enumerate(chunks,1):
                source.append([relative,number,chunk,content,''])
        if p.suffix=='.py':
            tree=ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                    description=FUNCTIONS.get(node.name) or ast.get_docstring(node) or ('Regression test: '+node.name.removeprefix('test_').replace('_',' ') if node.name.startswith('test_') else purpose)
                    functions.append([relative,node.name,node.lineno,node.end_lineno,description,''])
    out=ROOT/'outputs/code-review-20261107'; out.mkdir(parents=True,exist_ok=True)
    functions.extend(json.loads(subprocess.check_output(['node',str(ROOT/'tools/ts_inventory.cjs')],cwd=ROOT,text=True)))
    (out/'inventory.json').write_text(json.dumps({'files':files,'functions':functions,'source':source},indent=2),encoding='utf-8')
    print(f'{len(files)} files, {len(functions)} functions/classes, {len(source)} source rows')


if __name__=='__main__':
    main()
