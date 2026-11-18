"""Generate an experiment report from measured artifacts, never invented metrics."""
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def main():
    audit=read('reports/data_audit.json')
    cxr=read('reports/classifier_evaluation.json')
    ct=read('reports/segmentation_evaluation.json')
    integration=read('reports/integration_results.json')
    lines=['# Development experiment report — 7 November 2026','',
           'For education only, not clinical use. Results below describe local development runs, not official challenge benchmarks or clinical validation.','',
           '## Data and hardware','',
           '- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GB VRAM.',
           '- Python 3.12 environment: PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, MONAI 1.5.2.',
           '- Original datasets are preserved. SHA-256 fingerprints are recorded in reports/archive_verification.json.',
           '- Patient-level partitions were assigned before selecting a small classifier development cohort.','']
    if audit:
        lines += ['| Data | Train | Validation | Test |','|---|---:|---:|---:|']
        for label,counts in [('Full frontal X-rays',audit['classification']['images']),('X-ray patients',audit['classification']['patients']),('Labeled CT cases',audit['segmentation']['splits'])]:
            lines.append(f"| {label} | {counts['train']:,} | {counts['val']:,} | {counts['test']:,} |")
        lines += ['',f"Verified {audit['classification']['sampled_images_verified']} sampled X-ray files and {audit['segmentation']['geometry_matches']} CT/mask geometry pairs. X-ray patient overlap: zero.",'']
    lines += ['## X-ray development experiment','',
              'Training used 2,048 images from 1,946 patients; validation used 512 images from 490 patients; test used 512 images from 478 patients. Missing and uncertain labels were ignored. The DenseNet backbone was fine-tuned, not frozen.','']
    history=read('models/classifier-development/history.json')
    if history:
        best=max(history,key=lambda row:sum(v['auroc'] for v in row['validation'].values() if v['auroc'] is not None)/sum(v['auroc'] is not None for v in row['validation'].values()))
        lines += [f"Completed epochs: {len(history)}. Validation-selected epoch: {best['epoch']}. Recorded training/validation time: {history[-1]['elapsed_seconds']:.1f} seconds.",'']
    if cxr:
        lines += ['Thresholds were selected using maximum Youden J on validation labels only. Test performance:','',
                  '| Finding | Known test labels | Positive | Negative | AUROC | Average precision | Sensitivity | Specificity |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for name,r in cxr['findings'].items():
            if 'auroc' in r:
                lines.append(f"| {name} | {r['n']} | {r['positive']} | {r['n']-r['positive']} | {r['auroc']:.3f} | {r['average_precision']:.3f} | {r['sensitivity']:.3f} | {r['specificity']:.3f} |")
            else:
                lines.append(f"| {name} | {r['n']} | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |")
        lines += ['','Small known-negative counts make some results unstable. Average precision can look high in a positive-heavy labeled subset. These metrics do not establish calibration, subgroup fairness, or generalization. The full label counts, thresholds, confusion matrices and Brier scores are in classifier_evaluation.json.','']
    else:
        lines += ['Held-out evaluation has not completed.','']
    lines += ['## Spleen development experiment','']
    history=read('models/spleen-development/history.json')
    if history:
        best=max(history,key=lambda row:row['mean_dice'])
        lines += [f"Completed epochs: {len(history)}. Best validation epoch: {best['epoch']}. Best mean validation Dice in transformed space: {best['mean_dice']:.4f}.",'',
                  'Training uses 64-cube patches; validation uses full resampled volumes. Native-space test evaluation restores original image geometry.','']
    if ct:
        lines += [f"Native-space held-out mean Dice across {len(ct['cases'])} cases: **{ct['mean_dice']:.4f}**.",'',
                  '| Case | Native-space Dice | Inference seconds | Geometry matches |','|---|---:|---:|---|']
        for i,r in enumerate(ct['cases'],1):
            lines.append(f"| Test case {i} | {r['dice']:.4f} | {r['seconds']:.1f} | {r['geometry_match']} |")
        lines += ['','This is an early baseline. Low Dice indicates substantial segmentation error; do not use these masks as reliable anatomy annotations. More training and validation-led development are required.','']
    else:
        lines += ['Native-space held-out evaluation has not completed.','']
    lines += ['## Real model application checks','']
    if integration:
        lines += ['| Workflow | Prediction | Overlay HTTP | Review HTTP | Seconds |','|---|---|---:|---:|---:|']
        for r in integration:
            lines.append(f"| {r['kind']} | {r['prediction']} | {r['overlay_http_status']} | {r['review_http_status']} | {r['inference_seconds']:.1f} |")
        lines += ['','The automated reviewer explicitly rejected each output as a software test. This is not radiologist validation.','']
    else:
        lines += ['Real-checkpoint API integration checks have not completed.','']
    lines += ['## Remaining work before final submission','',
              '- Expand classifier training and validation beyond the small development cohort. Preserve final-test isolation for subsequent model selection.',
              '- Continue and improve spleen training using validation results, then perform a properly scoped final evaluation.',
              '- Add confidence intervals, calibration and subgroup/failure analysis where metadata supports them.',
              '- Verify Docker/Orthanc end to end; assemble complete CT DICOM series with geometry checks.',
              '- Add authenticated reviewers before shared deployment. Current application is local and identities are self-declared.',
              '- Record demo, complete presentation/reflection and configure authorized repository/submission destinations.','']
    Path('reports/EXPERIMENT_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Generated reports/EXPERIMENT_REPORT.md from measured artifacts.')


if __name__=='__main__':
    main()
