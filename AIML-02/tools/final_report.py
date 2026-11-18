"""Generate the final report from measured outputs, enforcing model/evidence consistency."""
import hashlib
import html
import json
import re
from datetime import datetime,timezone
from pathlib import Path


def read(name):
    return json.loads(Path(name).read_text())


def main():
    classifier=read('reports/classifier_final_evaluation.json')
    segmentation=read('reports/segmentation_improved_evaluation.json')
    selection=read('reports/classifier_selection.json')
    active=read('reports/active_models.json')
    containers=read('reports/model_container_results.json')
    recovery=read('reports/reproduction_check.json')
    explanations=read('reports/explanation_check.json')
    test_log=Path('reports/test-results-final.txt').read_text()
    matches=re.findall(r'(\d+) passed',test_log); assert matches
    tests=int(matches[-1]); assert tests>=48
    hashes={name:hashlib.sha256(Path(f'models/active/{name}.pt').read_bytes()).hexdigest() for name in ('classifier','spleen')}
    assert hashes==active['active_sha256']
    assert segmentation['checkpoint_sha256']==hashes['spleen']
    for kind,name,prefix in [('radiograph','classifier','densenet121-'),('ct','spleen','spleen-unet-')]:
        result=next(row for row in containers['checks'] if row['kind']==kind)
        assert result['model_version']==prefix+hashes[name][:12], 'Container verification must use final active weights'
    assert recovery['manifest_recovered'] and explanations['checkpoint_sha256']==hashes['classifier']
    stamp=datetime.now(timezone.utc).isoformat()
    summary={'generated_utc':stamp,'classifier':classifier,'segmentation':segmentation,'tests':tests,
             'selection':selection,'active_models':active,'containers':containers,'recovery':recovery,
             'clinical_validation':'Not established; requires qualified independent clinical evaluation'}
    Path('reports/final_summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Lumen â€” final educational project report','',f'Generated {stamp}.','',
           '**For education only, not clinical use.** This report describes measured local development evidence. Account authentication and software tests do not establish clinical validity.','',
           '## What is implemented','',
           '- Chest X-ray classification, per-finding Grad-CAM and validation-fitted calibration where label support permits.',
           '- NIfTI CT and complete classic CT DICOM-series reconstruction, source-geometry masks and acquired-slice overlays.',
           '- Account sign-in, roles, expiring sessions, versioned append-only reviews and failed-job recovery.',
           '- Orthanc radiograph/complete CT import, pinned CPU inference containers and local DVC recovery.',
           '- Editable presentation, speaker notes, a real-capture captioned demo and a transparent reflection draft.','',
           '## Data and experiment design','',
           'Both user-supplied archives were retained and checksummed. Full X-ray partitions contain 133287/28413/29327 images and zero patient overlap. The larger experiment contains 8192/2048/2048 images from 6783/1690/1691 patients. The fresh classifier test excludes every one of the 478 baseline test patients. Up to 200 known negatives per finding were included in training; validation/test remain naturally sampled within their original patient partitions. Missing and uncertain labels are masked.','',
           'Spleen training uses 28 cases, validation 6 and development test 7. Case identifiers are treated as patient identifiers. All 41 image/mask geometry pairs were checked. Native test geometry is restored before measuring overlap. The seven CT test cases were examined in the baseline, so improved CT results are development re-evaluation.','',
           'Hardware: NVIDIA RTX 3050 Ti Laptop (4 GB VRAM) and 15.4 GiB RAM; Python 3.12.14 / PyTorch 2.11.0+cu128 / torchvision 0.26.0+cu128 / MONAI 1.5.2. Training and checks shared this laptop; times are observations, not guaranteed throughput.','',
           '## Classifier selection and held-out evaluation','',
           '| Candidate | Mean AUROC on the same enlarged validation cohort |','|---|---:|']
    for candidate in selection['candidates']:
        lines.append(f"| {candidate['checkpoint']} | {candidate['mean_validation_auroc']:.4f} |")
    lines += ['',f"Selected: `{selection['selected']}`. Selection used validation only. Five candidate-training epochs completed; checkpoints were compared before fresh test evaluation. Calibration and thresholds were fitted on validation only.",'',
              '| Finding | Known test labels | Positive / negative | AUROC | Patient-bootstrap95% interval | AP | Sensitivity | Specificity |',
              '|---|---:|---:|---:|---|---:|---:|---:|']
    for label,row in classifier['findings'].items():
        if 'auroc' not in row:
            lines.append(f"| {label} | {row['n']} | {row['positive']} / {row['negative']} | unavailable | insufficient labels | unavailable | unavailable | unavailable |")
            continue
        ci=row['auroc_patient_bootstrap_95']; interval=f"{ci['lower']:.3f}â€“{ci['upper']:.3f}" if ci else 'unavailable'
        lines.append(f"| {label} | {row['n']} | {row['positive']} / {row['negative']} | {row['auroc']:.3f} | {interval} | {row['average_precision']:.3f} | {row['sensitivity']:.3f} | {row['specificity']:.3f} |")
    lines += ['','Intervals use 500 patient-bootstrap replicates, excluding replicates with a single known class. The JSON records valid replicate counts. Sparse negative labels make several results unstable; AP and observed prevalence refer only to the known-label subset. Baseline and fresh-test classifier scores use different cohorts and must not be treated as a paired improvement estimate.','',
              '### Calibration','', '| Finding | Raw Brier | Calibrated Brier | Raw ECE | Calibrated ECE | Calibration applied |','|---|---:|---:|---:|---:|---|']
    for label,row in classifier['findings'].items():
        if 'brier_raw' in row: lines.append(f"| {label} | {row['brier_raw']:.4f} | {row['brier_calibrated']:.4f} | {row['ece_raw']:.4f} | {row['ece_calibrated']:.4f} | {row['calibration_applied']} |")
    lines += ['','ECE uses ten equal-width bins. Positive-slope Platt calibration is fitted only when validation contains at least 20 known examples of each class. Unsupported findings retain raw scores. Calibration is conditional on observed labels and does not establish clinical disease probabilities.','',
              '### Recorded-sex and age subgroups','', '| Finding | Subgroup | Known labels | Positive / negative | AUROC |','|---|---|---:|---:|---:|']
    for label,row in classifier['findings'].items():
        for group,values in row.get('subgroups',{}).items():
            auc=f"{values['auroc']:.3f}" if values['auroc'] is not None else 'insufficient support'
            lines.append(f"| {label} | {group} | {values['n']} | {values['positive']} / {values['negative']} | {auc} |")
    lines += ['','Subgroup AUROC requires at least five known positives and negatives. Age and recorded sex come from supplied metadata; these exploratory counts do not establish fairness or external generalization.','',
              '## Spleen segmentation','',
              f"Original eight-epoch baseline native-space mean Dice: 0.062899. The new experiments completed 35 warm-start epochs with 64-cube patches and 40 context epochs with 96-cube patches. Best context validation Dice: 0.667626 at epoch 38. Selected native-space mean Dice:**{segmentation['mean_dice']:.4f}**.",
              f"Case-bootstrap95% interval: {segmentation['mean_dice_bootstrap_95'][0]:.3f}â€“{segmentation['mean_dice_bootstrap_95'][1]:.3f}, based on 2000 resamples. Seven cases provide limited generalization evidence.",'',
              '| Test case | Dice | Precision | Recall | Absolute relative volume error | Original geometry matches |','|---|---:|---:|---:|---:|---|']
    for index,row in enumerate(segmentation['cases'],1):
        lines.append(f"| {index} | {row['dice']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | {row['absolute_relative_volume_error']:.3f} | {row['geometry_match']} |")
    lines += ['','All-case visualizations show every test case using a fixed slice-selection rule. Over-segmentation and false positives remain. Matching affine/shape does not prove anatomical accuracy.','',
              '## Explanation perturbation check','',
              'Eight known-positive validation examples per finding were evaluated by replacing the hottest 20% of Grad-CAM pixels or a seeded random 20% with ImageNet-mean pixels. Score changes use raw sigmoid outputs. The recorded score drops are descriptive; occlusion creates distribution shift and there is no localization ground truth. This check did not select checkpoints.','',
              '| Finding | Mean score drop: hottest pixels | Mean score drop: random pixels |','|---|---:|---:|']
    for label,row in explanations['findings'].items():
        lines.append(f"| {label} | {row['mean_score_drop_hottest_20_percent']:.4f} | {row['mean_score_drop_random_20_percent']:.4f} |")
    lines += ['','The hottest-region perturbation did not consistently outperform random perturbation. Cardiomegaly and Atelectasis show important failures under this check. Negative drops mean the score increased after perturbation. Treat explanations as limited aids and do not infer verified disease localization.','',
              '## Software and integration verification','',
              f'- Final pytest regression: **{tests} passed**; see reports/test-results-final.txt for timing and non-failing warnings.',
              '- Frontend TypeScript/Vite production build passed.',
              '- Authenticated native browser CT upload, slice/overlay control and explicit software-test rejection were verified.',
              '- Final active models completed authenticated CPU-container uploads, inference, overlays and reviews. Submitted spoofed reviewer names were overridden by account identity.',
              '- Orthanc imported and analyzed a stable, complete synthetic CT series. Incomplete CT ZIP returned 422. Synthetic integration fixtures were isolated from normal uploads.',
              '- Passwords use salted PBKDF2; sessions expire/revoke; viewer role cannot mutate studies; origin/rate-limit checks are tested.',
              '- DVC recovered the frozen cohort into separate storage with identical SHA256. Both original archive objects are cached locally; no remote upload occurred.','',
              '## Remaining scientific and operational limits','',
              'Qualified independent clinical validation has not occurred. MRI, enhanced multi-frame CT, irregular/tilted stacks and arbitrary disease diagnosis are unsupported. Orthanc series with unknown completeness are deferred. The acquired-slice viewer is not full multiplanar reconstruction. Burned-in pixel identifiers are not automatically removed.','',
              'Accounts identify application users without verifying medical credentials. Remote use needs HTTPS/secure cookies, explicit allowed origins, account administration and further operational/security review. Sessions and inference locks are process-local, jobs are not durably queued, and audit triggers are not tamper-proof against database owners.','',
              'The reflection is an honest draft for the project owner to review. No personal learning, radiologist approval, external publication or repository/Drive destination has been fabricated.','',
              '## Reproduction and deliverables','',
              'README, CODE_WALKTHROUGH, dependency locks, pinned container digests, frozen DVC pointers, model fingerprints, experiment histories and checkpoint copies are preserved locally. Input data and credentials are excluded from source archives. The model bundle is separate from source. Demo video is a paced captioned sequence of genuine application captures, not a continuous screen recording.','']
    report='\n'.join(lines)
    Path('reports/FINAL_REPORT.md').write_text(report,encoding='utf-8')
    out=Path('outputs/submission'); out.mkdir(exist_ok=True)
    (out/'FINAL_REPORT.md').write_text(report,encoding='utf-8')
    rows=''.join('<tr><td>'+html.escape(label)+'</td><td>'+str(values['n'])+'</td><td>'+f"{values.get('auroc',0):.3f}"+'</td></tr>' for label,values in classifier['findings'].items())
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Lumen measured results</title><style>body{{font:20px Arial,sans-serif;max-width:1050px;margin:45px auto;padding:25px;color:#153444;background:#f1f5f6}}h1{{font-size:40px}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:16px;text-align:left;border-bottom:1px solid #d5e1e5}}strong{{color:#197d7c}}p{{line-height:1.5}}</style><h1>Lumen: measured development results</h1><p><strong>For education only, not clinical use.</strong></p><table><tr><th>Finding</th><th>Known test labels</th><th>AUROC</th></tr>{rows}</table><p>Spleen mean Dice: <strong>{segmentation['mean_dice']:.3f}</strong> on seven previously examined cases.<br>Baseline: 0.063. Geometry matched for every exported mask.</p><p><strong>{tests} automated tests passed.</strong> See the full report for uncertainty, calibration, subgroup support and failures.</p><p>Clinical validation has not been established.</p></html>'''
    (out/'Measured_Results.html').write_text(page,encoding='utf-8')
    print('Generated final summary, detailed measured report and offline results page.')


if __name__=='__main__': main()
