"""Preserve evaluated sources and atomically activate validation-selected models."""
import hashlib
import json
import shutil
from pathlib import Path
import torch


def main():
    comparison=json.loads(Path('reports/classifier_selection.json').read_text())
    evaluation=json.loads(Path('reports/classifier_final_evaluation.json').read_text())
    source=Path('models/classifier-selected/best.pt')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==evaluation['checkpoint_sha256']
    saved=torch.load(source,map_location='cpu',weights_only=True)
    saved['calibrators']=evaluation['calibrators']
    saved['calibration_source']='Validation-only Platt scaling; coefficients frozen before test scoring'
    saved['uncalibrated_checkpoint_sha256']=evaluation['checkpoint_sha256']
    active=Path('models/active'); active.mkdir(exist_ok=True)
    torch.save(saved,active/'classifier.pending'); (active/'classifier.pending').replace(active/'classifier.pt')
    candidates=[Path('models/spleen-context'),Path('models/spleen-improved'),Path('models/spleen-development')]
    selected=max(candidates,key=lambda p:max(r['mean_dice'] for r in json.loads((p/'history.json').read_text())))
    shutil.copyfile(selected/'best.pt',active/'spleen.pending'); (active/'spleen.pending').replace(active/'spleen.pt')
    record={'classifier_selection':comparison['selected'],'classifier_calibration':'Validation labels only',
            'spleen_selection':str(selected/'best.pt'),'spleen_selection_policy':'Best mean validation Dice across experiments, no test-based selection',
            'active_sha256':{name:hashlib.sha256((active/(name+'.pt')).read_bytes()).hexdigest() for name in ('classifier','spleen')}}
    Path('reports/active_models.json').write_text(json.dumps(record,indent=2)); print(json.dumps(record,indent=2))


if __name__=='__main__': main()
