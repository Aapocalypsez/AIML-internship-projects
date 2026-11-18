"""Select baseline or candidate using only the same enlarged validation cohort."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from ml.classifier import build
from ml.train_classifier import read_manifest,Images,evaluate


def main():
    manifest=Path('data/splits/classifier_improved.csv'); rows=read_manifest(manifest)
    torch.set_num_threads(4); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader=DataLoader(Images(rows,'val'),batch_size=16)
    comparisons=[]
    for name in ('models/releases/20261107/classifier.pt','models/classifier-improved/best.pt'):
        saved=torch.load(name,map_location='cpu',weights_only=True); model=build().to(device)
        model.load_state_dict(saved['state_dict']); metrics=evaluate(model,loader,device)
        comparisons.append({'checkpoint':name,'checkpoint_sha256':hashlib.sha256(Path(name).read_bytes()).hexdigest(),
                            'mean_validation_auroc':float(np.mean([m['auroc'] for m in metrics.values() if m['auroc'] is not None])),
                            'validation':metrics})
        del model
    selected=max(comparisons,key=lambda row:row['mean_validation_auroc'])
    saved=torch.load(selected['checkpoint'],map_location='cpu',weights_only=True)
    saved['evaluation_manifest_sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest()
    saved['selection_source_sha256']=selected['checkpoint_sha256']
    output=Path('models/classifier-selected'); output.mkdir(exist_ok=True)
    torch.save(saved,output/'best.pt')
    report={'selection_policy':'Highest mean AUROC on the same improved validation cohort; no test predictions used',
            'candidates':comparisons,'selected':selected['checkpoint'],'selected_source_sha256':selected['checkpoint_sha256'],
            'evaluation_manifest_sha256':saved['evaluation_manifest_sha256']}
    Path('reports/classifier_selection.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))


if __name__=='__main__': main()
