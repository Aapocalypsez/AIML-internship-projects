"""MONAI spleen baseline. JSON records: image, label, patient_id, split.

Validation uses full resampled volumes; ml.evaluate_segmentation evaluates native-space masks.
"""
import argparse
import json
import hashlib
import time
from pathlib import Path

import torch
from monai.data import Dataset, DataLoader, PersistentDataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.transforms import Compose, SpatialPadd, RandCropByPosNegLabeld, RandFlipd, EnsureTyped
from monai.utils import set_determinism
from ml.segmentation import build, preprocessing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--output', default='models/spleen')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--initialize', help='Warm-start weights as a new experiment; optimizer and history restart')
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--cache-dir', help='Private on-disk cache of deterministic preprocessing')
    parser.add_argument('--roi-size',type=int,default=64)
    args = parser.parse_args()
    if args.epochs < 1 or args.learning_rate <= 0 or args.roi_size not in (64,96) or (args.resume and args.initialize):
        parser.error('Positive epochs/rate required; resume and initialize are mutually exclusive')
    records = json.loads(Path(args.manifest).read_text())
    patients = {}
    for r in records:
        if r['split'] not in ('train', 'val', 'test') or not r['patient_id']:
            raise ValueError('Invalid split or patient identifier')
        if patients.setdefault(r['patient_id'], r['split']) != r['split']:
            raise ValueError('Patient leakage between splits')
    if not all(any(r['split']==s for r in records) for s in ('train','val','test')):
        raise ValueError('Train, validation and held-out test partitions are required')
    set_determinism(seed=42)
    torch.set_num_threads(4)
    keys = ['image', 'label']
    base = preprocessing(keys) + [EnsureTyped(keys, track_meta=False)]
    roi=(args.roi_size,)*3
    train = Compose(base + [SpatialPadd(keys, spatial_size=roi), RandCropByPosNegLabeld(keys, label_key='label', spatial_size=roi, pos=1, neg=1, num_samples=2), RandFlipd(keys, prob=.5, spatial_axis=0)])
    fingerprint=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    dataset_type=PersistentDataset if args.cache_dir else Dataset
    cache_args={'cache_dir':str(Path(args.cache_dir)/fingerprint)} if args.cache_dir else {}
    loaders = {s: DataLoader(dataset_type([r for r in records if r['split']==s], train if s=='train' else Compose(base), **cache_args), batch_size=1, shuffle=s=='train', num_workers=0) for s in ('train','val')}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build().to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    if args.initialize:
        initial=torch.load(args.initialize,map_location='cpu',weights_only=True)
        if initial['manifest_sha256'] != fingerprint:
            raise ValueError('Warm-start manifest changed')
        model.load_state_dict(initial['state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    history, best, first_epoch = [], -1., 0
    if args.resume:
        saved=torch.load(output/'last.pt',map_location=device,weights_only=True)
        if saved['manifest_sha256'] != fingerprint:
            raise ValueError('Resume manifest changed')
        if saved.get('roi_size',64)!=args.roi_size: raise ValueError('Resume patch geometry changed')
        model.load_state_dict(saved['state_dict']); optimizer.load_state_dict(saved['optimizer'])
        history,best,first_epoch=saved['history'],saved['best'],saved['epoch']
    started=time.perf_counter()
    for epoch in range(first_epoch,args.epochs):
        model.train(); losses=[]
        for batch in loaders['train']:
            optimizer.zero_grad()
            loss=loss_fn(model(batch['image'].to(device)),batch['label'].to(device))
            loss.backward(); optimizer.step(); losses.append(loss.item())
        model.eval(); scores=[]
        with torch.no_grad():
            for batch in loaders['val']:
                # Keep full volumes on CPU to bound memory on a 4 GB GPU.
                prediction=sliding_window_inference(batch['image'],roi,1 if args.roi_size==96 else 2,model,sw_device=device,device='cpu').argmax(1,keepdim=True)>0
                target=batch['label']>0
                denominator=prediction.sum()+target.sum()
                dice=float((2*(prediction & target).sum()/denominator).item()) if denominator else 1.
                scores.append(dice)
        mean=sum(scores)/len(scores)
        history.append({'epoch':epoch+1,'loss':sum(losses)/len(losses),'validation_dice':scores,'mean_dice':mean,'elapsed_seconds':time.perf_counter()-started})
        if mean>best:
            best=mean; torch.save({'state_dict':model.state_dict(),'epoch':epoch+1,'spacing':[1.5,1.5,2.],'window':[-57,164],'manifest_sha256':fingerprint,'roi_size':args.roi_size},output/'best.pt')
        torch.save({'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,'best':best,'history':history,'manifest_sha256':fingerprint,'roi_size':args.roi_size},output/'last.tmp')
        (output/'last.tmp').replace(output/'last.pt')
        (output/'history.json').write_text(json.dumps(history,indent=2))
        print(json.dumps(history[-1]),flush=True)
    (output/'run.json').write_text(json.dumps({'args':vars(args),'device':str(device),'torch':str(torch.__version__),'manifest_sha256':fingerprint,'resume_note':'Optimizer restored; augmentation RNG restarts at seed 42 on resumed processes','test_evaluated':False},indent=2))


if __name__=='__main__':
    main()
