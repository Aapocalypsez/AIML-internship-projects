"""Run from repository root: python -m ml.train_classifier --manifest data/classifier.csv."""
import argparse
import csv
import json
import random
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader

from ml.classifier import LABELS, build, transform


def read_manifest(path):
    rows = list(csv.DictReader(Path(path).open(newline='', encoding='utf-8')))
    patients = {}
    for row in rows:
        if row['split'] not in ('train', 'val', 'test') or not row['patient_id'].strip():
            raise ValueError('Every row needs a patient_id and train/val/test split')
        previous = patients.setdefault(row['patient_id'], row['split'])
        if previous != row['split']:
            raise ValueError('Patient leakage between splits')
        for label in LABELS:
            if row[label] not in ('0', '1', '-1', ''):
                raise ValueError(f'Invalid {label} label')
    if not all(any(r['split'] == s for r in rows) for s in ('train', 'val', 'test')):
        raise ValueError('Manifest must include train, val, and held-out test patients')
    return rows


class Images(Dataset):
    def __init__(self, rows, split):
        self.rows = [r for r in rows if r['split'] == split]
        self.transform = transform(split == 'train')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        values = [float(row[k]) if row[k] else -1 for k in LABELS]
        labels = torch.tensor(values)
        with Image.open(row['path']) as image:
            x = self.transform(image.convert('RGB'))
        return x, labels.clamp(min=0), labels >= 0


def evaluate(model, loader, device):
    model.eval()
    predictions, targets, masks = [], [], []
    with torch.no_grad():
        for x, y, mask in loader:
            predictions.append(model(x.to(device)).sigmoid().cpu().numpy())
            targets.append(y.numpy())
            masks.append(mask.numpy())
    p, y, m = map(np.concatenate, (predictions, targets, masks))
    metrics = {}
    for i, label in enumerate(LABELS):
        yi, pi = y[m[:, i], i], p[m[:, i], i]
        valid = len(np.unique(yi)) == 2
        metrics[label] = {'n': len(yi), 'auroc': float(roc_auc_score(yi, pi)) if valid else None, 'average_precision': float(average_precision_score(yi, pi)) if valid else None}
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--output', default='models/classifier')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--balance-known', action='store_true', help='Balance known positive/negative loss per finding using training counts')
    p.add_argument('--freeze-backbone', action='store_true', help='Train the classification head only for a fast transfer baseline')
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        p.error('Epochs and batch size must be positive')
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    rows = read_manifest(args.manifest)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    model = build(pretrained=not args.resume).to(device)
    if args.freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    loaders = {s: DataLoader(Images(rows, s), batch_size=args.batch_size, shuffle=s=='train', num_workers=0) for s in ('train', 'val')}
    optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=1e-4)
    balance=torch.ones(len(LABELS),device=device)
    if args.balance_known:
        counts=[(sum(r[k]=='0' for r in rows if r['split']=='train'),sum(r[k]=='1' for r in rows if r['split']=='train')) for k in LABELS]
        if any(min(pair)==0 for pair in counts): raise ValueError('Balanced loss needs both known classes for every finding')
        balance=torch.tensor([negative/positive for negative,positive in counts],device=device)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    history, best, start = [], float('-inf'), 0
    fingerprint = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    if args.resume:
        saved = torch.load(output/'last.pt', map_location=device, weights_only=True)
        if saved['manifest_sha256'] != fingerprint or saved['freeze_backbone'] != args.freeze_backbone or saved.get('balance_known',False)!=args.balance_known:
            raise ValueError('Resume manifest or training mode changed')
        model.load_state_dict(saved['state_dict']); optimizer.load_state_dict(saved['optimizer'])
        start, best, history = saved['epoch'], saved['best'], saved['history']
        torch.set_rng_state(saved['rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
    started = time.perf_counter()
    for epoch in range(start, args.epochs):
        model.train()
        if args.freeze_backbone:
            model.features.eval()
        total, count = 0., 0
        for x, y, mask in loaders['train']:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            if not mask.any():
                continue
            optimizer.zero_grad()
            losses = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y, reduction='none',pos_weight=balance)
            loss = losses[mask].mean()
            loss.backward(); optimizer.step()
            total += loss.item(); count += 1
        if not count:
            raise ValueError('No known labels in training data')
        metrics = evaluate(model, loaders['val'], device)
        aucs = [v['auroc'] for v in metrics.values() if v['auroc'] is not None]
        if not aucs:
            raise ValueError('Validation needs positive and negative examples for at least one label')
        score = float(np.mean(aucs))
        history.append({'epoch': epoch+1, 'training_loss': total/count, 'validation': metrics, 'elapsed_seconds': time.perf_counter()-started})
        if score > best:
            best = score
            torch.save({'state_dict': model.state_dict(), 'labels': LABELS, 'epoch': epoch+1, 'seed': 42, 'manifest_sha256': fingerprint, 'freeze_backbone': args.freeze_backbone,'balance_known':args.balance_known}, output/'best.pt')
        saved = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1,
                 'best': best, 'history': history, 'manifest_sha256': fingerprint, 'freeze_backbone': args.freeze_backbone,
                 'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'balance_known':args.balance_known}
        torch.save(saved, output/'last.tmp'); (output/'last.tmp').replace(output/'last.pt')
        print(json.dumps(history[-1]), flush=True)
        (output/'history.json').write_text(json.dumps(history, indent=2))
    (output/'run.json').write_text(json.dumps({'device': str(device), 'gpu': torch.cuda.get_device_name(0) if device.type=='cuda' else None, 'torch': str(torch.__version__), 'args': vars(args), 'manifest_sha256': fingerprint, 'uncertainty_policy': 'Ignore uncertain (-1) and missing labels', 'test_evaluated': False}, indent=2))


if __name__ == '__main__':
    main()
