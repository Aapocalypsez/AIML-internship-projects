"""Select thresholds on validation, then measure an untouched test split once."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from ml.classifier import LABELS, build
from ml.train_classifier import Images, read_manifest


def predict(model, rows, split, device):
    collected = [[], [], []]
    with torch.no_grad():
        for x, y, mask in DataLoader(Images(rows, split), batch_size=16):
            for bucket, value in zip(collected, [model(x.to(device)).sigmoid().cpu().numpy(), y.numpy(), mask.numpy()]):
                bucket.append(value)
    return [np.concatenate(values) for values in collected]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', default='reports/classifier_evaluation.json')
    args=parser.parse_args()
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    saved=torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    fingerprint=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    if saved['labels'] != LABELS or saved['manifest_sha256'] != fingerprint:
        raise ValueError('Checkpoint and evaluation manifest must match')
    model=build().to(device); model.load_state_dict(saved['state_dict']); model.eval()
    rows=read_manifest(args.manifest)
    vp, vy, vm=predict(model, rows, 'val', device)
    tp, ty, tm=predict(model, rows, 'test', device)
    result={'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(), 'manifest_sha256':fingerprint,
            'threshold_policy':'Maximum Youden J on validation known labels only',
            'scope':'Internal held-out development cohort; not the official CheXpert benchmark', 'findings':{}}
    for i,label in enumerate(LABELS):
        vlabels, vscores=vy[vm[:,i],i],vp[vm[:,i],i]
        labels,scores=ty[tm[:,i],i],tp[tm[:,i],i]
        if len(np.unique(vlabels))<2 or len(np.unique(labels))<2:
            result['findings'][label]={'n':len(labels),'status':'Insufficient known positive/negative labels'}
            continue
        fpr,tpr,thresholds=roc_curve(vlabels,vscores)
        finite=np.isfinite(thresholds)
        threshold=float(thresholds[finite][np.argmax((tpr-fpr)[finite])])
        tn,fp,fn,tp_count=confusion_matrix(labels,scores>=threshold,labels=[0,1]).ravel()
        result['findings'][label]={'n':len(labels),'positive':int(labels.sum()),'auroc':float(roc_auc_score(labels,scores)),
            'average_precision':float(average_precision_score(labels,scores)), 'brier_score':float(np.mean((scores-labels)**2)),
            'threshold':threshold,'sensitivity':float(tp_count/(tp_count+fn)), 'specificity':float(tn/(tn+fp)),
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp_count)}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
