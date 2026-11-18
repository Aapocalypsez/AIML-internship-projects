"""Evaluate held-out CT volumes in their original voxel geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from ml.segmentation import segment


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',default='reports/segmentation_evaluation.json')
    p.add_argument('--prediction-root',default='data/predictions/spleen')
    args=p.parse_args()
    fingerprint=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if saved['manifest_sha256'] != fingerprint:
        raise ValueError('Checkpoint and evaluation manifest do not match')
    results=[]
    for row in json.loads(Path(args.manifest).read_text()):
        if row['split']!='test':
            continue
        destination=Path(args.prediction_root)/row['patient_id']; destination.mkdir(parents=True,exist_ok=True)
        info=segment(row['image'],args.checkpoint,destination)
        prediction=nib.load(destination/'mask.nii.gz'); target=nib.load(row['label'])
        if prediction.shape != target.shape or not np.allclose(prediction.affine,target.affine):
            raise ValueError('Native-space geometry mismatch')
        predicted=np.asarray(prediction.dataobj)>0; truth=np.asarray(target.dataobj)>0
        denominator=predicted.sum()+truth.sum()
        dice=float(2*np.count_nonzero(predicted & truth)/denominator) if denominator else 1.
        intersection=int(np.count_nonzero(predicted & truth))
        results.append({'case':row['patient_id'],'dice':dice,'seconds':info['seconds'],'geometry_match':True,
                        'precision':intersection/int(predicted.sum()) if predicted.any() else 0.,
                        'recall':intersection/int(truth.sum()) if truth.any() else None,
                        'predicted_voxels':int(predicted.sum()),'true_voxels':int(truth.sum()),
                        'absolute_relative_volume_error':float(abs(int(predicted.sum())-int(truth.sum()))/truth.sum()) if truth.any() else None})
        print(json.dumps(results[-1]),flush=True)
    if not results:
        raise ValueError('No held-out cases')
    rng=np.random.default_rng(42); dice_values=np.array([r['dice'] for r in results])
    interval=np.percentile(rng.choice(dice_values,size=(2000,len(results)),replace=True).mean(axis=1),[2.5,97.5])
    report={'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
            'manifest_sha256':fingerprint,'scope':'Internal patient-separated test split; not the unlabeled MSD challenge test set',
            'mean_dice':float(dice_values.mean()),'mean_dice_bootstrap_95':[float(v) for v in interval],
            'uncertainty_note':'2000 case-bootstrap replicates; seven cases are insufficient for strong generalization claims; these cases were examined during baseline evaluation',
            'prediction_root':args.prediction_root,'cases':results}
    Path(args.output).write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
