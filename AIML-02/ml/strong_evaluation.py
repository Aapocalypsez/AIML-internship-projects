"""Patient-bootstrap uncertainty and validation-only calibration for a frozen classifier."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve,confusion_matrix
from ml.classifier import LABELS,build
from ml.evaluate_classifier import predict
from ml.train_classifier import read_manifest


def patient_interval(labels,scores,patients,repeats=500):
    rng=np.random.default_rng(42); unique=np.unique(patients)
    groups={p:np.flatnonzero(patients==p) for p in unique}; values=[]
    for _ in range(repeats):
        chosen=np.concatenate([groups[p] for p in rng.choice(unique,len(unique),replace=True)])
        if len(np.unique(labels[chosen]))==2: values.append(roc_auc_score(labels[chosen],scores[chosen]))
    return {'lower':float(np.percentile(values,2.5)),'upper':float(np.percentile(values,97.5)),
            'valid_replicates':len(values),'requested_replicates':repeats} if values else None


def ece(labels,scores):
    result=0.
    for index in range(10):
        selected=(scores>=index/10)&(scores<((index+1)/10) if index<9 else scores<=1)
        if selected.any(): result+=selected.mean()*abs(scores[selected].mean()-labels[selected].mean())
    return float(result)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--manifest',required=True); parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True); args=parser.parse_args()
    torch.set_num_threads(4); rows=read_manifest(args.manifest)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    fingerprint=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    if checkpoint.get('evaluation_manifest_sha256',checkpoint['manifest_sha256'])!=fingerprint: raise ValueError('Manifest mismatch')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=build().to(device); model.load_state_dict(checkpoint['state_dict']); model.eval()
    vp,vy,vm=predict(model,rows,'val',device); tp,ty,tm=predict(model,rows,'test',device)
    patients=np.array([r['patient_id'] for r in rows if r['split']=='test'])
    with Path('data/raw/chexpert/extracted/train.csv').open(newline='') as stream:
        metadata={r['Path'].split('/',1)[-1]:r for r in csv.DictReader(stream)}
    testing=[r for r in rows if r['split']=='test']
    attributes=[metadata.get(Path(r['path']).relative_to(Path('data/raw/chexpert/extracted').resolve()).as_posix(),{}) for r in testing]
    results={'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),'manifest_sha256':fingerprint,
             'scope':'Fresh patient-disjoint internal test cohort; excludes all baseline test patients',
             'selection':'Checkpoint selected on validation AUROC; calibration fitted on validation only; test evaluated after selection',
             'findings':{}}
    calibrators={}
    logit=lambda p:np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
    for index,label in enumerate(LABELS):
        vlabels,vscores=vy[vm[:,index],index],vp[vm[:,index],index]
        labels,scores=ty[tm[:,index],index],tp[tm[:,index],index]
        record={'n':len(labels),'positive':int(labels.sum()),'negative':int(len(labels)-labels.sum())}
        record['validation_known']={'n':len(vlabels),'positive':int(vlabels.sum()),'negative':int(len(vlabels)-vlabels.sum())}
        if len(np.unique(labels))<2 or len(np.unique(vlabels))<2:
            record['status']='Insufficient known positive/negative labels'; results['findings'][label]=record; continue
        calibrated=scores.copy(); validation=vscores.copy()
        if min(int(vlabels.sum()),len(vlabels)-int(vlabels.sum()))>=20:
            reg=LogisticRegression(C=1.,solver='lbfgs').fit(logit(vscores).reshape(-1,1),vlabels)
            if reg.coef_[0,0]>0:
                calibrated=reg.predict_proba(logit(scores).reshape(-1,1))[:,1]
                validation=reg.predict_proba(logit(vscores).reshape(-1,1))[:,1]
                calibrators[label]={'slope':float(reg.coef_[0,0]),'intercept':float(reg.intercept_[0])}
        fpr,tpr,thresholds=roc_curve(vlabels,validation); valid=np.isfinite(thresholds)
        threshold=float(thresholds[valid][np.argmax((tpr-fpr)[valid])])
        tn,fp,fn,tpos=confusion_matrix(labels,calibrated>=threshold,labels=[0,1]).ravel()
        record.update({'auroc':float(roc_auc_score(labels,scores)),'auroc_patient_bootstrap_95':patient_interval(labels,scores,patients[tm[:,index]]),
                       'average_precision':float(average_precision_score(labels,scores)),'threshold':threshold,
                       'sensitivity':float(tpos/(tpos+fn)),'specificity':float(tn/(tn+fp)),
                       'confusion':{'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tpos)},
                       'brier_raw':float(np.mean((scores-labels)**2)),'brier_calibrated':float(np.mean((calibrated-labels)**2)),
                       'ece_raw':ece(labels,scores),'ece_calibrated':ece(labels,calibrated),'calibration_applied':label in calibrators})
        results['findings'][label]=record
        subgroup={}
        for group in ('Male','Female','age_under_65','age_65_plus'):
            def matches(row):
                if group in ('Male','Female'): return row.get('Sex')==group
                try: age=float(row.get('Age',''))
                except ValueError: return False
                return np.isfinite(age) and 0<=age<=120 and ((age<65) if group=='age_under_65' else (age>=65))
            chosen=np.array([matches(row) for row in attributes])&tm[:,index]
            target=ty[chosen,index]; score=tp[chosen,index]
            subgroup[group]={'n':len(target),'positive':int(target.sum()),'negative':int(len(target)-target.sum()),
                             'auroc':float(roc_auc_score(target,score)) if min(int(target.sum()),int(len(target)-target.sum()))>=5 else None}
        record['subgroups']=subgroup
    results['calibrators']=calibrators
    np.savez_compressed(Path(args.output).with_suffix('.npz'),test_scores=tp,test_targets=ty,test_known=tm,patients=patients)
    # Store the fitted coefficients separately; original evaluated weights are preserved.
    Path(args.output).write_text(json.dumps(results,indent=2)); print(json.dumps(results,indent=2))


if __name__=='__main__': main()
