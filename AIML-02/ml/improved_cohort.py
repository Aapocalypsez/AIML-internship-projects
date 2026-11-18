"""Freeze a larger experiment, excluding all previously examined test patients."""
import csv
import hashlib
import json
import random
from pathlib import Path
from ml.classifier import LABELS


def main():
    source=Path('data/splits/classifier.csv')
    with source.open(newline='') as f: rows=list(csv.DictReader(f))
    with Path('data/splits/classifier_development.csv').open(newline='') as f: exposed={r['patient_id'] for r in csv.DictReader(f) if r['split']=='test'}
    output=Path('data/splits/classifier_improved.csv')
    if output.exists(): raise ValueError('Frozen improved cohort already exists; preserve it')
    selected=[]
    for split,count in [('train',8192),('val',2048),('test',2048)]:
        candidates=[r for r in rows if r['split']==split and not (split=='test' and r['patient_id'] in exposed)]
        random.Random(20261107).shuffle(candidates)
        # Include known negative training examples for rare findings; validation/test stay naturally sampled.
        chosen={}
        if split=='train':
            for label in LABELS:
                for row in [r for r in candidates if r[label]=='0'][:200]: chosen[row['path']]=row
        for row in candidates:
            if len(chosen)>=count: break
            chosen[row['path']]=row
        assert len(chosen)==count
        selected.extend(chosen.values())
    with output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(selected)
    groups={s:{r['patient_id'] for r in selected if r['split']==s} for s in ('train','val','test')}
    assert not groups['train']&groups['val'] and not groups['test']&(groups['train']|groups['val']|exposed)
    report={'seed':20261107,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'manifest_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'images':{s:sum(r['split']==s for r in selected) for s in groups},
            'patients':{s:len(group) for s,group in groups.items()},'excluded_exposed_test_patients':len(exposed),
            'policy':'Training includes up to 200 known negatives per finding; validation/test naturally sampled inside original patient partitions. New test patients exclude every baseline test patient.'}
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report))


if __name__=='__main__': main()
