"""Create reproducible patient-grouped manifests without copying source images."""
import argparse
import csv
import json
import random
import re
from pathlib import Path

LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']


def normalize_label(value):
    if value == '':
        return ''
    number = float(value)
    if number not in (-1, 0, 1):
        raise ValueError('Labels must be -1, 0, 1 or missing')
    return str(int(number))


def partition(patient_ids):
    patients = sorted(set(patient_ids))
    if len(patients) < 10:
        raise ValueError('Use at least 10 patients for a development split')
    random.Random(42).shuffle(patients)
    n_train, n_val = int(.7*len(patients)), max(1,int(.15*len(patients)))
    return {p:'train' if i<n_train else 'val' if i<n_train+n_val else 'test' for i,p in enumerate(patients)}


def chexpert(csv_path, root, output):
    records=[]
    root = Path(root).resolve()
    has_prefix = (root/'CheXpert-v1.0-small').is_dir()
    with Path(csv_path).open(newline='',encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            if row['Frontal/Lateral'] != 'Frontal':
                continue
            match = re.search(r'(patient\d+)',row['Path'])
            if not match:
                raise ValueError('Cannot determine patient identifier')
            relative = Path(row['Path'])
            if relative.is_absolute() or relative.drive or '..' in relative.parts:
                raise ValueError('Image path escapes dataset root')
            if relative.parts[0] == 'CheXpert-v1.0-small' and not has_prefix:
                relative = Path(*relative.parts[1:])
            path = root/relative
            if not path.is_file():
                raise FileNotFoundError(f'{path}: root must be the parent of CheXpert-v1.0-small')
            records.append({'path':str(path),'patient_id':match[1],**{k:normalize_label(row[k]) for k in LABELS}})
    splits=partition(r['patient_id'] for r in records)
    with Path(output).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['path','patient_id','split']+LABELS)
        writer.writeheader()
        writer.writerows({**r,'split':splits[r['patient_id']]} for r in records)


def spleen(root, output):
    records=[]
    for image in sorted((Path(root)/'imagesTr').glob('*.nii.gz')):
        if image.name.startswith('._'):
            continue
        label=Path(root)/'labelsTr'/image.name
        if not label.is_file():
            raise FileNotFoundError(label)
        records.append({'image':str(image.resolve()),'label':str(label.resolve()),'patient_id':image.name.removesuffix('.nii.gz')})
    splits=partition(r['patient_id'] for r in records)
    Path(output).write_text(json.dumps([{**r,'split':splits[r['patient_id']]} for r in records],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('dataset',choices=['chexpert','spleen'])
    parser.add_argument('--root',required=True)
    parser.add_argument('--csv')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    if args.dataset=='chexpert':
        if not args.csv:
            parser.error('--csv is required for CheXpert')
        chexpert(args.csv,args.root,args.output)
    else:
        spleen(args.root,args.output)
