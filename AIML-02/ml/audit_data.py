"""Audit split membership, label counts, sampled images and all paired CT geometry."""
import csv
import json
import random
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from ml.prepare_data import LABELS


def main():
    with Path('data/splits/classifier.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    patients={s:{r['patient_id'] for r in rows if r['split']==s} for s in ('train','val','test')}
    assert not (patients['train'] & patients['val'] or patients['train'] & patients['test'] or patients['val'] & patients['test'])
    sampled=random.Random(42).sample(rows,min(256,len(rows)))
    shapes=[]
    for row in sampled:
        with Image.open(row['path']) as image:
            shapes.append(image.size); image.verify()
    cxr={'images':dict(Counter(r['split'] for r in rows)), 'patients':{s:len(p) for s,p in patients.items()},
         'sampled_images_verified':len(sampled),'sample_width_range':[min(s[0] for s in shapes),max(s[0] for s in shapes)],
         'labels':{s:{label:dict(Counter(r[label] or 'missing' for r in rows if r['split']==s)) for label in LABELS} for s in patients},
         'patient_overlap':0,'scope':'Frontal images from source train.csv; official valid.csv preserved separately'}
    volumes=json.loads(Path('data/splits/spleen.json').read_text())
    records=[]
    for row in volumes:
        image,mask=nib.load(row['image']),nib.load(row['label'])
        assert image.shape==mask.shape and np.allclose(image.affine,mask.affine), 'CT/mask geometry mismatch'
        values=np.unique(np.asarray(mask.dataobj))
        assert set(values).issubset({0,1}), 'Unexpected spleen class'
        records.append({'case':row['patient_id'],'split':row['split'],'shape':list(image.shape),
                        'spacing_mm':nib.affines.voxel_sizes(image.affine).tolist(),'mask_voxels':int(np.count_nonzero(np.asarray(mask.dataobj)))})
    report={'classification':cxr,'segmentation':{'cases':len(records),'splits':dict(Counter(r['split'] for r in records)),
            'geometry_matches':len(records),'records':records,'patient_identity_policy':'One MSD spleen case treated as one patient'}}
    Path('reports').mkdir(exist_ok=True)
    Path('reports/data_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'classification':cxr['images'],'sampled_images_verified':len(sampled),'CT_pairs_verified':len(records)},indent=2))


if __name__=='__main__':
    main()
