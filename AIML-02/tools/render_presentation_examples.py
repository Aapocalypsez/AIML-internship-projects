"""Prepare faithful image/overlay layers from the active models and validation inputs."""
import csv
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from PIL import Image
from backend.volume_io import slice_png
from ml.inference import classify
from ml.segmentation import segment


def main():
    out=Path('reports/presentation-example'); out.mkdir(exist_ok=True)
    with Path('data/splits/classifier_improved.csv').open(newline='') as f: row=next(r for r in csv.DictReader(f) if r['split']=='val')
    with Image.open(row['path']) as image: image.convert('RGB').save(out/'xray.png')
    classifier=classify(out/'xray.png',Path('models/active/classifier.pt'),out)
    row=next(r for r in json.loads(Path('data/splits/spleen.json').read_text()) if r['split']=='val')
    raw=nib.load(row['image']); volume=nib.as_closest_canonical(nib.Nifti1Image(raw.get_fdata(dtype=np.float32),raw.affine))
    truth=nib.as_closest_canonical(nib.load(row['label']))
    assert truth.shape==volume.shape and np.allclose(truth.affine,volume.affine)
    index=int(np.asarray(truth.dataobj).sum(axis=(0,1)).argmax())
    nib.save(volume,out/'ct.nii.gz'); segmentation=segment(out/'ct.nii.gz',Path('models/active/spleen.pt'),out)
    (out/'ct.png').write_bytes(slice_png(out/'ct.nii.gz',index))
    (out/'ct-overlay.png').write_bytes(slice_png(out/'mask.nii.gz',index,overlay=True))
    evidence={'source':'Validation inputs, not synthetic scans','slice_rule':'Maximum supplied validation spleen area; not selected from test performance',
              'slice_index':index,'classifier_model_version':classifier['model_version'],'segmentation_model_version':segmentation['model_version']}
    (out/'evidence.json').write_text(json.dumps(evidence,indent=2)); print(json.dumps(evidence))


if __name__=='__main__': main()
