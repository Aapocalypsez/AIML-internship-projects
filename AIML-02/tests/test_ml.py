from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from PIL import Image
from monai.transforms import Compose, Invertd
from ml.classifier import LABELS, build
from ml.inference import classify
from ml.segmentation import preprocessing
from ml.train_classifier import read_manifest


def test_gradcam_outputs_align_with_source(tmp_path):
    torch.manual_seed(42)
    model=build()
    weights=tmp_path/'fixture.pt'
    torch.save({'state_dict':model.state_dict(),'labels':LABELS,'calibrators':{LABELS[0]:{'slope':.5,'intercept':.2}}},weights)
    pixels=np.random.default_rng(42).integers(0,255,(80,96,3),dtype=np.uint8)
    image=tmp_path/'image.png'; Image.fromarray(pixels).save(image)
    result=classify(image,weights,tmp_path)
    assert len(result['findings'])==5
    first=result['findings'][0]
    expected=1/(1+np.exp(-(.5*np.log(first['raw_score']/(1-first['raw_score']))+.2)))
    assert first['calibrated'] and np.isclose(first['score'],expected)
    for finding in result['findings']:
        assert 0<=finding['score']<=1
        overlay=Image.open(tmp_path/finding['heatmap'])
        assert overlay.size==(96,80) and overlay.mode=='RGBA'


def test_inverse_transform_restores_native_shape_and_affine(tmp_path):
    data=np.full((24,30,18),-1000,dtype=np.float32)
    data[4:20,5:25,3:15]=80
    affine=np.array([[-2,0,0,40],[0,3,0,-20],[0,0,4,10],[0,0,0,1]],dtype=float)
    path=tmp_path/'ct.nii.gz'; nib.save(nib.Nifti1Image(data,affine),path)
    transform=Compose(preprocessing(['image']))
    item=transform({'image':str(path)})
    item['prediction']=(item['image']>.5).float()
    restored=Invertd(keys='prediction',transform=transform,orig_keys='image',nearest_interp=True,to_tensor=True)(item)['prediction']
    assert tuple(restored.shape[1:])==data.shape
    assert np.allclose(restored.affine,affine)


def test_manifest_rejects_patient_leakage(tmp_path):
    import csv
    import pytest
    path=tmp_path/'leak.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['path','patient_id','split']+LABELS)
        writer.writeheader()
        for split in ('train','val','test'):
            writer.writerow({'path':'unused','patient_id':'same-person','split':split,**{k:'0' for k in LABELS}})
    with pytest.raises(ValueError,match='leakage'):
        read_manifest(path)
