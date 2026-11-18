import gzip
import io

import nibabel as nib
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from backend import app as module
from backend.volume_io import validate_volume, slice_png


def volume_bytes():
    data=np.zeros((16,20,8),dtype=np.float32)
    data[4:10,6:12,3:5]=100
    return gzip.compress(nib.Nifti1Image(data,np.diag([-2,3,4,1])).to_bytes(), mtime=0)


def test_ct_orientation_and_metadata():
    volume=validate_volume(volume_bytes())
    assert nib.aff2axcodes(volume.affine)==('R','A','S')
    assert volume.shape==(16,20,8)


def test_ct_upload_review_guard_and_slice_bounds(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'DATA',tmp_path)
    monkeypatch.delenv('SEGMENTATION_CHECKPOINT',raising=False)
    with TestClient(module.app) as client:
        item=client.post('/api/volumes',files={'file':('deidentified.nii.gz',volume_bytes())})
        assert item.status_code==201
        s=item.json(); assert s['kind']=='ct' and s['status']=='awaiting_model' and s['slices']==8
        assert client.get(f"/api/studies/{s['id']}/slices/3").status_code==200
        assert client.get(f"/api/studies/{s['id']}/slices/8").status_code==404
        assert client.get(f"/api/studies/{s['id']}/slices/3?overlay=true").status_code==404
        assert client.post('/api/volumes',files={'file':('other.nii.gz',volume_bytes())}).json()['id']==s['id']
        assert client.post('/api/volumes',files={'file':('bad.nii.gz',b'bad')}).status_code==422


def test_slice_mask_and_image_have_same_physical_aspect(tmp_path):
    source=validate_volume(volume_bytes())
    image=tmp_path/'image.nii.gz'; mask=tmp_path/'mask.nii.gz'
    nib.save(source,image)
    nib.save(nib.Nifti1Image((source.get_fdata()>0).astype(np.uint8),source.affine),mask)
    a=Image.open(io.BytesIO(slice_png(image,3)))
    b=Image.open(io.BytesIO(slice_png(mask,3,True)))
    assert a.size==b.size==(16,30)


def test_axial_display_places_patient_right_on_screen_left(tmp_path):
    values=np.zeros((4,4,2),dtype=np.float32)
    values[-1,-1,:]=240
    path=tmp_path/'orientation.nii.gz'
    nib.save(nib.Nifti1Image(values,np.eye(4)),path)
    image=Image.open(io.BytesIO(slice_png(path,0)))
    assert image.getpixel((0,0))==255
    assert image.getpixel((3,3))<255
