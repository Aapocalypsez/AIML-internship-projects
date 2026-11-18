import io
import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import app as module


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'DATA', tmp_path)
    monkeypatch.delenv('CLASSIFIER_CHECKPOINT', raising=False)
    with TestClient(module.app) as c:
        yield c


def png():
    stream=io.BytesIO()
    Image.new('RGB',(64,64),'gray').save(stream,format='PNG')
    return stream.getvalue()


def test_upload_deduplicates_and_never_invents_prediction(client):
    response=client.post('/api/studies',files={'file':('patient-name.png',png(),'image/png')})
    assert response.status_code==201
    first=response.json()
    assert first['status']=='awaiting_model' and first['result'] is None
    assert 'patient-name' not in first['name']
    assert client.post('/api/studies',files={'file':('scan.png',png())}).json()['id']==first['id']
    assert len(client.get('/api/studies').json())==1
    assert client.post(f"/api/studies/{first['id']}/reviews",json={'reviewer':'Test','decision':'approve','model_version':'fake'}).status_code==409


def test_raster_upload_discards_exif_metadata(client):
    image=Image.new('RGB',(64,64),'gray'); metadata=image.getexif()
    metadata[270]='SYNTHETIC identifier used only for privacy regression'
    stream=io.BytesIO(); image.save(stream,format='JPEG',exif=metadata)
    response=client.post('/api/studies',files={'file':('synthetic-exif.jpg',stream.getvalue())})
    assert response.status_code==201
    with Image.open(module.DATA/response.json()['id']/'image.png') as stored:
        assert not stored.getexif() and 'exif' not in stored.info


@pytest.mark.parametrize('name,content',[('x.png',b'broken'),('x.nii',b'volume'),('x.jpg',b''),('x.dcm',b'not dicom')])
def test_invalid_files(client,name,content):
    assert client.post('/api/studies',files={'file':(name,content)}).status_code==422


def test_review_history_and_version_check(client):
    s=client.post('/api/demo').json()
    assert s['demo'] and s['result']['kind']=='synthetic'
    url=f"/api/studies/{s['id']}/reviews"
    body={'reviewer':'Demo reviewer','decision':'approve','comment':'Workflow test','model_version':'synthetic-fixture-v1'}
    assert client.post(url,json={**body,'model_version':'stale'}).status_code==409
    assert client.post(url,json={**body,'reviewer':'   '}).status_code==422
    assert client.post(url,json=body).status_code==201
    assert client.post(url,json={**body,'decision':'reject'}).status_code==201
    assert [r['decision'] for r in client.get(f"/api/studies/{s['id']}").json()['reviews']]==['approve','reject']
    with module.db() as c:
        with pytest.raises(sqlite3.IntegrityError):
            c.execute("UPDATE reviews SET decision='approve'")
    assert client.get(f"/api/studies/{s['id']}/files/image.png").status_code==200
    assert client.get(f"/api/studies/{s['id']}/files/review.db").status_code==404


def test_restart_marks_interrupted_job(client):
    s=client.post('/api/demo').json()
    with module.db() as c:
        c.execute("UPDATE studies SET status='processing' WHERE id=?",(s['id'],))
    module.initialize()
    assert client.get(f"/api/studies/{s['id']}").json()['status']=='failed'


def test_dicom_conversion_and_modality_rejection(client):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    meta=FileMetaDataset()
    meta.TransferSyntaxUID=ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID='1.2.840.10008.5.1.4.1.1.1'
    meta.MediaStorageSOPInstanceUID=generate_uid()
    ds=FileDataset(None,{},file_meta=meta,preamble=b'\0'*128)
    ds.Modality='DX'; ds.Rows=64; ds.Columns=64; ds.SamplesPerPixel=1
    ds.PhotometricInterpretation='MONOCHROME1'; ds.BitsAllocated=16; ds.BitsStored=16; ds.HighBit=15; ds.PixelRepresentation=0
    ds.PixelData=np.arange(4096,dtype=np.uint16).tobytes()
    stream=io.BytesIO(); ds.save_as(stream,enforce_file_format=True)
    img=module.read_image(stream.getvalue(),'.dcm')
    assert img.getpixel((0,0))[0]==255 and img.getpixel((63,63))[0]==0
    assert client.post('/api/studies',files={'file':('test.dcm',stream.getvalue())}).status_code==201
    ds.Modality='CT'; stream=io.BytesIO(); ds.save_as(stream,enforce_file_format=True)
    assert client.post('/api/studies',files={'file':('test.dcm',stream.getvalue())}).status_code==422
