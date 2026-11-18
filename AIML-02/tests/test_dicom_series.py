"""Synthetic CT stacks with known patient coordinates, not patient scans."""
import io
import json
import zipfile
import numpy as np
import pytest
from pydicom.dataset import FileDataset,FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian,CTImageStorage,generate_uid
from backend.dicom_series import assemble


def series_zip(mutate=None,expected=3):
    study,series,frame=generate_uid(),generate_uid(),generate_uid()
    datasets=[]
    for index in range(3):
        meta=FileMetaDataset(); meta.TransferSyntaxUID=ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID=CTImageStorage; meta.MediaStorageSOPInstanceUID=generate_uid()
        ds=FileDataset(None,{},file_meta=meta,preamble=b'\0'*128)
        ds.SOPClassUID=CTImageStorage; ds.SOPInstanceUID=meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID=study; ds.SeriesInstanceUID=series; ds.FrameOfReferenceUID=frame
        ds.PatientName='SYNTHETIC^CT_TEST'; ds.PatientID='SYNTHETIC-NOT-A-PATIENT'
        ds.InstanceNumber=index+1; ds.ImagesInAcquisition=3; ds.NumberOfTemporalPositions=1
        ds.Modality='CT'; ds.Rows=4; ds.Columns=5; ds.SamplesPerPixel=1
        ds.PhotometricInterpretation='MONOCHROME2'; ds.BitsAllocated=16; ds.BitsStored=16; ds.HighBit=15; ds.PixelRepresentation=1
        ds.ImageOrientationPatient=[1,0,0,0,1,0]; ds.ImagePositionPatient=[10,20,index*3]
        ds.PixelSpacing=[2,1]; ds.RescaleSlope=2; ds.RescaleIntercept=-1000
        ds.PixelData=np.full((4,5),index*10,dtype=np.int16).tobytes(); datasets.append(ds)
    if mutate: mutate(datasets)
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as archive:
        archive.writestr('manifest.json',json.dumps({'expected_slices':expected}))
        for index,ds in enumerate(reversed(datasets)):
            content=io.BytesIO(); ds.save_as(content,enforce_file_format=True)
            archive.writestr(f'{index}.dcm',content.getvalue())
    return stream.getvalue()


def test_ct_sorting_hu_and_ras_coordinates():
    volume=assemble(series_zip())
    assert volume.shape==(5,4,3)
    assert np.allclose(volume.get_fdata()[0,0,:],[-1000,-980,-960])
    assert np.allclose(volume.affine@np.array([4,3,0,1]),[-10,-20,0,1])
    assert np.allclose(volume.header.get_zooms(),[1,2,3])


@pytest.mark.parametrize('mutate',[
    lambda rows:setattr(rows[1],'ImagePositionPatient',[10,20,0]),
    lambda rows:setattr(rows[1],'ImagePositionPatient',[10,20,4]),
    lambda rows:setattr(rows[1],'ImagePositionPatient',[11,20,3]),
    lambda rows:setattr(rows[1],'SeriesInstanceUID',generate_uid()),
    lambda rows:setattr(rows[1],'ImageOrientationPatient',[0,1,0,1,0,0]),
    lambda rows:setattr(rows[1],'PixelSpacing',[1,1]),
    lambda rows:setattr(rows[1],'RescaleSlope',0),
    lambda rows:delattr(rows[1],'RescaleIntercept'),
])
def test_ct_rejects_invalid_geometry(mutate):
    with pytest.raises(ValueError): assemble(series_zip(mutate))


def test_ct_rejects_incomplete_series():
    with pytest.raises(ValueError,match='Complete series'): assemble(series_zip(expected=4))


def test_series_api_deduplicates_without_invented_output(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from backend import app as module
    monkeypatch.setattr(module,'DATA',tmp_path); monkeypatch.delenv('SEGMENTATION_CHECKPOINT',raising=False)
    raw=series_zip()
    with TestClient(module.app) as client:
        first=client.post('/api/ct-series',files={'file':('ct.zip',raw)})
        assert first.status_code==201 and first.json()['kind']=='ct' and first.json()['status']=='awaiting_model'
        second=client.post('/api/ct-series',files={'file':('ct.zip',raw)})
        assert first.json()['id']==second.json()['id']
        assert client.post('/api/ct-series',files={'file':('ct.zip',series_zip(expected=4))}).status_code==422


@pytest.mark.parametrize('status,stable,expected', [('Complete',False,3),('Unknown',True,None),('Missing',True,4)])
def test_orthanc_waits_for_declared_complete_stable_series(status,stable,expected):
    import httpx
    from backend.orthanc_import import import_ct
    calls=[]
    def orthanc_handler(request):
        calls.append(request.url.path)
        return httpx.Response(200,json=['test'] if request.url.path=='/series' else {'MainDicomTags':{'Modality':'CT'},'IsStable':stable,'Status':status,'ExpectedNumberOfInstances':expected,'Instances':['a','b','c']})
    with httpx.Client(transport=httpx.MockTransport(orthanc_handler),base_url='http://orthanc') as orthanc,httpx.Client() as api:
        state={}; import_ct(orthanc,api,state)
    assert not state and calls==['/series','/series/test']
