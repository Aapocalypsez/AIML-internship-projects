"""Verify local Docker/Orthanc import using a visibly synthetic DICOM fixture."""
import io
import json
import time
from pathlib import Path
import httpx
import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, DigitalXRayImageStorageForPresentation, generate_uid


def main():
    password=next(line.split('=',1)[1] for line in Path('.env').read_text().splitlines() if line.startswith('ORTHANC_PASSWORD='))
    meta=FileMetaDataset()
    meta.TransferSyntaxUID=ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID=DigitalXRayImageStorageForPresentation
    meta.MediaStorageSOPInstanceUID=generate_uid()
    ds=FileDataset(None,{},file_meta=meta,preamble=b'\0'*128)
    ds.SOPClassUID=meta.MediaStorageSOPClassUID; ds.SOPInstanceUID=meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID=generate_uid(); ds.SeriesInstanceUID=generate_uid()
    ds.PatientName='SYNTHETIC^CONTAINER_TEST'; ds.PatientID='SYNTHETIC-NOT-A-PATIENT'
    ds.Modality='DX'; ds.Rows=64; ds.Columns=64; ds.SamplesPerPixel=1
    ds.PhotometricInterpretation='MONOCHROME2'; ds.BitsAllocated=16; ds.BitsStored=12
    ds.HighBit=11; ds.PixelRepresentation=0
    ds.PixelData=np.arange(4096,dtype=np.uint16).reshape(64,64).tobytes()
    stream=io.BytesIO(); ds.save_as(stream,enforce_file_format=True)
    with httpx.Client(timeout=30) as client:
        assert client.get('http://127.0.0.1:8080').status_code==200
        health=client.get('http://127.0.0.1:8080/api/health'); health.raise_for_status()
        assert health.json()['education_only'] and not health.json()['classifier_configured']
        response=client.post('http://127.0.0.1:8042/instances',auth=('research',password),content=stream.getvalue())
        response.raise_for_status()
        instance=response.json()['ID']
        imported=None
        for attempt in range(30):
            state=Path('data/importer/orthanc-import.json')
            seen=json.loads(state.read_text()) if state.exists() else {}
            if instance in seen:
                imported=seen[instance]; break
            time.sleep(1)
        assert imported and imported['status']=='imported', imported
        study=client.get('http://127.0.0.1:8000/api/studies/'+imported['detail']).json()
        assert study['status']=='awaiting_model' and study['result'] is None
        duplicate=client.post('http://127.0.0.1:8000/api/studies',files={'file':('synthetic.dcm',stream.getvalue())})
        assert duplicate.json()['id']==study['id']
    report={'viewer_http':200,'proxied_health_http':200,'orthanc_upload_http':response.status_code,
            'importer':'imported','byte_deduplication':True,'prediction':'awaiting_model',
            'fixture':'Synthetic 64x64 DX image, not patient data','scope':'Container workflow only; containerized ML inference is not configured'}
    Path('reports/container_results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':
    main()
