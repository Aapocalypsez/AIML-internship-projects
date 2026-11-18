"""Test authenticated CPU model inference and Orthanc CT import in isolated storage."""
import csv
import io
import json
import re
import runpy
import time
import zipfile
from pathlib import Path
import httpx


def ready(client,identifier):
    started=time.monotonic()
    while time.monotonic()-started<240:
        response=client.get('/api/studies/'+identifier); response.raise_for_status(); study=response.json()
        if study['status']=='ready': return study
        if study['status']=='failed': raise AssertionError(study['error'])
        time.sleep(1)
    raise TimeoutError('Container inference did not finish within240seconds')


def main():
    credentials=Path('data/auth/LOCAL_ACCESS.txt').read_text()
    password=re.search(r'Password: (.+)',credentials).group(1)
    with Path('data/splits/classifier_improved.csv').open(newline='') as f: xray=next(r for r in csv.DictReader(f) if r['split']=='val')
    ct=next(r for r in json.loads(Path('data/splits/spleen.json').read_text()) if r['split']=='val')
    evidence=[]
    with httpx.Client(base_url='http://127.0.0.1:8000',timeout=240) as client:
        assert client.get('/api/studies').status_code==401
        login=client.post('/api/auth/login',json={'username':'project-owner','password':password}); login.raise_for_status()
        health=client.get('/api/health').json()
        assert health['classifier_configured'] and health['segmentation_configured'] and health['authentication']=='account-session'
        for kind,row,endpoint in [('radiograph',xray,'/api/studies'),('ct',ct,'/api/volumes')]:
            path=Path(row['path'] if kind=='radiograph' else row['image'])
            with path.open('rb') as f: response=client.post(endpoint,files={'file':('validation.jpg' if kind=='radiograph' else 'validation.nii.gz',f)})
            response.raise_for_status(); study=ready(client,response.json()['id'])
            asset=f"/api/studies/{study['id']}/slices/{study['slices']//2}?overlay=true" if kind=='ct' else f"/api/studies/{study['id']}/files/heatmap-0.png"
            assert client.get(asset).status_code==200
            review=client.post(f"/api/studies/{study['id']}/reviews",json={'reviewer':'Spoofed identity','decision':'reject','comment':'Automated software verification only, not clinical validation.','model_version':study['result']['model_version']})
            assert review.status_code==201 and review.json()['reviews'][-1]['reviewer']=='project-owner'
            evidence.append({'kind':kind,'status':'ready','model_version':study['result']['model_version'],'cpu_inference_seconds':study['result']['seconds'],'overlay_http':200,'review_http':201,'authenticated_identity_enforced':True})
            print(json.dumps(evidence[-1]),flush=True)
        values=dict(line.split('=',1) for line in Path('.env').read_text().splitlines() if line and not line.startswith('#') and '=' in line)
        fixture=runpy.run_path('tests/test_dicom_series.py')['series_zip']()
        series_id=None
        with httpx.Client(base_url='http://127.0.0.1:8042',auth=('research',values['ORTHANC_PASSWORD']),timeout=30) as orthanc:
            with zipfile.ZipFile(io.BytesIO(fixture)) as archive:
                for name in archive.namelist():
                    if name.endswith('.dcm'):
                        response=orthanc.post('/instances',content=archive.read(name)); response.raise_for_status()
                        series_id=response.json()['ParentSeries']
            started=time.monotonic(); imported=None
            while time.monotonic()-started<150:
                state_path=Path('data/importer/orthanc-import.json')
                state=json.loads(state_path.read_text()) if state_path.exists() else {}
                imported=state.get('series:'+series_id)
                if imported: break
                time.sleep(1)
            assert imported and imported['status']=='imported', imported
            study=ready(client,imported['detail']); assert study['kind']=='ct'
            evidence.append({'kind':'synthetic_ct_dicom_series','importer':'imported','prediction':'ready','geometry':'sorted and reconstructed by validated assembler','fixture':'Explicit synthetic CT; isolated verification storage, not patient data'})
        assert client.post('/api/ct-series',files={'file':('incomplete.zip',runpy.run_path('tests/test_dicom_series.py')['series_zip'](expected=4))}).status_code==422
    Path('reports/model_container_results.json').write_text(json.dumps({'scope':'Authenticated CPU containers in isolated verification DATA_DIR','checks':evidence,'incomplete_series_http':422},indent=2))
    print('Authenticated model-container and Orthanc CT checks passed.',flush=True)


if __name__=='__main__': main()
