"""Poll Orthanc instances and submit supported radiographs to the local API.

The API deduplicates file bytes. Unsupported instances are recorded as skipped.
CT import requires Orthanc's stable, complete series and a known expected count.
"""
import json
import os
import time
import io
import zipfile
from pathlib import Path

import httpx


def read_instance(orthanc,instance):
    data=bytearray()
    with orthanc.stream('GET','/instances/'+instance+'/file') as response:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            data.extend(chunk)
            if len(data)>32*1024*1024: raise ValueError('DICOM instance exceeds32MB importer bound')
    return bytes(data)


def import_ct(orthanc,api,seen):
    response=orthanc.get('/series'); response.raise_for_status()
    for identifier in response.json():
        key='series:'+identifier
        if key in seen: continue
        response=orthanc.get('/series/'+identifier); response.raise_for_status(); series=response.json()
        if series.get('MainDicomTags',{}).get('Modality')!='CT': continue
        expected=series.get('ExpectedNumberOfInstances')
        if not series.get('IsStable') or series.get('Status')!='Complete' or not isinstance(expected,int) or expected!=len(series['Instances']):
            continue  # A quiet series alone is not evidence that every slice has arrived.
        if not 2<=expected<=1024:
            seen[key]={'status':'skipped','detail':'CT instance count exceeds supported bounds'}; continue
        buffer=io.BytesIO(); total=0
        try:
            with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('manifest.json',json.dumps({'expected_slices':expected}))
                for index,instance in enumerate(series['Instances']):
                    raw=read_instance(orthanc,instance)
                    total+=len(raw)
                    if total>256*1024*1024: raise ValueError('CT series exceeds expanded-byte limit')
                    archive.writestr(str(index)+'.dcm',raw)
        except ValueError as error:
            seen[key]={'status':'skipped','detail':str(error)}; continue
        result=api.post(os.environ.get('API_URL','http://localhost:8000')+'/api/ct-series',files={'file':('series.zip',buffer.getvalue(),'application/zip')})
        if result.status_code not in (201,422,413): result.raise_for_status()
        seen[key]={'status':'imported' if result.status_code==201 else 'skipped','detail':result.json()['id'] if result.status_code==201 else result.json()}


def main():
    state = Path(os.environ.get('ORTHANC_STATE', 'data/orthanc-import.json'))
    state.parent.mkdir(parents=True, exist_ok=True)
    seen = json.loads(state.read_text()) if state.exists() else {}
    auth = (os.environ.get('ORTHANC_USER', 'research'), os.environ['ORTHANC_PASSWORD'])
    with httpx.Client(base_url=os.environ.get('ORTHANC_URL','http://localhost:8042'), auth=auth, timeout=60) as orthanc, httpx.Client(timeout=120) as api:
        last_login=0.
        while True:
            try:
                if os.environ.get('API_REVIEW_USERNAME') and (last_login==0 or time.monotonic()-last_login>7*3600):
                    login=api.post(os.environ.get('API_URL','http://localhost:8000')+'/api/auth/login',json={'username':os.environ['API_REVIEW_USERNAME'],'password':os.environ['API_REVIEW_PASSWORD']})
                    login.raise_for_status(); last_login=time.monotonic()
                response=orthanc.get('/instances'); response.raise_for_status()
                for instance in response.json():
                    if instance in seen:
                        continue
                    tags=orthanc.get(f'/instances/{instance}/simplified-tags'); tags.raise_for_status()
                    if tags.json().get('Modality')=='CT': continue
                    raw=orthanc.get(f'/instances/{instance}/file'); raw.raise_for_status()
                    result=api.post(os.environ.get('API_URL','http://localhost:8000')+'/api/studies',files={'file':('scan.dcm',raw.content,'application/dicom')})
                    if result.status_code not in (201,422,413):
                        result.raise_for_status()
                    seen[instance]={'status':'imported' if result.status_code==201 else 'skipped','detail':result.json() if result.status_code!=201 else result.json()['id']}
                    temp=state.with_suffix('.tmp'); temp.write_text(json.dumps(seen,indent=2)); temp.replace(state)
                import_ct(orthanc,api,seen)
                temp=state.with_suffix('.tmp'); temp.write_text(json.dumps(seen,indent=2)); temp.replace(state)
            except httpx.HTTPError as e:
                if isinstance(e,httpx.HTTPStatusError) and e.response.status_code==401: last_login=0.
                print(f'Orthanc connection/import failed: {type(e).__name__}',flush=True)
            time.sleep(10)


if __name__=='__main__':
    main()
