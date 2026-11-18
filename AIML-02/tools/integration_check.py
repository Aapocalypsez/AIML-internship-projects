"""Run real saved checkpoints through both API upload/review workflows."""
import csv
import json
import os
from pathlib import Path
import tempfile

from fastapi.testclient import TestClient
from backend import app as module


def main():
    root=Path.cwd()
    os.environ['CLASSIFIER_CHECKPOINT']=str(root/'models/classifier-development/best.pt')
    os.environ['SEGMENTATION_CHECKPOINT']=str(root/'models/spleen-development/best.pt')
    with Path('data/splits/classifier_development.csv').open(newline='') as stream:
        xray=next(r for r in csv.DictReader(stream) if r['split']=='val')
    ct=next(r for r in json.loads(Path('data/splits/spleen.json').read_text()) if r['split']=='val')
    results=[]
    with tempfile.TemporaryDirectory(prefix='integration-',dir=root/'data') as folder:
        module.DATA=Path(folder)
        with TestClient(module.app) as client:
            for kind,row,endpoint in [('radiograph',xray,'/api/studies'),('ct',ct,'/api/volumes')]:
                path=Path(row['path'] if kind=='radiograph' else row['image'])
                with path.open('rb') as stream:
                    response=client.post(endpoint,files={'file':(path.name,stream)})
                response.raise_for_status()
                sid=response.json()['id']
                study=client.get(f'/api/studies/{sid}').json()
                assert study['status']=='ready', study
                assert not study['demo']
                if kind=='ct':
                    asset=f'/api/studies/{sid}/slices/{study["slices"]//2}?overlay=true'
                else:
                    assert len(study['result']['findings'])==5
                    asset=f'/api/studies/{sid}/files/heatmap-0.png'
                assert client.get(asset).status_code==200
                review=client.post(f'/api/studies/{sid}/reviews',json={'reviewer':'Automated integration test','decision':'reject',
                    'comment':'Software verification only. Not clinical validation.','model_version':study['result']['model_version']})
                assert review.status_code==201 and len(review.json()['reviews'])==1
                results.append({'kind':kind,'prediction':'ready','overlay_http_status':200,'review_http_status':201,
                                'model_version':study['result']['model_version'],'inference_seconds':study['result']['seconds']})
                print(json.dumps(results[-1]),flush=True)
    Path('reports/integration_results.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
