"""Descriptive Grad-CAM occlusion check on known-positive validation examples."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ml.classifier import LABELS,build,transform
from ml.train_classifier import read_manifest


def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--manifest',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); torch.set_num_threads(4); rng=np.random.default_rng(42)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=build(); model.load_state_dict(saved['state_dict']); model.eval()
    rows=read_manifest(a.manifest); findings={}
    for index,label in enumerate(LABELS):
        candidates=[r for r in rows if r['split']=='val' and r[label]=='1']; rng.shuffle(candidates)
        hot_drops=[]; random_drops=[]
        for row in candidates[:8]:
            with Image.open(row['path']) as image: x=transform()(image.convert('RGB')).unsqueeze(0)
            features=F.relu(model.features(x),inplace=False)
            logit=model.classifier(F.adaptive_avg_pool2d(features,(1,1)).flatten(1))[0,index]
            gradient=torch.autograd.grad(logit,features)[0]
            cam=(gradient.mean((2,3),keepdim=True)*features).sum(1).relu()
            cam=F.interpolate(cam.unsqueeze(1),size=(224,224),mode='bilinear',align_corners=False).detach().flatten().numpy()
            count=int(.2*len(cam)); hottest=np.argsort(cam,kind='stable')[-count:]; random=rng.choice(len(cam),count,replace=False)
            pair=x.repeat(2,1,1,1)
            pair[0].reshape(3,-1)[:,hottest]=0; pair[1].reshape(3,-1)[:,random]=0
            with torch.no_grad(): occluded=model(pair).sigmoid()[:,index].numpy()
            baseline=float(logit.sigmoid().detach())
            hot_drops.append(baseline-float(occluded[0])); random_drops.append(baseline-float(occluded[1]))
        findings[label]={'known_positive_validation_examples':len(hot_drops),
                         'mean_score_drop_hottest_20_percent':float(np.mean(hot_drops)) if hot_drops else None,
                         'mean_score_drop_random_20_percent':float(np.mean(random_drops)) if random_drops else None}
    result={'checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),'findings':findings,
            'method':'Eight known-positive validation examples per finding; replace hottest or random20% pixels with ImageNet mean (zero normalized tensor).',
            'limitations':'Descriptive perturbation check, not localization ground truth or clinical validation; occlusion creates distribution shifts. Not used for checkpoint selection.'}
    Path(a.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
