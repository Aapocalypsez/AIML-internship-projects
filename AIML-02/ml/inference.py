import hashlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ml.classifier import LABELS, build, transform


def classify(image_path, checkpoint, output):
    started = time.perf_counter()
    torch.set_num_threads(4)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved.get('labels') != LABELS:
        raise ValueError('Checkpoint label order does not match the application')
    model = build()
    model.load_state_dict(saved['state_dict'])
    model.eval()
    original = Image.open(image_path).convert('RGB')
    x = transform()(original).unsqueeze(0)
    features = F.relu(model.features(x), inplace=False)
    logits = model.classifier(F.adaptive_avg_pool2d(features, (1, 1)).flatten(1))
    scores = logits.sigmoid().detach()[0].tolist()
    calibrated=logits.detach().clone()
    for index,label in enumerate(LABELS):
        coefficients=saved.get('calibrators',{}).get(label)
        if coefficients:
            slope,intercept=float(coefficients['slope']),float(coefficients['intercept'])
            if not np.isfinite([slope,intercept]).all() or slope<=0: raise ValueError('Invalid calibration coefficients')
            calibrated[0,index]=slope*calibrated[0,index]+intercept
    displayed_scores=calibrated.sigmoid()[0].tolist()
    findings = []
    for i, label in enumerate(LABELS):
        gradient = torch.autograd.grad(logits[0, i], features, retain_graph=i < len(LABELS)-1)[0]
        cam = (gradient.mean((2, 3), keepdim=True) * features).sum(1).relu()
        cam = cam / cam.max().clamp(min=1e-8)
        cam = F.interpolate(cam.unsqueeze(1), size=(original.height, original.width), mode='bilinear', align_corners=False)[0, 0].detach().numpy()
        rgba = np.zeros((*cam.shape, 4), dtype=np.uint8)
        rgba[..., 0], rgba[..., 1] = 255, 130
        rgba[..., 3] = (cam * 220).astype(np.uint8)
        filename = f'heatmap-{i}.png'
        Image.fromarray(rgba).save(output / filename)
        findings.append({'label': label, 'score': displayed_scores[i], 'raw_score':scores[i], 'calibrated':label in saved.get('calibrators',{}),'heatmap': filename})
    note='Validation-fitted calibration where supported; scores are not established clinical probabilities.' if saved.get('calibrators') else 'Uncalibrated model outputs.'
    return {'model_version': 'densenet121-' + hashlib.sha256(checkpoint.read_bytes()).hexdigest()[:12], 'kind': 'trained', 'findings': findings, 'seconds': time.perf_counter()-started, 'note': note+' Grad-CAM is not a localization ground truth.'}
