"""Shared spleen architecture and geometry-preserving inference."""
from pathlib import Path
import os
import hashlib
import time

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, EnsureTyped, Invertd)


def build():
    return UNet(spatial_dims=3,in_channels=1,out_channels=2,channels=(16,32,64,128,256),strides=(2,2,2,2),num_res_units=2)


def preprocessing(keys):
    return [LoadImaged(keys), EnsureChannelFirstd(keys), Orientationd(keys, axcodes='RAS'),
            Spacingd(keys, pixdim=(1.5,1.5,2.), mode='bilinear' if len(keys)==1 else ('bilinear','nearest')),
            ScaleIntensityRanged('image', a_min=-57, a_max=164, b_min=0., b_max=1., clip=True),
            CropForegroundd(keys, source_key='image'), EnsureTyped(keys)]


def segment(image_path, checkpoint, destination):
    start=time.perf_counter()
    torch.set_num_threads(4)
    device=torch.device(os.environ.get('INFERENCE_DEVICE','cuda' if torch.cuda.is_available() else 'cpu'))
    model=build().to(device)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(saved['state_dict']); model.eval()
    transform=Compose(preprocessing(['image']))
    data=transform({'image':str(image_path)})
    roi=int(saved.get('roi_size',64))
    if roi not in (64,96): raise ValueError('Unsupported checkpoint patch size')
    with torch.no_grad():
        logits=sliding_window_inference(data['image'].unsqueeze(0),(roi,)*3,1 if roi==96 else 2,model,sw_device=device,device='cpu')
    data['prediction']=logits.argmax(1).to(torch.float32)
    restored=Invertd(keys='prediction',transform=transform,orig_keys='image',nearest_interp=True,to_tensor=True)(data)
    mask=np.asarray(restored['prediction'][0].cpu(),dtype=np.uint8)
    source=nib.load(str(image_path))
    if mask.shape != source.shape:
        raise ValueError('Inverted mask shape does not match input geometry')
    output=nib.Nifti1Image(mask,source.affine,source.header.copy())
    output.set_data_dtype(np.uint8)
    nib.save(output,str(Path(destination)/'mask.nii.gz'))
    return {'kind':'segmentation','model_version':'spleen-unet-'+hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()[:12],
            'findings':[], 'seconds':time.perf_counter()-start, 'mask_voxels':int(mask.sum()),
            'note':'Spleen anatomy segmentation. No disease diagnosis or diagnostic confidence is provided.'}
