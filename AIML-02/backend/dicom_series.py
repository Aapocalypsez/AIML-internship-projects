"""Assemble one complete, regular single-frame CT stack using patient coordinates.

Geometry follows DICOM PS3.3 C.7.6.2. Only classic, non-tilted CT is supported.
"""
import io
import json
import zipfile
import nibabel as nib
import numpy as np
import pydicom


def assemble(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members=archive.infolist()
        if len(members)>1025 or sum(m.file_size for m in members)>256*1024*1024:
            raise ValueError('CT archive exceeds slice or expanded-byte bounds')
        if len({m.filename for m in members})!=len(members):
            raise ValueError('Duplicate ZIP entries')
        if archive.getinfo('manifest.json').file_size>4096:
            raise ValueError('CT manifest is too large')
        manifest=json.loads(archive.read('manifest.json'))
        if not isinstance(manifest,dict): raise ValueError('CT manifest must be an object')
        expected=manifest.get('expected_slices')
        slices=[m for m in members if not m.is_dir() and m.filename.lower().endswith('.dcm')]
        if type(expected) is not int or not 2<=expected<=1024 or len(slices)!=expected:
            raise ValueError('Complete series and expected_slices between 2 and 1024 are required')
        datasets=[pydicom.dcmread(io.BytesIO(archive.read(m))) for m in slices]
    first=datasets[0]
    orientation=np.asarray(first.ImageOrientationPatient,dtype=float)
    spacing=np.asarray(first.PixelSpacing,dtype=float)
    if orientation.shape!=(6,) or spacing.shape!=(2,) or not np.isfinite(orientation).all() or not np.isfinite(spacing).all() or min(spacing)<=0:
        raise ValueError('Invalid CT orientation or spacing')
    x,y=orientation[:3],orientation[3:]
    if not np.allclose([np.linalg.norm(x),np.linalg.norm(y),np.dot(x,y)],[1,1,0],atol=1e-4):
        raise ValueError('CT direction cosines must be orthonormal')
    normal=np.cross(x,y)
    shape=(int(first.Rows),int(first.Columns))
    if min(shape)<2 or shape[0]*shape[1]*expected>64_000_000:
        raise ValueError('CT voxel limit exceeded')
    identities=set(); located=[]
    for ds in datasets:
        if ds.Modality!='CT' or int(getattr(ds,'NumberOfFrames',1))!=1 or ds.PhotometricInterpretation!='MONOCHROME2' or 'LOCALIZER' in list(getattr(ds,'ImageType',[])):
            raise ValueError('Only single-frame CT without localizers is supported')
        for key in ('StudyInstanceUID','SeriesInstanceUID','FrameOfReferenceUID'):
            if not getattr(ds,key,None) or getattr(ds,key)!=getattr(first,key):
                raise ValueError('Mixed CT series or coordinate frames')
        if str(ds.SOPInstanceUID) in identities:
            raise ValueError('Duplicate CT instance')
        identities.add(str(ds.SOPInstanceUID))
        if (int(ds.Rows),int(ds.Columns))!=shape or not np.allclose(ds.ImageOrientationPatient,orientation,atol=1e-4) or not np.allclose(ds.PixelSpacing,spacing,atol=1e-4):
            raise ValueError('Inconsistent CT slice geometry')
        position=np.asarray(ds.ImagePositionPatient,dtype=float)
        if position.shape!=(3,) or not np.isfinite(position).all():
            raise ValueError('Invalid CT position')
        located.append((float(position@normal),position,ds))
    located.sort(key=lambda item:item[0])
    distances=np.diff([item[0] for item in located]); step=float(np.median(distances))
    if step<=0 or not np.allclose(distances,step,rtol=.01,atol=.01):
        raise ValueError('Duplicate, missing or irregularly spaced CT slices')
    origin=located[0][1]
    arrays=[]
    for index,(_,position,ds) in enumerate(located):
        if not np.allclose(position,origin+normal*step*index,atol=.05):
            raise ValueError('Gantry tilt or in-plane slice displacement is unsupported')
        if not hasattr(ds,'RescaleSlope') or not hasattr(ds,'RescaleIntercept') or str(getattr(ds,'RescaleType','HU')).upper()!='HU':
            raise ValueError('CT requires explicit HU rescaling')
        slope=float(ds.RescaleSlope); intercept=float(ds.RescaleIntercept)
        if not np.isfinite([slope,intercept]).all() or slope==0:
            raise ValueError('Invalid CT rescaling')
        values=np.asarray(ds.pixel_array,dtype=np.float32)*slope+intercept
        if values.shape!=shape or not np.isfinite(values).all():
            raise ValueError('Invalid CT pixels')
        arrays.append(values)
    affine=np.eye(4); affine[:3,0]=x*spacing[1]; affine[:3,1]=y*spacing[0]
    affine[:3,2]=normal*step; affine[:3,3]=origin
    affine=np.diag([-1,-1,1,1])@affine
    return nib.as_closest_canonical(nib.Nifti1Image(np.stack(arrays,axis=2).transpose(1,0,2),affine))
