"""Bounded NIfTI decoding and consistently oriented axial PNG slices."""
import gzip
import io

import nibabel as nib
import numpy as np
from PIL import Image


def validate_volume(raw):
    # Limit expanded bytes before parsing compressed files supplied by clients.
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as source:
        expanded = source.read(256 * 1024 * 1024 + 1)
    if len(expanded) > 256 * 1024 * 1024:
        raise ValueError('Expanded CT volume exceeds 256 MB')
    try:
        volume = nib.Nifti1Image.from_bytes(expanded)
        if len(volume.shape) != 3 or min(volume.shape) < 2 or np.prod(volume.shape) > 64_000_000:
            raise ValueError('Expected a 3D volume with at most 64 million voxels')
        affine = volume.affine
        if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3,:3])) < 1e-8:
            raise ValueError('Invalid voxel geometry')
        data = volume.get_fdata(dtype=np.float32)
        if not np.isfinite(data).all():
            raise ValueError('CT contains nonfinite intensities')
        clean = nib.Nifti1Image(data, affine)
        return nib.as_closest_canonical(clean)
    except (nib.filebasedimages.ImageFileError, KeyError) as e:
        raise ValueError('Invalid NIfTI volume') from e


def slice_png(path, index, overlay=False):
    volume = nib.load(str(path))
    # Canonical RAS: display anterior at top; image and mask share this transform.
    values = np.asarray(volume.dataobj[:, :, index]).T[::-1, ::-1]
    if overlay:
        rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
        rgba[..., 0], rgba[..., 1] = 255, 150
        rgba[..., 3] = (values > .5).astype(np.uint8) * 180
        image = Image.fromarray(rgba)
    else:
        image = Image.fromarray((np.clip((values + 160) / 400, 0, 1) * 255).astype(np.uint8))
    # Preserve physical in-plane aspect ratio for anisotropic volumes.
    spacing = nib.affines.voxel_sizes(volume.affine)
    width = min(1024, image.width)
    height = max(1, min(2048, round(image.height * width / image.width * spacing[1] / spacing[0])))
    image = image.resize((width, height), resample=Image.Resampling.NEAREST if overlay else Image.Resampling.BILINEAR)
    stream = io.BytesIO(); image.save(stream, format='PNG')
    return stream.getvalue()
