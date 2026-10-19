"""Subtract matched dark frames from RAW spectral images before alignment.

Output is lossless 16-bit PNG (image<wl>.png). Clipping below zero is recorded.
Use matching wavelength and identical sensor settings/exposure. JPG input remains
quantized/compressed; quantitative workflows should preferably use raw PNG/TIFF.
"""
from pathlib import Path
import numpy as np
import cv2
from correct_flat_field import _index_images, _read_gray


def dark_correct_images(images_folder, dark_folder, output_folder, *, overwrite=True):
    sources = _index_images(images_folder)
    darks = _index_images(dark_folder)
    if set(sources) != set(darks):
        raise ValueError(f'Dark wavelengths mismatch: missing={sorted(set(sources)-set(darks))}; extra={sorted(set(darks)-set(sources))}')
    out = Path(output_folder)
    if out.resolve() in (Path(images_folder).resolve(), Path(dark_folder).resolve()):
        raise ValueError('Output folder cannot be one of the input folders')
    out.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    for wl, src in sources.items():
        dest = out / f'image{wl}.png'
        if dest.is_file() and not overwrite:
            continue
        a = _read_gray(src)
        d = _read_gray(darks[wl], expected_shape=a.shape)
        if not np.issubdtype(a.dtype, np.integer) or not np.issubdtype(d.dtype, np.integer):
            raise TypeError('Expected integer-valued source/dark images')
        diff = a.astype(np.float32)-d.astype(np.float32)
        negative_fraction = float((diff < 0).mean())
        # Preserve possible 16-bit dynamic range; no contrast normalization.
        clipped = np.clip(diff, 0, 65535).astype(np.uint16)
        if not cv2.imwrite(str(dest), clipped):
            raise OSError(f'Cannot save {dest}')
        diagnostics.append({'wavelength':wl,'negative_fraction_clipped':negative_fraction,'output':str(dest)})
    import pandas as pd
    df = pd.DataFrame(diagnostics)
    if len(df):
        df.to_csv(out/'dark_correction_diagnostics.csv', index=False)
    return out, df
