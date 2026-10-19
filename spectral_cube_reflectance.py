"""Radiometric cube generation from aligned bands (no P5/P95 transform).

Modes:
 - flat + Spectralon: R = (O/F) * mean(F)/S
 - Spectralon flat:   R = O/F
 - Spectralon only:   R = O/S (no spatial correction)
 - No reference:      R = O (raw signal, NOT reflectance)
S is Spectralon DN (dark corrected if appropriate) from a masked reference,
mean(F) computed in valid region. Assumes matching exposure and illumination,
spatially uniform flat reflectance, and Spectralon rho=1 by convention.
Results are only relative reflectances when such assumptions hold.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import cv2
from correct_flat_field import _index_images, _read_gray

# Defaults can be configured once in a notebook without changing the pipeline.
DEFAULT_PREVIEW_FORMAT = "png"
DEFAULT_JPEG_QUALITY = 95

def configure_preview_format(preview_format="png", jpeg_quality=95):
    """Set preview settings for subsequent generate_reflectance_cube calls.

    Handy when process_wheat_study invokes generate_reflectance_cube internally.
    """
    global DEFAULT_PREVIEW_FORMAT, DEFAULT_JPEG_QUALITY
    fmt = str(preview_format).lower().lstrip(".")
    if fmt == "jpeg":
        fmt = "jpg"
    if fmt not in ("png", "jpg"):
        raise ValueError("preview_format must be png or jpg")
    if isinstance(jpeg_quality, bool) or not isinstance(jpeg_quality, int) or not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be an integer between 1 and 100")
    DEFAULT_PREVIEW_FORMAT = fmt
    DEFAULT_JPEG_QUALITY = jpeg_quality



def load_quant_cube(path):
    with np.load(path, allow_pickle=False) as dat:
        cube = dat['cube'].astype(np.float32)
        wavelengths = dat['wavelengths'].astype(np.int32)
        valid_mask = dat['valid_mask'].astype(bool)
        data_type = str(dat['data_type'].item()) if 'data_type' in dat.files else 'signal'
    if valid_mask.shape != cube.shape:
        raise ValueError(f'Invalid valid_mask shape in {path}')
    return cube, wavelengths, valid_mask, data_type


def generate_reflectance_cube(
    images_folder, output_folder, *, flat_folder=None, spectralon_csv=None,
    flat_is_spectralon=False, spatial_correction=True, spectral_correction=True,
    min_flat_fraction=0.01, save_float_cube=True, cube_dtype='float16',
    preview_percentiles=(1, 99), preview_format=None, jpeg_quality=None,
):
    if preview_format is None:
        preview_format = DEFAULT_PREVIEW_FORMAT
    if jpeg_quality is None:
        jpeg_quality = DEFAULT_JPEG_QUALITY
    preview_format = str(preview_format).lower().lstrip(".")
    if preview_format == "jpeg":
        preview_format = "jpg"
    if preview_format not in ("png", "jpg"):
        raise ValueError("preview_format must be png or jpg")
    if isinstance(jpeg_quality, bool) or not isinstance(jpeg_quality, int) or not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be an integer between 1 and 100")
    files = _index_images(images_folder)
    flat = _index_images(flat_folder) if flat_folder is not None else None
    if spatial_correction and flat is None:
        raise ValueError('Spatial correction needs flat_folder')
    if flat is not None and spatial_correction and set(files) != set(flat):
        raise ValueError('Object/flat wavelengths differ')
    if spectral_correction and not flat_is_spectralon and spectralon_csv is None:
        raise ValueError('Spectral correction requires spectralon_csv or flat_is_spectralon')
    if flat_is_spectralon and (flat is None or not spatial_correction):
        raise ValueError('flat_is_spectralon needs flat_folder and spatial_correction=True')
    if cube_dtype not in ('float16','float32'):
        raise ValueError('cube_dtype must be float16 or float32')
    S = None
    if spectral_correction and not flat_is_spectralon:
        df = pd.read_csv(spectralon_csv)
        if not {'wavelength','mean_intensity'}.issubset(df.columns):
            raise ValueError('Spectralon CSV needs wavelength, mean_intensity')
        if df['wavelength'].duplicated().any():
            raise ValueError('Duplicate Spectralon wavelengths')
        S = dict(zip(df.wavelength.astype(int), df.mean_intensity.astype(float)))
    output = Path(output_folder)
    if output.resolve() == Path(images_folder).resolve():
        raise ValueError('Output folder must differ from images_folder')
    output.mkdir(parents=True, exist_ok=True)
    arrays, masks, report = [], [], []
    shape = None
    for wl, source in files.items():
        original = _read_gray(source)
        if shape is None: shape = original.shape
        if original.shape != shape: raise ValueError('Inconsistent image sizes')
        O = original.astype(np.float32)
        # Zero fill after affine alignment is invalid; this also excludes genuine zeros.
        valid = np.isfinite(O) & (O > 0)
        C = O.copy()
        mean_flat = np.nan
        if spatial_correction:
            F = _read_gray(flat[wl], expected_shape=shape).astype(np.float32)
            positives = F[F > 0]
            if not positives.size: raise ValueError(f'Flat has no positive values at {wl}')
            bound = max(float(np.median(positives))*min_flat_fraction, 1e-6)
            valid_flat = np.isfinite(F) & (F > bound)
            mean_flat = float(np.mean(F[valid_flat])) if valid_flat.any() else np.nan
            valid &= valid_flat
            if not np.isfinite(mean_flat): raise ValueError(f'No valid pixels at {wl}')
            if flat_is_spectralon and spectral_correction:
                # Large uniform Spectralon flat: divide pixelwise directly.
                C = np.divide(O,F,out=np.full_like(O,np.nan),where=valid)
            else:
                C = np.divide(O*mean_flat,F,out=np.full_like(O,np.nan),where=valid)
        if spectral_correction and not flat_is_spectralon:
            if wl not in S or not np.isfinite(S[wl]) or S[wl] <= 0:
                raise ValueError(f'No positive Spectralon signal for {wl} nm')
            C = C / float(S[wl])
        C[~valid] = np.nan
        arrays.append(C)
        masks.append(valid & np.isfinite(C))
        report.append({'wavelength':wl,'n_valid':int(np.isfinite(C).sum()),'mean_flat':mean_flat,
                       'spectralon_signal':S[wl] if S is not None else np.nan})
    cube = np.stack(arrays,axis=-1).astype(np.float32)
    valid_mask = np.stack(masks,axis=-1)
    wavelengths = np.array(list(files), dtype=np.int32)
    data_type = 'reflectance' if spectral_correction else 'signal'
    # No per-band or global *additive* normalization of numeric cube.
    if save_float_cube:
        path = output/'corrected_cube.npz'
        np.savez_compressed(path, cube=cube.astype(cube_dtype), wavelengths=wavelengths,
                            valid_mask=valid_mask, data_type=np.array(data_type))
    # Display-only images: use one global non-linear-free scaling with percentiles;
    # independent band scaling is intentionally avoided.
    finite = cube[np.isfinite(cube)]
    if not finite.size: raise ValueError('No valid corrected pixels')
    lo, hi = np.percentile(finite, preview_percentiles)
    if hi <= lo: hi = lo + 1e-6
    # Remove stale previews from the alternate format to avoid duplicate bands
    # (the downstream readers require exactly one image per wavelength).
    old_exts = ("jpg", "jpeg") if preview_format == "png" else ("png", "jpeg")
    for wl in wavelengths:
        for ext in old_exts:
            old = output / f"image{wl}.{ext}"
            if old.exists():
                old.unlink()
    for i, wl in enumerate(wavelengths):
        image = (cube[:,:,i]-lo)*255.0/(hi-lo)
        image = np.clip(np.nan_to_num(image,nan=0.0,posinf=255,neginf=0),0,255).astype(np.uint8)
        destination = output / f'image{wl}.{preview_format}'
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if preview_format == 'jpg' else []
        if not cv2.imwrite(str(destination), image, params):
            raise OSError(f'Cannot save {destination}')
    diag = pd.DataFrame(report)
    diag.to_csv(output/'radiometric_diagnostics.csv', index=False)
    (output/'radiometry_metadata.json').write_text(json.dumps({
        'data_type':data_type,'flat_is_spectralon':flat_is_spectralon,
        'spatial_correction':spatial_correction,'spectral_correction':spectral_correction,
        'preview_percentiles':list(preview_percentiles),
        'preview_format':preview_format,'jpeg_quality':jpeg_quality if preview_format == 'jpg' else None,
        'note':'Spectralon reference set to 1; not certified absolute reflectance.'},indent=2),encoding='utf-8')
    return output, diag
