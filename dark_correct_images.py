"""Dark-frame averaging and subtraction for a wavelength-resolved image series.

Wavelengths use image<integer>.jpg/.jpeg/.png names. The average of >=2 dark
series is calculated independently, pixel by pixel, at each wavelength and is
saved as lossless grayscale PNG in the study's dark_average/ directory.

PNG is integer-valued: averaged pixel values are rounded to the nearest DN.
For 8-bit JPEG sources this is usually sufficient; it does *not* recover the
original linear sensor signal lost during JPEG encoding.

Dark subtraction can be pixelwise or wavelength-specific scalar (whole-frame
mean of the corresponding dark image). Negative results are clipped to zero,
as in v5, and the corrected images are written as 16-bit PNG.
"""
from __future__ import annotations

from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
import cv2

from correct_flat_field import _index_images, _read_gray

_AVERAGE_META = 'dark_average_manifest.json'
_CORRECTION_META = 'dark_correction_manifest.json'
_AVERAGE_SCHEMA = 1
_CORRECTION_SCHEMA = 1
_MODES = ('pixelwise', 'scalar')


def _file_info(path: Path) -> dict:
    path = Path(path).resolve()
    st = path.stat()
    return {'path': str(path), 'size': int(st.st_size), 'mtime_ns': int(st.st_mtime_ns)}


def _signature(image_folders) -> list:
    """Capture source identity and timestamps without reading large images."""
    return [
        {'folder': str(Path(folder).resolve()),
         'images': [{'wavelength': int(wl), **_file_info(path)}
                    for wl, path in _index_images(folder).items()]}
        for folder in image_folders
    ]


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def dark_average_ready(dark_folders, output_folder) -> bool:
    """True only if PNGs and the source manifest still match the input series."""
    folders = [Path(p).resolve() for p in dark_folders]
    out = Path(output_folder).resolve()
    if len(folders) < 2 or not out.is_dir():
        return False
    try:
        manifest = _read_json(out / _AVERAGE_META)
        if manifest is None or manifest.get('schema') != _AVERAGE_SCHEMA:
            return False
        if manifest.get('sources') != _signature(folders):
            return False
        bands = set(_index_images(folders[0]))
        return set(_index_images(out)) == bands
    except (OSError, ValueError):
        return False


def average_dark_folders(dark_folders, output_folder, *, overwrite=True):
    """Average >=2 dark Spectral_Cube folders, per wavelength and pixel.

    Returns (output_directory, diagnostics_dataframe). Input wavelengths and
    frame shapes must match exactly. Each output image is image<wl>.png.
    """
    folders = [Path(p).expanduser().resolve() for p in dark_folders]
    if len(folders) < 2:
        raise ValueError('Averaging requires at least two distinct dark folders')
    if len(set(folders)) != len(folders):
        raise ValueError('Duplicate dark folders supplied')
    out = Path(output_folder).expanduser().resolve()
    if out in folders:
        raise ValueError('dark_average must not be an input dark folder')
    image_sets = [_index_images(folder) for folder in folders]
    wavelengths = set(image_sets[0])
    for folder, images in zip(folders[1:], image_sets[1:]):
        if set(images) != wavelengths:
            raise ValueError(
                f'Dark wavelengths mismatch for {folder}: '
                f'missing={sorted(wavelengths - set(images))}, '
                f'extra={sorted(set(images) - wavelengths)}'
            )
    if not overwrite and dark_average_ready(folders, out):
        print(f'Using existing averaged dark frames: {out}', flush=True)
        report = out / 'dark_average_report.csv'
        return out, pd.read_csv(report) if report.is_file() else pd.DataFrame()

    out.mkdir(parents=True, exist_ok=True)
    # The source manifest is written LAST: an interrupted write is never reused.
    (out / _AVERAGE_META).unlink(missing_ok=True)
    rows = []
    for wl in sorted(wavelengths):
        acc = None
        dtype = None
        shape = None
        for images in image_sets:
            img = _read_gray(images[wl], expected_shape=shape)
            if img.dtype not in (np.dtype('uint8'), np.dtype('uint16')):
                raise TypeError(f'Expected uint8/uint16 grayscale dark image, got {img.dtype}: {images[wl]}')
            if shape is None:
                shape, dtype = img.shape, img.dtype
                acc = np.zeros(shape, dtype=np.float64)
            elif img.dtype != dtype:
                raise TypeError(f'Mixed bit depths for {wl} nm dark frames')
            acc += img  # exact integer addition in float64 for usual series sizes
        avg = acc / len(folders)
        dest = out / f'image{wl}.png'
        quantized = np.rint(avg).astype(dtype)
        if not cv2.imwrite(str(dest), quantized):
            raise OSError(f'Failed to save averaged dark image: {dest}')
        rows.append({
            'wavelength': wl,
            'n_dark_frames': len(folders),
            'height': int(shape[0]), 'width': int(shape[1]),
            'output_dtype': str(dtype),
            'mean_dark_DN_unrounded': float(avg.mean()),
            'mean_dark_DN_saved': float(quantized.mean()),
            'max_rounding_error_DN': float(np.max(np.abs(avg - quantized))),
            'output_file': dest.name,
        })
    # Clear any PNGs from previous averages with different wavelength lists.
    for path in out.glob('image*.png'):
        if path.name not in {f'image{wl}.png' for wl in wavelengths}:
            path.unlink()
    df = pd.DataFrame(rows)
    df.to_csv(out / 'dark_average_report.csv', index=False)
    _write_json_atomic(out / _AVERAGE_META, {
        'schema': _AVERAGE_SCHEMA, 'n_dark_folders': len(folders),
        'averaging': 'pixelwise_arithmetic_mean_per_wavelength',
        'png_rounding': 'nearest_integer_DN',
        'sources': _signature(folders),
    })
    print(f'Averaged {len(folders)} dark series -> {out} ({len(wavelengths)} bands)', flush=True)
    return out, df


def dark_correction_ready(images_folder, dark_folder, output_folder, *, mode='pixelwise') -> bool:
    """Check bandset, method, and source metadata before reusing corrected PNGs."""
    if mode not in _MODES:
        raise ValueError(f'mode must be one of {_MODES}, not {mode!r}')
    out = Path(output_folder)
    if not out.is_dir():
        return False
    try:
        meta = _read_json(out / _CORRECTION_META)
        return (
            meta is not None and meta.get('schema') == _CORRECTION_SCHEMA and
            meta.get('mode') == mode and
            meta.get('sources') == _signature([images_folder, dark_folder]) and
            set(_index_images(out)) == set(_index_images(images_folder))
        )
    except (OSError, ValueError):
        return False


def dark_correct_images(images_folder, dark_folder, output_folder, *,
                        overwrite=True, mode='pixelwise'):
    """Subtract matched darks using full images or one scalar per wavelength.

    mode='pixelwise': O(x,y,wl) - D(x,y,wl)
    mode='scalar':    O(x,y,wl) - mean_xy[D(x,y,wl)]
    Returns (output_directory, diagnostics_dataframe).
    """
    if mode not in _MODES:
        raise ValueError(f'mode must be one of {_MODES}, not {mode!r}')
    sources = _index_images(images_folder)
    darks = _index_images(dark_folder)
    if set(sources) != set(darks):
        raise ValueError(f'Dark wavelengths mismatch: missing={sorted(set(sources)-set(darks))}; extra={sorted(set(darks)-set(sources))}')
    out = Path(output_folder).expanduser().resolve()
    if out in (Path(images_folder).resolve(), Path(dark_folder).resolve()):
        raise ValueError('Output folder cannot be an input folder')
    if not overwrite and dark_correction_ready(images_folder, dark_folder, out, mode=mode):
        report = out / 'dark_correction_diagnostics.csv'
        return out, pd.read_csv(report) if report.is_file() else pd.DataFrame()
    out.mkdir(parents=True, exist_ok=True)
    (out / _CORRECTION_META).unlink(missing_ok=True)
    diagnostics = []
    for wl, src in sources.items():
        dest = out / f'image{wl}.png'
        a = _read_gray(src)
        d = _read_gray(darks[wl], expected_shape=a.shape)
        if a.dtype not in (np.dtype('uint8'), np.dtype('uint16')) or d.dtype not in (np.dtype('uint8'), np.dtype('uint16')):
            raise TypeError('Expected uint8 or uint16 grayscale source/dark images')
        dark_mean = float(np.mean(d, dtype=np.float64))
        subtraction = dark_mean if mode == 'scalar' else d.astype(np.float32)
        diff = a.astype(np.float32) - subtraction
        clipped_fraction = float((diff < 0).mean())
        # PNG is integer-valued: round the scalar-corrected result to the nearest DN.
        # Pixelwise subtraction of integer images is unaffected by this rounding.
        corrected = np.rint(np.clip(diff, 0, 65535)).astype(np.uint16)
        if not cv2.imwrite(str(dest), corrected):
            raise OSError(f'Cannot save {dest}')
        diagnostics.append({
            'wavelength': wl,
            'mode': mode,
            'mean_dark_DN': dark_mean,
            'negative_fraction_clipped': clipped_fraction,
            'output': str(dest),
        })
    df = pd.DataFrame(diagnostics)
    df.to_csv(out / 'dark_correction_diagnostics.csv', index=False)
    _write_json_atomic(out / _CORRECTION_META, {
        'schema': _CORRECTION_SCHEMA, 'mode': mode,
        'sources': _signature([images_folder, dark_folder]),
    })
    return out, df
