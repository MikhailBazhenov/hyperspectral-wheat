"""Measure masked sample spectrum from quantitative NPZ cube (preferred)."""
from pathlib import Path
import cv2
import numpy as np
from spectral_cube_reflectance import load_quant_cube


def measure_plant_spectrum(cube_path, mask_path):
    cube, wavelengths, valid, data_type = load_quant_cube(cube_path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != cube.shape[:2]:
        raise ValueError(f'Bad or missing mask: {mask_path}')
    selected = mask > 0
    if not selected.any(): raise ValueError('Empty plant mask')
    result = dict(wavelength=[], mean_intensity=[], std_intensity=[], n_pixels=[])
    for i, wl in enumerate(wavelengths):
        good = selected & valid[:,:,i] & np.isfinite(cube[:,:,i])
        vals = cube[:,:,i][good].astype(np.float64)
        result['wavelength'].append(int(wl))
        result['mean_intensity'].append(float(vals.mean()) if vals.size else np.nan)
        result['std_intensity'].append(float(vals.std()) if vals.size else np.nan)
        result['n_pixels'].append(int(vals.size))
    return result
