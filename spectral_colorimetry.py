"""Spectral CIELCh(ab) colour coordinates from calibrated reflectance.

Implements the user's spectrum_to_lch example with CIE 1931 2-degree observer,
D65 (or E) and a *matching truncated-grid reference white*. MUSES9-HS has
400..1000 nm; we integrate 400..780 nm, not the full canonical 380..780 nm.
Accordingly, hue is an approximate instrumental spectral-colour descriptor.

This is NOT the HSV hue of the stretch-RGB image used for mask segmentation.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=12)
def _weights(wavelengths: tuple, illuminant: str):
    try:
        import colour
    except ImportError as exc:
        raise ImportError('Hue_LCh requires colour-science: pip install colour-science') from exc
    wl = np.asarray(wavelengths, dtype=float)
    if len(wl) < 2 or np.any(np.diff(wl) <= 0) or not np.allclose(np.diff(wl), np.diff(wl)[0]):
        raise ValueError('CIELCh wavelength samples must be evenly spaced ascending')
    if wl[0] < 360 or wl[-1] > 780:
        raise ValueError('CIELCh supports observed wavelengths from 360 to 780 nm')
    shape = colour.SpectralShape(float(wl[0]), float(wl[-1]), float(wl[1] - wl[0]))
    cmf = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer'].copy().align(shape).values
    illuminant = illuminant.upper()
    if illuminant == 'D65':
        E = colour.SDS_ILLUMINANTS['D65'].copy().align(shape).values
    elif illuminant == 'E':
        E = np.ones_like(wl)
    else:
        raise ValueError('illuminant must be D65 or E')
    normalization = np.sum(E * cmf[:, 1])
    if normalization <= 0:
        raise ValueError('Invalid white reference')
    spectral_weights = (E[:, None] * cmf) / normalization
    xyz_white = spectral_weights.sum(axis=0)
    return spectral_weights, xyz_white


def _colour_slices(wavelengths):
    wl = np.asarray(wavelengths, dtype=float)
    if wl.ndim != 1:
        raise ValueError('Wavelengths must be one-dimensional')
    selection = (wl >= 400) & (wl <= 780)
    selected = wl[selection]
    if len(selected) < 2 or selected[0] > 400 or selected[-1] < 780:
        raise ValueError('Hue_LCh needs coverage of 400..780 nm; no extrapolation')
    if not np.allclose(np.diff(selected), np.diff(selected)[0]):
        raise ValueError('Hue_LCh needs uniformly sampled wavelengths 400..780')
    return selection, tuple(float(n) for n in selected)


def _xyz_to_lch(XYZ, white):
    # Implements exactly the piecewise Lab conversion in spectrum_to_lch(1).py.
    t = np.asarray(XYZ, dtype=float) / white
    delta = 6 / 29
    f = np.where(t > delta ** 3, np.cbrt(np.maximum(t, 0)),
                 t / (3 * delta ** 2) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    C = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a)) % 360
    hue = np.where(C < 1e-6, np.nan, hue)
    return L, C, hue


def spectral_lch(wavelengths, reflectance, *, illuminant='D65', valid_mask=None,
                 block_rows=128):
    """Return (L,C,h) over wavelength last axis; NaN if any required channel invalid.

    reflectance: (bands,), (objects,bands) or (height,width,bands).
    Negative or nonfinite bands are invalid; no clipping of R>1.
    """
    select, keys = _colour_slices(wavelengths)
    matrix, white = _weights(keys, illuminant.upper())
    R = np.asarray(reflectance)
    if R.shape[-1] != len(wavelengths):
        raise ValueError('Reflectance last dimension must match wavelengths')
    if valid_mask is not None and np.asarray(valid_mask).shape != R.shape:
        raise ValueError('valid_mask must match reflectance shape')
    if R.ndim in (1, 2):
        z = np.asarray(R[..., select], dtype=np.float64)
        good = np.isfinite(z).all(axis=-1) & (z >= 0).all(axis=-1)
        if valid_mask is not None:
            good &= np.asarray(valid_mask)[..., select].all(axis=-1)
        XYZ = np.einsum('...b,bc->...c', np.where(np.isfinite(z), z, 0), matrix)
        L, C, h = _xyz_to_lch(XYZ, white)
        return tuple(np.where(good, result, np.nan) for result in (L, C, h))
    if R.ndim != 3:
        raise ValueError('reflectance must have 1, 2 or 3 dimensions')
    h, w = R.shape[:2]
    L, C, Hue = (np.full((h, w), np.nan, np.float32) for _ in range(3))
    for begin in range(0, h, block_rows):
        end = min(begin + block_rows, h)
        z = np.asarray(R[begin:end, :, select], dtype=np.float32)
        good = np.isfinite(z).all(axis=-1) & (z >= 0).all(axis=-1)
        if valid_mask is not None:
            good &= np.asarray(valid_mask)[begin:end, :, select].all(axis=-1)
        xyz = np.einsum('hwb,bc->hwc', np.where(np.isfinite(z), z, 0), matrix)
        ls, cs, hs = _xyz_to_lch(xyz, white)
        for dest, values in ((L, ls), (C, cs), (Hue, hs)):
            dest[begin:end] = np.where(good, values, np.nan)
    return L, C, Hue


def spectral_hue(wavelengths, reflectance, *, illuminant='D65', valid_mask=None):
    return spectral_lch(wavelengths, reflectance, illuminant=illuminant,
                        valid_mask=valid_mask)[2]
