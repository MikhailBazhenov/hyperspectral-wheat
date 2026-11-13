"""Spectral variants and mean normalization with explicit NaN/zero handling."""
from __future__ import annotations
import numpy as np

SPECTRAL_VARIANTS = ('raw', 'raw_normalized', 'savgol', 'savgol_normalized')


def mean_normalize(spectrum):
    """Divide each individual wavelength vector by its own finite-band mean.

    Accepts 1D spectrum or matrix (n_wavelengths, n_objects). A column with
    no finite bands or nonpositive mean is all NaN. No extrapolation/interpolation.
    """
    a = np.asarray(spectrum, dtype=np.float64)
    if a.ndim not in (1, 2):
        raise ValueError('Expected 1D or 2D spectra with wavelengths first')
    good = np.isfinite(a)
    n = good.sum(axis=0)
    mean = np.divide(np.where(good, a, 0).sum(axis=0), n,
                     out=np.full_like(np.asarray(n, dtype=float), np.nan), where=n > 0)
    safe = np.isfinite(mean) & (mean > 1e-12)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(safe, a / mean, np.nan)
