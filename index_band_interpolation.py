"""Shared wavelength interpolation and availability logic for spectral indices."""
import numpy as np


def band_status(wavelengths, requested, interpolate=True):
    """Return (status, interpolated_centers, unavailable_centers).

    Exact bands are preferred. An interior missing band is linearly interpolated.
    Extrapolation is prohibited. `wavelengths` must be unique and finite.
    """
    w = np.asarray(wavelengths, dtype=float)
    if w.ndim != 1 or not len(w) or not np.isfinite(w).all() or len(np.unique(w)) != len(w):
        raise ValueError('Wavelengths must be a nonempty 1D array of unique finite numbers')
    w = np.sort(w)
    interpolated, unavailable = [], []
    for nm in requested:
        nm = float(nm)
        if np.any(np.isclose(w, nm, rtol=0, atol=1e-7)):
            continue
        if not interpolate or nm < w[0] or nm > w[-1]:
            unavailable.append(nm)
        else:
            interpolated.append(nm)
    return ('unavailable' if unavailable else 'interpolated' if interpolated else 'exact',
            interpolated, unavailable)


def sample_band(wavelengths, values, nm, axis=0, interpolate=True):
    """Select an exact channel or interpolate linearly; NaN when unavailable."""
    w = np.asarray(wavelengths, dtype=float)
    v = np.asarray(values)
    if v.shape[axis] != len(w):
        raise ValueError('Wavelength and spectral-axis lengths differ')
    order = np.argsort(w)
    w = w[order]
    v = np.take(v, order, axis=axis)
    nm = float(nm)
    exact = np.flatnonzero(np.isclose(w, nm, rtol=0, atol=1e-7))
    if exact.size:
        return np.take(v, int(exact[0]), axis=axis)
    shape = v.shape[:axis] + v.shape[axis+1:]
    if not interpolate or nm < w[0] or nm > w[-1]:
        return np.full(shape, np.nan, dtype=np.float32)
    k = int(np.searchsorted(w, nm))
    a, b = w[k-1], w[k]
    return (np.take(v, k-1, axis=axis).astype(np.float32) * ((b-nm)/(b-a)) +
            np.take(v, k, axis=axis).astype(np.float32) * ((nm-a)/(b-a)))
