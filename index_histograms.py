"""Histograms of valid spatial vegetation-index pixels (not RGB map colours)."""
import numpy as np


def histogram_columns(values, *, n_bins=10, sample='', mask='', index=''):
    """Return (wide columns, long rows) for object-specific equal-width bins.

    Intervals are [lower, upper), except the final bin is [lower, upper].
    With a constant-valued map, all valid pixels go into bin 1 (width 0).
    NaN/Inf and pixels outside the object mask must be filtered by the caller.
    """
    if isinstance(n_bins, bool) or int(n_bins) != n_bins or not 1 <= n_bins <= 100:
        raise ValueError('histogram_bins must be an integer in 1..100')
    n_bins = int(n_bins)
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n_valid = int(values.size)
    minimum = float(values.min()) if n_valid else float('nan')
    maximum = float(values.max()) if n_valid else float('nan')
    width = (maximum - minimum) / n_bins if n_valid else float('nan')
    if n_valid and minimum == maximum:
        # No non-zero bin width exists; a deterministic, explicit convention.
        edges = np.full(n_bins + 1, minimum)
        counts = np.zeros(n_bins, dtype=np.int64)
        counts[0] = n_valid
    elif n_valid:
        edges = np.linspace(minimum, maximum, n_bins + 1)
        counts, _ = np.histogram(values, bins=edges)
    else:
        edges = np.full(n_bins + 1, np.nan)
        counts = np.zeros(n_bins, dtype=np.int64)
    if int(counts.sum()) != n_valid:
        raise AssertionError('Histogram must account for every valid pixel')
    wide = {'histogram_bins': n_bins, 'histogram_bin_width': width}
    long = []
    for i in range(n_bins):
        n = int(counts[i])
        frac = n / n_valid if n_valid else float('nan')
        wide[f'bin_{i+1:02d}_count'] = n
        long.append({'sample': sample, 'mask': mask, 'index': index,
                     'bin_number': i + 1, 'bin_lower': float(edges[i]),
                     'bin_upper': float(edges[i + 1]),
                     'upper_inclusive': i == n_bins - 1,
                     'pixel_count': n, 'pixel_fraction': frac,
                     'n_valid_pixels': n_valid,
                     'observed_min': minimum, 'observed_max': maximum})
    return wide, long
