"""Wavelength interpolation and efficient access to spectral cube bands.

Only the requested 2-D bands are extracted. In particular, we NEVER reorder
and copy an entire H x W x N cube merely to access one spectral wavelength.
"""
from collections import OrderedDict
import numpy as np

_ATOL = 1e-7


def _validate(wavelengths):
    w = np.asarray(wavelengths, dtype=float)
    if w.ndim != 1 or not len(w) or not np.isfinite(w).all() or len(np.unique(w)) != len(w):
        raise ValueError('Wavelengths must be a nonempty 1D array of unique finite numbers')
    order = np.argsort(w)
    return w[order], order


def _band_location(w, order, nm, interpolate=True):
    """Return ('exact', original_index) or ('interpolated', i1, i2, a, b).

    Interpolation expression is a*band1+b*band2. No extrapolation.
    """
    nm = float(nm)
    hits = np.flatnonzero(np.isclose(w, nm, rtol=0, atol=_ATOL))
    if hits.size:
        return ('exact', int(order[int(hits[0])]))
    if not interpolate or nm < w[0] or nm > w[-1]:
        return ('missing',)
    k = int(np.searchsorted(w, nm))
    if not 0 < k < len(w):
        return ('missing',)
    low, high = w[k - 1], w[k]
    return ('interpolated', int(order[k - 1]), int(order[k]),
            (high - nm) / (high - low), (nm - low) / (high - low))


def band_status(wavelengths, requested, interpolate=True):
    """Return (status, interpolated_centers, unavailable_centers)."""
    w, order = _validate(wavelengths)
    interpolated, unavailable = [], []
    for nm in requested:
        state = _band_location(w, order, nm, interpolate)[0]
        if state == 'interpolated':
            interpolated.append(float(nm))
        elif state == 'missing':
            unavailable.append(float(nm))
    return ('unavailable' if unavailable else 'interpolated' if interpolated else 'exact',
            interpolated, unavailable)


def sample_band(wavelengths, values, nm, axis=0, interpolate=True):
    """Select a view of an existing channel, or interpolate two source channels.

    Works even if the input wavelengths are unsorted and for any spectral axis.
    Out-of-range requests return NaNs of the correct spatial shape.
    """
    w, order = _validate(wavelengths)
    v = np.asarray(values)
    axis = np.core.numeric.normalize_axis_index(axis, v.ndim)
    if v.shape[axis] != len(w):
        raise ValueError('Wavelength and spectral-axis lengths differ')
    loc = _band_location(w, order, nm, interpolate)
    if loc[0] == 'exact':
        return np.take(v, loc[1], axis=axis)
    shape = v.shape[:axis] + v.shape[axis + 1:]
    if loc[0] == 'missing':
        return np.full(shape, np.nan, dtype=np.float32)
    _, i, j, a, b = loc
    # Cast before multiplication to avoid float16 arithmetic on persisted cubes.
    left = np.take(v, i, axis=axis).astype(np.float32, copy=False)
    right = np.take(v, j, axis=axis).astype(np.float32, copy=False)
    return left * a + right * b


class BandSampler:
    """Per-cube band reader with precomputed locations and bounded LRU cache.

    Existing bands are returned as views; interpolated bands need a single new
    H x W float32 array. A small LRU avoids keeping all wavelengths in RAM.
    """
    def __init__(self, wavelengths, cube, *, axis=2, interpolate=True, max_cached_bands=12):
        self.w, self.order = _validate(wavelengths)
        self.cube = np.asarray(cube)
        self.axis = np.core.numeric.normalize_axis_index(axis, self.cube.ndim)
        if self.cube.shape[self.axis] != len(self.w):
            raise ValueError('Wavelength and spectral-axis lengths differ')
        if max_cached_bands < 0:
            raise ValueError('max_cached_bands must be >= 0')
        self.interpolate = interpolate
        self.limit = int(max_cached_bands)
        self._locations = {}
        self._cache = OrderedDict()

    def location(self, nm):
        nm = float(nm)
        if nm not in self._locations:
            self._locations[nm] = _band_location(self.w, self.order, nm, self.interpolate)
        return self._locations[nm]

    def __call__(self, nm):
        nm = float(nm)
        if nm in self._cache:
            self._cache.move_to_end(nm)
            return self._cache[nm]
        loc = self.location(nm)
        if loc[0] == 'exact':
            # Basic slice indexing returns a view and does not copy the whole cube.
            sl = [slice(None)] * self.cube.ndim
            sl[self.axis] = loc[1]
            result = self.cube[tuple(sl)]
        elif loc[0] == 'missing':
            shape = self.cube.shape[:self.axis] + self.cube.shape[self.axis+1:]
            result = np.full(shape, np.nan, dtype=np.float32)
        else:
            _, i, j, a, b = loc
            sl = [slice(None)] * self.cube.ndim
            sl[self.axis] = i
            left = self.cube[tuple(sl)]
            sl[self.axis] = j
            right = self.cube[tuple(sl)]
            # Efficient float32 result; matching the old interpolation formula.
            result = (left.astype(np.float32, copy=False) * a +
                      right.astype(np.float32, copy=False) * b)
        if self.limit:
            self._cache[nm] = result
            if len(self._cache) > self.limit:
                self._cache.popitem(last=False)
        return result
