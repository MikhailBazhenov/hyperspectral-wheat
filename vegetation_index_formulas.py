"""Pixelwise index math reusing the exact published implementation from project."""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from pathlib import Path
from calculate_vegetation_indices import INDEX_INFO
from index_band_interpolation import sample_band
_SIGNAL_OK = {k for k, v in INDEX_INFO.items() if v[3]} | {'REP'}

def _div(a, b):
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return np.where(np.abs(b) > 1e-12, a / b, np.nan)


def _make_R_function(wavelengths: np.ndarray, cube: np.ndarray, interpolate=True, allow_extrapolation=False):
    """Return a callable R(nm) -> HxW float32 map, cached by wavelength."""
    cache = {}
    def R(nm):
        if nm not in cache:
            cache[nm] = sample_band(wavelengths, cube, nm, axis=2, interpolate=interpolate)
        return cache[nm]
    return R


def compute_index_maps(wavelengths, cube, data_type='signal', interpolate=True, allow_extrapolation=False):
    if allow_extrapolation:
        raise ValueError('Index calculation never extrapolates beyond measured wavelengths')
    if data_type not in ('signal', 'signal_exploratory', 'reflectance'):
        raise ValueError("data_type must be 'signal', 'signal_exploratory' or 'reflectance'")
    if data_type == 'signal_exploratory':
        warnings.warn(
            'Exploratory mode: indices requiring reflectance or absolute signal scale '
            'are calculated numerically from DN; interpret them with caution.',
            UserWarning, stacklevel=2)
    if data_type == 'reflectance':
        finite = cube[np.isfinite(cube)]
        if finite.size and finite.min() < 0:
            raise ValueError('Reflectance mode expects nonnegative values')

    R = _make_R_function(wavelengths, cube, interpolate=interpolate, allow_extrapolation=allow_extrapolation)
    req_wls = sorted({v for entry in INDEX_INFO.values() for v in entry[2]})
    r = {v: R(v) for v in req_wls}
    x = lambda n: r[n]
    nd = lambda a, b: _div(x(a) - x(b), x(a) + x(b))

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        mcari = ((x(700) - x(670)) - 0.2 * (x(700) - x(550))) * _div(x(700), x(670))
        tcari = 3 * ((x(700) - x(670)) - 0.2 * (x(700) - x(550)) * _div(x(700), x(670)))
        osavi = _div(1.16 * (x(800) - x(670)), x(800) + x(670) + 0.16)
        radicand = (2 * x(800) + 1) ** 2 - 8 * (x(800) - x(670))
        values = {
            'NDVI': nd(800, 680),
            'NDVI_670': nd(800, 670),
            'GNDVI': nd(800, 550),
            'NDRE': nd(790, 720),
            'NDVI705': nd(750, 705),
            'CIgreen': _div(x(790), x(550)) - 1,
            'CIrededge': _div(x(790), x(710)) - 1,
            'SR': _div(x(800), x(680)),
            'SR705': _div(x(750), x(710)),
            'VOG1': _div(x(740), x(720)),
            'MTCI': _div(x(750) - x(710), x(710) - x(680)),
            'PRI': nd(531, 570),
            'PRI570_530': nd(530, 570),
            'PSRI': _div(x(680) - x(500), x(750)),
            'SIPI': _div(x(800) - x(445), x(800) - x(680)),
            'ARI1': _div(1, x(550)) - _div(1, x(700)),
            'ARI2': x(800) * (_div(1, x(550)) - _div(1, x(700))),
            'CRI1': _div(1, x(510)) - _div(1, x(550)),
            'CRI2': _div(1, x(510)) - _div(1, x(700)),
            'WBI': _div(x(900), x(970)),
            'NRI': nd(570, 670),
            'RGI': _div(x(690), x(550)),
            'R750_R700': _div(x(750), x(700)),
            'mND705': _div(x(750) - x(705), x(750) + x(705) - 2 * x(445)),
            'MSR705': _div(x(750) - x(445), x(705) - x(445)),
            'REP': 700 + 40 * _div((x(670) + x(780)) / 2 - x(700), x(740) - x(700)),
            'MCARI': mcari,
            'TCARI': tcari,
            'TVI': 0.5 * (120 * (x(750) - x(550)) - 200 * (x(670) - x(550))),
            'DVI': x(800) - x(680),
            'RDVI': _div(x(800) - x(670), np.sqrt(np.maximum(x(800) + x(670), 0))),
            'OSAVI': osavi,
            'SAVI': _div(1.5 * (x(800) - x(680)), x(800) + x(680) + 0.5),
            'EVI': _div(2.5 * (x(800) - x(680)), x(800) + 6 * x(680) - 7.5 * x(450) + 1),
            'MSAVI': (2 * x(800) + 1 - np.sqrt(np.where(radicand >= 0, radicand, np.nan))) / 2,
            'MCARI_OSAVI': _div(mcari, osavi),
            'TCARI_OSAVI': _div(tcari, osavi),
        }

    if data_type == 'signal':
        for name in list(values):
            if name not in _SIGNAL_OK:
                values[name] = np.full(cube.shape[:2], np.nan, dtype=np.float32)

    for name, arr in values.items():
        arr = arr.astype(np.float32, copy=False)
        arr[~np.isfinite(arr)] = np.nan
        values[name] = arr
    return values


def save_map(
    arr: np.ndarray,
    out_path: Path,
    title: str,
    cmap='viridis',
    robust_percentiles=(2, 98),
    dpi=180,
    *,
    vmin=None,
    vmax=None,
):
    """Save a PNG index map with an explicit, study-wide color scale.

    vmin/vmax take priority over robust_percentiles. Masked or non-finite
    pixels are shown in black. Use a separate all-white image for indices
    unavailable owing to an out-of-range wavelength.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(arr, dtype=np.float32)
    good = np.isfinite(data)

    # An entirely unavailable or invalid map is a plain white image of exactly
    # the same dimensions as the original spatial cube, without axes or margins.
    if not good.any():
        from PIL import Image
        Image.new('RGB', (data.shape[1], data.shape[0]), 'white').save(out_path)
        return

    vals = data[good]
    if vmin is None or vmax is None:
        lo, hi = np.nanpercentile(vals, robust_percentiles)
        if lo == hi:
            lo, hi = np.nanmin(vals), np.nanmax(vals)
        if lo == hi:
            lo, hi = lo - 1e-6, hi + 1e-6
        vmin, vmax = float(lo), float(hi)
    if not (np.isfinite(vmin) and np.isfinite(vmax) and float(vmin) < float(vmax)):
        raise ValueError(f'Invalid color bounds: {vmin}, {vmax}')

    palette = plt.get_cmap(cmap).copy()
    palette.set_bad('black')
    # Linear normalization, identical for all samples of one index.
    norm = colors.Normalize(vmin=float(vmin), vmax=float(vmax), clip=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(np.ma.masked_invalid(data), cmap=palette, norm=norm)
    ax.set_title(title)
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def compute_single_index_map(wavelengths, cube, index, data_type='signal', interpolate=True):
    """Calculate ONE spectral index without holding all 37 full-size maps in RAM.

    Uses the same implemented expressions as compute_index_maps. This is the
    preferred API for batch map creation on large hyperspectral cubes.
    """
    if index not in INDEX_INFO:
        raise KeyError(f'Unknown vegetation index: {index}')
    if data_type not in ('signal', 'signal_exploratory', 'reflectance'):
        raise ValueError('Unknown data_type')
    if data_type == 'signal' and index not in _SIGNAL_OK:
        return np.full(cube.shape[:2], np.nan, dtype=np.float32)

    R = _make_R_function(wavelengths, cube, interpolate=interpolate)
    x = lambda n: R(n)
    nd = lambda a,b: _div(x(a)-x(b), x(a)+x(b))
    mcari = lambda: ((x(700)-x(670))-0.2*(x(700)-x(550)))*_div(x(700),x(670))
    tcari = lambda: 3*((x(700)-x(670))-0.2*(x(700)-x(550))*_div(x(700),x(670)))
    osavi = lambda: _div(1.16*(x(800)-x(670)), x(800)+x(670)+0.16)

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        if index == 'NDVI': value = nd(800,680)
        elif index == 'NDVI_670': value = nd(800,670)
        elif index == 'GNDVI': value = nd(800,550)
        elif index == 'NDRE': value = nd(790,720)
        elif index == 'NDVI705': value = nd(750,705)
        elif index == 'CIgreen': value = _div(x(790),x(550))-1
        elif index == 'CIrededge': value = _div(x(790),x(710))-1
        elif index == 'SR': value = _div(x(800),x(680))
        elif index == 'SR705': value = _div(x(750),x(710))
        elif index == 'VOG1': value = _div(x(740),x(720))
        elif index == 'MTCI': value = _div(x(750)-x(710),x(710)-x(680))
        elif index == 'PRI': value = nd(531,570)
        elif index == 'PRI570_530': value = nd(530,570)
        elif index == 'PSRI': value = _div(x(680)-x(500),x(750))
        elif index == 'SIPI': value = _div(x(800)-x(445),x(800)-x(680))
        elif index == 'ARI1': value = _div(1,x(550))-_div(1,x(700))
        elif index == 'ARI2': value = x(800)*(_div(1,x(550))-_div(1,x(700)))
        elif index == 'CRI1': value = _div(1,x(510))-_div(1,x(550))
        elif index == 'CRI2': value = _div(1,x(510))-_div(1,x(700))
        elif index == 'WBI': value = _div(x(900),x(970))
        elif index == 'NRI': value = nd(570,670)
        elif index == 'RGI': value = _div(x(690),x(550))
        elif index == 'R750_R700': value = _div(x(750),x(700))
        elif index == 'mND705': value = _div(x(750)-x(705),x(750)+x(705)-2*x(445))
        elif index == 'MSR705': value = _div(x(750)-x(445),x(705)-x(445))
        elif index == 'REP': value = 700+40*_div((x(670)+x(780))/2-x(700),x(740)-x(700))
        elif index == 'MCARI': value = mcari()
        elif index == 'TCARI': value = tcari()
        elif index == 'TVI': value = 0.5*(120*(x(750)-x(550))-200*(x(670)-x(550)))
        elif index == 'DVI': value = x(800)-x(680)
        elif index == 'RDVI': value = _div(x(800)-x(670),np.sqrt(np.maximum(x(800)+x(670),0)))
        elif index == 'OSAVI': value = osavi()
        elif index == 'SAVI': value = _div(1.5*(x(800)-x(680)),x(800)+x(680)+0.5)
        elif index == 'EVI': value = _div(2.5*(x(800)-x(680)),x(800)+6*x(680)-7.5*x(450)+1)
        elif index == 'MSAVI':
            radicand = (2*x(800)+1)**2-8*(x(800)-x(670))
            value = (2*x(800)+1-np.sqrt(np.where(radicand>=0,radicand,np.nan)))/2
        elif index == 'MCARI_OSAVI': value = _div(mcari(), osavi())
        elif index == 'TCARI_OSAVI': value = _div(tcari(), osavi())
        else:
            raise RuntimeError('Not implemented: ' + index)
    value = np.asarray(value, dtype=np.float32)
    return np.where(np.isfinite(value), value, np.nan)
