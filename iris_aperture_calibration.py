"""Study-level iris/aperture exposure calibration for the wheat v6 pipeline.

Images of the SAME stationary reference in exactly ONE folder containing
open_iris and ONE folder containing closed_iris are compared wavelength by wavelength. The correction is the mean across a selected wavelength interval of robust
per-pixel *closed/open* ratios, using only pixels inside a safe DN interval
in BOTH input images.
This multiplier is applied once to the already radiometrically corrected
quantitative object cube BEFORE float16 quantization.

Caveats: JPEG may be nonlinear; exposure/gain/lighting must match; aperture
changes may affect vignetting; a single scalar cannot remove wavelength- or
position-dependent variation. Diagnostic CSV/PNG reports are saved per study.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import warnings

import numpy as np
import pandas as pd

from correct_flat_field import _index_images, _read_gray

SCHEMA = 3
REPORT_NAME = 'iris_correction_report.json'
PER_PAIR_NAME = 'iris_correction_by_pair.csv'
PER_BAND_NAME = 'iris_correction_by_wavelength.csv'
PLOT_NAME = 'iris_correction_spectrum.png'


def _resolve_iris_images(folder: Path) -> Path:
    """Use imageNNN files directly, or files inside the child Spectral_Cube."""
    nested = folder / 'Spectral_Cube'
    candidates = []
    for candidate in (folder, nested):
        if candidate.is_dir():
            try:
                _index_images(candidate)
            except (OSError, ValueError, NotADirectoryError):
                continue
            candidates.append(candidate.resolve())
    if len(candidates) == 2:
        raise ValueError(
            f'Calibration folder {folder} contains spectral images both directly '
            'and inside Spectral_Cube. Please keep only one layout.')
    if not candidates:
        raise ValueError(
            f'No image<wl>.jpg/png files in {folder} or {nested}')
    return candidates[0]


def discover_iris_pairs(main_folder, *, open_marker='open_iris',
                        closed_marker='closed_iris', verbose=False):
    """Find ONE folder containing each iris marker (case-insensitive).

    Folder names may contain prefixes or suffixes, e.g. ``ref_open_iris_01``
    and ``ref_closed_iris_01``. Exactly one match per marker is allowed
    within each study; multiple matches raise an error instead of choosing
    arbitrarily. Each reference folder may contain spectral images directly
    or in a ``Spectral_Cube`` subfolder.

    Return (pairs, unmatched), preserving the previous public interface.
    """
    main = Path(main_folder).expanduser().resolve()
    if not main.is_dir():
        raise NotADirectoryError(main)
    op_marker = str(open_marker).casefold()
    cl_marker = str(closed_marker).casefold()
    if not op_marker or not cl_marker or op_marker == cl_marker:
        raise ValueError('Iris markers must be nonempty and distinct')

    matches = {'open': [], 'closed': []}
    for folder in main.rglob('*'):
        if not folder.is_dir():
            continue
        name = folder.name.casefold()
        open_hit = op_marker in name
        closed_hit = cl_marker in name
        if open_hit and closed_hit:
            raise ValueError(
                f'Ambiguous iris reference folder {folder}: its name contains '
                f'both markers {open_marker!r} and {closed_marker!r}.')
        if open_hit:
            matches['open'].append(folder)
        elif closed_hit:
            matches['closed'].append(folder)

    # A study must contain at most one reference folder for each marker.
    # Keep this safety check: we no longer combine multiple pairs.
    if len(matches['open']) > 1 or len(matches['closed']) > 1:
        raise ValueError(
            f'Expected at most ONE folder containing {open_marker!r} and ONE '
            f'containing {closed_marker!r} under {main}; found '
            f'{len(matches["open"])} and {len(matches["closed"])}. '
            'Rename or remove extra reference folders.')
    if len(matches['open']) == 1 and len(matches['closed']) == 1:
        op = _resolve_iris_images(matches['open'][0])
        cl = _resolve_iris_images(matches['closed'][0])
        return [(op, cl, 'single_pair')], []
    unmatched = matches['open'] + matches['closed']
    if verbose:
        print(f'WARNING: expected one folder containing {open_marker!r} '
              f'and one containing {closed_marker!r}; '
              f'found open={len(matches["open"])}, closed={len(matches["closed"])}.')
    return [], unmatched


def is_iris_reference_folder(cube_folder, main_folder, *, open_marker='open_iris',
                             closed_marker='closed_iris'):
    """Exclude reference series whose path has a component containing a marker."""
    main = Path(main_folder).expanduser().resolve()
    folder = Path(cube_folder).expanduser().resolve()
    try:
        relative = folder.relative_to(main)
    except ValueError:
        return False
    markers = (str(open_marker).casefold(), str(closed_marker).casefold())
    return any(marker and marker in part.casefold()
               for part in relative.parts for marker in markers)


def _file_signatures(pairs):
    sig = []
    for op, cl, key in pairs:
        for side, folder in (('open', op), ('closed', cl)):
            files = _index_images(folder)
            for wl, path in files.items():
                st = path.stat()
                sig.append((key, side, str(path.resolve()), int(wl), st.st_size, st.st_mtime_ns))
    return sig


def _identity(pairs, params):
    source = json.dumps({'schema': SCHEMA, 'files': _file_signatures(pairs),
                         'params': params}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(source.encode('utf-8')).hexdigest()


def _read_cache(report_path, identity):
    try:
        data = json.loads(Path(report_path).read_text(encoding='utf-8'))
        if data.get('schema') == SCHEMA and data.get('input_signature') == identity:
            k = float(data['coefficient'])
            if math.isfinite(k) and k > 0:
                return data
    except (FileNotFoundError, ValueError, KeyError, OSError, TypeError):
        pass
    return None


def _comparison(open_file, closed_file, *, valid_dn_range, min_signal_fraction,
                max_sample_pixels, min_valid_pixels, saturation_fraction=None):
    """Compare corresponding well-exposed pixels in BOTH source images.

    ``valid_dn_range=(10, 245)`` is expressed for 8-bit images; when two
    16-bit files are supplied, the limits are scaled by 65535/255 so that
    they represent the same fraction of the available digital range.
    An optional ``min_signal_fraction`` further excludes dark background
    in small-reference targets (e.g., a Spectralon disk).
    """
    a = _read_gray(open_file)
    b = _read_gray(closed_file, expected_shape=a.shape)
    if a.dtype != b.dtype or a.dtype not in (np.dtype('uint8'), np.dtype('uint16')):
        raise TypeError(f'Bit depth mismatch or unsupported image type: {open_file}, {closed_file}')
    max_dn = float(np.iinfo(a.dtype).max)
    scale = max_dn / 255.0
    low = float(valid_dn_range[0]) * scale
    high = float(valid_dn_range[1]) * scale
    if saturation_fraction is not None:
        high = min(high, float(saturation_fraction) * max_dn)

    stride = max(1, int(math.ceil(math.sqrt(a.size / max_sample_pixels))))
    open_d = a[::stride, ::stride].astype(np.float32)
    closed_d = b[::stride, ::stride].astype(np.float32)
    # A value outside [low, high] in EITHER frame disqualifies the pixel.
    in_range = ((open_d >= low) & (open_d <= high) &
                (closed_d >= low) & (closed_d <= high))
    candidates = closed_d[in_range]
    if not candidates.size:
        return None, {
            'n_valid_pixels': 0,
            'n_pixels_both_in_range': 0,
            'valid_fraction': 0.0,
            'range_fraction': 0.0,
            'open_below_min_fraction': float((open_d < low).mean()),
            'closed_below_min_fraction': float((closed_d < low).mean()),
            'open_above_max_fraction': float((open_d > high).mean()),
            'closed_above_max_fraction': float((closed_d > high).mean()),
            'open_saturated_fraction': float((open_d > high).mean()),
            'closed_saturated_fraction': float((closed_d > high).mean()),
            'comparison_stride': stride,
            'reference_threshold_DN': low,
            'valid_min_DN': low, 'valid_max_DN': high,
        }
    reference_level = float(np.percentile(candidates, 99))
    threshold = max(low, min_signal_fraction * reference_level)
    good = in_range & (open_d >= threshold) & (closed_d >= threshold)
    n = int(good.sum())
    common = {
        'n_valid_pixels': n,
        'n_pixels_both_in_range': int(in_range.sum()),
        'valid_fraction': float(good.mean()),
        'range_fraction': float(in_range.mean()),
        'open_below_min_fraction': float((open_d < low).mean()),
        'closed_below_min_fraction': float((closed_d < low).mean()),
        'open_above_max_fraction': float((open_d > high).mean()),
        'closed_above_max_fraction': float((closed_d > high).mean()),
        # Retain historical column names for existing diagnostic scripts.
        'open_saturated_fraction': float((open_d > high).mean()),
        'closed_saturated_fraction': float((closed_d > high).mean()),
        'comparison_stride': stride,
        'reference_threshold_DN': threshold,
        'valid_min_DN': low, 'valid_max_DN': high,
    }
    if n < min_valid_pixels:
        return None, common

    ratio = closed_d[good] / open_d[good]  # closed/open factor, no zero denominator
    median = float(np.median(ratio))
    mad = float(1.4826 * np.median(np.abs(ratio - median)))
    return median, {
        **common,
        'pixel_ratio_robust_cv': mad / median if median else np.nan,
        'open_median_DN': float(np.median(open_d[good])),
        'closed_median_DN': float(np.median(closed_d[good])),
    }


def plot_iris_calibration(main_folder, *, show_plot=True):
    """Plot all usable bands, highlighting bands used for the global factor."""
    import matplotlib.pyplot as plt
    folder = Path(main_folder).expanduser().resolve()
    series = pd.read_csv(folder / PER_BAND_NAME).sort_values('wavelength')
    report = json.loads((folder / REPORT_NAME).read_text(encoding='utf-8'))
    used = series['used_for_coefficient'].astype(bool)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(series['wavelength'], series['coefficient'], '-', lw=1, alpha=0.45,
            color='0.5', label='Available wavelengths (diagnostic)')
    if (~used).any():
        ax.scatter(series.loc[~used, 'wavelength'], series.loc[~used, 'coefficient'],
                   s=25, color='0.6', label='Not used in global factor', zorder=3)
    ax.scatter(series.loc[used, 'wavelength'], series.loc[used, 'coefficient'],
               s=32, color='tab:blue', label='Used for global factor', zorder=4)
    bounds = report.get('wavelength_range_nm')
    if bounds is not None:
        ax.axvspan(bounds[0], bounds[1], alpha=0.09, color='tab:blue',
                   label=f'Calculation range: {bounds[0]}–{bounds[1]} nm')
    ax.axhline(float(report['coefficient']), color='tab:red', linestyle='--',
               label=f"Global coefficient = {float(report['coefficient']):.5g}")
    ax.set(xlabel='Wavelength (nm)', ylabel='Closed / open iris intensity ratio',
           title='Iris aperture correction: spectral stability')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    filename = folder / PLOT_NAME
    fig.savefig(filename, dpi=160)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return filename


def calculate_iris_correction(main_folder, *, reference_pairs=None, open_marker='open_iris',
                              closed_marker='closed_iris',
                              valid_dn_range=(10, 245), wavelength_range=(500, 700),
                              min_signal_fraction=0.15, saturation_fraction=None,
                              max_sample_pixels=150_000, min_valid_pixels=100,
                              min_wavelengths=5, stability_cv_warning=0.10,
                              overwrite=False, show_plot=True):
    """Calculate one scalar per study from corresponding unsaturated pixels.

    Returns dictionary with coefficient (closed/open), diagnostics and paths;
    returns None with printed warning if a folder is missing.
    CSV/JSON/PNG reports are saved in the main study folder. No input changes.
    Cached results are valid only while image mtimes, sizes and params match.
    """
    main = Path(main_folder).expanduser().resolve()
    if (len(valid_dn_range) != 2 or not all(np.isfinite(v) for v in valid_dn_range)
            or not (0 <= valid_dn_range[0] < valid_dn_range[1] <= 255)):
        raise ValueError('valid_dn_range must be a pair 0 <= min < max <= 255')
    if wavelength_range is not None:
        if (len(wavelength_range) != 2 or not all(np.isfinite(v) for v in wavelength_range)
                or wavelength_range[0] > wavelength_range[1]):
            raise ValueError('wavelength_range must be (min_nm, max_nm), or None')
    if saturation_fraction is not None and not (0 < saturation_fraction <= 1):
        raise ValueError('saturation_fraction must be in (0, 1]')
    if not (0 <= min_signal_fraction < 1):
        raise ValueError('min_signal_fraction must be in [0,1)')
    if max_sample_pixels < 100 or min_valid_pixels < 1:
        raise ValueError('Too few sample pixels specified')
    if min_wavelengths < 1:
        raise ValueError('min_wavelengths must be positive')

    if reference_pairs is None:
        pairs, unmatched = discover_iris_pairs(
            main, open_marker=open_marker, closed_marker=closed_marker, verbose=False)
    else:
        # Internal route used by calculate_iris_pair() for direct comparison.
        if len(reference_pairs) != 1:
            raise ValueError('Exactly one iris reference pair is required')
        pairs = [(Path(o).expanduser().resolve(), Path(c).expanduser().resolve(), str(key))
                 for o, c, key in reference_pairs]
        unmatched = []
    if not pairs:
        print(f'WARNING [{main.name}]: no complete {open_marker}/{closed_marker} '
              'reference pair. Iris correction is skipped.')
        return None
    params = dict(open_marker=open_marker, closed_marker=closed_marker,
                  valid_dn_range=list(valid_dn_range),
                  wavelength_range=list(wavelength_range) if wavelength_range is not None else None,
                  saturation_fraction=saturation_fraction,
                  min_signal_fraction=min_signal_fraction,
                  max_sample_pixels=max_sample_pixels,
                  min_valid_pixels=min_valid_pixels, min_wavelengths=min_wavelengths)
    signature = _identity(pairs, params)
    cache = _read_cache(main / REPORT_NAME, signature) if not overwrite else None
    if cache is not None and (main / PER_BAND_NAME).is_file() and (main / PER_PAIR_NAME).is_file():
        result = dict(cache)
        result['cached'] = True
        print(f"[{main.name}] Using cached iris factor k={cache['coefficient']:.6f} "
              f"(CV={100*cache['wavelength_cv']:.2f}%, "
              f"range={cache['wavelength_range_nm']} nm)")
        if show_plot or not (main / PLOT_NAME).is_file():
            plot_iris_calibration(main, show_plot=show_plot)
        return result

    records = []
    for op, cl, pair_key in pairs:
        opens = _index_images(op)
        closeds = _index_images(cl)
        wavelengths = sorted(set(opens) & set(closeds))
        if set(opens) != set(closeds):
            warnings.warn(f'Iris pair {pair_key}: wavelength mismatch, using only common '
                          f'bands ({len(wavelengths)} bands)')
        for wl in wavelengths:
            k, diag = _comparison(opens[wl], closeds[wl],
                                  valid_dn_range=valid_dn_range,
                                  min_signal_fraction=min_signal_fraction,
                                  saturation_fraction=saturation_fraction,
                                  max_sample_pixels=max_sample_pixels,
                                  min_valid_pixels=min_valid_pixels)
            in_selected_range = (wavelength_range is None or
                                 wavelength_range[0] <= wl <= wavelength_range[1])
            records.append(dict(pair_id=pair_key, open_folder=str(op), closed_folder=str(cl),
                                wavelength=wl, coefficient=k if k is not None else np.nan,
                                accepted=(k is not None),
                                used_for_coefficient=bool(in_selected_range and k is not None),
                                **diag))
    by_pair = pd.DataFrame(records)
    if by_pair.empty:
        raise ValueError('No matching iris-calibration wavelengths found')
    valid = by_pair[by_pair.accepted & np.isfinite(by_pair.coefficient)].copy()
    if valid.empty:
        raise ValueError('No usable unsaturated iris calibration pixels. Adjust exposure or target coverage.')
    by_band = valid.groupby('wavelength', as_index=False).agg(
        coefficient=('coefficient', 'mean'),
        std_between_pairs=('coefficient', 'std'),
        n_reference_pairs=('coefficient', 'size'),
        n_valid_pixels=('n_valid_pixels', 'sum'),
        mean_pixel_ratio_cv=('pixel_ratio_robust_cv', 'mean'),
    ).sort_values('wavelength')
    by_band['used_for_coefficient'] = (
        True if wavelength_range is None else
        by_band.wavelength.between(wavelength_range[0], wavelength_range[1], inclusive='both')
    )
    selected = by_band.loc[by_band.used_for_coefficient].copy()
    if len(selected) < min_wavelengths:
        raise ValueError(
            f'Only {len(selected)} usable iris wavelengths in range '
            f'{wavelength_range}; need at least {min_wavelengths}. '
            'Check exposure, DN limits, or the wavelength interval.')
    # Equal weight per SELECTED wavelength; bands outside interval are diagnostic only.
    coef = float(selected.coefficient.mean())
    med = float(selected.coefficient.median())
    cv = float(selected.coefficient.std(ddof=0) / coef)
    difference = (selected.coefficient / coef - 1).abs()
    warning_msgs = []
    if cv > stability_cv_warning:
        warning_msgs.append(
            f'Wavelength-dependent iris factor: CV={cv:.1%} exceeds '
            f'threshold {stability_cv_warning:.0%}; one scalar may be inadequate.')
    if float(difference.max()) > 2 * stability_cv_warning:
        warning_msgs.append('Some channels differ strongly from the global correction factor.')
    if valid.loc[valid.used_for_coefficient, 'pixel_ratio_robust_cv'].median() > 0.10:
        warning_msgs.append('Spatial variation of the exposure ratio is large: '
                            'check target motion, aperture vignetting and nonlinearity.')
    for warning in warning_msgs:
        print('WARNING:', warning, flush=True)
    by_pair.to_csv(main / PER_PAIR_NAME, index=False)
    by_band.to_csv(main / PER_BAND_NAME, index=False)
    result = dict(schema=SCHEMA, input_signature=signature, coefficient=coef,
                  coefficient_direction='closed_iris / open_iris',
                  coefficient_median=med, wavelength_cv=cv,
                  wavelength_range_nm=list(wavelength_range) if wavelength_range is not None else None,
                  valid_dn_range_8bit=list(valid_dn_range),
                  wavelength_min=float(selected.coefficient.min()),
                  wavelength_max=float(selected.coefficient.max()),
                  max_relative_wavelength_deviation=float(difference.max()),
                  n_pairs=len(pairs), n_wavelengths=len(selected),
                  n_wavelengths_diagnostic=len(by_band),
                  pairs=[dict(open=str(o), closed=str(c), pair_id=key)
                         for o, c, key in pairs],
                  warnings=warning_msgs, cached=False,
                  note='Multiplier applied once to object corrected cube, before float16 saving. '
                       'Assumes objects shot with open iris and radiometric references with closed iris. '
                       'All other camera/exposure/illumination settings must match; JPEG may be nonlinear.')
    (main / REPORT_NAME).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    plot_iris_calibration(main, show_plot=show_plot)
    print(f'[{main.name}] Iris pair | selected bands: {len(selected)}/{len(by_band)} '
          f'| range={wavelength_range or "all"} nm | '
          f'DN={tuple(valid_dn_range)} (8-bit) | k=closed/open={coef:.6f} '
          f'| selected spectral CV={100*cv:.2f}%')
    return result


def calculate_iris_pair(open_folder, closed_folder, *, output_folder=None,
                        show_plot=True, **kwargs):
    """Compare TWO folders directly, using the same calibration as the study pipeline.

    Each input can be a folder of image<wl>.jpg/png files or its parent
    containing Spectral_Cube. This call uses exactly one pair. If output_folder is omitted, write reports
    alongside the two sibling folders (their common parent).

    Example:
        result = calculate_iris_pair(
            r'D:/study/open_iris', r'D:/study/closed_iris', show_plot=True)
        print(result['coefficient'])
    """
    def resolve(p):
        p = Path(p).expanduser().resolve()
        if (p / 'Spectral_Cube').is_dir():
            p = p / 'Spectral_Cube'
        _index_images(p)  # validate source directory
        return p
    op, cl = resolve(open_folder), resolve(closed_folder)
    if op == cl:
        raise ValueError('Open and closed iris folders must differ')
    if output_folder is None:
        # If both reference folders are siblings, this is the study root.
        output_folder = Path(__import__('os').path.commonpath([str(op.parent), str(cl.parent)]))
    main = Path(output_folder).expanduser().resolve()
    main.mkdir(parents=True, exist_ok=True)
    return calculate_iris_correction(
        main, reference_pairs=[(op, cl, 'direct_pair')],
        show_plot=show_plot, **kwargs)
