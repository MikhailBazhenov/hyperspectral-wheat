"""End-to-end multi-sample wheat pipeline with optional radiometric/geometric steps.

Order: RAW -> optional dark correction -> optional geometric alignment ->
optional flat/Spectralon normalization -> preview image stack + quantitative
float16/float32 cube -> segmentation on preview images -> masked spectrum from NPZ.

The function supports two complementary behaviors:
- reuse_existing=True: reuse already produced intermediate/final files whenever
  possible to resume processing after an error.
- overwrite_existing=True: force regeneration of all derived files, ignoring
  existing results. This flag has priority over reuse_existing.

Folders with markers calibration_target, spectralon, flat, dark are never
samples; folders whose relative path contains '#' are ignored entirely.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import logging
import traceback
from datetime import datetime

from correct_flat_field import _index_images
from dark_correct_images import dark_correct_images
from spectral_cube_reflectance import generate_reflectance_cube
from measure_plant_spectrum import measure_plant_spectrum


SPECTRUM_CACHE_NAME = 'plant_spectrum.csv'


def _find_one(cubes, marker, required):
    matching = [p for p in cubes if marker.casefold() in p.parent.name.casefold()]
    if len(matching) > 1:
        raise ValueError(f'Multiple {marker} folders: {matching}')
    if required and len(matching) != 1:
        raise ValueError(f'Expected exactly one {marker} Spectral_Cube, got {len(matching)}')
    return matching[0] if matching else None


def _same_bandset(folder_a, folder_b):
    """Return True if both folders contain the same wavelength keys."""
    try:
        a = set(_index_images(folder_a))
        b = set(_index_images(folder_b))
    except Exception:
        return False
    return bool(a) and a == b


def _output_ready(folder, source_folder):
    return Path(folder).is_dir() and _same_bandset(folder, source_folder)


def _corrected_ready(corrected_folder, source_folder):
    """Ensure saved cube is readable, has the required bands and preview files."""
    from spectral_cube_reflectance import load_quant_cube
    corrected_folder = Path(corrected_folder)
    cube_path = corrected_folder / 'corrected_cube.npz'
    if not cube_path.is_file() or not _same_bandset(corrected_folder, source_folder):
        return False
    try:
        cube, wls, valid, _ = load_quant_cube(cube_path)
        return (cube.ndim == 3 and cube.shape == valid.shape and
                list(map(int, wls)) == sorted(_index_images(source_folder)))
    except Exception:
        return False


def _read_cached_spectrum(path):
    df = pd.read_csv(path)
    expected = ['wavelength', 'mean_intensity', 'std_intensity', 'n_pixels']
    if not set(expected).issubset(df.columns):
        raise ValueError(f'Invalid cached spectrum file: {path}')
    return df[expected]


def process_wheat_study(
    main_folder, *,
    use_dark_correction=True, use_alignment=True,
    use_spatial_correction=True, use_spectral_correction=True,
    flat_is_spectralon=False,
    calibration_marker='calibration_target', spectralon_marker='spectralon',
    flat_marker='flat', dark_marker='dark', exclude_marker='#',
    reference_wavelength=700, spectralon_wavelength=400,
    spectralon_edge_margin_px=5, min_flat_fraction=0.01,
    rgb_wavelengths=(630, 530, 470), stretch_percentiles=(1, 99),
    classes=('leaf',), class_thresholds=None, tile_size=2048, overlap=512,
    save_float_cube=True, cube_dtype='float16',
    preview_percentiles=(1, 99), reuse_existing=False, overwrite_existing=False,
    continue_on_error=True, show_plot=True,
):
    """Process all sample cubes in a study folder.

    Returns mean, std, long, report DataFrames; per-sample quantitative cubes
    are stored as corrected_images/corrected_cube.npz.
    """
    from collect_aruco_folder import collect_aruco_folder
    from align_spectral_images import align_spectral_images
    from segment_spectralon import segment_spectralon
    from measure_spectralon import measure_spectralon
    from segment_wheat_gwfss_tiled import segment_wheat_gwfss_tiled

    main = Path(main_folder).expanduser().resolve()
    if not main.is_dir():
        raise NotADirectoryError(main)
    if not save_float_cube:
        raise ValueError('save_float_cube must be True: downstream spectra and index maps use the NPZ cube')

    logger = logging.getLogger('wheat_pipeline.' + main.name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # One file per study, append across runs.
    if not logger.handlers:
        handler = logging.FileHandler(main / 'processing_errors.log', mode='a', encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(handler)
    logger.info('START study=%s reuse=%s overwrite=%s', main, reuse_existing, overwrite_existing)

    # overwrite_existing has priority over reuse_existing.
    if overwrite_existing:
        reuse_existing = False

    markers = (calibration_marker, spectralon_marker, flat_marker, dark_marker, exclude_marker)
    if any(not isinstance(m, str) or not m for m in markers):
        raise ValueError('Markers must be non-empty strings')

    cubes = sorted(
        p for p in main.rglob('Spectral_Cube')
        if p.is_dir() and not any(exclude_marker.casefold() in piece.casefold() for piece in p.relative_to(main).parts)
    )
    if not cubes:
        raise ValueError('No Spectral_Cube folders found')

    cal = _find_one(cubes, calibration_marker, use_alignment)
    spec = _find_one(cubes, spectralon_marker, False)
    flat = _find_one(cubes, flat_marker, use_spatial_correction)
    dark = _find_one(cubes, dark_marker, use_dark_correction)

    if flat_is_spectralon and not use_spatial_correction:
        raise ValueError('flat_is_spectralon requires spatial correction')
    if use_spectral_correction and not flat_is_spectralon and spec is None:
        raise ValueError('Need Spectralon folder; or set flat_is_spectralon=True for a large white flat')

    service_markers = (calibration_marker, spectralon_marker, flat_marker, dark_marker)
    samples = [
        p for p in cubes
        if not any(m.casefold() in p.parent.name.casefold() for m in service_markers)
    ]
    if not samples:
        raise ValueError('No sample cubes')

    def dark_process(cube):
        if cube is None:
            return None
        if not use_dark_correction or cube == dark:
            return cube
        dest = cube.parent / 'dark_corrected_images'
        if not (reuse_existing and _output_ready(dest, cube)):
            dark_correct_images(cube, dark, dest)
        return dest

    centers = None
    if use_alignment:
        cal_src = dark_process(cal)
        centers = main / 'centers_long.csv'
        if not (reuse_existing and centers.is_file()):
            collect_aruco_folder(cal_src).to_csv(centers, index=False)

    def aligned(cube):
        if cube is None:
            return None
        src = dark_process(cube)
        if not use_alignment:
            return src
        dest = cube.parent / 'aligned_images'
        if not (reuse_existing and _output_ready(dest, src)):
            align_spectral_images(centers, reference_wavelength, src, dest)
        return dest

    flat_src = aligned(flat) if use_spatial_correction else None
    spectralon_csv = None
    if use_spectral_correction and not flat_is_spectralon:
        reference_src = aligned(spec)
        spec_mask = main / 'spectralon_mask.png'
        if not (reuse_existing and spec_mask.is_file()):
            sf = _index_images(reference_src)
            if int(spectralon_wavelength) not in sf:
                raise ValueError(f'Spectralon wavelength {spectralon_wavelength} unavailable')
            segment_spectralon(sf[int(spectralon_wavelength)], spec_mask,
                               edge_margin_px=spectralon_edge_margin_px)
        spectralon_csv = main / 'spectralon_spectrum.csv'
        if not (reuse_existing and spectralon_csv.is_file()):
            measure_spectralon(spec_mask, reference_src, spectralon_csv)

    rows, reports = [], []
    for cube in samples:
        label = cube.parent.relative_to(main).as_posix()
        print('Processing:', label, flush=True)
        mask = cube.parent / 'wheat_mask.png'
        corrected = cube.parent / 'corrected_images'
        spectrum_cache = cube.parent / SPECTRUM_CACHE_NAME
        stage = 'alignment'
        try:
            src = aligned(cube)
            stage = 'cube_generation'
            if not (reuse_existing and _corrected_ready(corrected, src)):
                generate_reflectance_cube(
                    src, corrected,
                    flat_folder=flat_src,
                    spectralon_csv=spectralon_csv,
                    flat_is_spectralon=flat_is_spectralon,
                    spatial_correction=use_spatial_correction,
                    spectral_correction=use_spectral_correction,
                    min_flat_fraction=min_flat_fraction,
                    save_float_cube=True,
                    cube_dtype=cube_dtype,
                    preview_percentiles=preview_percentiles,
                )
            stage = 'segmentation'
            if not (reuse_existing and mask.is_file()):
                segment_wheat_gwfss_tiled(
                    images_folder=corrected,
                    mask_path=mask,
                    rgb_wavelengths=rgb_wavelengths,
                    stretch_percentiles=stretch_percentiles,
                    classes=classes,
                    class_thresholds=class_thresholds,
                    tile_size=tile_size,
                    overlap=overlap,
                    rgb_path=cube.parent / 'wheat_rgb.png',
                    overlay_path=cube.parent / 'wheat_overlay.png',
                    class_map_path=cube.parent / 'wheat_classes.png',
                )
            stage = 'spectrum'
            if reuse_existing and spectrum_cache.is_file() and not overwrite_existing:
                df = _read_cached_spectrum(spectrum_cache)
            else:
                spectrum = measure_plant_spectrum(corrected / 'corrected_cube.npz', mask)
                df = pd.DataFrame(spectrum)
                df.to_csv(spectrum_cache, index=False)
            df.insert(0, 'sample', label)
            rows.append(df)
            reports.append({
                'sample': label,
                'status': 'ok',
                'stage': 'complete',
                'error': '',
                'n_wavelengths': len(df),
                'n_pixels': int(df['n_pixels'].iloc[0]) if len(df) else 0,
            })
        except Exception as exc:
            reports.append({
                'sample': label,
                'status': 'error',
                'stage': stage,
                'error': f'{type(exc).__name__}: {exc}',
                'n_wavelengths': 0,
                'n_pixels': 0,
            })
            logger.error('SAMPLE ERROR sample=%s stage=%s\n%s', label, stage, traceback.format_exc())
            pd.DataFrame(reports).to_csv(main / 'processing_report.csv', index=False)
            if not continue_on_error:
                pd.DataFrame(reports).to_csv(main / 'processing_report.csv', index=False)
                raise
            print(f'ERROR {label}: {exc}', flush=True)

    report = pd.DataFrame(reports)
    report.to_csv(main / 'processing_report.csv', index=False)
    if not rows:
        logger.error('No successful samples in study %s', main)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), report

    long = pd.concat(rows, ignore_index=True)
    mean = long.pivot(index='wavelength', columns='sample', values='mean_intensity').sort_index()
    std = long.pivot(index='wavelength', columns='sample', values='std_intensity').sort_index()
    mean.to_csv(main / 'spectra_mean.csv', index_label='wavelength')
    std.to_csv(main / 'spectra_std.csv', index_label='wavelength')
    long.to_csv(main / 'spectra_long.csv', index=False)

    try:
        import seaborn as sns
        sns.set_theme(style='whitegrid')
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=long, x='wavelength', y='mean_intensity', hue='sample', estimator=None, ax=ax)
    except ImportError:
        fig, ax = plt.subplots(figsize=(12, 6))
        for name in mean.columns:
            ax.plot(mean.index, mean[name], label=name)
    ax.set(
        xlabel='Wavelength, nm',
        ylabel=('Relative reflectance' if use_spectral_correction else 'Signal, DN'),
        title='Wheat spectra',
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig(main / 'wheat_spectra.png', dpi=170, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    logger.info('DONE study=%s ok=%s errors=%s', main, sum(report.status == 'ok'), sum(report.status == 'error'))
    return mean, std, long, report
