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
import json
import math
import pandas as pd
import matplotlib.pyplot as plt
import logging
import traceback
from datetime import datetime

from correct_flat_field import _index_images
from dark_correct_images import (
    dark_correct_images, dark_correction_ready, average_dark_folders,
)
from spectral_cube_reflectance import generate_reflectance_cube
from measure_plant_spectrum import measure_plant_spectrum
from iris_aperture_calibration import (
    calculate_iris_correction, is_iris_reference_folder, REPORT_NAME as IRIS_REPORT_NAME,
)


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


def _source_is_newer(target, source_folder):
    """Check a derived file against all bands of its upstream image set."""
    target = Path(target)
    if not target.is_file():
        return True
    cutoff = target.stat().st_mtime_ns
    return any(p.stat().st_mtime_ns > cutoff for p in _index_images(source_folder).values())


def _output_ready(folder, source_folder, dependencies=()):
    """Avoid reusing aligned images after correction/method changes."""
    folder = Path(folder)
    if not folder.is_dir() or not _same_bandset(folder, source_folder):
        return False
    # Source path matters: RAW -> DARK_CORRECTED -> RAW must not reuse stale alignment.
    origin = folder / 'pipeline_alignment_source.txt'
    if not origin.is_file() or origin.read_text(encoding='utf-8') != str(Path(source_folder).resolve()):
        return False
    sources = _index_images(source_folder)
    outputs = _index_images(folder)
    if any(outputs[wl].stat().st_mtime_ns < p.stat().st_mtime_ns for wl, p in sources.items()):
        return False
    if dependencies:
        earliest_output_ns = min(p.stat().st_mtime_ns for p in outputs.values())
        if any(Path(dep).stat().st_mtime_ns > earliest_output_ns for dep in dependencies):
            return False
    return True


def _corrected_ready(corrected_folder, source_folder, dependencies=(), *,
                     iris_enabled=False, iris_factor=1.0, iris_signature=None):
    """Ensure saved cube is readable, has the required bands and preview files."""
    from spectral_cube_reflectance import load_quant_cube
    corrected_folder = Path(corrected_folder)
    cube_path = corrected_folder / 'corrected_cube.npz'
    if not cube_path.is_file() or not _same_bandset(corrected_folder, source_folder):
        return False
    origin = corrected_folder / 'pipeline_cube_source.txt'
    if not origin.is_file() or origin.read_text(encoding='utf-8') != str(Path(source_folder).resolve()):
        return False
    if _source_is_newer(cube_path, source_folder):
        return False
    if any(Path(dep).stat().st_mtime_ns > cube_path.stat().st_mtime_ns for dep in dependencies):
        return False
    try:
        meta_path = corrected_folder / 'radiometry_metadata.json'
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if bool(meta.get('iris_correction_enabled', False)) != bool(iris_enabled):
            return False
        if not math.isclose(float(meta.get('iris_correction_factor', 1.0)), float(iris_factor),
                            rel_tol=1e-8, abs_tol=1e-10):
            return False
        if iris_enabled and meta.get('iris_calibration_signature') != iris_signature:
            return False
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
    dark_subtraction_mode='pixelwise',
    use_iris_correction=True, iris_open_marker='open_iris',
    iris_closed_marker='closed_iris',
    iris_stability_cv_warning=0.10,
    iris_valid_dn_range=(10, 245), iris_wavelength_range=(500, 700),
    calibration_marker='calibration_target', spectralon_marker='spectralon',
    flat_marker='flat', dark_marker='dark', exclude_marker='#',
    reference_wavelength=700, spectralon_wavelength=400,
    spectralon_edge_margin_px=5, min_flat_fraction=0.01,
    rgb_wavelengths=(630, 530, 470), stretch_percentiles=(1, 99),
    classes=('leaf',), class_thresholds=None, tile_size=2048, overlap=512,
    save_float_cube=True, cube_dtype='float16',
    preview_percentiles=(1, 99), reuse_existing=False, overwrite_existing=False,
    continue_on_error=True, show_plot=True, stop_after_cubes=False,
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
    if dark_subtraction_mode not in ('pixelwise', 'scalar'):
        raise ValueError("dark_subtraction_mode must be 'pixelwise' or 'scalar'")

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

    # Iris calibration series are not biological samples and must NOT be used
    # as ordinary flat/Spectralon/ArUco references, even if nested under such names.
    pipeline_cubes = [p for p in cubes if not is_iris_reference_folder(
        p, main, open_marker=iris_open_marker, closed_marker=iris_closed_marker)]
    cal = _find_one(pipeline_cubes, calibration_marker, use_alignment)
    spec = _find_one(pipeline_cubes, spectralon_marker, False)
    flat = _find_one(pipeline_cubes, flat_marker, use_spatial_correction)
    dark_folders = [p for p in pipeline_cubes if dark_marker.casefold() in p.parent.name.casefold()]
    dark = None
    if use_dark_correction:
        if not dark_folders:
            print(f"WARNING [{main.name}]: no dark folders (marker={dark_marker!r}); "
                  "continuing WITHOUT dark subtraction.", flush=True)
            logger.warning('No dark series: proceeding without dark correction')
        elif len(dark_folders) == 1:
            dark = dark_folders[0]
            print(f"Dark correction: using single source {dark.parent.name} "
                  f"(mode={dark_subtraction_mode})", flush=True)
        else:
            dark, average_info = average_dark_folders(
                dark_folders, main / 'dark_average', overwrite=not reuse_existing)
            print(f"Dark correction: {len(dark_folders)} folders -> "
                  f"{dark} (mode={dark_subtraction_mode})", flush=True)

    if flat_is_spectralon and not use_spatial_correction:
        raise ValueError('flat_is_spectralon requires spatial correction')
    if use_spectral_correction and not flat_is_spectralon and spec is None:
        raise ValueError('Need Spectralon folder; or set flat_is_spectralon=True for a large white flat')

    iris_result = None
    if use_iris_correction:
        iris_result = calculate_iris_correction(
            main, open_marker=iris_open_marker,
            closed_marker=iris_closed_marker,
            stability_cv_warning=iris_stability_cv_warning,
            valid_dn_range=iris_valid_dn_range,
            wavelength_range=iris_wavelength_range,
            overwrite=overwrite_existing, show_plot=False)
    iris_enabled = iris_result is not None
    iris_factor = float(iris_result['coefficient']) if iris_enabled else 1.0
    iris_signature = iris_result['input_signature'] if iris_enabled else None

    service_markers = (calibration_marker, spectralon_marker, flat_marker, dark_marker)
    samples = [
        p for p in pipeline_cubes
        if not any(m.casefold() in p.parent.name.casefold() for m in service_markers)
    ]
    if not samples:
        raise ValueError('No sample cubes')

    def dark_process(cube):
        if cube is None:
            return None
        if not use_dark_correction or dark is None or cube == dark:
            return cube
        dest = cube.parent / 'dark_corrected_images'
        if not (reuse_existing and dark_correction_ready(
                cube, dark, dest, mode=dark_subtraction_mode)):
            dark_correct_images(cube, dark, dest, mode=dark_subtraction_mode)
        return dest

    centers = None
    if use_alignment:
        cal_src = dark_process(cal)
        centers = main / 'centers_long.csv'
        if not (reuse_existing and centers.is_file() and
                not _source_is_newer(centers, cal_src)):
            collect_aruco_folder(cal_src).to_csv(centers, index=False)

    def aligned(cube):
        if cube is None:
            return None
        src = dark_process(cube)
        if not use_alignment:
            return src
        dest = cube.parent / 'aligned_images'
        if not (reuse_existing and _output_ready(dest, src, dependencies=(centers,))):
            (dest / 'pipeline_alignment_source.txt').unlink(missing_ok=True)
            align_spectral_images(centers, reference_wavelength, src, dest)
            (dest / 'pipeline_alignment_source.txt').write_text(
                str(Path(src).resolve()), encoding='utf-8')
        return dest

    flat_src = aligned(flat) if use_spatial_correction else None
    spectralon_csv = None
    if use_spectral_correction and not flat_is_spectralon:
        reference_src = aligned(spec)
        spec_mask = main / 'spectralon_mask.png'
        if not (reuse_existing and spec_mask.is_file() and
                not _source_is_newer(spec_mask, reference_src)):
            sf = _index_images(reference_src)
            if int(spectralon_wavelength) not in sf:
                raise ValueError(f'Spectralon wavelength {spectralon_wavelength} unavailable')
            segment_spectralon(sf[int(spectralon_wavelength)], spec_mask,
                               edge_margin_px=spectralon_edge_margin_px)
        spectralon_csv = main / 'spectralon_spectrum.csv'
        if not (reuse_existing and spectralon_csv.is_file() and
                not _source_is_newer(spectralon_csv, reference_src) and
                spectralon_csv.stat().st_mtime_ns >= spec_mask.stat().st_mtime_ns):
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
            radiometric_dependencies = []
            if use_spatial_correction:
                radiometric_dependencies.extend(_index_images(flat_src).values())
            if spectralon_csv is not None:
                radiometric_dependencies.append(spectralon_csv)
            if iris_enabled:
                radiometric_dependencies.append(main / IRIS_REPORT_NAME)
            if not (reuse_existing and _corrected_ready(
                    corrected, src, dependencies=radiometric_dependencies,
                    iris_enabled=iris_enabled, iris_factor=iris_factor,
                    iris_signature=iris_signature)):
                (corrected / 'pipeline_cube_source.txt').unlink(missing_ok=True)
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
                    iris_correction_enabled=iris_enabled,
                    iris_correction_factor=iris_factor,
                    iris_calibration_signature=iris_signature,
                )
                (corrected / 'pipeline_cube_source.txt').write_text(
                    str(Path(src).resolve()), encoding='utf-8')
            if stop_after_cubes:
                reports.append({'sample': label, 'status': 'ok', 'stage': 'cube_ready', 'error': '',
                                'n_wavelengths': 0, 'n_pixels': 0})
                continue
            stage = 'segmentation'
            cube_path = corrected / 'corrected_cube.npz'
            if not (reuse_existing and mask.is_file() and
                    mask.stat().st_mtime_ns >= cube_path.stat().st_mtime_ns):
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
            if (reuse_existing and spectrum_cache.is_file() and not overwrite_existing and
                spectrum_cache.stat().st_mtime_ns >= max(
                    cube_path.stat().st_mtime_ns, mask.stat().st_mtime_ns)):
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
        if not stop_after_cubes:
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
