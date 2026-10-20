"""Fast, study-comparable vegetation-index heatmaps and empirical CSV ranges.

v5 improvements over v4:
* reuse each input cube, wavelength lookup and bounded band cache per sample;
* draft scales: update statistics AND write maps in a single calculation pass;
* empirical scales: reuse up-to-date per-sample statistics, so only one
  index calculation pass is necessary after a completed draft run;
* PIL palette rendering is the default; Matplotlib renderer is optional;
* time spent in loading, calculation, statistics and PNG rendering is logged.

The scientific formulas, masked-pixel rules, statistics and color limits have
not been changed. Only persisted corrected_cube.npz data are read.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import cv2

from calculate_vegetation_indices import INDEX_INFO
from index_band_interpolation import BandSampler
from spectral_cube_reflectance import load_quant_cube
from vegetation_index_formulas import compute_single_index_map, save_map, save_map_fast
from index_color_scales import (
    PER_SAMPLE_NAME, EMPIRICAL_NAME,
    read_color_ranges, update_sample_statistics, summarize_empirical_ranges,
    select_color_bounds, index_statistics, atomic_write_csv,
)

_STATS_SCHEMA = 2
_STATS_METADATA = ('stats_schema','cube_mtime_ns','cube_size_bytes',
                   'mask_mtime_ns','mask_size_bytes','use_mask','mask_filename','data_type')


def _samples(main, corrected_folder_name, service_markers, exclude_marker):
    out = []
    for folder in sorted(main.rglob(corrected_folder_name)):
        if not folder.is_dir() or folder.name != corrected_folder_name:
            continue
        parts = folder.relative_to(main).parts
        if any(exclude_marker.casefold() in p.casefold() for p in parts):
            continue
        if any(m.casefold() in folder.parent.name.casefold() for m in service_markers):
            continue
        out.append((folder.parent.relative_to(main).as_posix(), folder))
    return out


def _source_meta(folder, mask_filename, use_mask, data_type):
    cube = folder / 'corrected_cube.npz'
    cstat = cube.stat()
    if use_mask:
        mstat = (folder.parent / mask_filename).stat()
        mask_ns, mask_size = mstat.st_mtime_ns, mstat.st_size
    else:
        mask_ns, mask_size = 0, 0
    return dict(stats_schema=_STATS_SCHEMA, cube_mtime_ns=cstat.st_mtime_ns,
                cube_size_bytes=cstat.st_size, mask_mtime_ns=mask_ns,
                mask_size_bytes=mask_size, use_mask=int(use_mask),
                mask_filename=str(mask_filename), data_type=str(data_type))


def _stats_current(old_stats, sample_name, names, metadata):
    if old_stats.empty or not set(_STATS_METADATA).issubset(old_stats.columns):
        return False
    subset = old_stats.loc[old_stats['sample'].astype(str) == str(sample_name)]
    if not set(names).issubset(set(subset['index'])):
        return False
    subset = subset.loc[subset['index'].isin(names)]
    if len(subset) != len(names):
        return False
    for col in _STATS_METADATA:
        if col in ('data_type', 'mask_filename'):
            if not subset[col].astype(str).eq(str(metadata[col])).all():
                return False
        else:
            if not pd.to_numeric(subset[col], errors='coerce').eq(metadata[col]).all():
                return False
    return True


def _load_sample(folder, data_type, use_mask, mask_filename, max_cached_bands):
    t0 = perf_counter()
    cube, wls, valid, kind = load_quant_cube(folder / 'corrected_cube.npz')
    if kind != 'reflectance' and data_type == 'reflectance':
        raise ValueError(f'{folder.parent}: cube is a signal, not relative reflectance')
    if cube.ndim != 3 or valid.shape != cube.shape:
        raise ValueError(f'Invalid cube dimensions in {folder}')
    plant = np.ones(cube.shape[:2], dtype=bool)
    if use_mask:
        path = folder.parent / mask_filename
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != cube.shape[:2]:
            raise ValueError(f'Missing or dimension-mismatched mask: {path}')
        plant = mask > 0
    # One location lookup table and bounded LRU for every selected index.
    reader = BandSampler(wls, cube, axis=2, max_cached_bands=max_cached_bands)
    return cube, wls, valid, plant, reader, perf_counter() - t0


def _valid_pixels(valid_cube, reader, wavelengths, plant, requested):
    result = plant.copy()
    for nm in requested:
        loc = reader.location(nm)
        if loc[0] == 'exact':
            result &= valid_cube[:, :, loc[1]]
        elif loc[0] == 'interpolated':
            result &= valid_cube[:, :, loc[1]] & valid_cube[:, :, loc[2]]
        else:
            result[:] = False
            break
    return result


def _calculate(name, cube, wls, valid_cube, plant, reader, data_type):
    requested = INDEX_INFO[name][2]
    status, interpolated, missing = reader_status(reader, requested)
    if missing:
        return np.full(cube.shape[:2], np.nan, dtype=np.float32), interpolated, missing
    arr = compute_single_index_map(wls, cube, name, data_type=data_type, band_reader=reader)
    arr[~_valid_pixels(valid_cube, reader, wls, plant, requested)] = np.nan
    return arr, interpolated, missing


def reader_status(reader, requested):
    interpolation, missing = [], []
    for nm in requested:
        state = reader.location(nm)[0]
        if state == 'interpolated':
            interpolation.append(float(nm))
        elif state == 'missing':
            missing.append(float(nm))
    return ('unavailable' if missing else 'interpolated' if interpolation else 'exact',
            interpolation, missing)


def _old_map_valid(old, dest, cube_path, mask_path, use_mask, missing, n_valid,
                   lo, hi, source, renderer, cmap):
    if old is None or not dest.is_file():
        return False
    try:
        expected = 'unavailable' if missing else 'no_valid_pixels' if not n_valid else 'ok'
        return (str(old['status']) == expected and
                str(old['scale_source']) == source and
                str(old['renderer']) == renderer and str(old['cmap']) == cmap and
                np.isclose(float(old['vmin']), lo, rtol=1e-10, atol=1e-10) and
                np.isclose(float(old['vmax']), hi, rtol=1e-10, atol=1e-10) and
                dest.stat().st_mtime_ns >= cube_path.stat().st_mtime_ns and
                (not use_mask or dest.stat().st_mtime_ns >= mask_path.stat().st_mtime_ns))
    except (ValueError, TypeError, KeyError, OSError):
        return False


def _write_one_map(arr, dest, title, missing, n_valid, cmap, renderer,
                   robust_percentiles, lo, hi):
    if missing or not n_valid:
        from PIL import Image
        Image.new('RGB', (arr.shape[1], arr.shape[0]), 'white').save(dest)
    elif renderer == 'fast':
        save_map_fast(arr, dest, title, cmap=cmap, vmin=lo, vmax=hi)
    else:
        save_map(arr, dest, title, cmap=cmap, vmin=lo, vmax=hi,
                 robust_percentiles=robust_percentiles)


def create_vegetation_index_maps(
    main_folder, *, indices=None,
    output_subfolder='index_maps', mask_filename='wheat_mask.png', use_mask=True,
    calibration_marker='calibration_target', spectralon_marker='spectralon',
    flat_marker='flat', dark_marker='dark', exclude_marker='#',
    corrected_folder_name='corrected_images', data_type='reflectance',
    robust_percentiles=(2,98), cmap='viridis', overwrite=True,
    color_config_csv=None, scale_source='draft', empirical_strategy='robust',
    renderer='fast', max_cached_bands=12,
):
    """Write maps with study-wide limits and collect observed per-index ranges.

    renderer='fast' (default) uses Pillow+the original Matplotlib colormap;
    renderer='matplotlib' retains the v4 plotting appearance (much slower).
    Statistical CSVs are always derived from valid numerical pixel values,
    never from quantized PNG colors. Reuse decisions preserve existing PNGs.
    """
    main = Path(main_folder).expanduser().resolve()
    if not main.is_dir():
        raise NotADirectoryError(main)
    names = list(INDEX_INFO) if indices is None else list(dict.fromkeys(indices))
    if any(n not in INDEX_INFO for n in names):
        raise ValueError('Unknown index requested')
    if scale_source not in ('draft', 'empirical'):
        raise ValueError("scale_source must be 'draft' or 'empirical'")
    if renderer not in ('fast', 'matplotlib'):
        raise ValueError("renderer must be 'fast' or 'matplotlib'")
    if data_type not in ('reflectance', 'signal', 'signal_exploratory'):
        raise ValueError('Unknown data_type')
    if not 0 <= int(max_cached_bands) <= 128:
        raise ValueError('max_cached_bands must be in 0..128')
    config = read_color_ranges(color_config_csv, required_indices=names)
    folders = _samples(main, corrected_folder_name,
        (calibration_marker, spectralon_marker, flat_marker, dark_marker), exclude_marker)
    if not folders:
        raise ValueError(f'No {corrected_folder_name} folders found in {main}')
    all_samples = [name for name, _ in folders]

    previous_report_path = main / 'vegetation_index_maps_report.csv'
    previous_report = pd.read_csv(previous_report_path) if previous_report_path.is_file() else pd.DataFrame()
    old_records = {}
    if {'sample','index'}.issubset(previous_report.columns):
        for _, row in previous_report.iterrows():
            old_records[(str(row['sample']),str(row['index']))] = row

    stats_path = main / PER_SAMPLE_NAME
    old_stats = pd.read_csv(stats_path) if stats_path.is_file() and stats_path.stat().st_size > 8 else pd.DataFrame()
    source_meta = {}
    stale = set()
    error_by_sample = {}
    time_rows = []
    rows = []

    # Pass 1 is needed only for EMPIRICAL rendering when some observed ranges
    # have not yet been calculated or were invalidated by a changed cube/mask.
    # A completed v5 draft pass has this metadata, so finalization is one pass.
    if scale_source == 'empirical':
        for sample_name, folder in folders:
            try:
                meta = _source_meta(folder, mask_filename, use_mask, data_type)
                source_meta[sample_name] = meta
                if not _stats_current(old_stats, sample_name, names, meta):
                    stale.add(sample_name)
            except Exception as exc:
                error_by_sample[sample_name] = f'{type(exc).__name__}: {exc}'
        for sample_name, folder in folders:
            if sample_name not in stale or sample_name in error_by_sample:
                continue
            started = perf_counter()
            load_s = calc_s = stat_s = 0.
            try:
                cube, wls, valid, plant, reader, load_s = _load_sample(
                    folder,data_type,use_mask,mask_filename,max_cached_bands)
                stats = []
                for name in names:
                    tic = perf_counter()
                    arr, _, _ = _calculate(name,cube,wls,valid,plant,reader,data_type)
                    calc_s += perf_counter()-tic
                    tic = perf_counter()
                    stats.append({**index_statistics(sample_name,name,arr),**source_meta[sample_name]})
                    stat_s += perf_counter()-tic
                    del arr
                update_sample_statistics(main, sample_name, stats, all_samples=all_samples)
            except Exception as exc:
                error_by_sample[sample_name] = f'{type(exc).__name__}: {exc}'
                print(f'ERROR calculating index ranges for {sample_name}: {exc}')
                update_sample_statistics(main, sample_name, [], all_samples=all_samples)
            time_rows.append(dict(sample=sample_name,phase='empirical_stats',
                                  load_seconds=load_s,calculate_seconds=calc_s,
                                  statistics_seconds=stat_s,render_seconds=0.,
                                  total_seconds=perf_counter()-started))
        empirical = summarize_empirical_ranges(main)
    else:
        empirical = pd.DataFrame()

    # Draft mode: one pass BOTH computes empirical stats and writes draft PNGs.
    # Empirical mode: redraws based on the study-wide ranges above.
    for sample_name, folder in folders:
        started = perf_counter()
        load_s = calc_s = stat_s = render_s = 0.
        cached_stats = source_meta.get(sample_name)
        try:
            if sample_name in error_by_sample:
                raise RuntimeError(error_by_sample[sample_name])
            if cached_stats is None:
                cached_stats = _source_meta(folder, mask_filename, use_mask, data_type)
            # Fast reuse: if statistics AND all output maps are current, the cube
            # never needs to be loaded.  Prior v4 reports do not include renderer
            # metadata and are automatically refreshed on their first v5 run.
            can_skip = (not overwrite and
                        _stats_current(old_stats,sample_name,names,cached_stats) and
                        sample_name not in stale)
            if can_skip:
                reused = []
                for n in names:
                    old = old_records.get((sample_name,n))
                    lo,hi,source = select_color_bounds(
                        config,empirical,n,scale_source=scale_source,
                        empirical_strategy=empirical_strategy)
                    dest = folder.parent / output_subfolder / (n+'.png')
                    if old is None or not _old_map_valid(
                            old,dest,folder/'corrected_cube.npz',
                            folder.parent/mask_filename,use_mask,
                            str(old.get('status',''))=='unavailable',
                            int(old.get('n_valid_pixels',0)),lo,hi,source,renderer,cmap):
                        can_skip = False
                        break
                    reused.append(dict(old))
                if can_skip:
                    rows.extend(reused)
                    print(f'{sample_name}: reused {len(reused)} maps (no cube read)', flush=True)
                    time_rows.append(dict(sample=sample_name,phase='reuse_maps',
                        load_seconds=0.,calculate_seconds=0.,statistics_seconds=0.,
                        render_seconds=0.,total_seconds=perf_counter()-started))
                    continue
            cube, wls, valid, plant, reader, load_s = _load_sample(
                folder,data_type,use_mask,mask_filename,max_cached_bands)
            out = folder.parent / output_subfolder
            out.mkdir(parents=True,exist_ok=True)
            sample_stats = []
            for name in names:
                tic = perf_counter()
                arr, interpolated, missing = _calculate(
                    name,cube,wls,valid,plant,reader,data_type)
                calc_s += perf_counter()-tic
                tic = perf_counter()
                n_valid = int(np.isfinite(arr).sum())
                if scale_source == 'draft':
                    sample_stats.append({**index_statistics(sample_name,name,arr),**cached_stats})
                stat_s += perf_counter()-tic
                lo, hi, source = select_color_bounds(
                    config, empirical, name, scale_source=scale_source,
                    empirical_strategy=empirical_strategy)
                dest = out / f'{name}.png'
                old = old_records.get((sample_name,name))
                reuse = not overwrite and _old_map_valid(
                    old,dest,folder/'corrected_cube.npz',folder.parent/mask_filename,
                    use_mask,missing,n_valid,lo,hi,source,renderer,cmap)
                if not reuse:
                    tic = perf_counter()
                    _write_one_map(arr,dest,f'{folder.parent.name} — {name}',missing,n_valid,
                                   cmap,renderer,robust_percentiles,lo,hi)
                    render_s += perf_counter()-tic
                rows.append(dict(sample=sample_name,index=name,output_png=str(dest),
                                 n_valid_pixels=n_valid,interpolated=bool(interpolated),
                                 missing_wavelengths_nm=','.join(str(nm) for nm in missing),
                                 status=('unavailable' if missing else 'no_valid_pixels' if not n_valid else 'ok'),
                                 error='',vmin=lo,vmax=hi,scale_source=source,
                                 renderer=renderer,cmap=cmap))
                del arr
            if scale_source == 'draft':
                update_sample_statistics(main,sample_name,sample_stats,all_samples=all_samples)
        except Exception as exc:
            print(f'ERROR rendering index maps for {sample_name}: {exc}')
            if scale_source == 'draft':
                update_sample_statistics(main, sample_name, [], all_samples=all_samples)
            # Remove any partial report rows created before the error.
            rows = [r for r in rows if r['sample'] != sample_name]
            for name in names:
                rows.append(dict(sample=sample_name,index=name,output_png='',n_valid_pixels=0,
                                 interpolated=False,missing_wavelengths_nm='',status='error',
                                 error=f'{type(exc).__name__}: {exc}',vmin=np.nan,vmax=np.nan,
                                 scale_source='',renderer=renderer,cmap=cmap))
        time_rows.append(dict(sample=sample_name,phase='render_maps',
                              load_seconds=load_s,calculate_seconds=calc_s,
                              statistics_seconds=stat_s,render_seconds=render_s,
                              total_seconds=perf_counter()-started))
        print(f'{sample_name}: load {load_s:.2f}s, math {calc_s:.2f}s, '
              f'statistics {stat_s:.2f}s, PNG {render_s:.2f}s', flush=True)

    if scale_source == 'draft':
        summarize_empirical_ranges(main)
    report = pd.DataFrame(rows)
    atomic_write_csv(report, previous_report_path)
    atomic_write_csv(pd.DataFrame(time_rows), main/'index_processing_timing.csv')
    return report
