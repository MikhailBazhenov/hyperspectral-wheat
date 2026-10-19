"""Study-consistent pixelwise vegetation index maps from quantitative NPZ cubes.

Draft bounds come from index_color_ranges.csv beside this script. Each processed
sample updates index_ranges_per_sample.csv and index_ranges_empirical.csv in the
study's root. scale_source='empirical' redraws all maps using a single common
range per index (except indices with policy='fixed').

No raw bands are used to calculate indices; only corrected_cube.npz is read.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import cv2

from calculate_vegetation_indices import INDEX_INFO
from index_band_interpolation import band_status
from spectral_cube_reflectance import load_quant_cube
from vegetation_index_formulas import compute_single_index_map, save_map
from index_color_scales import (
    PER_SAMPLE_NAME, EMPIRICAL_NAME,
    read_color_ranges, update_sample_statistics, summarize_empirical_ranges,
    select_color_bounds, index_statistics, atomic_write_csv,
)


def _index_validity(valid_cube, wavelengths, requested):
    """A valid interpolated pixel requires both contributing source channels."""
    w = np.asarray(wavelengths, dtype=float)
    order = np.argsort(w)
    w = w[order]
    valid = np.asarray(valid_cube, dtype=bool)[:, :, order]
    allowed = np.ones(valid.shape[:2], dtype=bool)
    for nm in requested:
        matching = np.flatnonzero(np.isclose(w, nm, atol=1e-7, rtol=0))
        if matching.size:
            allowed &= valid[:, :, int(matching[0])]
        elif w[0] < nm < w[-1]:
            k = int(np.searchsorted(w, nm))
            allowed &= valid[:, :, k-1] & valid[:, :, k]
        else:
            allowed[:] = False
    return allowed


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


def _iter_sample_maps(sample_name, corrected, selected, data_type, use_mask, mask_filename):
    sample_folder = corrected.parent
    cube_path = corrected / 'corrected_cube.npz'
    cube, wls, valid, kind = load_quant_cube(cube_path)
    if kind != 'reflectance' and data_type == 'reflectance':
        raise ValueError(f'{sample_name}: cube is a signal, not relative reflectance')
    if cube.ndim != 3 or valid.shape != cube.shape:
        raise ValueError(f'Invalid cube dimensions in {cube_path}')

    plant_mask = np.ones(cube.shape[:2], dtype=bool)
    if use_mask:
        mask_path = sample_folder / mask_filename
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != cube.shape[:2]:
            raise ValueError(f'Missing or dimension-mismatched mask: {mask_path}')
        plant_mask = mask > 0

    for name in selected:
        status, interpolated, missing = band_status(wls, INDEX_INFO[name][2])
        if missing:
            arr = np.full(cube.shape[:2], np.nan, dtype=np.float32)
        else:
            arr = compute_single_index_map(wls, cube, name, data_type=data_type)
            usable = _index_validity(valid, wls, INDEX_INFO[name][2]) & plant_mask
            arr[~usable] = np.nan
        yield name, arr, interpolated, missing


def create_vegetation_index_maps(
    main_folder, *, indices=None,
    output_subfolder='index_maps', mask_filename='wheat_mask.png', use_mask=True,
    calibration_marker='calibration_target', spectralon_marker='spectralon',
    flat_marker='flat', dark_marker='dark', exclude_marker='#',
    corrected_folder_name='corrected_images', data_type='reflectance',
    robust_percentiles=(2,98), cmap='viridis', overwrite=True,
    color_config_csv=None, scale_source='draft', empirical_strategy='robust',
):
    """Create PNG maps with a consistent color range for every index.

    Parameters added to the previous API:
      color_config_csv: CSV containing index,policy,draft_min,draft_max.
          None means index_color_ranges.csv next to this module (project root).
      scale_source: 'draft' uses project ranges; 'empirical' uses study-wide
          bounds for indices configured with policy='adaptive'.
      empirical_strategy: 'robust' uses min(sample P2)/max(sample P98), while
          'extrema' uses true sample min/max. Both use the same bounds for every
          sample in the study; clipped pixels do not change underlying indices.

    The per-sample and empirical CSVs are updated after EACH successfully
    analyzed sample. Existing maps are reused only when their earlier bounds,
    source cube and mask are still valid, and overwrite=False.
    """
    main = Path(main_folder).expanduser().resolve()
    if not main.is_dir():
        raise NotADirectoryError(main)
    names = list(INDEX_INFO) if indices is None else list(dict.fromkeys(indices))
    if any(n not in INDEX_INFO for n in names):
        raise ValueError('Unknown index requested')
    if scale_source not in ('draft','empirical'):
        raise ValueError("scale_source must be 'draft' or 'empirical'")
    config = read_color_ranges(color_config_csv, required_indices=names)
    folders = _samples(main, corrected_folder_name,
        (calibration_marker,spectralon_marker,flat_marker,dark_marker), exclude_marker)
    all_samples = [name for name,_ in folders]
    if not folders:
        raise ValueError(f'No {corrected_folder_name} folders found in {main}')

    previous_report_path = main / 'vegetation_index_maps_report.csv'
    previous_report = pd.read_csv(previous_report_path) if previous_report_path.is_file() else pd.DataFrame()
    old_map_records = {}
    for _, row in previous_report.iterrows():
        if 'sample' in row and 'index' in row:
            old_map_records[(str(row['sample']),str(row['index']))] = row

    problem_rows = []
    # Pass 1: collect *all* empirical bounds from masked, radiometrically valid
    # plant pixels. Immediately write a per-study empirical summary each sample.
    for sample_name, folder in folders:
        try:
            stats = []
            for name, arr, _, _ in _iter_sample_maps(
                sample_name,folder,names,data_type,use_mask,mask_filename):
                stats.append(index_statistics(sample_name, name, arr))
            update_sample_statistics(main, sample_name, stats, all_samples=all_samples)
        except Exception as exc:
            update_sample_statistics(main, sample_name, [], all_samples=all_samples)
            problem_rows.append(dict(sample=sample_name,index='',output_png='',n_valid_pixels=0,
                                     status='error',error=f'{type(exc).__name__}: {exc}'))
            print(f'ERROR in index statistics for {sample_name}: {exc}')

    empirical = summarize_empirical_ranges(main)

    # Pass 2: render the color maps with one scale per index for the whole study.
    rows = []
    failed = {r['sample'] for r in problem_rows}
    for sample_name, folder in folders:
        if sample_name in failed:
            for name in names:
                rows.append(dict(sample=sample_name,index=name,output_png='',n_valid_pixels=0,
                                 interpolated=False,missing_wavelengths_nm='',status='error',
                                 error=next(r['error'] for r in problem_rows if r['sample']==sample_name),
                                 vmin=np.nan,vmax=np.nan,scale_source=''))
            continue
        try:
            out = folder.parent / output_subfolder
            out.mkdir(parents=True, exist_ok=True)
            cube_path = folder / 'corrected_cube.npz'
            mask_path = folder.parent / mask_filename
            for name, arr, interpolated, missing in _iter_sample_maps(
                sample_name,folder,names,data_type,use_mask,mask_filename):
                lo, hi, source = select_color_bounds(
                    config, empirical, name, scale_source=scale_source,
                    empirical_strategy=empirical_strategy)
                dest = out / f'{name}.png'
                valid_count = int(np.isfinite(arr).sum())
                old = old_map_records.get((sample_name,name))
                old_valid = False
                if old is not None and not overwrite and dest.is_file():
                    try:
                        old_valid = (old['status'] == ('unavailable' if missing else 'ok') and
                            str(old['scale_source']) == source and
                            np.isclose(float(old['vmin']),lo,rtol=1e-10,atol=1e-10) and
                            np.isclose(float(old['vmax']),hi,rtol=1e-10,atol=1e-10) and
                            dest.stat().st_mtime >= cube_path.stat().st_mtime and
                            (not use_mask or dest.stat().st_mtime >= mask_path.stat().st_mtime))
                    except (ValueError,TypeError,KeyError):
                        old_valid = False
                if not old_valid:
                    if missing or not valid_count:
                        # White image with exact spatial dimensions, no legend.
                        from PIL import Image
                        Image.new('RGB',(arr.shape[1],arr.shape[0]),'white').save(dest)
                    else:
                        save_map(arr,dest,f'{folder.parent.name} — {name}',cmap=cmap,
                                 robust_percentiles=robust_percentiles,vmin=lo,vmax=hi)
                rows.append(dict(
                    sample=sample_name,index=name,output_png=str(dest),
                    n_valid_pixels=valid_count,interpolated=bool(interpolated),
                    missing_wavelengths_nm=','.join(str(x) for x in missing),
                    status=('unavailable' if missing else 'no_valid_pixels' if not valid_count else 'ok'),
                    error='',vmin=lo,vmax=hi,scale_source=source,
                ))
        except Exception as exc:
            print(f'ERROR while rendering maps for {sample_name}: {exc}')
            for name in names:
                rows.append(dict(sample=sample_name,index=name,output_png='',n_valid_pixels=0,
                                 interpolated=False,missing_wavelengths_nm='',status='error',
                                 error=f'{type(exc).__name__}: {exc}',vmin=np.nan,vmax=np.nan,
                                 scale_source=''))

    report = pd.DataFrame(rows)
    atomic_write_csv(report, previous_report_path)
    return report
