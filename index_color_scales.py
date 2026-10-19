"""Project-wide vegetation-index color limits and per-study observed statistics.

The editable index_color_ranges.csv lives beside the Python modules. Study
statistics are saved incrementally to each study's root folder. Explicit
limits ensure that matching colors mean matching index values across samples.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_CONFIG = Path(__file__).resolve().with_name('index_color_ranges.csv')
PER_SAMPLE_NAME = 'index_ranges_per_sample.csv'
EMPIRICAL_NAME = 'index_ranges_empirical.csv'
REQUIRED_CONFIG = {'index', 'policy', 'draft_min', 'draft_max'}


def read_color_ranges(config_csv=None, required_indices=None):
    """Load editable ranges. policy=fixed or adaptive; values must be finite."""
    path = Path(config_csv).expanduser().resolve() if config_csv is not None else DEFAULT_CONFIG
    if not path.is_file():
        raise FileNotFoundError(f'Index color configuration is missing: {path}')
    cfg = pd.read_csv(path, keep_default_na=False)
    if not REQUIRED_CONFIG.issubset(cfg.columns):
        raise ValueError(f'{path}: columns required: {sorted(REQUIRED_CONFIG)}')
    if cfg['index'].duplicated().any():
        raise ValueError('Duplicate indices in color configuration')
    for field in ('draft_min', 'draft_max'):
        cfg[field] = pd.to_numeric(cfg[field], errors='raise')
    valid_policy = cfg['policy'].isin(('fixed', 'adaptive'))
    if not valid_policy.all():
        raise ValueError(f'Unknown scale policy: {cfg.loc[~valid_policy,"policy"].tolist()}')
    if not (np.isfinite(cfg['draft_min']).all() and np.isfinite(cfg['draft_max']).all()):
        raise ValueError('All draft bounds must be finite')
    if not (cfg['draft_min'] < cfg['draft_max']).all():
        raise ValueError('Every draft_min must be smaller than draft_max')
    if required_indices is not None:
        missing = set(required_indices) - set(cfg['index'])
        if missing:
            raise ValueError(f'Indices missing from color configuration: {sorted(missing)}')
    return cfg.set_index('index')


def atomic_write_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    df.to_csv(temp, index=False, float_format='%.10g')
    temp.replace(path)


def update_sample_statistics(main_folder, sample_name, new_stats, all_samples=None):
    """Replace one sample's observations, then update the aggregate CSV.

    This is called *during* processing and writes both CSVs after each sample.
    No raw pixel data are saved to these reports.
    """
    main = Path(main_folder)
    path = main / PER_SAMPLE_NAME
    prev = pd.read_csv(path) if path.is_file() else pd.DataFrame()
    if 'sample' in prev.columns:
        prev = prev.loc[prev['sample'].astype(str) != str(sample_name)].copy()
        if all_samples is not None:
            prev = prev.loc[prev['sample'].isin(set(all_samples))].copy()
    now = pd.concat([prev, pd.DataFrame(new_stats)], ignore_index=True)
    if len(now):
        now = now.sort_values(['sample','index'], kind='stable').reset_index(drop=True)
    atomic_write_csv(now, path)
    return summarize_empirical_ranges(main)


def summarize_empirical_ranges(main_folder):
    """Summarize observed min/max and conservative per-sample robust bounds.

    robust_min = min(sample P2); robust_max = max(sample P98). These are NOT
    pooled per-pixel percentiles: they preserve every sample's typical range
    without depending on its image size.
    """
    main = Path(main_folder)
    path = main / PER_SAMPLE_NAME
    cols = ['index','n_samples','n_pixels','observed_min','observed_max',
            'robust_min','robust_max']
    if not path.is_file():
        result = pd.DataFrame(columns=cols)
        atomic_write_csv(result, main / EMPIRICAL_NAME)
        return result
    df = pd.read_csv(path)
    rows = []
    if df.empty:
        result = pd.DataFrame(columns=cols)
        atomic_write_csv(result, main / EMPIRICAL_NAME)
        return result
    for index, group in df.groupby('index', sort=False):
        group = group.loc[group['n_valid_pixels'].fillna(0) > 0]
        if group.empty:
            rows.append(dict(index=index,n_samples=0,n_pixels=0,observed_min=np.nan,
                             observed_max=np.nan,robust_min=np.nan,robust_max=np.nan))
            continue
        rows.append(dict(
            index=index,
            n_samples=int(group['sample'].nunique()),
            n_pixels=int(group['n_valid_pixels'].sum()),
            observed_min=float(group['observed_min'].min()),
            observed_max=float(group['observed_max'].max()),
            robust_min=float(group['p02'].min()),
            robust_max=float(group['p98'].max()),
        ))
    result = pd.DataFrame(rows, columns=cols).sort_values('index').reset_index(drop=True)
    atomic_write_csv(result, main / EMPIRICAL_NAME)
    return result


def select_color_bounds(config_df, empirical_df, index, scale_source='draft', empirical_strategy='robust'):
    """Select a study-consistent color range for one index."""
    if scale_source not in ('draft', 'empirical'):
        raise ValueError("scale_source must be 'draft' or 'empirical'")
    if empirical_strategy not in ('robust','extrema'):
        raise ValueError("empirical_strategy must be 'robust' or 'extrema'")
    conf = config_df.loc[index]
    base = float(conf['draft_min']), float(conf['draft_max'])
    if scale_source == 'draft' or conf['policy'] == 'fixed':
        return (*base, 'fixed' if conf['policy'] == 'fixed' else 'draft')
    if empirical_df is not None and len(empirical_df):
        match = empirical_df.loc[empirical_df['index'] == index]
        if len(match) == 1:
            lo_col, hi_col = ('robust_min','robust_max') if empirical_strategy == 'robust' else ('observed_min','observed_max')
            row = match.iloc[0]
            lo, hi = float(row[lo_col]),float(row[hi_col])
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                return lo, hi, f'empirical_{empirical_strategy}'
    return (*base, 'draft_fallback')


def index_statistics(sample, index, arr):
    finite = np.asarray(arr, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size:
        lo, hi = np.percentile(finite, (2,98))
        return dict(sample=sample,index=index,n_valid_pixels=int(finite.size),
                    observed_min=float(finite.min()),observed_max=float(finite.max()),
                    p02=float(lo),p98=float(hi))
    return dict(sample=sample,index=index,n_valid_pixels=0,
                observed_min=np.nan,observed_max=np.nan,p02=np.nan,p98=np.nan)
