"""Object-wise spectra, S-G smoothing, indices, pixelwise VI maps and PCA.

Every object is a distinct mask; no mask -> full valid frame by default.
Uses existing v5 index formulas, interpolation and color scale definitions.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from spectral_cube_reflectance import load_quant_cube
from hyperspectral_masks import list_object_masks,read_mask
from hyperspectral_pca_maps import generate_pca_maps
from calculate_vegetation_indices import INDEX_INFO,calculate_vegetation_indices
from index_presets import resolve_indices, OBSOLETE_INDEX_NAMES, FORMULA_FINGERPRINT
from create_vegetation_index_maps import _calculate
from index_band_interpolation import BandSampler
from index_color_scales import read_color_ranges,index_statistics
from vegetation_index_formulas import save_map_fast

SERVICE_MARKERS=('calibration_target','spectralon','flat','dark')

def find_samples(main_folder):
    main=Path(main_folder).expanduser().resolve()
    return sorted([p.parent.parent for p in main.rglob('corrected_cube.npz')
        if not any('#' in part or any(tag in part.lower() for tag in SERVICE_MARKERS)
                   for part in p.relative_to(main).parts[:-2])])

def selected_masks(sample,shape,*,mode='individual',fallback_full=True):
    """Individual masks, their union, or entire field (including background)."""
    if mode not in ('individual','union','full'):
        raise ValueError('mask mode: individual, union, full')
    if mode=='full':return {'__full__':np.ones(shape,dtype=bool)}
    paths=list_object_masks(sample)
    if not paths:
        if fallback_full:
            print(f'WARNING: no masks for {sample.name}; analyzing full frame')
            return {'__full__':np.ones(shape,dtype=bool)}
        raise ValueError(f'No masks for {sample}')
    masks={name:read_mask(path,shape) for name,path in paths.items()}
    if mode=='union':return {'__union__':np.logical_or.reduce(list(masks.values()))}
    return masks

def _smooth_spectrum(values,window,poly):
    result=np.array(values,dtype=float,copy=True)
    valid=np.isfinite(result)
    edges=np.diff(np.r_[False,valid,False].astype(np.int8))
    starts=np.flatnonzero(edges==1);stops=np.flatnonzero(edges==-1)
    for a,b in zip(starts,stops):
        n=b-a
        w=min(int(window),n if n%2==1 else n-1)
        if w>=3 and w>poly:
            result[a:b]=savgol_filter(result[a:b],window_length=w,polyorder=poly,mode='interp')
    return result

def _spectra_and_indices(main, samples, *, mode, window, poly, data_type, indices):
    out=main/'object_analysis';out.mkdir(exist_ok=True)
    all_data=[];metadata=[]
    for sample in samples:
        cube,wls,valid,kind=load_quant_cube(sample/'corrected_images'/'corrected_cube.npz')
        if data_type=='reflectance' and kind!='reflectance':
            raise ValueError(f'{sample.name}: cube contains signal, not reflectance')
        masks=selected_masks(sample,cube.shape[:2],mode=mode)
        for mask_id,region in masks.items():
            means=[];std=[];counts=[]
            for j in range(len(wls)):
                ok=region & valid[:,:,j] & np.isfinite(cube[:,:,j])
                values=cube[:,:,j][ok]
                means.append(float(np.mean(values)) if values.size else np.nan)
                std.append(float(np.std(values)) if values.size else np.nan)
                counts.append(int(values.size))
            smooth=_smooth_spectrum(means,window,poly)
            # S-G polynomial overshoot can introduce small unphysical negatives.
            # Only clamp those in calibrated-reflectance mode; retain NaNs.
            if data_type=='reflectance':smooth=np.maximum(smooth,0)
            uid=f'{sample.relative_to(main).as_posix()}::{mask_id}'
            metadata.append(dict(object=uid,sample=sample.relative_to(main).as_posix(),
                                 mask=mask_id,mask_pixels=int(region.sum())))
            for wl,m,s,n,sg in zip(wls,means,std,counts,smooth):
                all_data.append(dict(sample=uid,wavelength=int(wl),mean_intensity=m,
                                     std_intensity=s,n_pixels=n,sg_mean_intensity=float(sg)))
    df=pd.DataFrame(all_data)
    if df.empty:raise ValueError('No corrected cubes found')
    pd.DataFrame(metadata).to_csv(out/'objects.csv',index=False)
    df.to_csv(out/'spectra_long.csv',index=False)
    raw=df.pivot(index='wavelength',columns='sample',values='mean_intensity').sort_index()
    sg=df.pivot(index='wavelength',columns='sample',values='sg_mean_intensity').sort_index()
    raw.to_csv(out/'spectra_mean_raw.csv',index_label='wavelength')
    sg.to_csv(out/'spectra_mean_savgol.csv',index_label='wavelength')
    for data,name in [(raw,'raw'),(sg,'savgol')]:
        fig,ax=plt.subplots(figsize=(12,6))
        for col in data.columns:ax.plot(data.index,data[col],label=col,linewidth=1.2)
        ax.set(xlabel='Wavelength, nm',ylabel='Relative reflectance' if data_type=='reflectance' else 'Signal',
               title='Object mean spectra ('+name+')')
        if len(data.columns)<=20:ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),fontsize=7)
        fig.tight_layout();fig.savefig(out/f'spectra_{name}.png',dpi=160,bbox_inches='tight');plt.close(fig)
    for data,name in [(raw,'raw'),(sg,'savgol')]:
        calculate_vegetation_indices(out/f'spectra_mean_{name if name=="raw" else "savgol"}.csv',
            output_csv=out/f'indices_{name}.csv',data_type=data_type,indices=indices)
    return df,raw,sg

def _index_map_ranges(main,samples,mode,names,data_type,scale_config,empirical_strategy):
    """Pass 1: all-object per-index statistics, respecting the selected mask mode."""
    observations=[]
    for sample in samples:
        cube,wls,valid,kind=load_quant_cube(sample/'corrected_images'/'corrected_cube.npz')
        reader=BandSampler(wls,cube,axis=2,max_cached_bands=12)
        masks=selected_masks(sample,cube.shape[:2],mode=mode)
        for index in names:
            arr,interp,missing=_calculate(index,cube,wls,valid,np.ones(cube.shape[:2],bool),reader,data_type)
            for id,mask in masks.items():
                a=arr[mask & np.isfinite(arr)]
                observations.append(index_statistics(f'{sample.name}::{id}',index,a))
    stats=pd.DataFrame(observations)
    stats.to_csv(main/'object_analysis'/'index_ranges_by_object.csv',index=False)
    cfg=read_color_ranges(scale_config,required_indices=names)
    bounds={}
    for idx in names:
        c=cfg.loc[idx]
        if c['policy']=='fixed':
            bounds[idx]=(float(c.draft_min),float(c.draft_max),'fixed');continue
        x=stats[(stats['index']==idx)&stats['n_valid_pixels'].gt(0)]
        lo_key,hi_key=('p02','p98') if empirical_strategy=='robust' else ('observed_min','observed_max')
        lo=float(x[lo_key].min()) if len(x) else np.nan
        hi=float(x[hi_key].max()) if len(x) else np.nan
        if np.isfinite(lo) and np.isfinite(hi) and lo<hi:bounds[idx]=(lo,hi,'empirical')
        else:bounds[idx]=(float(c.draft_min),float(c.draft_max),'draft_fallback')
    pd.DataFrame([dict(index=n,vmin=a,vmax=b,source=c) for n,(a,b,c) in bounds.items()]).to_csv(
        main/'object_analysis'/'index_color_bounds.csv',index=False)
    return bounds

def render_object_index_maps(main_folder, *, mode='individual', indices=None, index_preset='medium',
                            scale_config=None, empirical_strategy='robust',
                            data_type='reflectance',overwrite=True):
    main=Path(main_folder).expanduser().resolve()
    samples=find_samples(main)
    names=resolve_indices(index_preset, indices)
    if empirical_strategy not in ('robust','extrema'):raise ValueError('empirical_strategy')
    if scale_config is None:scale_config=Path(__file__).resolve().parent/'index_color_ranges.csv'
    (main/'object_analysis').mkdir(exist_ok=True)
    bounds=_index_map_ranges(main,samples,mode,names,data_type,scale_config,empirical_strategy)
    rows=[]
    for sample in samples:
        cube,wls,valid,kind=load_quant_cube(sample/'corrected_images'/'corrected_cube.npz')
        if data_type=='reflectance' and kind!='reflectance':raise ValueError(f'Expected reflectance: {sample}')
        reader=BandSampler(wls,cube,axis=2,max_cached_bands=12)
        masks=selected_masks(sample,cube.shape[:2],mode=mode)
        for name in names:
            arr,interpolated,missing=_calculate(name,cube,wls,valid,np.ones(cube.shape[:2],bool),reader,data_type)
            lo,hi,source=bounds[name]
            for mask_id,region in masks.items():
                dest=sample/'object_analysis'/'index_maps'/mask_id/f'{name}.png'
                dest.parent.mkdir(parents=True,exist_ok=True)
                if overwrite or not dest.exists():
                    v=np.where(region,arr,np.nan)
                    save_map_fast(v,dest,title=f'{sample.name} / {mask_id} — {name}',vmin=lo,vmax=hi)
                rows.append(dict(sample=sample.relative_to(main).as_posix(),mask=mask_id,index=name,
                    output_png=str(dest),valid_pixels=int((region & np.isfinite(arr)).sum()),
                    vmin=lo,vmax=hi,scale=source,missing_bands=','.join(map(str,missing))))
        print(f'Index maps: {sample.name} / {len(masks)} mask(s)',flush=True)
    report=pd.DataFrame(rows)
    report.to_csv(main/'object_analysis'/'index_maps_report.csv',index=False)
    # Clean only pipeline-generated PNG names from earlier, larger presets;
    # leave masks, arbitrary files and user artwork untouched.
    dropped=set(INDEX_INFO).difference(names) | set(OBSOLETE_INDEX_NAMES)
    for sample in samples:
        map_base=sample/'object_analysis'/'index_maps'
        if map_base.exists():
            for name in dropped:
                for stale in map_base.glob(f'*/{name}.png'):
                    stale.unlink()
    return report

def _make_object_collages(main,report,n_columns=4):
    from PIL import Image,ImageDraw,ImageFont
    by_index=report.groupby('index',sort=True)
    dst=main/'object_analysis'/'collages';dst.mkdir(parents=True,exist_ok=True)
    try:font=ImageFont.truetype('DejaVuSans.ttf',16)
    except OSError:font=ImageFont.load_default()
    # Avoid misleading leftover collages when switching from maximum to minimal.
    shown=set(report['index'].unique()) if len(report) else set()
    for index in set(INDEX_INFO).difference(shown) | set(OBSOLETE_INDEX_NAMES):
        old=dst/f'collage_{index}.png'
        if old.is_file():old.unlink()
    for idx,group in by_index:
        rows=list(group.itertuples(index=False))
        width,height=320,275
        nr=(len(rows)+n_columns-1)//n_columns
        canvas=Image.new('RGB',(width*n_columns,height*nr),'white')
        draw=ImageDraw.Draw(canvas)
        for j,row in enumerate(rows):
            x=(j%n_columns)*width;y=(j//n_columns)*height
            with Image.open(row.output_png) as image:
                im=image.convert('RGB');im.thumbnail((width-12,height-42))
            canvas.paste(im,(x+(width-im.width)//2,y))
            draw.text((x+8,y+height-36),f'{row.sample} | {row.mask}',font=font,fill='black')
        canvas.save(dst/f'collage_{idx}.png')
    return dst

def analyze_study_objects(main_folder, *, mask_mode='individual',map_mode=None,
                          sg_window=9,sg_polyorder=2,n_pca_components=8,
                          n_clusters='auto',data_type='reflectance',
                          color_config_csv=None,indices=None,index_preset='medium',overwrite_maps=True,
                          empirical_strategy='robust',pca_maps=True,pca_map_mode=None):
    """Stage 3. For map_mode choose individual, union, full, or none.

    Spectra/indices/PCA objects use mask_mode. If no masks are present,
    selected_masks defaults to the whole valid image. Changing masks requires
    rerunning this function, not the cube calibration or segmentation.
    """
    names=resolve_indices(index_preset, indices)
    main=Path(main_folder).expanduser().resolve();samples=find_samples(main)
    if not samples:raise FileNotFoundError(f'No corrected_cube.npz found under {main}')
    if sg_window%2!=1 or sg_window<3 or sg_polyorder<0 or sg_polyorder>=sg_window:
        raise ValueError('sg_window must be odd >=3, sg_polyorder < window')
    df,raw,sg=_spectra_and_indices(main,samples,mode=mask_mode,window=sg_window,
                                   poly=sg_polyorder,data_type=data_type,indices=names)
    if pca_maps:
        spatial_mode=pca_map_mode or mask_mode
        for s in samples:
            try:
                cube_path=s/'corrected_images'/'corrected_cube.npz'
                shared_cube=load_quant_cube(cube_path)
                if spatial_mode=='full':
                    generate_pca_maps(cube_path,n_components=n_pca_components,
                                      output_dir=s/'pca_maps'/'__full__',cube_data=shared_cube,overwrite=True)
                else:
                    # Read spatial dimensions from one small already-created mask, or
                    # the full cube when mask files do not exist.
                    mask_files=list_object_masks(s)
                    if spatial_mode=='individual' and mask_files:
                        for mid,mpath in mask_files.items():
                            generate_pca_maps(cube_path,n_components=n_pca_components,
                                mask_path=mpath,output_dir=s/'pca_maps'/mid,cube_data=shared_cube,overwrite=True)
                    elif spatial_mode=='union' and mask_files:
                        import cv2
                        base=cv2.imread(str(next(iter(mask_files.values()))),cv2.IMREAD_GRAYSCALE)
                        if base is None:raise OSError('Cannot read first object mask')
                        union=base>0
                        for path in list(mask_files.values())[1:]:
                            union|=read_mask(path,union.shape)
                        generate_pca_maps(cube_path,n_components=n_pca_components,
                                mask_array=union,output_dir=s/'pca_maps'/'__union__',cube_data=shared_cube,overwrite=True)
                    else:
                        generate_pca_maps(cube_path,n_components=n_pca_components,
                                      output_dir=s/'pca_maps'/'__full__',cube_data=shared_cube,overwrite=True)
            except Exception as exc:print(f'WARNING PCA maps for {s.name}: {exc}')
    out=main/'object_analysis'
    # Record preset and exact wavelength/formula choices for reproducibility.
    # If formulas changed since the last run, old PNGs must be redrawn even
    # when the caller requested overwrite_maps=False.
    selection_path = out/'selected_indices.csv'
    try:
        old_selection = pd.read_csv(selection_path) if selection_path.is_file() else pd.DataFrame()
        old_formulas = dict(zip(old_selection['index'], old_selection['formula'])) if 'formula' in old_selection else {}
        formula_changed = any(old_formulas.get(n) != INDEX_INFO[n][1] for n in names)
    except Exception:
        formula_changed = True
    pd.DataFrame([dict(index=n,formula=INDEX_INFO[n][1],
                       wavelengths_nm=','.join(map(str,INDEX_INFO[n][2])),
                       formula_fingerprint=FORMULA_FINGERPRINT)
                  for n in names]).to_csv(selection_path,index=False)
    # Drop empty/incomplete object spectra for PCA only. They remain in the
    # main quantitative tables for traceability and user inspection.
    pca_candidates=raw.loc[:,raw.apply(lambda col: np.isfinite(col.to_numpy()).all())]
    if pca_candidates.shape[1]>=3 and pca_candidates.shape[0]>=2:
        from plot_spectral_pca import plot_spectral_pca
        try:
            pca_candidates.to_csv(out/'spectra_for_object_pca.csv',index_label='wavelength')
            scores,variance,loadings=plot_spectral_pca(out/'spectra_for_object_pca.csv',
                output_png=out/'objects_pca_kmeans.png',output_scores_csv=out/'objects_pca_scores.csv',
                n_clusters=n_clusters if n_clusters!='auto' or pca_candidates.shape[1]>3 else 2,
                show=False)
        except Exception as exc:print(f'WARNING object PCA/K-means: {exc}')
    else:
        print('PCA across objects skipped: requires at least 3 objects and 2 bands')
    if map_mode is not None:
        report=render_object_index_maps(main,mode=map_mode,indices=names,scale_config=color_config_csv,
            empirical_strategy=empirical_strategy,data_type=data_type,
            overwrite=(overwrite_maps or formula_changed))
        _make_object_collages(main,report)
    return df
