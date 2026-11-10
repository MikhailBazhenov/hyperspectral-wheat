"""Spatial PCA for a corrected hyperspectral NPZ cube (per-sample PCA).

Fitting uses sampled, finite, all-band-valid pixels; invalid scores are NaN.
PCA scores are *not* reflectance. Their signs can reverse between fits, and
separately fitted components must not be compared between different cubes.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from spectral_cube_reflectance import load_quant_cube


def generate_pca_maps(cube_path, *, n_components=8, output_dir=None,
                      fit_max_pixels=75000, batch_pixels=50000, seed=17,
                      mask_path=None, mask_array=None, cube_data=None, overwrite=True):
    import cv2
    cube_path = Path(cube_path)
    out = Path(output_dir) if output_dir else cube_path.parent.parent / 'pca_maps'
    if (out / 'explained_variance.csv').exists() and not overwrite:
        return out
    cube, wl, valid, kind = load_quant_cube(cube_path) if cube_data is None else cube_data
    h,w,b = cube.shape
    n_components = int(n_components)
    if not 1 <= n_components <= b:
        raise ValueError(f'n_components should be within 1..{b}')
    ok = np.all(valid & np.isfinite(cube), axis=2)
    if mask_path is not None and mask_array is not None:
        raise ValueError("Provide mask_path or mask_array, not both")
    if mask_array is not None:
        m=np.asarray(mask_array)
        if m.shape!=(h,w): raise ValueError(f"Mask shape {m.shape} differs from {(h,w)}")
        ok &= m.astype(bool)
    if mask_path is not None:
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None or m.shape != (h,w):
            raise ValueError(f'Invalid mask {mask_path}')
        ok &= m > 0
    flattened = cube.reshape(-1,b)
    chosen = np.flatnonzero(ok.ravel())
    if len(chosen) < n_components + 1:
        raise ValueError('Not enough all-band-valid pixels for PCA')
    rng = np.random.default_rng(seed)
    fit_ids = rng.choice(chosen, size=min(fit_max_pixels,len(chosen)),replace=False)
    model = PCA(n_components=n_components,svd_solver='randomized' if n_components < min(b,len(fit_ids)) else 'full',random_state=seed)
    model.fit(flattened[fit_ids].astype(np.float32))
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'component':range(1,n_components+1),'explained_variance_ratio':model.explained_variance_ratio_,
                  'cumulative_variance_ratio':np.cumsum(model.explained_variance_ratio_)}).to_csv(out/'explained_variance.csv',index=False)
    pd.DataFrame(model.components_, columns=[str(n) for n in wl],index=[f'PC{i+1}' for i in range(n_components)]).to_csv(out/'loadings.csv',index_label='component')
    # float32 scores stored separately for reproducibility; grayscale is display-only.
    scores = np.full((h*w,n_components),np.nan,dtype=np.float32)
    for start in range(0,len(chosen),batch_pixels):
        ids = chosen[start:start+batch_pixels]
        scores[ids] = model.transform(flattened[ids]).astype(np.float32)
    scores = scores.reshape(h,w,n_components)
    np.savez_compressed(out/'pca_scores.npz',scores=scores.astype('float32'),wavelengths=wl,
                        explained_variance_ratio=model.explained_variance_ratio_)
    channels=[]
    bounds=[]
    for k in range(n_components):
        plane=scores[:,:,k]
        pixels=plane[np.isfinite(plane)]
        lo,hi=np.percentile(pixels,[2,98])
        if hi<=lo: hi=lo+1e-6
        norm=(np.clip((plane-lo)/(hi-lo),0,1)*255)
        scaled=np.nan_to_num(norm,nan=0).astype('uint8')
        Image.fromarray(scaled).save(out/f'PC{k+1:02d}.png')
        channels.append(scaled)
        bounds.append((float(lo),float(hi)))
    for i in range(n_components-2):
        rgb=np.stack(channels[i:i+3],axis=2)
        rgb[~ok]=0
        Image.fromarray(rgb).save(out/f'RGB_PC{i+1}_{i+2}_{i+3}.png')
    pd.DataFrame({'component':range(1,n_components+1),'display_p02':[x[0] for x in bounds],
                  'display_p98':[x[1] for x in bounds]}).to_csv(out/'display_bounds.csv',index=False)
    return out
