"""Stage 4: date-wise experimental comparisons of object mean spectra and indices.

Dates, genotypes and environments come solely from user supplied folder-name
codes; cases not matching exactly one code in EACH dimension are excluded and
logged. No implicit inference of experimental design or replicate independence.
"""
from __future__ import annotations
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from simple_xlsx import read_dataframe_xlsx, write_dataframe_xlsx
from create_experiment_collages import (
    _match_code, _validate_codes, _safe_name, create_experiment_collages)
from spectral_preprocessing import SPECTRAL_VARIANTS


def _save(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout()
    fig.savefig(path,dpi=155,bbox_inches='tight')
    plt.close(fig)


def _labels(items,codes,displays):
    index,status=_match_code(items,codes)
    return (codes[index] if index is not None else '',
            displays[index] if index is not None else '', status)


def _classify(objects, gc,gd,ec,ed,dc,dd):
    rows=[]
    for item in objects.itertuples(index=False):
        source=str(item.sample)
        basename=Path(source).name
        g,gl,gs=_labels(basename,gc,gd)
        e,el,es=_labels(basename,ec,ed)
        d,dl,ds=_labels(basename,dc,dd)
        status='ok' if gs==es==ds=='ok' else f'genotype_{gs};environment_{es};date_{ds}'
        rows.append(dict(object=str(item.object),sample=source,mask=str(item.mask),
            genotype_code=g,genotype_display=gl,
            environment_code=e,environment_display=el,
            date_code=d,date_display=dl,status=status))
    return pd.DataFrame(rows)


def _plot_group_spectra(group, spectra, gd,ed, date_display, variant, path):
    wl=spectra.index.to_numpy(float)
    fig,axs=plt.subplots(len(gd),1,figsize=(12,max(4,4*len(gd))),sharex=True,
                         sharey=True,squeeze=False)
    records=[]
    for gi,glabel in enumerate(gd):
        ax=axs[gi,0]
        for elabel in ed:
            members=group[(group.genotype_display==glabel)&(group.environment_display==elabel)]
            ids=[u for u in members['object'] if u in spectra.columns]
            if not ids:continue
            a=spectra[ids].to_numpy(dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',category=RuntimeWarning)
                mean=np.nanmean(a,axis=1)
                sd=np.nanstd(a,axis=1,ddof=1) if len(ids)>1 else np.full(len(wl),np.nan)
            ax.plot(wl,mean,label=f'{elabel} (n={len(ids)})',linewidth=1.7)
            color=ax.lines[-1].get_color()
            if len(ids)>1:ax.fill_between(wl,mean-sd,mean+sd,color=color,alpha=.16)
            for w,m,s in zip(wl,mean,sd):
                records.append(dict(date_display=date_display,genotype_display=glabel,
                    environment_display=elabel,wavelength=w,n_objects=len(ids),
                    mean=m,sd=s))
        ax.set_title(glabel)
        ax.set_ylabel('Mean-normalized spectrum' if 'normalized' in variant else 'Relative reflectance / signal')
        ax.grid(alpha=.15)
        if ax.lines:ax.legend(fontsize=9)
    axs[-1,0].set_xlabel('Wavelength (nm)')
    fig.suptitle(f'{date_display} — mean spectra ± SD ({variant})',fontsize=14)
    _save(fig,path)
    return pd.DataFrame(records)


def _plot_index_boxscatter(group, scores, index, gd,ed,date_display,path):
    joined=group.merge(scores[['sample',index]],left_on='object',right_on='sample',how='left')
    fig,axs=plt.subplots(1,len(gd),figsize=(max(7,5*len(gd)),5),sharey=True,squeeze=False)
    rng=np.random.default_rng(7)
    for gi,glabel in enumerate(gd):
        ax=axs[0,gi]
        box_data=[];positions=[]
        for ei,elabel in enumerate(ed):
            vals=pd.to_numeric(joined.loc[(joined.genotype_display==glabel)&
                (joined.environment_display==elabel),index],errors='coerce').dropna().to_numpy(dtype=float)
            if len(vals):
                box_data.append(vals);positions.append(ei+1)
                ax.scatter(ei+1+rng.uniform(-.11,.11,len(vals)),vals,s=25,alpha=.75,zorder=3)
        if box_data:
            ax.boxplot(box_data,positions=positions,widths=.47,patch_artist=False,
                       showfliers=False,medianprops={'linewidth':1.6})
        ax.set_xticks(range(1,len(ed)+1),ed,rotation=30,ha='right')
        ax.set_title(glabel)
        ax.grid(axis='y',alpha=.2)
    axs[0,0].set_ylabel(f'{index} (object mean spectrum)')
    fig.suptitle(f'{date_display} — {index} | points=object masks',fontsize=13)
    _save(fig,path)


def _pca_by_date(group, spectra, date_display, folder, scaling='center'):
    ids=[u for u in group['object'] if u in spectra.columns]
    if len(ids)<3 or spectra.shape[0]<2:
        print(f'PCA skipped for {date_display}: fewer than 3 objects or 2 bands')
        return
    frame=spectra[ids].copy()
    frame.index=frame.index.astype(float)
    frame=frame.interpolate(method='index',axis=0,limit_area='inside')
    good=frame.apply(lambda col: np.isfinite(col.to_numpy(float)).all())
    frame=frame.loc[:,good]
    if frame.shape[1]<3:
        print(f'PCA skipped for {date_display}: fewer than 3 complete spectra')
        return
    X=frame.T.to_numpy(float)
    if scaling=='center':X=X-X.mean(axis=0,keepdims=True)
    elif scaling=='standardize':
        mu=X.mean(axis=0,keepdims=True)
        scale=X.std(axis=0,keepdims=True)
        X=np.divide(X-mu,np.where(scale>0,scale,1))
    else:raise ValueError('pca scaling: center or standardize')
    if not np.any(X.std(axis=0)>1e-12):
        print(f'PCA skipped for {date_display}: all spectra identical')
        return
    model=PCA(n_components=2,random_state=42)
    xy=model.fit_transform(X)
    lookup=group.set_index('object')
    report=[]
    # PCA labels display strings are user-provided rather than raw codes.
    fig,ax=plt.subplots(figsize=(10,7))
    gnames=list(dict.fromkeys(group['genotype_display']))
    enames=list(dict.fromkeys(group['environment_display']))
    colors=plt.get_cmap('tab10')
    markers=['o','s','^','D','v','P','X','*']
    for i,u in enumerate(frame.columns):
        entry=lookup.loc[u]
        gi=gnames.index(entry['genotype_display'])
        ei=enames.index(entry['environment_display'])
        label=f"{entry['genotype_display']} / {entry['environment_display']}"
        ax.scatter(xy[i,0],xy[i,1],color=colors(gi%10),marker=markers[ei%len(markers)],
                   s=65,label=label)
        # short traceability label while avoiding technical codes as main group names
        ax.annotate(str(entry['mask']),xy[i],xytext=(4,4),textcoords='offset points',fontsize=7)
        report.append(dict(object=u,date_display=date_display,
          genotype_display=entry['genotype_display'],environment_display=entry['environment_display'],
          PC1=float(xy[i,0]),PC2=float(xy[i,1])))
    handles,labels=ax.get_legend_handles_labels()
    unique=dict(zip(labels,handles))
    ax.legend(unique.values(),unique.keys(),loc='best',fontsize=8)
    ax.set(xlabel=f'PC1 ({model.explained_variance_ratio_[0]*100:.1f}%)',
           ylabel=f'PC2 ({model.explained_variance_ratio_[1]*100:.1f}%)',
           title=f'{date_display} — PCA of selected mean spectra')
    ax.axhline(0,color='gray',alpha=.3);ax.axvline(0,color='gray',alpha=.3)
    _save(fig,folder/'objects_pca.png')
    records=pd.DataFrame(report)
    # K-means on full selected spectral vectors, not merely first 2 PCs.
    if len(frame.columns)>=4 and np.any(X.std(axis=0)>1e-12):
        trials=[]
        for k in range(2,min(8,len(frame.columns)-1)+1):
            try:
                labels_k=KMeans(n_clusters=k,n_init=10,random_state=42).fit_predict(X)
                if len(set(labels_k))<2 or len(set(labels_k))>=len(labels_k):continue
                score=float(silhouette_score(X,labels_k))
                trials.append((score,k,labels_k))
            except ValueError:continue
        if trials:
            best=max(trials,key=lambda a:a[0])
            records['cluster']=np.asarray(best[2],dtype=int)+1
            write_dataframe_xlsx(pd.DataFrame([dict(k=t[1],silhouette=t[0]) for t in trials]),
                                 folder/'objects_pca_cluster_diagnostics.xlsx')
    write_dataframe_xlsx(records,folder/'objects_pca_scores.xlsx')
    write_dataframe_xlsx(pd.DataFrame(dict(component=['PC1','PC2'],
            explained_variance_ratio=model.explained_variance_ratio_)),
            folder/'objects_pca_variance.xlsx')
    write_dataframe_xlsx(pd.DataFrame(dict(wavelength=frame.index.to_numpy(float),
            PC1_loading=model.components_[0],PC2_loading=model.components_[1])),
            folder/'objects_pca_loadings.xlsx')


def analyze_experiment_by_date(main_folder, *,
        genotypes_codes, genotypes_display,
        environment_codes, environment_display,
        dates_codes, dates_display,
        indices=None, spectral_variant=None, pca_scaling='center',
        create_collages=True, per_cell_columns=2, overwrite=True):
    """Create date-specific collages, object PCA, spectra ± SD, index box+scatter.

    Stage 3 must already exist. The user-defined object mask is the descriptive
    observation; it is NOT claimed to be an independent plant/biological repeat.
    """
    main=Path(main_folder).expanduser().resolve()
    out=main/'object_analysis'
    gc,gd=_validate_codes(genotypes_codes,genotypes_display,'genotypes')
    ec,ed=_validate_codes(environment_codes,environment_display,'environment')
    dc,dd=_validate_codes(dates_codes,dates_display,'dates')
    settings_path=out/'analysis_settings.json'
    if not settings_path.exists():raise FileNotFoundError('Stage 3 settings missing; run Stage 3 first')
    stage3=json.loads(settings_path.read_text(encoding='utf-8'))
    if spectral_variant is None:spectral_variant=stage3['spectral_variant']
    if spectral_variant not in SPECTRAL_VARIANTS:raise ValueError('Invalid spectral_variant')
    if spectral_variant!=stage3['spectral_variant']:
        raise ValueError('Stage 4 variant differs from Stage 3; rerun Stage 3 with that variant')
    objects=read_dataframe_xlsx(out/'objects.xlsx')
    spectra_frame=read_dataframe_xlsx(out/f'spectra_mean_{spectral_variant}.xlsx')
    spectra=spectra_frame.set_index('wavelength')
    spectra=spectra.apply(pd.to_numeric,errors='coerce')
    spectra.index=spectra.index.astype(float)
    index_frame=read_dataframe_xlsx(out/'indices_selected.xlsx')
    selected=read_dataframe_xlsx(out/'selected_indices.xlsx')
    names=list(selected['index']) if indices is None else list(dict.fromkeys(indices))
    missing=[n for n in names if n not in index_frame.columns]
    if missing:raise ValueError(f'Selected index tables are missing {missing}; rerun Stage 3')
    classification=_classify(objects,gc,gd,ec,ed,dc,dd)
    write_dataframe_xlsx(classification,out/'experiment_objects.xlsx')
    unclassified=classification[classification.status!='ok']
    if not unclassified.empty:
        print('WARNING: sample folder codes unmatched/ambiguous (excluded):')
        for row in unclassified.itertuples():print(f'  {row.sample}: {row.status}')
    if create_collages:
        create_experiment_collages(main,genotypes_codes=gc,genotypes_display=gd,
            environment_codes=ec,environment_display=ed,dates_codes=dc,dates_display=dd,
            indices=names,per_cell_columns=per_cell_columns,overwrite=overwrite)
    base=out/'by_date'
    base.mkdir(exist_ok=True,parents=True)
    summary=[]
    for date,display in zip(dc,dd):
        group=classification[(classification.status=='ok')& (classification.date_code==date)]
        dest=base/_safe_name(date)
        dest.mkdir(parents=True,exist_ok=True)
        if group.empty:
            print(f'No classified objects for date {display}')
            continue
        full=group.merge(index_frame,left_on='object',right_on='sample',how='left')
        write_dataframe_xlsx(full,dest/'index_observations.xlsx')
        spectrum_stats=_plot_group_spectra(group,spectra,gd,ed,display,spectral_variant,
                                           dest/'spectra_by_genotype_environment.png')
        write_dataframe_xlsx(spectrum_stats,dest/'group_spectrum_stats.xlsx')
        _pca_by_date(group,spectra,display,dest/'pca',scaling=pca_scaling)
        path=dest/'index_boxplots';path.mkdir(exist_ok=True)
        for name in names:
            _plot_index_boxscatter(group,index_frame,name,gd,ed,display,
                                   path/f'{_safe_name(name)}.png')
        summary.append(dict(date_code=date,date_display=display,n_object_masks=len(group),
                            n_samples=group['sample'].nunique(),n_indices=len(names),
                            spectrum_variant=spectral_variant))
        print(f'Completed date {display}: {len(group)} object masks, {len(names)} index plots')
    from datetime import datetime,timezone
    manifest={'stage4_completed_utc':datetime.now(timezone.utc).isoformat(),
        'stage3_completed_utc':stage3.get('stage3_completed_utc'),
        'stage4_stale_after_stage3_rerun':False,
        'stage':'date-wise experimental comparison',
        'spectral_variant':spectral_variant,'genotypes_codes':gc,'genotypes_display':gd,
        'environment_codes':ec,'environment_display':ed,
        'dates_codes':dc,'dates_display':dd,
        'independence_warning':'Mask/object rows and pixels are NOT automatically independent biological replicates.',
        'mean_spectra_statistics':'mean +/- sample SD of OBJECT mean spectra within each genotype/environment/date',
        'index_observations':'indices of selected mean spectra; use with care for non scale-invariant indices',
        'colour_hue':'Hue_LCh on RAW reflectance only, D65 and 400-780 nm; != HSV segmentation hue',
        'pca':'PCA fitted per observation date on selected mean spectra, not comparable coordinate systems between dates'}
    (out/'experiment_comparison_config.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    report=pd.DataFrame(summary)
    if not report.empty:write_dataframe_xlsx(report,out/'experiment_comparison_summary.xlsx')
    return report
