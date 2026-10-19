"""PCA and optional K-means clustering of sample spectra.

Dependencies: numpy, pandas, matplotlib, seaborn, scikit-learn
Input: spectra_mean.csv with `wavelength` rows and one column per sample.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def plot_spectral_pca(
    spectra_csv,
    output_png=None,
    output_scores_csv=None,
    n_clusters=None,
    cluster_range=(2, 8),
    output_cluster_scores_csv=None,
    scaling="center",
    normalize_spectra=False,
    wavelength_range=None,
    random_state=42,
    annotate=True,
    show=True,
    figsize=(10, 7),
):
    """Perform PCA on spectra and optionally cluster samples with K-means.

    Parameters
    ----------
    spectra_csv : str or Path
        CSV with wavelength column and sample columns (wide format).
    output_png : str or Path, optional
        Save figure as PNG.
    output_scores_csv : str or Path, optional
        Save sample scores (PC1, PC2, cluster).
    n_clusters : int, "auto", or None
        Number of K-means clusters. "auto" selects the largest silhouette score;
        None disables clustering.
    cluster_range : tuple (min_k, max_k)
        Inclusive K range checked in auto mode; limited to n_samples - 1.
    output_cluster_scores_csv : str or Path, optional
        Save silhouette and inertia for each tested K in auto mode.
    scaling : {'center', 'standardize'}
        'center' subtracts band means (default). 'standardize' also divides by
        each band's standard deviation, increasing the weight of low-variance bands.
    normalize_spectra : bool
        If True, divide each sample spectrum by its positive mean *before* PCA;
        focuses on spectral shape rather than overall intensity.
    wavelength_range : tuple (low, high) or None
        Restrict wavelength range in nm, inclusive.
    random_state : int
        Seed for reproducible K-means initialization.
    annotate : bool
        Label each point with the sample name.
    show : bool
        Display figure in notebook (plt.show()).

    Returns
    -------
    scores : pandas.DataFrame
        One row per sample with PC1, PC2, and optional cluster (1-based labels).
    explained_variance : pandas.DataFrame
        Components and explained variance ratios.
    loadings : pandas.DataFrame
        Wavelengths and the PC1/PC2 coefficients.
    Notes: scores.attrs["cluster_diagnostics"] provides the auto-selection
    table; scores.attrs["selected_n_clusters"] provides the chosen K.

    Notes
    -----
    K-means operates on the SAME preprocessed full spectral data used by PCA,
    not just the first two PC scores. The plotted clusters are a 2D projection.
    Missing interior wavelength values are linearly interpolated; missing values
    at spectral endpoints cause an error (no extrapolation).
    """
    df = pd.read_csv(spectra_csv)
    if "wavelength" not in df.columns:
        raise ValueError("CSV must contain a 'wavelength' column.")
    if df.columns.duplicated().any():
        raise ValueError("Duplicate CSV column names.")
    df["wavelength"] = pd.to_numeric(df["wavelength"], errors="raise")
    df = df.sort_values("wavelength")
    if df["wavelength"].duplicated().any():
        raise ValueError("Duplicate wavelengths in CSV.")
    if wavelength_range is not None:
        lo, hi = wavelength_range
        df = df.loc[df["wavelength"].between(lo, hi)]
    if len(df) < 2:
        raise ValueError("At least two wavelength bands are required.")

    names = [c for c in df.columns if c != "wavelength"]
    if len(names) < 3:
        raise ValueError("At least three samples are required for a PC1-PC2 plot.")
    spectra = df[names].apply(pd.to_numeric, errors="raise")
    spectra.index = df["wavelength"].to_numpy(dtype=float)
    spectra = spectra.interpolate(method="index", axis=0, limit_area="inside")
    if not np.isfinite(spectra.to_numpy(dtype=float)).all():
        raise ValueError("Spectra contain NaN/inf at an endpoint or invalid values.")

    x = spectra.to_numpy(dtype=float).T  # samples x wavelengths
    if normalize_spectra:
        mean = x.mean(axis=1, keepdims=True)
        if np.any(mean <= 0):
            raise ValueError("Spectrum-mean normalization requires positive sample means.")
        x = x / mean
    if scaling == "center":
        x_scaled = StandardScaler(with_mean=True, with_std=False).fit_transform(x)
    elif scaling == "standardize":
        x_scaled = StandardScaler(with_mean=True, with_std=True).fit_transform(x)
    else:
        raise ValueError("scaling must be 'center' or 'standardize'.")

    pca = PCA(n_components=2)
    xy = pca.fit_transform(x_scaled)
    scores = pd.DataFrame({"sample": names, "PC1": xy[:, 0], "PC2": xy[:, 1]})
    cluster_diagnostics = None
    if n_clusters == "auto":
        if len(names) < 3:
            raise ValueError("Auto-clustering requires at least 3 samples.")
        min_k, max_k = map(int, cluster_range)
        if min_k < 2 or max_k < min_k:
            raise ValueError("cluster_range must satisfy 2 <= min_k <= max_k")
        max_k = min(max_k, len(names) - 1)
        if min_k > max_k:
            raise ValueError("No valid K in cluster_range for this sample count.")
        evaluations = []
        for k in range(min_k, max_k + 1):
            test_km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
            labels = test_km.fit_predict(x_scaled)
            distinct = len(np.unique(labels))
            sil = (float(silhouette_score(x_scaled, labels))
                   if 2 <= distinct < len(names) else np.nan)
            evaluations.append({"n_clusters": k, "silhouette_score": sil,
                                "inertia": float(test_km.inertia_)})
        cluster_diagnostics = pd.DataFrame(evaluations)
        valid_scores = cluster_diagnostics.dropna(subset=["silhouette_score"])
        if valid_scores.empty:
            raise ValueError("No valid clustering: data may contain identical spectra.")
        n_clusters = int(valid_scores.sort_values(
            ["silhouette_score", "n_clusters"], ascending=[False, True]
        ).iloc[0]["n_clusters"])
        print(f"Automatic K-means selection: K={n_clusters} "
              f"(silhouette={valid_scores.set_index('n_clusters').loc[n_clusters, 'silhouette_score']:.3f})")
    if n_clusters is not None:
        n_clusters = int(n_clusters)
        if not 2 <= n_clusters <= len(names):
            raise ValueError(f"n_clusters must be between 2 and {len(names)}.")
        # Cluster in original preprocessed spectral feature space, not the plot projection.
        km = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
        scores["cluster"] = km.fit_predict(x_scaled) + 1

    if output_cluster_scores_csv is not None and cluster_diagnostics is not None:
        p = Path(output_cluster_scores_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        cluster_diagnostics.to_csv(p, index=False)
    scores.attrs["cluster_diagnostics"] = cluster_diagnostics
    scores.attrs["selected_n_clusters"] = n_clusters

    explained_variance = pd.DataFrame({
        "component": ["PC1", "PC2"],
        "explained_variance_ratio": pca.explained_variance_ratio_,
    })
    loadings = pd.DataFrame({
        "wavelength": spectra.index.astype(int),
        "PC1": pca.components_[0],
        "PC2": pca.components_[1],
    })

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=figsize)
    if n_clusters is None:
        sns.scatterplot(data=scores, x="PC1", y="PC2", ax=ax, s=95, color="tab:blue")
    else:
        scores["cluster"] = scores["cluster"].astype(str)
        sns.scatterplot(data=scores, x="PC1", y="PC2", hue="cluster", palette="tab10",
                        s=100, ax=ax, edgecolor="white", linewidth=0.7)
        ax.legend(title="K-means cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
        scores["cluster"] = scores["cluster"].astype(int)
    if annotate:
        for row in scores.itertuples(index=False):
            ax.annotate(row.sample, (row.PC1, row.PC2), xytext=(5, 4),
                        textcoords="offset points", fontsize=8)
    ax.axhline(0, color="gray", lw=0.7, alpha=0.45)
    ax.axvline(0, color="gray", lw=0.7, alpha=0.45)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.set_title("PCA of mean hyperspectral spectra" + (" + K-means" if n_clusters else ""))
    fig.tight_layout()
    if output_png is not None:
        p = Path(output_png)
        if p.suffix.lower() != ".png":
            raise ValueError("output_png must be a PNG file")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")
    if output_scores_csv is not None:
        p = Path(output_scores_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(p, index=False)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return scores, explained_variance, loadings
