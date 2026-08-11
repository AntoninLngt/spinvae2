"""
Visualization tool for Dexed IMGEP discoveries produced by ExperimentPipeline
(adtool.callbacks.on_discovery_callbacks.save_discovery_on_disk.SaveDiscoveryOnDisk).

Reads every discovery.json in a discoveries/ folder and produces:
  - for low-dimensional Z (<=12 dims, e.g. the old 7D ACTM-only space): a pairwise scatter
    matrix of every dimension, coloured by discovery order.
  - for higher-dimensional Z (e.g. the 38D canonical ACTM+TT space): a correlation heatmap
    instead (an NxN scatter matrix is impractical above ~12 dims).
  - a 2D PCA projection of Z, coloured by discovery order (shows exploration progression),
    for any dimensionality.
  - simple coverage metrics (grid occupancy, mean nearest-neighbour distance) in native Z
    space, using the same methodology as spinvae2's dataset audit notebook -- so the numbers
    are directly comparable to the human-preset baseline (when Z is the 38D canonical space).

Z's dimension names are pulled from DexedStatistics.Z_FEATURE_NAMES when the dimensionality
matches (38D, current canonical space); falls back to the legacy 7D ACTM-only names, then to
generic dim_i labels for anything else (e.g. old runs, or future Z spaces).

Usage:
    python examples/dexed/visualize_discoveries.py --discoveries_dir examples/dexed/run_500/discoveries
    python examples/dexed/visualize_discoveries.py --discoveries_dir examples/dexed/run_500/discoveries \
        --out_dir examples/dexed/run_500/figures
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ACTM_ONLY_DIMS = ["hardness", "depth", "brightness", "roughness", "warmth", "sharpness", "boominess"]

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _dim_names(n_dims: int):
    try:
        from maps.DexedStatistics import Z_FEATURE_NAMES
        if n_dims == len(Z_FEATURE_NAMES):
            return list(Z_FEATURE_NAMES)
    except Exception:
        pass
    if n_dims == len(ACTM_ONLY_DIMS):
        return list(ACTM_ONLY_DIMS)
    return [f"dim_{i}" for i in range(n_dims)]


def load_discoveries(discoveries_dir: Path):
    """ Returns (Z, order, wav_paths) sorted by discovery order (folder name's idx_N). """
    entries = []
    for folder in sorted(discoveries_dir.iterdir()):
        discovery_file = folder / "discovery.json"
        if not discovery_file.exists():
            continue
        with open(discovery_file) as f:
            d = json.load(f)
        z = np.asarray(d["output"], dtype=float)
        # folder name looks like "<timestamp>_exp_0_idx_7_seed_42" -- idx gives discovery order
        idx = int(folder.name.split("_idx_")[1].split("_seed_")[0])
        wav_path = folder / "visu.wav"
        entries.append((idx, z, wav_path if wav_path.exists() else None))
    entries.sort(key=lambda e: e[0])
    order = np.array([e[0] for e in entries])
    Z = np.stack([e[1] for e in entries])
    wav_paths = [e[2] for e in entries]
    return Z, order, wav_paths


def plot_scatter_matrix(Z: np.ndarray, order: np.ndarray, out_path: Path, dim_names=None, max_dims: int = 12):
    n_dims = Z.shape[1]
    dim_names = dim_names or _dim_names(n_dims)

    if n_dims > max_dims:
        plot_correlation_heatmap(Z, out_path, dim_names)
        return

    fig, axes = plt.subplots(n_dims, n_dims, figsize=(2.2 * n_dims, 2.2 * n_dims))
    sc = None
    for i in range(n_dims):
        for j in range(n_dims):
            ax = axes[i, j]
            if i == j:
                ax.hist(Z[:, i], bins=min(10, len(Z)), color="steelblue")
            else:
                sc = ax.scatter(Z[:, j], Z[:, i], c=order, cmap="viridis", s=25)
            if i == n_dims - 1:
                ax.set_xlabel(dim_names[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(dim_names[i], fontsize=8)
            ax.tick_params(labelsize=6)
    if sc is not None:
        fig.colorbar(sc, ax=axes, label="discovery order", shrink=0.6)
    fig.suptitle(f"Z feature pairs across {len(Z)} discoveries (colour = discovery order)")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_correlation_heatmap(Z: np.ndarray, out_path: Path, dim_names=None):
    """ NxN scatter matrices are impractical past ~12 dims -- a correlation heatmap scales
    much better and still shows redundancy/structure across the 38D canonical space. """
    n_dims = Z.shape[1]
    dim_names = dim_names or _dim_names(n_dims)
    corr = np.corrcoef(Z, rowvar=False)

    fig, ax = plt.subplots(figsize=(0.35 * n_dims + 3, 0.35 * n_dims + 3))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n_dims)); ax.set_xticklabels(dim_names, rotation=90, fontsize=6)
    ax.set_yticks(range(n_dims)); ax.set_yticklabels(dim_names, fontsize=6)
    plt.colorbar(im, ax=ax, label="Pearson correlation", shrink=0.8)
    ax.set_title(f"Correlation between Z dimensions ({len(Z)} discoveries)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_pca(Z: np.ndarray, order: np.ndarray, out_path: Path):
    if len(Z) < 3:
        print(f"Skipping PCA plot: only {len(Z)} discoveries (need >= 3)")
        return
    from sklearn.decomposition import PCA
    Zc = Z - Z.mean(axis=0)
    pca = PCA(n_components=min(2, Zc.shape[1]))
    Zp = pca.fit_transform(Zc)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(Zp[:, 0], Zp[:, 1] if Zp.shape[1] > 1 else np.zeros(len(Zp)),
                     c=order, cmap="viridis", s=40)
    for i, (x, y) in enumerate(zip(Zp[:, 0], Zp[:, 1] if Zp.shape[1] > 1 else np.zeros(len(Zp)))):
        ax.annotate(str(order[i]), (x, y), fontsize=7, alpha=0.7)
    plt.colorbar(sc, label="discovery order")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"PCA of Z ({Z.shape[1]}D, {len(Z)} discoveries) -- "
                 f"{100*pca.explained_variance_ratio_.sum():.0f}% variance explained")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")


def coverage_metrics(Z: np.ndarray, n_bins: int = 10):
    """ Same methodology as spinvae2/notebooks/dexed_audit_TT_ACTM.ipynb (native-space NN
    distance + grid occupancy on the first 2 PCA components), for direct comparability. """
    if len(Z) < 2:
        return None
    from scipy.spatial import cKDTree
    tree = cKDTree(Z)
    dists, _ = tree.query(Z, k=2)
    mean_nn_dist = float(np.mean(dists[:, 1]))

    if len(Z) >= 3:
        from sklearn.decomposition import PCA
        Zp = PCA(n_components=2).fit_transform(Z - Z.mean(axis=0))
        x_edges = np.linspace(Zp[:, 0].min(), Zp[:, 0].max(), n_bins + 1)
        y_edges = np.linspace(Zp[:, 1].min(), Zp[:, 1].max(), n_bins + 1)
        H, _, _ = np.histogram2d(Zp[:, 0], Zp[:, 1], bins=[x_edges, y_edges])
        occ_ratio = (H > 0).sum() / (n_bins * n_bins)
    else:
        occ_ratio = None

    return {"n_discoveries": len(Z), "mean_nn_distance": mean_nn_dist,
            "grid_occupancy": occ_ratio, "n_bins": n_bins}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--discoveries_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None,
                         help="Defaults to <discoveries_dir>/../figures")
    args = parser.parse_args()

    discoveries_dir = Path(args.discoveries_dir)
    out_dir = Path(args.out_dir) if args.out_dir else discoveries_dir.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    Z, order, wav_paths = load_discoveries(discoveries_dir)
    dim_names = _dim_names(Z.shape[1])
    print(f"Loaded {len(Z)} discoveries from {discoveries_dir}")
    print(f"Z shape: {Z.shape} (dims: {dim_names})")

    plot_scatter_matrix(Z, order, out_dir / "z_scatter_matrix.png", dim_names)
    plot_pca(Z, order, out_dir / "z_pca.png")

    metrics = coverage_metrics(Z)
    print("\n--- Coverage metrics (native Z space) ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
