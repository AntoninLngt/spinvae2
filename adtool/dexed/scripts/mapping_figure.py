"""Intervention-space <-> behaviour-space mapping figure, adapted from Figure 4 of
"AI-driven Automated Discovery Tools Reveal Diverse Behavioral Competencies of Biological
Networks" (Etcheverry et al.) and its companion tutorial notebook.

Idea: cluster the reached behaviours (Z, 38D z-scored) with HDBSCAN, then paint the SAME
cluster memberships onto a 2D projection of the sampled parameters (Theta, 155D). If distinct
parameter regions land in the same behaviour cluster, the map is redundant -- the regime
where goal-directed exploration is supposed to beat random search.

Three rows: `full` (the confirmed fix -- logit_mutation + carrier_floor +
niche_curiosity_bias + filter_degenerate_parents + algorithm_mutation_prob). Since
2026-08-14 the `full` row is subsampled from the 4000-iteration long run
(runs/long/full_7h, fresh seed 7) rather than from a 1000-iteration confirmation run:
same configuration, but the 1000 points drawn for the map now come from a 4x larger and
seed-independent pool, so the map is less hostage to one trajectory. See
theta_investigation.ipynb sections 12-14), Random, and the Human preset corpus (added as the "what humans
consider musically relevant" reference -- Theta comes from dexed_presets.df.pickle, joined on
preset_UID with the 38D features already used everywhere else in this project). This
replaces the original IMGEP baseline row (runs/main/discoveries, the unfixed operator) now
that `full` is the reference configuration throughout the investigation -- the old baseline's
collapsed mapping is documented elsewhere (theta_distribution.png, theta_heatmap.png) and
does not need to be re-shown here.

Run twice (t-SNE and UMAP) -- same clustering, two projection methods, to check the
qualitative reading isn't a t-SNE artefact. UMAP better preserves global structure than
t-SNE (see spinvae2/notebooks/dexed_audit_TT_ACTM_ttb.ipynb), so a diverging story between
the two would be a red flag.
"""

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, '.')

import hdbscan
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from shapely.geometry import Point
from shapely.ops import unary_union

N_PER_METHOD = 4000
SEED = 0
PROJECTION_FIT = 'figures/projection_fit.pickle'
FULL_DIR = 'runs/long/full_7h'
RANDOM_DIR = 'runs/random/big4000'

HUMAN_PRESETS_PICKLE = Path.home() / 'projects/spinvae2/synth/dexed_presets.df.pickle'
HUMAN_FEATURES_PICKLE = Path('/data2/anasynth_nonbp/longeot/spinvae2_datasets/Dexed/raw_timbre_features_ttb.df.pickle')


def load_theta_z(discoveries_dir):
    files = sorted(Path(discoveries_dir).rglob('discovery.json'))
    thetas, zs = [], []
    for f in files:
        d = json.load(f.open())
        thetas.append([v for _, v in d['params']['dynamic_params']['preset']])
        zs.append(d['output'])
    return np.asarray(thetas, dtype=float), np.asarray(zs, dtype=float)


def load_human_theta_z(z_feature_names, rng, n):
    presets = pd.read_pickle(HUMAN_PRESETS_PICKLE)[['preset_UID', 'params_values']]
    features = pd.read_pickle(HUMAN_FEATURES_PICKLE)
    features = features[features['variation'] == 0]  # base presets only, no augmentation
    df = features.merge(presets, on='preset_UID', how='inner')
    idx = rng.choice(len(df), n, replace=False)
    df = df.iloc[idx]
    theta = np.stack(df['params_values'].to_numpy())
    z = df[z_feature_names].to_numpy(dtype=float)
    return theta, z


def contour_polys(points, eps):
    poly = unary_union([Point(p).buffer(eps) for p in points])
    poly = poly.buffer(eps * 5, join_style=1).buffer(-eps * 5, join_style=1)
    geoms = poly.geoms if poly.geom_type == 'MultiPolygon' else [poly]
    return [np.array(g.exterior.coords.xy).T for g in geoms]


def load_projection_fit():
    """The frozen UMAP fit (scripts/fit_projection.py). Shared by every mapping figure so that
    they use one coordinate system and one set of axis bounds."""
    with open(PROJECTION_FIT, 'rb') as f:
        return pickle.load(f)


def project(X, method, seed, space, fit=None):
    """space: 'theta' or 'z'.

    umap  -> transforms into the FROZEN embedding, so coordinates are comparable across
             figures and across runs of this script.
    tsne  -> joint fit, recomputed every time (scikit-learn's t-SNE has no out-of-sample
             transform). Internally consistent only: NEVER compare a t-SNE panel to another
             figure's, and do not read absolute positions from it.
    """
    if method == 'tsne':
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, random_state=seed).fit_transform(X)
    elif method == 'umap':
        reducer = fit['reducer_theta'] if space == 'theta' else fit['reducer_z']
        return reducer.transform(X)
    raise ValueError(method)


def make_figure(method):
    from maps.DexedStatistics import Z_FEATURE_NAMES

    rng = np.random.default_rng(SEED)
    fit = load_projection_fit()

    print(f'[{method}] loading discoveries...')
    theta_full, z_full = load_theta_z(FULL_DIR)
    theta_rs, z_rs = load_theta_z(RANDOM_DIR)

    for name in ('full', 'rs'):
        theta, z = (theta_full, z_full) if name == 'full' else (theta_rs, z_rs)
        keep = ~np.isnan(z).any(axis=1)
        n = min(N_PER_METHOD, keep.sum())
        idx = rng.choice(np.where(keep)[0], n, replace=False)
        if name == 'full':
            theta_full, z_full = theta[idx], z[idx]
        else:
            theta_rs, z_rs = theta[idx], z[idx]

    print(f'[{method}] loading human corpus...')
    theta_human, z_human = load_human_theta_z(Z_FEATURE_NAMES, rng, N_PER_METHOD)

    print(f'full: {z_full.shape}, Random: {z_rs.shape}, Human: {z_human.shape}')
    n_full, n_rs, n_human = len(z_full), len(z_rs), len(z_human)

    # z-scoring taken from the FROZEN fit, not recomputed here: recomputing it per figure was
    # the second reason two mapping figures could not be compared (different mu/sigma ->
    # different geometry, even with the same reducer).
    mu, sigma = fit['mu'], fit['sigma']
    zn_full = (z_full - mu) / sigma
    zn_rs = (z_rs - mu) / sigma
    zn_human = (z_human - mu) / sigma

    print(f'[{method}] clustering (HDBSCAN, 38D z-scored)...')
    def cluster(Z, min_cluster_size, min_samples, cluster_selection_epsilon=0.0):
        return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                                cluster_selection_epsilon=cluster_selection_epsilon).fit_predict(Z)
    # RETUNED 2026-08-14 (scratchpad/tune_hdbscan_joint.py grid search, same exercise as
    # mapping_figure_humanfit.py's retuning but redone here since the JOINT z-scoring reference
    # gives a different geometry): min_cluster_size=8/min_samples=1 roughly triples noise
    # reduction on full (3.1%->0.9%) and Random (7.7%->1.6%) at IDENTICAL cluster counts (3 and
    # 4 respectively). Human keeps its original config (15/5/eps=0) -- it was already the best
    # found for Human specifically (2 clusters, 1.6% noise); every retuned variant tried traded
    # that away for more, noisier clusters, which is not an improvement when the starting point
    # is already this clean.
    # HDBSCAN's min_cluster_size/min_samples are ABSOLUTE counts, so a config tuned at n=1000
    # silently becomes 4x stricter in relative terms at n=4000 -- which is exactly what
    # happened when this figure moved to 4000 points (human noise jumped 1.9% -> 10.5%).
    # Scaling them with n keeps the density criterion constant across figure sizes.
    k = N_PER_METHOD / 1000.0
    labels_full = cluster(zn_full, max(5, round(8 * k)), max(1, round(1 * k)))
    labels_rs = cluster(zn_rs, max(5, round(8 * k)), max(1, round(1 * k)))
    labels_human = cluster(zn_human, max(5, round(15 * k)), max(1, round(5 * k)))
    for name, labels in [('full', labels_full), ('Random', labels_rs), ('Human', labels_human)]:
        print(f'  {name}: {labels.max() + 1} clusters, {np.sum(labels < 0)} noise pts')

    print(f'[{method}] projecting Theta (155D)...')
    theta_2d = project(np.concatenate([theta_full, theta_rs, theta_human]), method, SEED,
                        'theta', fit)
    ti_full = theta_2d[:n_full]
    ti_rs = theta_2d[n_full:n_full + n_rs]
    ti_human = theta_2d[n_full + n_rs:]

    print(f'[{method}] projecting Z (38D z-scored)...')
    z_2d = project(np.concatenate([zn_full, zn_rs, zn_human]), method, SEED, 'z', fit)
    zi_full = z_2d[:n_full]
    zi_rs = z_2d[n_full:n_full + n_rs]
    zi_human = z_2d[n_full + n_rs:]

    print(f'[{method}] drawing...')
    cluster_colors = ['#4c78a8', '#f58518', '#54a24b', '#e45756', '#72b7b2',
                      '#eeca3b', '#b279a2', '#ff9da6', '#9d755d']
    noise_color = '#bbbbbb'

    fig = Figure(figsize=(11, 15), dpi=130)
    axes = fig.subplots(3, 2)
    rows = [('(a) Curiosity search (full)', labels_full, ti_full, zi_full),
            ('(b) Random search', labels_rs, ti_rs, zi_rs),
            ('(c) Human preset corpus', labels_human, ti_human, zi_human)]

    eps_theta = 0.015 * (theta_2d.max() - theta_2d.min())
    eps_z = 0.015 * (z_2d.max() - z_2d.min())

    # Backdrop of every dataset, drawn faintly in every panel. Without it, shared bounds make
    # each panel mostly empty: in Theta the parametric runs occupy ~18% of the shared width
    # (both centred at the same spot) while the human corpus sprawls over ~88% elsewhere, so a
    # panel showing only one of them is 80% white space with no indication of where it sits.
    backdrop_theta = np.concatenate([ti_full, ti_rs, ti_human])
    backdrop_z = np.concatenate([zi_full, zi_rs, zi_human])

    for r, (title, labels, ti, zi) in enumerate(rows):
        ax_theta, ax_z = axes[r]
        ax_theta.scatter(backdrop_theta[:, 0], backdrop_theta[:, 1], s=0.8, color='#e4e4e4',
                         zorder=0, rasterized=True)
        ax_z.scatter(backdrop_z[:, 0], backdrop_z[:, 1], s=0.8, color='#e4e4e4',
                     zorder=0, rasterized=True)
        for label in sorted(set(labels)):
            mask = labels == label
            if label < 0:
                color, zorder, do_contour = noise_color, 1, False
            else:
                color = cluster_colors[label % len(cluster_colors)]
                zorder, do_contour = 2, True
            for ax, pts, eps in ((ax_theta, ti, eps_theta), (ax_z, zi, eps_z)):
                # A contour around a handful of widely-scattered points draws a huge polygon
                # over a near-empty region, which reads as "this method covers all of that".
                # Only outline clusters dense enough for the outline to mean something.
                if do_contour and mask.sum() >= max(10, 0.005 * N_PER_METHOD):
                    for coords in contour_polys(pts[mask], eps):
                        ax.fill(coords[:, 0], coords[:, 1], color=color, alpha=0.15, zorder=0)
                        ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=1, zorder=0)
                ax.scatter(pts[mask][:, 0], pts[mask][:, 1], s=6, color=color, zorder=zorder)
        proj_name = 't-SNE' if method == 'tsne' else 'UMAP'
        ax_theta.set_title(f'{title}\nParameter space $\\Theta$ ({proj_name} of 155 params)', fontsize=10)
        ax_z.set_title(f'{title}\nBehaviour space $Z$ ({proj_name} of 38D descriptors)', fontsize=10)
        for ax, bg, key in ((ax_theta, '#fdf6dd', 'bounds_theta'), (ax_z, '#e8f1fb', 'bounds_z')):
            ax.set_facecolor(bg)
            ax.set_xticks([]); ax.set_yticks([])
            # Identical bounds on every row AND across figures (frozen fit). Without this the
            # panels auto-scale to their own content and a compact cloud looks as wide as a
            # sprawling one.
            if method == 'umap':
                b_lo, b_hi = fit[key]
                ax.set_xlim(b_lo[0], b_hi[0]); ax.set_ylim(b_lo[1], b_hi[1])

    proj_name = 't-SNE' if method == 'tsne' else 'UMAP'
    fig.suptitle('Mapping between parameter space and behaviour space\n'
                 f'(HDBSCAN clusters computed in 38D z-scored Z; {proj_name}; '
                 f'full n={n_full}, random n={n_rs}, human n={n_human}; grey = HDBSCAN noise)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = f'figures/mapping_theta_z_figure_{method}.png'
    FigureCanvasAgg(fig).print_png(out)
    print(f'saved {out}')


if __name__ == '__main__':
    methods = sys.argv[1:] or ['tsne', 'umap']
    for m in methods:
        make_figure(m)
