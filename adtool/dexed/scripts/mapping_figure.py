"""Intervention-space <-> behaviour-space mapping figure, adapted from Figure 4 of
"AI-driven Automated Discovery Tools Reveal Diverse Behavioral Competencies of Biological
Networks" (Etcheverry et al.) and its companion tutorial notebook.

Idea: cluster the reached behaviours (Z, 38D z-scored) with HDBSCAN, then paint the SAME
cluster memberships onto a 2D projection of the sampled parameters (Theta, 155D). If distinct
parameter regions land in the same behaviour cluster, the map is redundant -- the regime
where goal-directed exploration is supposed to beat random search.

Three rows this time: IMGEP, Random, and the Human preset corpus (added as the "what humans
consider musically relevant" reference -- Theta comes from dexed_presets.df.pickle, joined on
preset_UID with the 38D features already used everywhere else in this project).

Run twice (t-SNE and UMAP) -- same clustering, two projection methods, to check the
qualitative reading isn't a t-SNE artefact. UMAP better preserves global structure than
t-SNE (see spinvae2/notebooks/dexed_audit_TT_ACTM_ttb.ipynb), so a diverging story between
the two would be a red flag.
"""

import json
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

N_PER_METHOD = 1000
SEED = 0

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


def project(X, method, seed):
    if method == 'tsne':
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, random_state=seed).fit_transform(X)
    elif method == 'umap':
        import umap
        return umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=seed).fit_transform(X)
    raise ValueError(method)


def make_figure(method):
    from maps.DexedStatistics import Z_FEATURE_NAMES

    rng = np.random.default_rng(SEED)

    print(f'[{method}] loading discoveries...')
    theta_imgep, z_imgep = load_theta_z('run_2000_v2/discoveries')
    theta_rs, z_rs = load_theta_z('random_2000_combined/discoveries')

    for name in ('imgep', 'rs'):
        theta, z = (theta_imgep, z_imgep) if name == 'imgep' else (theta_rs, z_rs)
        keep = ~np.isnan(z).any(axis=1)
        idx = rng.choice(np.where(keep)[0], N_PER_METHOD, replace=False)
        if name == 'imgep':
            theta_imgep, z_imgep = theta[idx], z[idx]
        else:
            theta_rs, z_rs = theta[idx], z[idx]

    print(f'[{method}] loading human corpus...')
    theta_human, z_human = load_human_theta_z(Z_FEATURE_NAMES, rng, N_PER_METHOD)

    print(f'IMGEP: {z_imgep.shape}, Random: {z_rs.shape}, Human: {z_human.shape}')

    z_all = np.concatenate([z_imgep, z_rs, z_human])
    mu, sigma = np.nanmean(z_all, axis=0), np.nanstd(z_all, axis=0)
    sigma[sigma == 0] = 1.0
    zn_imgep = (z_imgep - mu) / sigma
    zn_rs = (z_rs - mu) / sigma
    zn_human = (z_human - mu) / sigma

    print(f'[{method}] clustering (HDBSCAN, 38D z-scored)...')
    def cluster(Z):
        # Tuned empirically (2026-08-10): the three datasets have very different local densities
        # (Human = one compact blob, Random = spread out), so no single HDBSCAN config suits all
        # three -- min_cluster_size=10/eps=0.1 gave low noise on Human but IMGEP/Random still had
        # 14-17% unassigned points. min_samples=5 (denser core requirement) + eps=0.0 (no post-hoc
        # merging) was the best compromise found: noise drops to ~2-17% across all three while
        # cluster counts stay stable (not 0 or 70+, as with other configs tried).
        return hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, cluster_selection_epsilon=0.0).fit_predict(Z)
    labels_imgep, labels_rs, labels_human = cluster(zn_imgep), cluster(zn_rs), cluster(zn_human)
    for name, labels in [('IMGEP', labels_imgep), ('Random', labels_rs), ('Human', labels_human)]:
        print(f'  {name}: {labels.max() + 1} clusters, {np.sum(labels < 0)} noise pts')

    print(f'[{method}] projecting Theta (joint, 155D)...')
    theta_2d = project(np.concatenate([theta_imgep, theta_rs, theta_human]), method, SEED)
    ti_imgep = theta_2d[:N_PER_METHOD]
    ti_rs = theta_2d[N_PER_METHOD:2 * N_PER_METHOD]
    ti_human = theta_2d[2 * N_PER_METHOD:]

    print(f'[{method}] projecting Z (joint, 38D z-scored)...')
    z_2d = project(np.concatenate([zn_imgep, zn_rs, zn_human]), method, SEED)
    zi_imgep = z_2d[:N_PER_METHOD]
    zi_rs = z_2d[N_PER_METHOD:2 * N_PER_METHOD]
    zi_human = z_2d[2 * N_PER_METHOD:]

    print(f'[{method}] drawing...')
    cluster_colors = ['#4c78a8', '#f58518', '#54a24b', '#e45756', '#72b7b2',
                      '#eeca3b', '#b279a2', '#ff9da6', '#9d755d']
    noise_color = '#bbbbbb'

    fig = Figure(figsize=(11, 15), dpi=130)
    axes = fig.subplots(3, 2)
    rows = [('(a) Curiosity search (IMGEP)', labels_imgep, ti_imgep, zi_imgep),
            ('(b) Random search', labels_rs, ti_rs, zi_rs),
            ('(c) Human preset corpus', labels_human, ti_human, zi_human)]

    eps_theta = 0.015 * (theta_2d.max() - theta_2d.min())
    eps_z = 0.015 * (z_2d.max() - z_2d.min())

    for r, (title, labels, ti, zi) in enumerate(rows):
        ax_theta, ax_z = axes[r]
        for label in sorted(set(labels)):
            mask = labels == label
            if label < 0:
                color, zorder, do_contour = noise_color, 1, False
            else:
                color = cluster_colors[label % len(cluster_colors)]
                zorder, do_contour = 2, True
            for ax, pts, eps in ((ax_theta, ti, eps_theta), (ax_z, zi, eps_z)):
                if do_contour and mask.sum() >= 3:
                    for coords in contour_polys(pts[mask], eps):
                        ax.fill(coords[:, 0], coords[:, 1], color=color, alpha=0.15, zorder=0)
                        ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=1, zorder=0)
                ax.scatter(pts[mask][:, 0], pts[mask][:, 1], s=6, color=color, zorder=zorder)
        proj_name = 't-SNE' if method == 'tsne' else 'UMAP'
        ax_theta.set_title(f'{title}\nParameter space $\\Theta$ ({proj_name} of 155 params)', fontsize=10)
        ax_z.set_title(f'{title}\nBehaviour space $Z$ ({proj_name} of 38D descriptors)', fontsize=10)
        for ax, bg in ((ax_theta, '#fdf6dd'), (ax_z, '#e8f1fb')):
            ax.set_facecolor(bg)
            ax.set_xticks([]); ax.set_yticks([])

    proj_name = 't-SNE' if method == 'tsne' else 'UMAP'
    fig.suptitle('Mapping between parameter space and behaviour space\n'
                 f'(HDBSCAN clusters computed in 38D z-scored Z; {proj_name}; n={N_PER_METHOD} per '
                 'method; grey = HDBSCAN noise)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = f'figures/mapping_theta_z_figure_{method}.png'
    FigureCanvasAgg(fig).print_png(out)
    print(f'saved {out}')


if __name__ == '__main__':
    methods = sys.argv[1:] or ['tsne', 'umap']
    for m in methods:
        make_figure(m)
