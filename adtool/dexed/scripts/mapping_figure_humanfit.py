"""Variant of mapping_figure.py: UMAP is fit ONCE on the full human corpus (all presets, not
subsampled -- so its shape is faithful to spinvae2/notebooks/dexed_audit_TT_ACTM_ttb.ipynb),
then IMGEP and Random are projected into that SAME learned embedding via reducer.transform()
(the exact pattern already used in imgep_coverage_vs_human_baseline.ipynb). Answers a
different question than mapping_figure.py's joint fit: not "how do the three compare on a
shared map fit on all of them", but "where do IMGEP/Random discoveries fall relative to the
human corpus's own, already-known geometry".

t-SNE has no out-of-sample transform in scikit-learn, so this variant is UMAP-only.
HDBSCAN clustering (for the Theta<->Z colour coding) is still computed independently in
native 38D per dataset, same as mapping_figure.py -- unaffected by the projection choice.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, '.')

import hdbscan
import numpy as np
import pandas as pd
import umap
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from shapely.geometry import Point
from shapely.ops import unary_union

N_DISCOVERIES = 1000  # IMGEP / Random, transformed into the human-fit embedding
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


def load_human_theta_z_full(z_feature_names):
    presets = pd.read_pickle(HUMAN_PRESETS_PICKLE)[['preset_UID', 'params_values']]
    features = pd.read_pickle(HUMAN_FEATURES_PICKLE)
    features = features[features['variation'] == 0]
    df = features.merge(presets, on='preset_UID', how='inner')
    theta = np.stack(df['params_values'].to_numpy())
    z = df[z_feature_names].to_numpy(dtype=float)
    return theta, z


def contour_polys(points, eps):
    poly = unary_union([Point(p).buffer(eps) for p in points])
    poly = poly.buffer(eps * 5, join_style=1).buffer(-eps * 5, join_style=1)
    geoms = poly.geoms if poly.geom_type == 'MultiPolygon' else [poly]
    return [np.array(g.exterior.coords.xy).T for g in geoms]


def main():
    from maps.DexedStatistics import Z_FEATURE_NAMES

    rng = np.random.default_rng(SEED)

    print('loading discoveries...')
    theta_imgep, z_imgep = load_theta_z('run_2000_v2/discoveries')
    theta_rs, z_rs = load_theta_z('random_2000_combined/discoveries')
    for name in ('imgep', 'rs'):
        theta, z = (theta_imgep, z_imgep) if name == 'imgep' else (theta_rs, z_rs)
        keep = ~np.isnan(z).any(axis=1)
        idx = rng.choice(np.where(keep)[0], N_DISCOVERIES, replace=False)
        if name == 'imgep':
            theta_imgep, z_imgep = theta[idx], z[idx]
        else:
            theta_rs, z_rs = theta[idx], z[idx]

    print('loading FULL human corpus (no subsampling)...')
    theta_human, z_human = load_human_theta_z_full(Z_FEATURE_NAMES)
    print(f'Human: {z_human.shape} (full corpus)')

    # A handful of tt_* values are NaN in the raw human corpus (same known issue as the
    # DexedStatistics NaN fix elsewhere in this project -- harmonic descriptors failing on a
    # few edge-case presets). np.nanmean/nanstd below fix the aggregate stats, but UMAP itself
    # (unlike HDBSCAN, which silently drops NaN rows) refuses any NaN input -- drop the
    # affected rows explicitly, negligible loss out of 30145.
    finite = ~np.isnan(z_human).any(axis=1)
    n_dropped = (~finite).sum()
    if n_dropped:
        print(f'dropping {n_dropped} human rows with NaN features (of {len(z_human)})')
    theta_human, z_human = theta_human[finite], z_human[finite]

    # z-score using the human corpus alone as reference (it is the one UMAP is fit on)
    mu, sigma = z_human.mean(axis=0), z_human.std(axis=0)
    sigma[sigma == 0] = 1.0
    zn_human, zn_imgep, zn_rs = (z_human - mu) / sigma, (z_imgep - mu) / sigma, (z_rs - mu) / sigma

    print('clustering (HDBSCAN, native 38D z-scored, per dataset)...')
    def cluster(Z, min_cluster_size, min_samples):
        # min_cluster_size/min_samples tuned PER DATASET (2026-08-10), not globally: IMGEP and
        # Random need min_cluster_size=15/min_samples=5 or Random explodes into 20+ spurious
        # clusters (its 1000 points are spread thin across the space). But that same config,
        # applied to the human 1000-point subsample below, produces 87% noise -- subsampling
        # 1000/30142 from an already-compact corpus leaves that subsample locally sparse in a way
        # the full corpus isn't, so a "15 points within eps, 5 as core" density threshold is far
        # too strict there; min_cluster_size=10/min_samples=1 was needed. No single config was
        # found that works for all three (see investigation notebook 4.2/4.3 for the grid search)
        # -- this is a genuine density mismatch between datasets, not a bug to hide behind one
        # global default.
        return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                                cluster_selection_epsilon=0.0).fit_predict(Z)
    labels_imgep, labels_rs = cluster(zn_imgep, 15, 5), cluster(zn_rs, 15, 5)
    # human corpus is large -- HDBSCAN on 30k points is slow with these settings, subsample for
    # the colour-coding only (does not affect the UMAP fit, which stays on the full corpus)
    human_sub_idx = rng.choice(len(zn_human), N_DISCOVERIES, replace=False)
    labels_human = cluster(zn_human[human_sub_idx], 10, 1)
    for name, labels in [('IMGEP', labels_imgep), ('Random', labels_rs), ('Human (n=1000 subset)', labels_human)]:
        print(f'  {name}: {labels.max() + 1} clusters, {np.sum(labels < 0)} noise pts')

    print('fitting UMAP on Theta (human corpus, full), transforming IMGEP/Random...')
    reducer_theta = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=SEED)
    ti_human_full = reducer_theta.fit_transform(theta_human)
    ti_imgep = reducer_theta.transform(theta_imgep)
    ti_rs = reducer_theta.transform(theta_rs)
    ti_human = ti_human_full[human_sub_idx]

    print('fitting UMAP on Z (human corpus, full, z-scored), transforming IMGEP/Random...')
    reducer_z = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=SEED)
    zi_human_full = reducer_z.fit_transform(zn_human)
    zi_imgep = reducer_z.transform(zn_imgep)
    zi_rs = reducer_z.transform(zn_rs)
    zi_human = zi_human_full[human_sub_idx]

    print('drawing...')
    cluster_colors = ['#4c78a8', '#f58518', '#54a24b', '#e45756', '#72b7b2',
                      '#eeca3b', '#b279a2', '#ff9da6', '#9d755d']
    noise_color = '#bbbbbb'

    fig = Figure(figsize=(11, 15), dpi=130)
    axes = fig.subplots(3, 2)
    rows = [('(a) Curiosity search (IMGEP)', labels_imgep, ti_imgep, zi_imgep),
            ('(b) Random search', labels_rs, ti_rs, zi_rs),
            ('(c) Human preset corpus (n=1000 of 30145)', labels_human, ti_human, zi_human)]

    # bounds/eps from the FULL human embedding, since IMGEP/Random are transformed into it
    # (they may fall outside its convex hull -- that is itself an informative signal)
    all_theta = np.concatenate([ti_human_full, ti_imgep, ti_rs])
    all_z = np.concatenate([zi_human_full, zi_imgep, zi_rs])
    eps_theta = 0.015 * (all_theta.max() - all_theta.min())
    eps_z = 0.015 * (all_z.max() - all_z.min())

    for r, (title, labels, ti, zi) in enumerate(rows):
        ax_theta, ax_z = axes[r]
        # faint backdrop of the full human corpus on every row, for scale reference
        if r < 2:
            ax_theta.scatter(ti_human_full[:, 0], ti_human_full[:, 1], s=1, color='#dddddd', zorder=0)
            ax_z.scatter(zi_human_full[:, 0], zi_human_full[:, 1], s=1, color='#dddddd', zorder=0)
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
        ax_theta.set_title(f'{title}\n$\\Theta$ (UMAP fit on human corpus, others transformed)', fontsize=9.5)
        ax_z.set_title(f'{title}\n$Z$ (UMAP fit on human corpus, others transformed)', fontsize=9.5)
        for ax, bg in ((ax_theta, '#fdf6dd'), (ax_z, '#e8f1fb')):
            ax.set_facecolor(bg)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle('Where do IMGEP / Random discoveries fall in the human corpus\'s own map?\n'
                 '(UMAP fit on the full 30145-preset human corpus; IMGEP/Random projected in via '
                 '.transform(); light grey = full human corpus backdrop; HDBSCAN clusters in native 38D)',
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = 'figures/mapping_theta_z_figure_umap_humanfit.png'
    FigureCanvasAgg(fig).print_png(out)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
