"""Variant of mapping_figure_humanfit.py: instead of colouring points by HDBSCAN cluster
(unsupervised, density-based, and unstable across datasets of very different local density --
see the "min_cluster_size tuned PER DATASET" comment in mapping_figure_humanfit.py and its
"full: 4 clusters, 365 noise pts" result), colour by GROUND-TRUTH instrument category, the
same approach used for Contribution II's latent_category_mapping.png
(spinvae2/notebooks/stage2_ablation_compare.ipynb, section 9): the human corpus carries real
category labels (instrument_labels_str in dexed_presets.df.pickle, one preset can have several,
we keep the first that matches the 8 best-populated categories: piano, string, bass, brass,
harmonic_perc, percussive, organ, guitar; anything else -> 'other').

`full` and Random discoveries have no such ground truth (they are synthesised presets, not
part of the human corpus), so each is given a PROXY category: the category of its nearest
human-corpus neighbour in native 38D z-scored Z space (k=1, cosine... no, Euclidean on
z-scored features, consistent with every other distance computation in this project). This is
an approximation, not a measurement -- flagged explicitly in the figure caption.

Same UMAP-fit-on-human-corpus-then-.transform() geometry as mapping_figure_humanfit.py, so the
two figures answer the same "where do full/Random fall in the human corpus's own map" question,
just with an interpretable label instead of an unstable unsupervised cluster id.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import umap
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.neighbors import NearestNeighbors

N_DISCOVERIES = 1000
SEED = 0

HUMAN_PRESETS_PICKLE = Path.home() / 'projects/spinvae2/synth/dexed_presets.df.pickle'
HUMAN_FEATURES_PICKLE = Path('/data2/anasynth_nonbp/longeot/spinvae2_datasets/Dexed/raw_timbre_features_ttb.df.pickle')

TOP_CATEGORIES = ['piano', 'string', 'bass', 'brass', 'harmonic_perc', 'percussive', 'organ', 'guitar']
CATEGORY_COLORS = {
    'piano': '#4c78a8', 'string': '#f58518', 'bass': '#54a24b', 'brass': '#e45756',
    'harmonic_perc': '#72b7b2', 'percussive': '#eeca3b', 'organ': '#b279a2', 'guitar': '#ff9da6',
    'other': '#cccccc',
}


def load_theta_z(discoveries_dir):
    files = sorted(Path(discoveries_dir).rglob('discovery.json'))
    thetas, zs = [], []
    for f in files:
        d = json.load(f.open())
        thetas.append([v for _, v in d['params']['dynamic_params']['preset']])
        zs.append(d['output'])
    return np.asarray(thetas, dtype=float), np.asarray(zs, dtype=float)


def top_category(labels):
    for cat in TOP_CATEGORIES:
        if cat in labels:
            return cat
    return 'other'


def load_human_theta_z_category(z_feature_names):
    presets = pd.read_pickle(HUMAN_PRESETS_PICKLE)[['preset_UID', 'params_values', 'instrument_labels_str']]
    features = pd.read_pickle(HUMAN_FEATURES_PICKLE)
    features = features[features['variation'] == 0]
    df = features.merge(presets, on='preset_UID', how='inner')
    theta = np.stack(df['params_values'].to_numpy())
    z = df[z_feature_names].to_numpy(dtype=float)
    category = np.array([top_category(labels) for labels in df['instrument_labels_str']])
    return theta, z, category


def contour_polys(points, eps):
    from shapely.geometry import Point
    from shapely.ops import unary_union
    poly = unary_union([Point(p).buffer(eps) for p in points])
    poly = poly.buffer(eps * 5, join_style=1).buffer(-eps * 5, join_style=1)
    geoms = poly.geoms if poly.geom_type == 'MultiPolygon' else [poly]
    return [np.array(g.exterior.coords.xy).T for g in geoms]


def main():
    from maps.DexedStatistics import Z_FEATURE_NAMES

    rng = np.random.default_rng(SEED)

    print('loading discoveries...')
    theta_full, z_full = load_theta_z('runs/confirm/full/seed0/discoveries')
    theta_rs, z_rs = load_theta_z('runs/random/combined/discoveries')
    for name in ('full', 'rs'):
        theta, z = (theta_full, z_full) if name == 'full' else (theta_rs, z_rs)
        keep = ~np.isnan(z).any(axis=1)
        n = min(N_DISCOVERIES, keep.sum())
        idx = rng.choice(np.where(keep)[0], n, replace=False)
        if name == 'full':
            theta_full, z_full = theta[idx], z[idx]
        else:
            theta_rs, z_rs = theta[idx], z[idx]

    print('loading FULL human corpus with category labels (no subsampling)...')
    theta_human, z_human, cat_human = load_human_theta_z_category(Z_FEATURE_NAMES)
    print(f'Human: {z_human.shape} (full corpus)')
    for cat in TOP_CATEGORIES + ['other']:
        print(f'  {cat:16s} {(cat_human == cat).sum()}')

    finite = ~np.isnan(z_human).any(axis=1)
    n_dropped = (~finite).sum()
    if n_dropped:
        print(f'dropping {n_dropped} human rows with NaN features (of {len(z_human)})')
    theta_human, z_human, cat_human = theta_human[finite], z_human[finite], cat_human[finite]

    # z-score using the human corpus alone as reference (it is the one UMAP is fit on, and the
    # one nearest-neighbour category lookup runs in)
    mu, sigma = z_human.mean(axis=0), z_human.std(axis=0)
    sigma[sigma == 0] = 1.0
    zn_human, zn_full, zn_rs = (z_human - mu) / sigma, (z_full - mu) / sigma, (z_rs - mu) / sigma

    print('assigning proxy categories to full/random via nearest human neighbour (native 38D)...')
    nn = NearestNeighbors(n_neighbors=1).fit(zn_human)
    cat_full = cat_human[nn.kneighbors(zn_full, return_distance=False)[:, 0]]
    cat_rs = cat_human[nn.kneighbors(zn_rs, return_distance=False)[:, 0]]
    for name, cats in [('full (proxy)', cat_full), ('Random (proxy)', cat_rs)]:
        print(f'  {name}:', {c: int((cats == c).sum()) for c in TOP_CATEGORIES + ['other']})

    # human corpus is large -- subsample for the scatter/legend only (does not affect the UMAP
    # fit, which stays on the full corpus)
    human_sub_idx = rng.choice(len(zn_human), N_DISCOVERIES, replace=False)

    print('fitting UMAP on Theta (human corpus, full), transforming full/Random...')
    reducer_theta = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=SEED)
    ti_human_full = reducer_theta.fit_transform(theta_human)
    ti_full = reducer_theta.transform(theta_full)
    ti_rs = reducer_theta.transform(theta_rs)
    ti_human = ti_human_full[human_sub_idx]

    print('fitting UMAP on Z (human corpus, full, z-scored), transforming full/Random...')
    reducer_z = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=SEED)
    zi_human_full = reducer_z.fit_transform(zn_human)
    zi_full = reducer_z.transform(zn_full)
    zi_rs = reducer_z.transform(zn_rs)
    zi_human = zi_human_full[human_sub_idx]

    print('drawing...')
    fig = Figure(figsize=(11, 15), dpi=130)
    axes = fig.subplots(3, 2)
    rows = [('(a) Curiosity search (full)', cat_full, ti_full, zi_full),
            ('(b) Random search', cat_rs, ti_rs, zi_rs),
            ('(c) Human preset corpus (n=1000 of 30145)', cat_human[human_sub_idx], ti_human, zi_human)]

    all_theta = np.concatenate([ti_human_full, ti_full, ti_rs])
    all_z = np.concatenate([zi_human_full, zi_full, zi_rs])
    eps_theta = 0.015 * (all_theta.max() - all_theta.min())
    eps_z = 0.015 * (all_z.max() - all_z.min())

    for r, (title, cats, ti, zi) in enumerate(rows):
        ax_theta, ax_z = axes[r]
        if r < 2:
            ax_theta.scatter(ti_human_full[:, 0], ti_human_full[:, 1], s=1, color='#eeeeee', zorder=0)
            ax_z.scatter(zi_human_full[:, 0], zi_human_full[:, 1], s=1, color='#eeeeee', zorder=0)
        # draw 'other' first (background), named categories on top with contours
        order = ['other'] + TOP_CATEGORIES
        for cat in order:
            mask = cats == cat
            if mask.sum() == 0:
                continue
            color = CATEGORY_COLORS[cat]
            zorder, do_contour = (1, False) if cat == 'other' else (2, True)
            for ax, pts, eps in ((ax_theta, ti, eps_theta), (ax_z, zi, eps_z)):
                if do_contour and mask.sum() >= 3:
                    for coords in contour_polys(pts[mask], eps):
                        ax.fill(coords[:, 0], coords[:, 1], color=color, alpha=0.15, zorder=0)
                        ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=1, zorder=0)
                ax.scatter(pts[mask][:, 0], pts[mask][:, 1], s=6, color=color, zorder=zorder,
                           label=cat if r == 2 else None)
        ax_theta.set_title(f'{title}\n$\\Theta$ (UMAP fit on human corpus, others transformed)', fontsize=9.5)
        ax_z.set_title(f'{title}\n$Z$ (UMAP fit on human corpus, others transformed)', fontsize=9.5)
        for ax, bg in ((ax_theta, '#fdf6dd'), (ax_z, '#e8f1fb')):
            ax.set_facecolor(bg)
            ax.set_xticks([]); ax.set_yticks([])

    axes[2, 1].legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    fig.suptitle('Where do full / Random discoveries fall relative to human instrument categories?\n'
                 '(UMAP fit on the full 30145-preset human corpus; full/Random projected in via '
                 '.transform(); ground-truth category for Human, nearest-human-neighbour PROXY '
                 'category for full/Random; grey = light human backdrop / "other" category)',
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])
    out = 'figures/mapping_theta_z_figure_category.png'
    FigureCanvasAgg(fig).print_png(out)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
