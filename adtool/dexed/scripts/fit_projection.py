"""Fits and FREEZES the UMAP projections used by every mapping figure, so that separately
generated figures share one coordinate system and one set of axis bounds.

Why this exists: each mapping script used to fit its own UMAP on whatever data it happened to
load. Two figures produced that way cannot be compared point-to-point -- a cluster on the left
of one plot has nothing to do with the left of another, and the axis ranges differ. Fitting
once and calling .transform() everywhere afterwards fixes both problems.

Same convention as scripts/fit_grid.py (which freezes the PCA-4D coverage grid): saved once,
reused forever. Re-run it only if the reference pool itself changes, and regenerate every
mapping figure afterwards -- a stale pickle silently makes old and new figures incomparable.

Two projections are frozen, because the two spaces do NOT have the same scope:
  - Z  (38-D behaviour): shared by BOTH studies, so the fit pool includes the parametric runs,
        the human corpus AND the four latent variants. This is the projection in which
        study-1 and study-2 discoveries can legitimately be compared.
  - Theta (155-D parameters): parametric only. The latent runs' genome is a 256-D z, which is
        neither commensurable with Theta nor with the other variants' latent spaces, so it
        cannot enter this fit and must never be plotted on these axes.

t-SNE deliberately gets no frozen fit: scikit-learn's implementation has no out-of-sample
transform, so t-SNE figures are internally consistent only and can never be compared across
figures. UMAP is therefore the canonical projection for anything cross-figure.
"""
import sys
sys.path.insert(0, '.')
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import umap

SEED = 0
N_HUMAN = 4000
OUT = 'figures/projection_fit.pickle'

HUMAN_PRESETS_PICKLE = Path.home() / 'projects/spinvae2/synth/dexed_presets.df.pickle'
HUMAN_FEATURES_PICKLE = Path('/data2/anasynth_nonbp/longeot/spinvae2_datasets/Dexed/raw_timbre_features_ttb.df.pickle')

FULL_DIR = 'runs/long/full_7h'
RANDOM_DIR = 'runs/random/big4000'
LATENT_DIRS = {v: f'runs/latent/{v}' for v in
               ['avec_tout', 'sans_timbre', 'sans_KL', 'sans_les_deux']}


def load_theta_z(d):
    files = sorted(Path(d).rglob('discovery.json'))
    th, z = [], []
    for f in files:
        p = json.load(f.open())
        dyn = p['params']['dynamic_params']
        th.append([v for _, v in dyn['preset']])
        z.append(p['output'])
    return np.asarray(th, dtype=float), np.asarray(z, dtype=float)


def drop_nan(*arrays):
    m = ~np.isnan(arrays[-1]).any(axis=1)
    return [a[m] for a in arrays]


def main():
    from maps.DexedStatistics import Z_FEATURE_NAMES
    rng = np.random.default_rng(SEED)

    print('loading parametric runs...', flush=True)
    th_full, z_full = drop_nan(*load_theta_z(FULL_DIR))
    th_rand, z_rand = drop_nan(*load_theta_z(RANDOM_DIR))
    print(f'  full   {z_full.shape}')
    print(f'  random {z_rand.shape}')

    print('loading human corpus...', flush=True)
    presets = pd.read_pickle(HUMAN_PRESETS_PICKLE)[['preset_UID', 'params_values']]
    feats = pd.read_pickle(HUMAN_FEATURES_PICKLE)
    feats = feats[feats['variation'] == 0]
    df = feats.merge(presets, on='preset_UID', how='inner')
    th_human_all = np.stack(df['params_values'].to_numpy())
    z_human_all = df[Z_FEATURE_NAMES].to_numpy(dtype=float)
    th_human_all, z_human_all = drop_nan(th_human_all, z_human_all)
    idx = rng.choice(len(z_human_all), min(N_HUMAN, len(z_human_all)), replace=False)
    th_human, z_human = th_human_all[idx], z_human_all[idx]
    print(f'  human  {z_human.shape} (echantillon de {len(z_human_all)})')

    print('loading latent runs (Z only -- leur Theta est un z 256-D, hors de cette carte)...',
          flush=True)
    z_latent = {}
    for name, d in LATENT_DIRS.items():
        if Path(d).exists():
            _, z = load_theta_z(d)
            z_latent[name] = drop_nan(z)[0]
            print(f'  {name:14s} {z_latent[name].shape}')

    # ---- z-scoring reference: EVERYTHING that will ever be projected into Z ----
    z_pool = np.concatenate([z_full, z_rand, z_human] + list(z_latent.values()))
    mu, sigma = z_pool.mean(axis=0), z_pool.std(axis=0)
    sigma[sigma == 0] = 1.0
    zs = lambda M: (M - mu) / sigma

    print('\nfitting UMAP on Z (38-D z-score, pool complet)...', flush=True)
    reducer_z = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.0, random_state=SEED)
    emb_z_pool = reducer_z.fit_transform(zs(z_pool))

    print('fitting UMAP on Theta (155-D, parametrique seulement)...', flush=True)
    th_pool = np.concatenate([th_full, th_rand, th_human])
    reducer_th = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.0, random_state=SEED)
    emb_th_pool = reducer_th.fit_transform(th_pool)

    def bounds(E, pad=0.04):
        lo, hi = E.min(axis=0), E.max(axis=0)
        span = hi - lo
        return lo - pad * span, hi + pad * span

    fit = {
        'seed': SEED,
        'mu': mu, 'sigma': sigma,
        'reducer_z': reducer_z, 'reducer_theta': reducer_th,
        'bounds_z': bounds(emb_z_pool),
        'bounds_theta': bounds(emb_th_pool),
        'z_feature_names': list(Z_FEATURE_NAMES),
        'pool': {'full': FULL_DIR, 'random': RANDOM_DIR,
                 'human_n': int(len(z_human)), 'latent': list(z_latent.keys())},
        'n_pool_z': int(len(z_pool)), 'n_pool_theta': int(len(th_pool)),
    }
    with open(OUT, 'wb') as f:
        pickle.dump(fit, f)

    print(f'\nsaved {OUT}')
    print(f'  pool Z     : {len(z_pool)} points ({len(z_latent)} variantes latentes incluses)')
    print(f'  pool Theta : {len(th_pool)} points (parametrique uniquement)')
    print(f'  bornes Z     x {fit["bounds_z"][0][0]:.2f}..{fit["bounds_z"][1][0]:.2f}'
          f'  y {fit["bounds_z"][0][1]:.2f}..{fit["bounds_z"][1][1]:.2f}')
    print(f'  bornes Theta x {fit["bounds_theta"][0][0]:.2f}..{fit["bounds_theta"][1][0]:.2f}'
          f'  y {fit["bounds_theta"][0][1]:.2f}..{fit["bounds_theta"][1][1]:.2f}')


if __name__ == '__main__':
    main()
