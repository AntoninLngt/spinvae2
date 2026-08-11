"""Cumulative grid-coverage curve (Cully & Demiris-style QD diversity-vs-samples plot).

For each method, tracks how many distinct cells of a fixed grid (built on a low-dim PCA
projection of Z) have been visited after the first n discoveries, processed in chronological
order (not shuffled) -- as opposed to Vendi Score, which is order-independent and only looks at
a fixed-size endpoint sample. Repeated over multiple independent seeds per method to get a
mean +/- std band, since a single trajectory's coverage curve is itself a noisy realization
(see investigation notebook section 9's seed-variance finding).

Random has no natural "seed" structure in the same sense (each draw is iid, no goal-directed
trajectory) -- its band is built by bootstrap-resampling random orderings of the same fixed
2000-preset pool instead of relaunching runs.
"""
import sys
sys.path.insert(0, '.')
from pathlib import Path
import json
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

N_ITER = 200
BOOTSTRAP_SIZE = 100
CHECKPOINTS = list(range(5, N_ITER + 1, 5))
N_BINS_PER_DIM = 8
N_PCA_DIMS = 4
N_RANDOM_RESAMPLES = 20


def load_z_chronological(discoveries_dir, cap=None):
    files = sorted(Path(discoveries_dir).rglob('discovery.json'))
    if cap is not None:
        files = files[:cap]
    rows = []
    for f in files:
        d = json.load(f.open())
        rows.append(d['output'])
    return np.asarray(rows, dtype=float)


print('loading baseline/algomut seeds (0-19, 200 iter each, chronological order)...')
baseline_runs, algomut_runs = [], []
for seed in range(20):
    b_dir = 'baseline_seed0_true/discoveries' if seed == 0 else (
        f'run_2000_seed{seed}/discoveries' if seed in (1, 2) else f'coverage/baseline/seed{seed}/discoveries')
    a_dir = 'sweep_algomut/discoveries' if seed == 0 else (
        f'sweep_algomut_seed{seed}/discoveries' if seed in (1, 2) else f'coverage/algomut/seed{seed}/discoveries')
    baseline_runs.append(load_z_chronological(b_dir, cap=N_ITER))
    algomut_runs.append(load_z_chronological(a_dir, cap=N_ITER))

print('loading random pool...')
random_pool = load_z_chronological('random_2000_combined/discoveries')

# Drop NaN rows (a handful of degenerate presets) -- keep chronological order within each run
def drop_nan_keep_order(M):
    return M[~np.isnan(M).any(axis=1)]

baseline_runs = [drop_nan_keep_order(M) for M in baseline_runs]
algomut_runs = [drop_nan_keep_order(M) for M in algomut_runs]
random_pool = drop_nan_keep_order(random_pool)

print('fitting PCA (4D) on pooled data...')
pooled = np.concatenate(baseline_runs + algomut_runs + [random_pool])
mu, sigma = pooled.mean(axis=0), pooled.std(axis=0)
sigma[sigma == 0] = 1.0
pooled_n = (pooled - mu) / sigma
pca = PCA(n_components=N_PCA_DIMS, random_state=0).fit(pooled_n)

def to_pca(M):
    return pca.transform((M - mu) / sigma)

baseline_pca = [to_pca(M) for M in baseline_runs]
algomut_pca = [to_pca(M) for M in algomut_runs]
random_pool_pca = to_pca(random_pool)

print('defining grid bounds (percentile, on pooled PCA projection)...')
pooled_pca = np.concatenate(baseline_pca + algomut_pca + [random_pool_pca])
lo = np.percentile(pooled_pca, 0.5, axis=0)
hi = np.percentile(pooled_pca, 99.5, axis=0)


def cumulative_coverage(M_pca, checkpoints):
    clipped = np.clip(M_pca, lo, hi)
    edges = [np.linspace(lo[d], hi[d], N_BINS_PER_DIM + 1) for d in range(N_PCA_DIMS)]
    idx = np.stack([np.clip(np.digitize(clipped[:, d], edges[d][1:-1]), 0, N_BINS_PER_DIM - 1)
                     for d in range(N_PCA_DIMS)], axis=1)
    seen = set()
    out = []
    ptr = 0
    for k in checkpoints:
        k = min(k, len(idx))
        while ptr < k:
            seen.add(tuple(idx[ptr]))
            ptr += 1
        out.append(len(seen))
    return out


print('computing cumulative coverage curves...')
baseline_curves = np.array([cumulative_coverage(M, CHECKPOINTS) for M in baseline_pca])
algomut_curves = np.array([cumulative_coverage(M, CHECKPOINTS) for M in algomut_pca])

rng = np.random.default_rng(0)
random_curves = []
for _ in range(N_RANDOM_RESAMPLES):
    perm = rng.permutation(len(random_pool_pca))[:N_ITER]
    random_curves.append(cumulative_coverage(random_pool_pca[perm], CHECKPOINTS))
random_curves = np.array(random_curves)

print('drawing...')
fig = Figure(figsize=(8, 5.5), dpi=130)
ax = fig.add_subplot(111)

for curves, label, color in [
    (baseline_curves, 'baseline (n=20 seeds)', '#4c78a8'),
    (algomut_curves, 'algomut (n=20 seeds)', '#f58518'),
    (random_curves, 'random (n=20 resamples)', '#54a24b'),
]:
    m = curves.mean(axis=0)
    s = curves.std(axis=0)
    ax.plot(CHECKPOINTS, m, color=color, label=label, linewidth=2)
    ax.fill_between(CHECKPOINTS, m - s, m + s, color=color, alpha=0.2)

ax.axvline(BOOTSTRAP_SIZE, color='gray', linestyle='--', linewidth=1, label=f'fin du bootstrap (n={BOOTSTRAP_SIZE})')
ax.set_xlabel('nombre de decouvertes (ordre chronologique)')
ax.set_ylabel(f'cellules distinctes visitees (grille {N_BINS_PER_DIM}^{N_PCA_DIMS} sur PCA-4D de Z)')
ax.set_title('Couverture cumulative de grille vs nombre de decouvertes\n(moyenne +/- ecart-type, style Cully & Demiris)')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()

out = 'figures/grid_coverage_evolution.png'
FigureCanvasAgg(fig).print_png(out)
print('saved', out)

import pickle
pickle.dump({
    'checkpoints': CHECKPOINTS,
    'baseline_curves': baseline_curves,
    'algomut_curves': algomut_curves,
    'random_curves': random_curves,
}, open('figures/grid_coverage_evolution.pickle', 'wb'))
print('saved grid_coverage_evolution.pickle')
