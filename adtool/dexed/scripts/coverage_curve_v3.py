"""v3 of coverage_curve.py: unified 5-curve comparison -- baseline, algomut, big_jump,
silence_sink, random -- on the same PCA fit / grid bounds for a fair, single-figure summary.

silence_sink runs get an extra filtering step: their silence-sink points (offset +1e6, see
DexedStatistics.silence_sink) are dropped before both the PCA fit and the cumulative coverage
count, the same way drop_nan drops NaN rows -- they're a deliberately out-of-band marker, not a
real behaviour. Checkpoints are still indexed against the ORIGINAL 200-iteration budget (a
silence_sink run that only produced 160 valid points after filtering simply plateaus over the
last 40 checkpoints, same as running out of chronological data) -- this is intentionally a fair
like-for-like comparison against how much of the ITERATION BUDGET (not the valid-sample budget)
each method actually spent finding new cells.
"""
import sys
sys.path.insert(0, '.')
from pathlib import Path
import json
import pickle
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


def drop_nan_keep_order(M):
    return M[~np.isnan(M).any(axis=1)]


def drop_sink_keep_order(M):
    return M[~(np.abs(M) > 1e4).any(axis=1)]


print('loading baseline/algomut/bigjump/silencesink seeds (0-19, 200 iter each, chronological order)...')
baseline_runs, algomut_runs, bigjump_runs, silencesink_runs = [], [], [], []
for seed in range(20):
    b_dir = 'runs/multiseed/baseline_seed0_true/discoveries' if seed == 0 else (
        f'runs/multiseed/seed{seed}/discoveries' if seed in (1, 2) else f'runs/coverage/baseline/seed{seed}/discoveries')
    a_dir = 'runs/algomut/seed0/discoveries' if seed == 0 else (
        f'runs/algomut/seed{seed}/discoveries' if seed in (1, 2) else f'runs/coverage/algomut/seed{seed}/discoveries')
    j_dir = f'runs/coverage/bigjump/seed{seed}/discoveries'
    s_dir = f'runs/coverage/silencesink/seed{seed}/discoveries'
    baseline_runs.append(load_z_chronological(b_dir, cap=N_ITER))
    algomut_runs.append(load_z_chronological(a_dir, cap=N_ITER))
    bigjump_runs.append(load_z_chronological(j_dir, cap=N_ITER))
    silencesink_runs.append(load_z_chronological(s_dir, cap=N_ITER))

print('loading random pool...')
random_pool = load_z_chronological('runs/random/combined/discoveries')

baseline_runs = [drop_nan_keep_order(M) for M in baseline_runs]
algomut_runs = [drop_nan_keep_order(M) for M in algomut_runs]
bigjump_runs = [drop_nan_keep_order(M) for M in bigjump_runs]
silencesink_runs = [drop_sink_keep_order(drop_nan_keep_order(M)) for M in silencesink_runs]
random_pool = drop_nan_keep_order(random_pool)

print('fitting PCA (4D) on pooled data...')
pooled = np.concatenate(baseline_runs + algomut_runs + bigjump_runs + silencesink_runs + [random_pool])
mu, sigma = pooled.mean(axis=0), pooled.std(axis=0)
sigma[sigma == 0] = 1.0
pooled_n = (pooled - mu) / sigma
pca = PCA(n_components=N_PCA_DIMS, random_state=0).fit(pooled_n)


def to_pca(M):
    return pca.transform((M - mu) / sigma)


baseline_pca = [to_pca(M) for M in baseline_runs]
algomut_pca = [to_pca(M) for M in algomut_runs]
bigjump_pca = [to_pca(M) for M in bigjump_runs]
silencesink_pca = [to_pca(M) for M in silencesink_runs]
random_pool_pca = to_pca(random_pool)

print('defining grid bounds (percentile, on pooled PCA projection)...')
pooled_pca = np.concatenate(baseline_pca + algomut_pca + bigjump_pca + silencesink_pca + [random_pool_pca])
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
bigjump_curves = np.array([cumulative_coverage(M, CHECKPOINTS) for M in bigjump_pca])
silencesink_curves = np.array([cumulative_coverage(M, CHECKPOINTS) for M in silencesink_pca])

rng = np.random.default_rng(0)
random_curves = []
for _ in range(N_RANDOM_RESAMPLES):
    perm = rng.permutation(len(random_pool_pca))[:N_ITER]
    random_curves.append(cumulative_coverage(random_pool_pca[perm], CHECKPOINTS))
random_curves = np.array(random_curves)

print('drawing...')
fig = Figure(figsize=(8.5, 6), dpi=130)
ax = fig.add_subplot(111)

for curves, label, color in [
    (baseline_curves, 'baseline (n=20 seeds)', '#4c78a8'),
    (algomut_curves, 'algomut (n=20 seeds)', '#f58518'),
    (bigjump_curves, 'big_jump p=0.15 x8 (n=20 seeds)', '#e45756'),
    (silencesink_curves, 'silence_sink (n=20 seeds)', '#b279a2'),
    (random_curves, 'random (n=20 resamples)', '#54a24b'),
]:
    m = curves.mean(axis=0)
    s = curves.std(axis=0)
    ax.plot(CHECKPOINTS, m, color=color, label=label, linewidth=2)
    ax.fill_between(CHECKPOINTS, m - s, m + s, color=color, alpha=0.2)

ax.axvline(BOOTSTRAP_SIZE, color='gray', linestyle='--', linewidth=1, label=f'fin du bootstrap (n={BOOTSTRAP_SIZE})')
ax.set_xlabel('nombre de decouvertes (ordre chronologique)')
ax.set_ylabel(f'cellules distinctes visitees (grille {N_BINS_PER_DIM}^{N_PCA_DIMS} sur PCA-4D de Z)')
ax.set_title('Couverture cumulative de grille -- tous les leviers testes\n(moyenne +/- ecart-type, style Cully & Demiris)')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()

out = 'figures/grid_coverage_evolution_all.png'
FigureCanvasAgg(fig).print_png(out)
print('saved', out)

pickle.dump({
    'checkpoints': CHECKPOINTS,
    'baseline_curves': baseline_curves,
    'algomut_curves': algomut_curves,
    'bigjump_curves': bigjump_curves,
    'silencesink_curves': silencesink_curves,
    'random_curves': random_curves,
}, open('figures/grid_coverage_evolution_all.pickle', 'wb'))
print('saved grid_coverage_evolution_all.pickle')
