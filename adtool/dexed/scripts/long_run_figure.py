"""Figure for the 4000-iteration long run (full config, seed 7, bootstrap 100).

Two questions this answers that no 1000-iteration run could:
  - does the result replicate on a fresh seed? (seed 7 vs the 5 confirmation seeds)
  - does coverage saturate once the budget is 4x anything run before?
"""
import sys
sys.path.insert(0, '.')
from pathlib import Path
import json
import numpy as np
from sklearn.decomposition import PCA
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from coverage_diversity import vendi_score

N_BOOT, SEED = 100, 0
CHECKPOINTS = list(range(100, 4001, 100))


def load(d, cap=None):
    files = sorted(Path(d).rglob('discovery.json'))
    if cap is not None:
        files = files[:cap]
    return np.asarray([json.load(f.open())['output'] for f in files], dtype=float)


def drop_nan(M):
    return M[~np.isnan(M).any(axis=1)]


print('loading...', flush=True)
long_M = drop_nan(load('runs/long/full_7h'))
confirm = [drop_nan(load(f'runs/confirm/full/seed{s}')) for s in range(5)]
random_pool = drop_nan(load('runs/random/combined/discoveries'))
print(f'  long {len(long_M)} | confirm 5x{len(confirm[0])} | random {len(random_pool)}')

ref = np.concatenate([long_M] + confirm + [random_pool])
mu, sd = ref.mean(axis=0), ref.std(axis=0)
sd[sd == 0] = 1.0
zs = lambda M: (M - mu) / sd
rng = np.random.default_rng(SEED)

pca = PCA(n_components=4, random_state=SEED).fit(zs(ref))
pp = pca.transform(zs(ref))
lo, hi = np.percentile(pp, 0.5, axis=0), np.percentile(pp, 99.5, axis=0)
edges = [np.linspace(lo[d], hi[d], 9) for d in range(4)]


def cells(M):
    P = np.clip(pca.transform(zs(M)), lo, hi)
    idx = np.stack([np.clip(np.digitize(P[:, d], edges[d][1:-1]), 0, 7) for d in range(4)], axis=1)
    return len({tuple(r) for r in idx})


print('courbes du run long...', flush=True)
v_long = [vendi_score(zs(long_M[:n])) for n in CHECKPOINTS]
c_long = [cells(long_M[:n]) for n in CHECKPOINTS]

print('courbes de reference (5 seeds + random)...', flush=True)
cp = [n for n in CHECKPOINTS if n <= 1000]
v_conf = np.array([[vendi_score(zs(M[:n])) for n in cp] for M in confirm])
c_conf = np.array([[cells(M[:n]) for n in cp] for M in confirm])

rand_cp = [n for n in CHECKPOINTS if n <= len(random_pool)]
perms = [rng.permutation(len(random_pool)) for _ in range(5)]
v_rand = np.array([[vendi_score(zs(random_pool[p[:n]])) for n in rand_cp] for p in perms])
c_rand = np.array([[cells(random_pool[p[:n]]) for n in rand_cp] for p in perms])

print('=== gain marginal de couverture (cellules / 1000 iterations) ===')
marg_x, marg_y = [], []
prev_n = prev_c = 0
for n, c in zip(CHECKPOINTS, c_long):
    if n % 500 == 0:
        marg_y.append((c - prev_c) / ((n - prev_n) / 1000))
        marg_x.append(n)
        print(f'  n={n:5d}  couverture {c:4d}   +{marg_y[-1]:6.1f}/1000')
        prev_n, prev_c = n, c

fig = Figure(figsize=(14, 4.3), dpi=180)
ax1, ax2, ax3 = fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)

for ax, y_long, conf, rand, ylab, title in (
        (ax1, v_long, v_conf, v_rand, 'Vendi Score cumulatif', 'Vendi Score'),
        (ax2, c_long, c_conf, c_rand, 'cellules de grille occupees', 'Couverture de grille')):
    ax.plot(CHECKPOINTS, y_long, color='#54a24b', linewidth=2, label='full, seed 7 (n=4000)')
    m, s = conf.mean(axis=0), conf.std(axis=0)
    ax.plot(cp, m, '--', color='#9c755f', linewidth=1.6, label='full, seeds 0-4 (n=1000)')
    ax.fill_between(cp, m - s, m + s, color='#9c755f', alpha=0.18)
    m, s = rand.mean(axis=0), rand.std(axis=0)
    ax.plot(rand_cp, m, '--', color='#4c78a8', linewidth=1.6, label='random')
    ax.fill_between(rand_cp, m - s, m + s, color='#4c78a8', alpha=0.18)
    ax.axvline(N_BOOT, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('nombre de decouvertes'); ax.set_ylabel(ylab); ax.set_title(title)
    ax.legend(fontsize=7.5, loc='lower right')

ax3.bar([str(x) for x in marg_x], marg_y, color='#54a24b')
for i, y in enumerate(marg_y):
    ax3.annotate(f'{y:.0f}', (i, y), textcoords='offset points', xytext=(0, 3),
                 ha='center', fontsize=7.5)
ax3.set_xlabel('fin du bloc de 500 iterations'); ax3.set_ylabel('nouvelles cellules / 1000 iter')
ax3.set_title('Gain marginal de couverture')

for ax in (ax1, ax2, ax3):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.25)
fig.suptitle('Run long : full, graine 7, 4000 iterations, bootstrap 100 '
             '(pointilles = references a n<=1000/2000, bande = +/- 1 ecart-type)', fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
FigureCanvasAgg(fig).print_png('figures/long_run_4000.png')
print('\nsaved figures/long_run_4000.png')
