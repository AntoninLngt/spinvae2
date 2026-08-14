"""Figures for the study-2 latent-space exploration notebook (latent_investigation.ipynb).

Produces four figures from the 4 x 500-iteration latent runs, with the study-1 parametric
runs as reference:
  latent_overview.png    silence / Vendi / coverage, 4 variants vs parametric references
  latent_evolution.png   cumulative Vendi and coverage vs n -- the collapse signature
  latent_locality.png    mutation step size in Z, bootstrap vs guided (the sigma caveat)
  latent_projection.png  joint UMAP of the behaviour space Z, per variant, human backdrop

Note on what is and is not comparable: the four variants have DIFFERENT 256-D latent spaces
(z is not commensurable across them, nor with study 1's 155-D Theta), so every cross-method
comparison here lives in the shared 38-D behaviour space Z. That is the whole point of
holding the behaviour space fixed across both studies.
"""
import sys
sys.path.insert(0, '.')
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from coverage_diversity import vendi_score

N_BOOT, SEED = 100, 0
VARIANTS = ['avec_tout', 'sans_timbre', 'sans_KL', 'sans_les_deux']
CHECKPOINTS = list(range(50, 501, 50))
HUMAN_FEATURES_PICKLE = Path('/data2/anasynth_nonbp/longeot/spinvae2_datasets/Dexed/raw_timbre_features_ttb.df.pickle')

COLORS = {
    'avec_tout': '#4c78a8', 'sans_timbre': '#f58518', 'sans_KL': '#e45756',
    'sans_les_deux': '#b279a2', 'full (param)': '#54a24b', 'random (param)': '#72b7b2',
}


def load(d, cap=None):
    files = sorted(Path(d).rglob('discovery.json'))
    if cap is not None:
        files = files[:cap]
    return np.asarray([json.load(f.open())['output'] for f in files], dtype=float)


def drop_nan(M):
    return M[~np.isnan(M).any(axis=1)]


def sil(M):
    return 100.0 * (np.abs(M).max(axis=1) < 1e-9).mean()


print('loading...', flush=True)
sets = {v: drop_nan(load(f'runs/latent/{v}')) for v in VARIANTS}
sets['full (param)'] = drop_nan(load('runs/confirm/full/seed0', cap=500))
random_pool = drop_nan(load('runs/random/combined/discoveries'))
sets['random (param)'] = random_pool[:500]
for k, v in sets.items():
    print(f'  {k:16s} {len(v)}')

ORDER = VARIANTS + ['full (param)', 'random (param)']

# ------------------------------------------------------------------ shared geometry
ref = np.concatenate(list(sets.values()) + [random_pool])
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


def style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.25)


# ============================================================== 1. overview
print('figure 1/4 : overview', flush=True)
fig = Figure(figsize=(13, 4.2), dpi=180)
ax1, ax2, ax3 = fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)
x = np.arange(len(ORDER))
w = 0.38

boot = [sil(sets[k][:N_BOOT]) for k in ORDER]
guid = [sil(sets[k][N_BOOT:]) for k in ORDER]
ax1.bar(x - w / 2, boot, w, label='bootstrap', color='#bbbbbb')
ax1.bar(x + w / 2, guid, w, label='phase guidee', color=[COLORS[k] for k in ORDER])
for xi, g in zip(x, guid):
    ax1.annotate(f'{g:.0f}%', (xi + w / 2, g), textcoords='offset points', xytext=(0, 3),
                 ha='center', fontsize=7.5)
ax1.set_title('Taux de silence'); ax1.set_ylabel('% de decouvertes quasi-silencieuses')
ax1.legend(fontsize=7.5)

for ax, fn, title in ((ax2, lambda M: vendi_score(zs(M)), 'Vendi Score'),
                      (ax3, cells, 'Couverture de grille')):
    vals = [fn(sets[k]) for k in ORDER]
    ax.bar(x, vals, color=[COLORS[k] for k in ORDER])
    for xi, v in zip(x, vals):
        ax.annotate(f'{v:.1f}' if title.startswith('Couv') else f'{v:.2f}', (xi, v),
                    textcoords='offset points', xytext=(0, 3), ha='center', fontsize=7.5)
    ax.set_title(title)
    ax.set_ylabel('cellules occupees' if title.startswith('Couv') else 'Vendi Score')

for ax in (ax1, ax2, ax3):
    ax.set_xticks(x); ax.set_xticklabels(ORDER, rotation=25, ha='right', fontsize=7.5)
    style(ax)
fig.suptitle('Exploration en coordonnees latentes : 4 variantes SPINVAE-2 (n=500, bootstrap=100), '
             'contre les references parametriques au meme n', fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
FigureCanvasAgg(fig).print_png('figures/latent_overview.png')

# ============================================================== 2. evolution
print('figure 2/4 : evolution', flush=True)
fig = Figure(figsize=(11.5, 4.4), dpi=180)
ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
for k in ORDER:
    M = sets[k]
    v = [vendi_score(zs(M[:n])) for n in CHECKPOINTS if n <= len(M)]
    c = [cells(M[:n]) for n in CHECKPOINTS if n <= len(M)]
    xs = [n for n in CHECKPOINTS if n <= len(M)]
    ls = '--' if 'param' in k else '-'
    ax1.plot(xs, v, ls, label=k, color=COLORS[k], linewidth=1.8)
    ax2.plot(xs, c, ls, label=k, color=COLORS[k], linewidth=1.8)
for ax, ylab, title in ((ax1, 'Vendi Score cumulatif', 'Vendi Score'),
                        (ax2, 'cellules de grille occupees', 'Couverture de grille')):
    ax.axvline(N_BOOT, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('nombre de decouvertes'); ax.set_ylabel(ylab); ax.set_title(title)
    ax.legend(fontsize=7.5); style(ax)
fig.suptitle("Evolution cumulative (pointilles = references parametriques, "
             "ligne verticale = fin du bootstrap)", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
FigureCanvasAgg(fig).print_png('figures/latent_evolution.png')

# ============================================================== 3. locality
print('figure 3/4 : localite', flush=True)
fig = Figure(figsize=(11.5, 4.2), dpi=180)
ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
ratios = []
for k in ORDER:
    Z = zs(sets[k])
    b = np.linalg.norm(np.diff(Z[:N_BOOT], axis=0), axis=1)
    g = np.linalg.norm(np.diff(Z[N_BOOT:], axis=0), axis=1)
    ratios.append(b.mean() / g.mean())
    ax1.hist(g, bins=40, alpha=0.45, label=k, color=COLORS[k], density=True)
ax1.set_xlabel('distance L2 entre decouvertes consecutives (phase guidee)')
ax1.set_ylabel('densite'); ax1.set_title('Amplitude du pas de mutation dans Z')
ax1.legend(fontsize=7.5); style(ax1)

bars = ax2.bar(x, ratios, color=[COLORS[k] for k in ORDER])
ax2.axhline(1.0, color='k', linestyle='--', linewidth=1)
ax2.annotate('1.0 = pas guide aussi large que le bootstrap', (len(ORDER) - 0.4, 1.03),
             ha='right', fontsize=7.5)
for xi, r in zip(x, ratios):
    ax2.annotate(f'{r:.2f}x', (xi, r), textcoords='offset points', xytext=(0, 3),
                 ha='center', fontsize=7.5)
ax2.set_xticks(x); ax2.set_xticklabels(ORDER, rotation=25, ha='right', fontsize=7.5)
ax2.set_ylabel('pas bootstrap / pas guide'); ax2.set_title('Localite relative de la mutation')
style(ax2)
fig.suptitle('Diagnostic de calibration : a sigma=0.1 les runs latents sont ~3x plus locaux '
             'que leur propre bootstrap', fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
FigureCanvasAgg(fig).print_png('figures/latent_locality.png')

# ============================================================== 4. projection
print('figure 4/4 : projection UMAP (peut prendre 1-2 min)', flush=True)
import umap
from maps.DexedStatistics import Z_FEATURE_NAMES

feat = pd.read_pickle(HUMAN_FEATURES_PICKLE)
feat = feat[feat['variation'] == 0]
z_human = feat[Z_FEATURE_NAMES].to_numpy(dtype=float)
z_human = z_human[~np.isnan(z_human).any(axis=1)]
human_sub = z_human[rng.choice(len(z_human), 1500, replace=False)]

blocks = [sets[k] for k in ORDER] + [human_sub]
joint = np.concatenate(blocks)
emb = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.0, random_state=SEED).fit_transform(zs(joint))
off, spans = 0, {}
for k, b in zip(ORDER + ['human'], blocks):
    spans[k] = emb[off:off + len(b)]; off += len(b)

fig = Figure(figsize=(12.5, 8), dpi=170)
axes = fig.subplots(2, 3)
for ax, k in zip(axes.ravel(), ORDER):
    ax.scatter(spans['human'][:, 0], spans['human'][:, 1], s=2, color='#e8e8e8', zorder=0)
    ax.scatter(spans['full (param)'][:, 0], spans['full (param)'][:, 1], s=3,
               color='#cccccc', zorder=1)
    ax.scatter(spans[k][:, 0], spans[k][:, 1], s=7, color=COLORS[k], zorder=2)
    ax.set_title(k, fontsize=10)
for ax, k in zip(axes.ravel()[4:], ['full (param)', 'random (param)']):
    ax.scatter(spans['human'][:, 0], spans['human'][:, 1], s=2, color='#e8e8e8', zorder=0)
    ax.scatter(spans[k][:, 0], spans[k][:, 1], s=7, color=COLORS[k], zorder=2)
    ax.set_title(k, fontsize=10)
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor('#f7fafd')
fig.suptitle("Projection UMAP conjointe de l'espace de comportement Z (38-D z-score)\n"
             "gris clair = corpus humain (n=1500), gris moyen = full parametrique ; "
             "n=500 par methode", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
FigureCanvasAgg(fig).print_png('figures/latent_projection.png')

print('\nfigures ecrites dans figures/ :')
for f in ('latent_overview', 'latent_evolution', 'latent_locality', 'latent_projection'):
    print('  ', f + '.png')
