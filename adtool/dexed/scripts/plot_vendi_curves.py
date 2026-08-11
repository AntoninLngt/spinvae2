import pickle
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

data = pickle.load(open('/tmp/claude-4367/-net-home-u-anasynth-longeot-projects/67caa9c6-dea8-4188-b083-0eeb82af2b34/scratchpad/vendi_curves.pickle', 'rb'))
checkpoints, curves = data['checkpoints'], data['curves']

fig = Figure(figsize=(8, 5), dpi=130)
ax = fig.add_subplot(111)

styles = {
    'baseline seed0': dict(color='#4c78a8', linestyle='-'),
    'baseline seed1': dict(color='#4c78a8', linestyle='--'),
    'baseline seed2': dict(color='#4c78a8', linestyle=':'),
    'algomut seed0': dict(color='#f58518', linestyle='-'),
    'algomut seed1': dict(color='#f58518', linestyle='--'),
    'algomut seed2': dict(color='#f58518', linestyle=':'),
    'random': dict(color='#54a24b', linestyle='-', linewidth=2.5),
}

for label, scores in curves.items():
    ax.plot(checkpoints, scores, label=label, **styles[label])

ax.set_xlabel('nombre de decouvertes')
ax.set_ylabel('Vendi Score (cumulatif)')
ax.set_title("Evolution du Vendi Score au cours de l'exploration\n(z-score fixe, reference = union des 7 jeux tronques a 1000)")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)
fig.tight_layout()

out = '/net/home/u.anasynth/longeot/projects/spinvae2/adtool/dexed/figures/vendi_score_evolution.png'
FigureCanvasAgg(fig).print_png(out)
print('saved', out)
