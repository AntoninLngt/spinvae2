"""Does constraining the loop to note-like sounds make the exploration produce more of them?

2x2 factorial (goal box restricted / parent pool restricted), 10 independent seeds, 200
iterations each, bootstrap 50 excluded.

Read PAIRED, per seed, not pooled. The first pass of this experiment pooled three seeds and
reported a 16.7% note-like rate for the control; the per-seed rates behind it were 1.3%, 48.7%
and 0.0%. The outcome is bimodal -- a run either finds a note-like region or never does -- so
a pooled mean describes none of the runs and is dominated by whichever seed got lucky. Every
comparison below is therefore within-seed, and reported as a median of per-seed values plus
the number of seeds moving each way (a sign test, which assumes nothing about the shape).
"""
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

BOOT = 50
SEEDS = [1000 * (i + 1) for i in range(10)]
CONDS = ['box_nu', 'box_goalbox', 'box_parents', 'box_both']
LABEL = {'box_nu': 'aucune contrainte', 'box_goalbox': 'buts note-like',
         'box_parents': 'parents note-like', 'box_both': 'buts + parents'}
RANDOM_RUN = Path('runs/random/big4000')
HUMAN = Path('/data2/anasynth_nonbp/longeot/spinvae2_datasets/Dexed/raw_timbre_features_ttb.df.pickle')


def vendi(Z, n_max=200, seed=0):
    """exp(Shannon entropy of the eigenvalues of the normalised similarity matrix)."""
    if len(Z) < 2:
        return float('nan')
    if len(Z) > n_max:
        Z = Z[np.random.default_rng(seed).choice(len(Z), n_max, replace=False)]
    S = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    D = ((S[:, None, :] - S[None, :, :]) ** 2).sum(-1)
    K = np.exp(-D / (np.median(D[D > 0]) + 1e-12))
    w = np.linalg.eigvalsh(K / len(K))
    w = w[w > 1e-12]
    return float(np.exp(-(w * np.log(w)).sum()))


def load(path):
    fs = sorted(Path(path).rglob('discovery.json'))[BOOT:]
    if not fs:
        return np.zeros((0, 38))
    Z = np.asarray([json.load(f.open())['output'] for f in fs], dtype=float)
    return Z


def main():
    from maps.notelike import distance, threshold
    from maps.DexedStatistics import Z_FEATURE_NAMES
    thr = threshold()

    rows = []
    for c in CONDS:
        for s in SEEDS:
            Z = load(f'runs/notelike_v2/{c}/seed{s}')
            n_raw = len(Z)
            silent = (np.abs(Z).max(axis=1) < 1e-9)
            Z = Z[~np.isnan(Z).any(axis=1)]
            d = distance(Z)
            rows.append(dict(cond=c, seed=s, n=n_raw,
                             notelike=100 * (d <= thr).mean(),
                             silence=100 * silent.mean(),
                             dmed=float(np.median(d)),
                             vendi=vendi(Z, seed=s)))
    df = pd.DataFrame(rows)

    print(f'seuil note-like = {thr:.2f}  (95% du corpus humain accepte)')
    print(f'{len(SEEDS)} graines independantes x 200 iterations, bootstrap {BOOT} exclu\n')

    print('TAUX NOTE-LIKE PAR GRAINE (%)')
    piv = df.pivot(index='cond', columns='seed', values='notelike').loc[CONDS]
    print(piv.round(1).to_string())
    print()

    print(f'{"condition":20s} {"mediane":>9s} {"moyenne":>9s} {"min-max":>13s} '
          f'{"silence":>9s} {"d_med":>8s} {"Vendi":>13s}')
    for c in CONDS:
        g = df[df.cond == c]
        print(f'{LABEL[c]:20s} {g.notelike.median():8.1f}% {g.notelike.mean():8.1f}% '
              f'{g.notelike.min():5.1f}-{g.notelike.max():5.1f}% {g.silence.mean():8.1f}% '
              f'{g.dmed.median():8.1f} {g.vendi.mean():7.2f}+/-{g.vendi.std():.2f}')

    # ---- references ----------------------------------------------------------------------
    Zr = load(RANDOM_RUN)
    Zr = Zr[~np.isnan(Zr).any(axis=1)]
    dr = distance(Zr)
    print(f'{"random (n=" + str(len(Zr)) + ")":20s} {100 * (dr <= thr).mean():8.1f}% '
          f'{"":9s} {"":13s} {"":9s} {np.median(dr):8.1f} {vendi(Zr):7.2f}')
    fe = pd.read_pickle(HUMAN)
    zh = fe[fe['variation'] == 0][Z_FEATURE_NAMES].to_numpy(float)
    zh = zh[~np.isnan(zh).any(axis=1)]
    dh = distance(zh)
    print(f'{"humain (n=" + str(len(zh)) + ")":20s} {100 * (dh <= thr).mean():8.1f}% '
          f'{"":9s} {"":13s} {"":9s} {np.median(dh):8.1f} {vendi(zh):7.2f}')

    # ---- paired comparisons against the control ------------------------------------------
    print('\nCOMPARAISONS APPARIEES vs "aucune contrainte" (une paire par graine)')
    base = df[df.cond == 'box_nu'].set_index('seed').notelike
    for c in CONDS[1:]:
        v = df[df.cond == c].set_index('seed').notelike
        delta = (v - base).reindex(SEEDS)
        up, down, tie = (delta > 0).sum(), (delta < 0).sum(), (delta == 0).sum()
        print(f'  {LABEL[c]:20s} delta median {delta.median():+6.1f} pts   '
              f'{up} graines en hausse, {down} en baisse, {tie} ex aequo')

    # ---- main effects of the 2x2 ---------------------------------------------------------
    print('\nEFFETS PRINCIPAUX (2x2, medianes sur graines)')
    w = df.pivot(index='seed', columns='cond', values='notelike')
    gb = ((w.box_goalbox - w.box_nu) + (w.box_both - w.box_parents)) / 2
    pa = ((w.box_parents - w.box_nu) + (w.box_both - w.box_goalbox)) / 2
    inter = (w.box_both - w.box_parents) - (w.box_goalbox - w.box_nu)
    for nom, e in (('restreindre les buts', gb), ('restreindre les parents', pa),
                   ('interaction', inter)):
        print(f'  {nom:24s} {e.median():+6.1f} pts   ({(e > 0).sum()}/{len(SEEDS)} graines positives)')

    df.to_csv('figures/notelike_v2_par_graine.csv', index=False)
    print('\nper-seed -> figures/notelike_v2_par_graine.csv')


if __name__ == '__main__':
    main()
