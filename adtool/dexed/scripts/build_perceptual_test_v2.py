"""Blind listening test for the note-like criterion, now that the loop OPTIMISES it.

The question this test answers, and why the numbers cannot. Constraining the goal box and the
parent pool to note-like sounds raised the note-like rate from a median of 0.0% to 39.3% over
ten seeds. But the criterion is now the search's own target, so that figure is no longer an
independent measurement of anything: it could mean the system produces better sounds, or
merely that it produces sounds satisfying a Mahalanobis distance on nine envelope descriptors.
Goodhart's law applies to a criterion the moment you optimise it. Only a listener can separate
the two readings.

Design. Five groups, all rendered through the same chain and peak-normalised for playback, so
loudness cannot be the cue:

  box_both_notelike  (12) -- what the constrained system produces and the criterion accepts.
  box_both_rejete     (6) -- same runs, criterion REJECTS. The within-condition control: if
                             the criterion tracks anything audible, this group must rate below
                             the one above. Same generator, so the comparison isolates the
                             criterion rather than the configuration.
  box_nu             (10) -- the unconstrained loop: what we produced before this change.
  random              (6) -- the baseline any exploration has to beat.
  humain              (6) -- calibration. These must rate high; if they do not, the rating
                             scale or the rendering chain is at fault, not the criterion.
                             In the previous test all six scored 2/2, which is what made the
                             rest of that test readable.

Sampling spreads across the ten seeds rather than drawing from one, because the note-like rate
is strongly seed-dependent (per-seed range 8.7-94.7% for box_both).

Output: figures/perceptual_test_v2/ -- the wavs, reponses.csv to fill in, and cle.csv, which
is the answer key and must not be opened before rating.
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

SEED = 20260815
BOOT = 50
SEEDS = [1000 * (i + 1) for i in range(10)]
OUT = Path('figures/perceptual_test_v2')
N = {'box_both_notelike': 12, 'box_both_rejete': 6, 'box_nu': 10, 'random': 6, 'humain': 6}
HUMAN_FEATURES = Path('/data2/anasynth_nonbp/longeot/spinvae2_datasets/Dexed/'
                      'raw_timbre_features_ttb.df.pickle')
HUMAN_PRESETS = Path.home() / 'projects/spinvae2/synth/dexed_presets.df.pickle'

_sim = [None]


def render(preset):
    if _sim[0] is None:
        from systems.Dexed import DexedSimulation
        _sim[0] = DexedSimulation(output_Fs=16000, midi_note_duration_s=3.0,
                                  render_duration_s=4.0, midi_pitch=56, midi_velocity=75)
    d = _sim[0].map({'params': {'dynamic_params': {'preset': preset}}})
    au = np.asarray(d.get('output') if isinstance(d, dict) else d).squeeze()
    return au[0] if au.ndim > 1 else au


def collect(cond):
    """Every discovery of a condition, with its wav path and note-like verdict."""
    from maps.notelike import distance, threshold
    thr = threshold()
    items = []
    for s in SEEDS:
        for f in sorted(Path(f'runs/notelike_v2/{cond}/seed{s}').rglob('discovery.json'))[BOOT:]:
            wav = f.parent / 'visu.wav'
            if not wav.exists():
                continue
            z = np.asarray(json.load(f.open())['output'], dtype=float)
            if np.isnan(z).any():
                continue
            d = float(distance(z[None])[0])
            items.append({'wav': wav, 'dist': d, 'notelike': d <= thr, 'seed': s})
    return items


def main():
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    print('lecture des decouvertes...', flush=True)
    both, nu = collect('box_both'), collect('box_nu')
    pools = {
        'box_both_notelike': [i for i in both if i['notelike']],
        'box_both_rejete': [i for i in both if not i['notelike']],
        'box_nu': nu,
    }
    for k, v in pools.items():
        print(f'  {k:20s} {len(v)} disponibles')

    items = []
    for cat, pool in pools.items():
        chosen = rng.permutation(len(pool))[:N[cat]]
        for i in chosen:
            items.append((cat, pool[i]['wav'], None, pool[i]['dist'], pool[i]['seed']))
        got = sorted({pool[i]['seed'] for i in chosen})
        print(f'  {cat:20s} -> {len(chosen)} extraits, {len(got)} graines distinctes')

    print('\nrendu des references...', flush=True)
    fs = sorted(Path('runs/random/big4000').rglob('discovery.json'))
    for i in rng.permutation(len(fs))[:N['random']]:
        p = json.load(fs[i].open())['params']['dynamic_params']['preset']
        items.append(('random', None, render(p), float('nan'), -1))

    fe = pd.read_pickle(HUMAN_FEATURES)
    fe = fe[fe['variation'] == 0]
    pres = pd.read_pickle(HUMAN_PRESETS)[['preset_UID', 'params_values']]
    df = fe.merge(pres, on='preset_UID', how='inner')
    for i in rng.permutation(len(df))[:N['humain']]:
        items.append(('humain', None, render(list(enumerate(df.iloc[i]['params_values']))),
                      float('nan'), -1))

    order = rng.permutation(len(items))
    key_rows, resp_rows = [], []
    print(f'\necriture de {len(items)} extraits anonymises...')
    for rank, idx in enumerate(order, start=1):
        cat, wav, arr, dist, seed = items[idx]
        if arr is None:
            au, sr = sf.read(wav)
            au = au[:, 0] if au.ndim > 1 else au
        else:
            au, sr = arr, 16000
        peak = float(np.abs(au).max())
        if peak > 0:
            au = au / peak * 0.95          # loudness must not be a cue
        item_id = f'son_{rank:02d}'
        sf.write(OUT / f'{item_id}.wav', au.astype(np.float32), sr)
        key_rows.append({'id': item_id, 'categorie': cat, 'distance_notelike': f'{dist:.2f}',
                         'graine': seed, 'crete_origine': f'{peak:.6f}'})
        resp_rows.append({'id': item_id, 'note': '', 'commentaire': ''})

    with open(OUT / 'cle.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'categorie', 'distance_notelike', 'graine',
                                          'crete_origine'])
        w.writeheader(); w.writerows(key_rows)
    with open(OUT / 'reponses.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'note', 'commentaire'])
        w.writeheader(); w.writerows(resp_rows)

    print(f'\necrit dans {OUT}/')
    print(f'  {len(items)} wav + reponses.csv (a remplir) + cle.csv (a NE PAS ouvrir avant)')
    print('  composition :', dict(Counter(k['categorie'] for k in key_rows)))
    print('\n  bareme : 0 = inutilisable / 1 = passable / 2 = son musical utilisable')


if __name__ == '__main__':
    main()
