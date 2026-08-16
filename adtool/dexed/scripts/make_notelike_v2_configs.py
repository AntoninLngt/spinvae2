"""Regenerates the note-like experiment with independent seeds.

Two faults in the first pass (runs/notelike) are corrected here.

1. The parent filter never ran. `_vector_search_for_goal` returned early unless
   `filter_degenerate_parents` was set, so `filter_non_notelike_parents` was stored on the
   instance and never read -- the `box_parents` and `box_both` conditions were identical to
   `box_nu` by construction. Fixed by gating that path on `_parent_filtering_enabled`.

2. The three seeds were not independent replicates. DexedParameterMap.sample() draws
   `get_random_preset(seed=self.seed + self._rng_counter)`, so consecutive config seeds 0/1/2
   walk through nearly the same preset sequence, offset by one draw. With 200 iterations the
   counter reaches ~200, so seeds must be spaced by more than that; they are spaced by 1000
   here, which also leaves room to extend a run to 1000 iterations later without collision.

Ten seeds rather than three, because the outcome turned out to be bimodal across seeds: on
the first pass the note-like rate of `box_nu` was 1.3%, 48.7% and 0.0%. A three-seed mean of
16.7% describes none of those runs, and the pooled figure is dominated by whichever seed
happened to find a note-like region.
"""
import json
from pathlib import Path

CONFIG_SRC = Path('configs/notelike/box_nu_seed0.json')
OUT = Path('configs/notelike_v2')
SEEDS = [1000 * (i + 1) for i in range(10)]
CONDITIONS = {
    # nom            note_like_goal_box, filter_non_notelike_parents
    'box_nu':        (False, False),   # reference: no note-like constraint anywhere
    'box_goalbox':   (True,  False),   # goals drawn only from note-like territory
    'box_parents':   (False, True),    # only note-like discoveries may be mutated
    'box_both':      (True,  True),
}


def set_in(node, key, value):
    """Sets `key` wherever it occurs in the nested config; returns how many times."""
    n = 0
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                node[k] = value
                n += 1
            else:
                n += set_in(v, key, value)
    elif isinstance(node, list):
        for v in node:
            n += set_in(v, key, value)
    return n


def main():
    base = json.loads(CONFIG_SRC.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    written = 0
    for cond, (goalbox, parents) in CONDITIONS.items():
        for s in SEEDS:
            cfg = json.loads(json.dumps(base))
            checks = {
                'seed': set_in(cfg, 'seed', s),
                'note_like_goal_box': set_in(cfg, 'note_like_goal_box', goalbox),
                'filter_non_notelike_parents': set_in(cfg, 'filter_non_notelike_parents', parents),
                'save_location': set_in(cfg, 'save_location', f'runs/notelike_v2/{cond}/seed{s}'),
                'goal_sampling': set_in(cfg, 'goal_sampling', 'box'),
            }
            missing = [k for k, v in checks.items() if v != 1]
            if missing:
                raise SystemExit(f'ABANDON {cond}/seed{s}: champs absents ou dupliques {missing} '
                                 f'({checks}) -- une config muette rendrait la condition nulle, '
                                 f'ce qui est exactement le bug de la premiere passe')
            (OUT / f'{cond}_seed{s}.json').write_text(json.dumps(cfg, indent=1))
            written += 1
    print(f'{written} configs ecrites dans {OUT}/')
    print(f'  conditions : {", ".join(CONDITIONS)}')
    print(f'  graines    : {SEEDS}')


if __name__ == '__main__':
    main()
