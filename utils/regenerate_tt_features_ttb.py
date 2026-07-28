"""
Regenerates raw_timbre_features.df.pickle's 'tt_*' columns using ttb (Python, MATLAB-free)
instead of MATLAB TimbreToolbox, keeping the 'ac_*' (AudioCommons/ACTM) columns and metadata
untouched (only TT is being replaced -- ACTM was never MATLAB-dependent).

Usage:
    python utils/regenerate_tt_features_ttb.py --n_presets 100 --out /tmp/raw_tt_test.pickle
    python utils/regenerate_tt_features_ttb.py --out /data2/.../raw_timbre_features_ttb.df.pickle
"""
import argparse
import pathlib
import sys
import multiprocessing

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

AUDIO_DIR = pathlib.Path('/data2/anasynth_nonbp/longeot/spinvae2_datasets/Dexed/Audio')
RAW_PICKLE = pathlib.Path('/data2/anasynth_nonbp/longeot/spinvae2_datasets/Dexed/raw_timbre_features.df.pickle')


def _compute_one(row):
    from utils.ttb_extractor import compute_tt_features  # imported per-worker (spawn context)
    uid, variation, pitch, vel = row['preset_UID'], row['variation'], row['midi_pitch'], row['midi_vel']
    wav = AUDIO_DIR / f"{int(uid):06d}_pitch{int(pitch):03d}vel{int(vel):03d}_var{int(variation):03d}.wav"
    if not wav.exists():
        return None
    try:
        audio, sr = sf.read(wav)
        tt_feats = compute_tt_features(audio, sr)
    except Exception as e:
        print(f"UID {uid} var {variation} failed: {e}")
        return None
    return {f"tt_{k}": v for k, v in tt_feats.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_presets", type=int, default=None, help="Limit to N presets (for testing)")
    parser.add_argument("--n_workers", type=int, default=32)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    raw_df = pd.read_pickle(RAW_PICKLE)
    # Only variation 0 has its own physical wav file (other variations are delayed-audio
    # variants without a separate file, per data/dexeddataset.py) -- and it's the only one
    # used anywhere downstream (the audit notebook filters on variation==0 too).
    raw_df = raw_df[raw_df['variation'] == 0].reset_index(drop=True)
    if args.n_presets is not None:
        uids = raw_df['preset_UID'].drop_duplicates().sample(n=args.n_presets, random_state=args.seed)
        raw_df = raw_df[raw_df['preset_UID'].isin(uids)].reset_index(drop=True)

    print(f"Recomputing tt_* features for {len(raw_df)} rows ({raw_df['preset_UID'].nunique()} presets), "
          f"{args.n_workers} workers...")

    rows = raw_df.to_dict('records')
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(args.n_workers) as pool:
        new_tt_cols = pool.map(_compute_one, rows)

    n_failed = sum(1 for r in new_tt_cols if r is None)
    print(f"{n_failed}/{len(rows)} rows failed (kept OLD MATLAB tt_* values for those)")

    old_tt_cols = [c for c in raw_df.columns if c.startswith('tt_')]
    new_df = raw_df.drop(columns=old_tt_cols).reset_index(drop=True)
    # Replace failures with an empty dict so pd.DataFrame doesn't choke on None entries;
    # their old MATLAB tt_* values are restored below via combine_first.
    new_tt_cols = [r if r is not None else {} for r in new_tt_cols]
    tt_df = pd.DataFrame(new_tt_cols, columns=old_tt_cols)
    tt_df = tt_df.combine_first(raw_df[old_tt_cols].reset_index(drop=True))
    new_df = pd.concat([new_df, tt_df[old_tt_cols]], axis=1)

    out_path = pathlib.Path(args.out)
    new_df.to_pickle(out_path)
    print(f"Saved {out_path} ({new_df.shape})")


if __name__ == "__main__":
    main()
