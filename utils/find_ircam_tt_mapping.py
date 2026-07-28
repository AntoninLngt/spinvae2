"""
One-off script: systematically correlates every ircamdescriptor feature (after median/IQR
aggregation) against every one of the 32 canonical TT features, on a sample of real dataset
presets with known ground-truth TT values, to find the best-matching ircamdescriptor code for
each TT feature -- rather than guessing sub-dimension indices by hand (which turned out wrong
for several descriptors on a first 30-preset spot check).
"""
import pathlib
import sys
import multiprocessing

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from utils.ircam_descriptors import ircam_descriptor_extractor  # noqa: E402

AUDIO_DIR = pathlib.Path('/data2/anasynth_nonbp/longeot/spinvae2_datasets/Dexed/Audio')
RAW_PICKLE = pathlib.Path('/data2/anasynth_nonbp/longeot/spinvae2_datasets/Dexed/raw_timbre_features.df.pickle')
N_SAMPLE = 300
N_WORKERS = 16

TT_TARGETS = [
    'tt_SpecCent_IQR', 'tt_SpecCrest_med', 'tt_SpecCrest_IQR', 'tt_SpecDecr_med', 'tt_SpecDecr_IQR',
    'tt_SpecFlat_med', 'tt_SpecFlat_IQR', 'tt_SpecKurt_med', 'tt_SpecKurt_IQR', 'tt_SpecRollOff_med',
    'tt_SpecRollOff_IQR', 'tt_SpecSpread_med', 'tt_SpecSpread_IQR', 'tt_SpecVar_med', 'tt_SpecVar_IQR',
    'tt_F0_med', 'tt_F0_IQR', 'tt_HarmErg_IQR', 'tt_InHarm_med', 'tt_InHarm_IQR', 'tt_OddEvenRatio_med',
    'tt_OddEvenRatio_IQR', 'tt_AmpMod', 'tt_Att', 'tt_Dec', 'tt_DecSlope', 'tt_EffDur', 'tt_FreqMod',
    'tt_LAT', 'tt_RMSEnv_med', 'tt_RMSEnv_IQR', 'tt_TempCent',
]


def _compute_one(uid: int):
    wav = AUDIO_DIR / f"{uid:06d}_pitch056vel075_var000.wav"
    if not wav.exists():
        return None
    audio, fs = sf.read(wav)
    feats = ircam_descriptor_extractor(audio, fs)
    feats['preset_UID'] = uid
    return feats


def main():
    raw_df = pd.read_pickle(RAW_PICKLE)
    sub = raw_df[raw_df['variation'] == 0].sample(n=N_SAMPLE, random_state=0).reset_index(drop=True)
    uids = sub['preset_UID'].astype(int).tolist()

    print(f"Computing ircamdescriptor features for {len(uids)} presets, {N_WORKERS} workers...")
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(N_WORKERS) as pool:
        rows = pool.map(_compute_one, uids)
    rows = [r for r in rows if r is not None]
    ircam_df = pd.DataFrame(rows)
    print("ircam_df shape:", ircam_df.shape)
    ircam_df.to_pickle('/tmp/ircam_df_sample300.pickle')

    merged = sub.merge(ircam_df, on='preset_UID')
    merged.to_pickle('/tmp/merged_sample300.pickle')
    print("merged shape:", merged.shape)

    ircam_cols = [c for c in ircam_df.columns if c != 'preset_UID']
    ircam_numeric = merged[ircam_cols].apply(pd.to_numeric, errors='coerce')
    # drop constant/degenerate columns (e.g. the WindowLength-metadata columns found earlier)
    ircam_numeric = ircam_numeric.loc[:, ircam_numeric.std(skipna=True) > 1e-9]
    print(f"Non-degenerate ircam columns: {ircam_numeric.shape[1]} / {len(ircam_cols)}")

    results = []
    for target in TT_TARGETS:
        if target not in merged.columns:
            print(f"{target}: MISSING from raw_df")
            continue
        corrs = ircam_numeric.corrwith(merged[target]).abs().sort_values(ascending=False)
        best_col = corrs.index[0]
        best_r = corrs.iloc[0]
        second_col = corrs.index[1]
        second_r = corrs.iloc[1]
        results.append((target, best_col, best_r, second_col, second_r))
        print(f"{target:22s} -> {best_col:16s} r={best_r:.3f}   (2nd: {second_col:16s} r={second_r:.3f})")

    result_df = pd.DataFrame(results, columns=['tt_feature', 'best_ircam_col', 'best_r', 'second_col', 'second_r'])
    result_df.to_csv('/tmp/tt_ircam_mapping.csv', index=False)
    print("\nSaved mapping to /tmp/tt_ircam_mapping.csv")


if __name__ == "__main__":
    main()
