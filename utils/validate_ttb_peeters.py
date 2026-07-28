"""
Validates geoffroypeeters/ttb (Python reimplementation of TimbreToolbox) against real MATLAB-TT
ground truth already computed for the Dexed dataset, on a sample of the dataset. ttb's feature
names match spinvae2's tt_* column names directly (Att, Dec, DecSlope, LAT, TempCent, EffDur,
FreqMod, AmpMod, RMSEnv, SpecCent, SpecSpread, SpecSkew, SpecKurt, SpecSlope, SpecDecr,
SpecRollOff, SpecVar, FrameErg, SpecFlat, SpecCrest, F0, InHarm, HarmErg, OddEveRatio) -- but
several are computed on 5 different representations (STFTmag, STFTpow, Harmonic, ERBfft,
ERBgam), and it is not obvious a priori which one matches MATLAB TT's own convention. This
script tries all 5 representations for the ambiguous (spectral-shape) descriptors and picks
the best-correlated one per feature, rather than assuming.
"""
import pathlib
import sys
import multiprocessing

import numpy as np
import pandas as pd
import soundfile as sf

TTB_ROOT = pathlib.Path.home() / "projects" / "ttb_peeters"
sys.path.insert(0, str(TTB_ROOT))

AUDIO_DIR = pathlib.Path('/data2/anasynth_nonbp/longeot/spinvae2_datasets/Dexed/Audio')
RAW_PICKLE = pathlib.Path('/data2/anasynth_nonbp/longeot/spinvae2_datasets/Dexed/raw_timbre_features.df.pickle')
N_SAMPLE = 300
N_WORKERS = 16

SPECTRAL_GROUPS = ['STFTmag', 'STFTpow', 'Harmonic', 'ERBfft', 'ERBgam']
SPECTRAL_KEYS = ['SpecCent', 'SpecSpread', 'SpecSkew', 'SpecKurt', 'SpecSlope',
                  'SpecDecr', 'SpecRollOff', 'SpecVar', 'FrameErg', 'SpecFlat', 'SpecCrest']

# Unambiguous, single-representation descriptors (no group search needed)
DIRECT_MAPPING = [
    ('tt_F0_med', 'Harmonic', 'F0', 'median'),
    ('tt_F0_IQR', 'Harmonic', 'F0', 'iqr'),
    ('tt_HarmErg_IQR', 'Harmonic', 'HarmErg', 'iqr'),
    ('tt_InHarm_med', 'Harmonic', 'InHarm', 'median'),
    ('tt_InHarm_IQR', 'Harmonic', 'InHarm', 'iqr'),
    ('tt_OddEvenRatio_med', 'Harmonic', 'OddEveRatio', 'median'),
    ('tt_OddEvenRatio_IQR', 'Harmonic', 'OddEveRatio', 'iqr'),
    ('tt_AmpMod', 'TEE', 'AmpMod', 'median'),
    ('tt_Att', 'TEE', 'Att', 'median'),
    ('tt_Dec', 'TEE', 'Dec', 'median'),
    ('tt_DecSlope', 'TEE', 'DecSlope', 'median'),
    ('tt_EffDur', 'TEE', 'EffDur', 'median'),
    ('tt_FreqMod', 'TEE', 'FreqMod', 'median'),
    ('tt_LAT', 'TEE', 'LAT', 'median'),
    ('tt_RMSEnv_med', 'TEE', 'RMSEnv', 'median'),
    ('tt_RMSEnv_IQR', 'TEE', 'RMSEnv', 'iqr'),
    ('tt_TempCent', 'TEE', 'TempCent', 'median'),
]

# tt_feature -> (spectral_key) to search across all 5 groups
SEARCH_MAPPING = [
    ('tt_SpecCent_IQR', 'SpecCent', 'iqr'),
    ('tt_SpecCrest_med', 'SpecCrest', 'median'),
    ('tt_SpecCrest_IQR', 'SpecCrest', 'iqr'),
    ('tt_SpecDecr_med', 'SpecDecr', 'median'),
    ('tt_SpecDecr_IQR', 'SpecDecr', 'iqr'),
    ('tt_SpecFlat_med', 'SpecFlat', 'median'),
    ('tt_SpecFlat_IQR', 'SpecFlat', 'iqr'),
    ('tt_SpecKurt_med', 'SpecKurt', 'median'),
    ('tt_SpecKurt_IQR', 'SpecKurt', 'iqr'),
    ('tt_SpecRollOff_med', 'SpecRollOff', 'median'),
    ('tt_SpecRollOff_IQR', 'SpecRollOff', 'iqr'),
    ('tt_SpecSpread_med', 'SpecSpread', 'median'),
    ('tt_SpecSpread_IQR', 'SpecSpread', 'iqr'),
    ('tt_SpecVar_med', 'SpecVar', 'median'),
    ('tt_SpecVar_IQR', 'SpecVar', 'iqr'),
]


def _extract_scalar(desc_entry, stat):
    v = desc_entry['median'] if stat == 'median' else desc_entry['iqr']
    return float(np.asarray(v).flatten()[0])


def _compute_one(uid: int):
    import peeaudiolight  # noqa: F401
    import peeTimbreToolbox

    wav = AUDIO_DIR / f"{uid:06d}_pitch056vel075_var000.wav"
    if not wav.exists():
        return None
    audio, sr = sf.read(wav)
    try:
        descHub_d = peeTimbreToolbox.F_computeAllDescriptor(audio, sr)
        descHub_d = peeTimbreToolbox.F_temporalModeling(descHub_d)
    except Exception as e:
        print(f"UID {uid} failed: {e}")
        return None

    row = {'preset_UID': uid}
    for tt_col, group, key, stat in DIRECT_MAPPING:
        try:
            row[f"ttb_{tt_col}"] = _extract_scalar(descHub_d[group][key], stat)
        except Exception:
            row[f"ttb_{tt_col}"] = np.nan
    for group in SPECTRAL_GROUPS:
        for key in SPECTRAL_KEYS:
            try:
                entry = descHub_d[group][key]
                row[f"ttb_{group}_{key}_med"] = _extract_scalar(entry, 'median')
                row[f"ttb_{group}_{key}_iqr"] = _extract_scalar(entry, 'iqr')
            except Exception:
                row[f"ttb_{group}_{key}_med"] = np.nan
                row[f"ttb_{group}_{key}_iqr"] = np.nan
    return row


def main():
    raw_df = pd.read_pickle(RAW_PICKLE)
    sub = raw_df[raw_df['variation'] == 0].sample(n=N_SAMPLE, random_state=0).reset_index(drop=True)
    uids = sub['preset_UID'].astype(int).tolist()

    print(f"Computing ttb (Python TT) features for {len(uids)} presets, {N_WORKERS} workers...")
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(N_WORKERS) as pool:
        rows = pool.map(_compute_one, uids)
    rows = [r for r in rows if r is not None]
    ttb_df = pd.DataFrame(rows)
    print("ttb_df shape:", ttb_df.shape, f"({len(uids) - len(rows)} failures)")
    ttb_df.to_pickle('/tmp/ttb_df_sample300.pickle')

    merged = sub.merge(ttb_df, on='preset_UID')
    merged.to_pickle('/tmp/merged_ttb_sample300.pickle')

    results = []
    print(f"\n{'tt_feature':22s} {'best_source':30s} {'r':>8s}")
    for tt_col, group, key, stat in DIRECT_MAPPING:
        ttb_col = f"ttb_{tt_col}"
        r = merged[tt_col].corr(merged[ttb_col])
        results.append((tt_col, f"{group}.{key}", r))
        print(f"{tt_col:22s} {group + '.' + key:30s} {r:8.3f}")

    for tt_col, key, stat in SEARCH_MAPPING:
        suffix = 'med' if stat == 'median' else 'iqr'
        candidates = [f"ttb_{group}_{key}_{suffix}" for group in SPECTRAL_GROUPS]
        corrs = merged[candidates].corrwith(merged[tt_col]).abs().sort_values(ascending=False)
        best_col = corrs.index[0]
        best_r_signed = merged[tt_col].corr(merged[best_col])
        results.append((tt_col, best_col.replace('ttb_', ''), best_r_signed))
        print(f"{tt_col:22s} {best_col.replace('ttb_', ''):30s} {best_r_signed:8.3f}")

    result_df = pd.DataFrame(results, columns=['tt_feature', 'best_source', 'r'])
    result_df.to_csv('/tmp/ttb_validation_v2.csv', index=False)
    print(f"\nMean |r|: {result_df['r'].abs().mean():.3f}")
    print(f"Features with |r| > 0.9: {(result_df['r'].abs() > 0.9).sum()} / {len(result_df)}")
    print(f"Features with |r| > 0.7: {(result_df['r'].abs() > 0.7).sum()} / {len(result_df)}")
    print(f"Features with |r| < 0.5: {(result_df['r'].abs() < 0.5).sum()} / {len(result_df)}")


if __name__ == "__main__":
    main()
