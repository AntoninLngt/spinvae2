"""
Computes TimbreToolbox-compatible raw features using geoffroypeeters/ttb (Python port of
MATLAB TimbreToolbox), as a MATLAB-free replacement for utils/timbretoolbox.py.

Output keys exactly match the ones produced by the original MATLAB pipeline (before the
'tt_' prefix is added by abstractbasedataset.py's compute_and_store_timbre_features), so this
is a drop-in replacement for utils.timbretoolbox.TimbreToolboxResults.read_stats_csv's output.

Representation choice matters: several descriptors (SpecCent, SpecSpread, SpecSkew, SpecKurt,
SpecSlope, SpecDecr, SpecRollOff, SpecVar, FrameErg, SpecFlat, SpecCrest) can be computed from
several input representations (STFTmag, STFTpow, ERBfft, ERBgam, Harmonic). The production
MATLAB config (spinvae2/utils/timbre.m) only evaluates STFTpow (not STFTmag) and ERBfft (not
ERBgam), and utils/timbretoolbox.py's CSV parser overwrites same-named descriptors as it reads
representation blocks in the order they appear in the CSV: ERB, Harmonic, STFT, AudioSignal,
TEE -- so STFT (i.e. STFTpow) always wins for shared descriptors, confirmed by inspecting a
real CSV export. Harmonic-only descriptors (F0, InHarm, HarmErg, NoiseErg, Noisiness,
OddEveRatio, HarmDev) naturally come from the Harmonic representation.

Known limitations (see spinvae2/notebooks validation, session of 2026-07-24):
  - F0, InHarm, HarmErg, OddEvenRatio: moderate-to-weak correlation with the real MATLAB
    ground truth (r ~0.13-0.9 depending on the feature). Root cause: the MATLAB toolbox
    installed on bonsho (VincentPerreault0/timbretoolbox) uses a per-frame-variable
    inharmonicity coefficient with a corrected objective function, while ttb implements the
    original 2011 JASA paper's algorithm (constant coefficient, uncorrected objective). This
    is an inherent algorithmic difference, not a portage bug.
  - SpecKurt: weak correlation (r~0.25) even after fixing the STFT frequency-axis bug --
    kurtosis (4th moment) appears especially sensitive to remaining fine implementation
    differences (windowing edge effects, FFT size subtleties).
"""
import sys
import pathlib
from typing import Dict

import numpy as np

TTB_ROOT = pathlib.Path.home() / "projects" / "ttb_peeters"
if str(TTB_ROOT) not in sys.path:
    sys.path.insert(0, str(TTB_ROOT))

import peeaudiolight  # noqa: F401,E402  (side-effect import expected by peeTimbreToolbox)
import peeTimbreToolbox as _ttb  # noqa: E402

# descriptor -> stats to export, matching the *_med/_IQR/_min/_max columns already present in
# spinvae2's raw_timbre_features.df.pickle
_STAT_SUFFIXES = {"median": "med", "iqr": "IQR", "min": "min", "max": "max"}

# global (single-value) TEE descriptors -- no _med/_IQR/_min/_max suffix
_TEE_GLOBAL = ["Att", "Dec", "Rel", "LAT", "AttSlope", "DecSlope", "TempCent", "EffDur", "FreqMod", "AmpMod"]
# time-varying TEE descriptor -- gets the usual _med/_IQR/_min/_max suffixes
_TEE_TIMEVARYING = ["RMSEnv"]

# shared spectral-shape descriptors: always taken from STFTpow (see module docstring)
_STFTPOW_DESCRIPTORS = ["SpecCent", "SpecSpread", "SpecSkew", "SpecKurt", "SpecSlope", "SpecDecr",
                          "SpecRollOff", "SpecVar", "FrameErg", "SpecFlat", "SpecCrest"]

# harmonic-only descriptors: only ever computed from the Harmonic representation
_HARMONIC_DESCRIPTORS = ["F0", "InHarm", "HarmErg", "NoiseErg", "Noisiness", "OddEveRatio", "HarmDev"]


def _scalar(entry: dict, stat: str) -> float:
    return float(np.asarray(entry[stat]).flatten()[0])


def compute_tt_features(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """ Computes the TimbreToolbox-compatible raw feature dict for one audio array, matching
    the column names (without the 'tt_' prefix) of spinvae2's raw_timbre_features.df.pickle. """
    desc_hub = _ttb.F_computeAllDescriptor(audio, sr)
    desc_hub = _ttb.F_temporalModeling(desc_hub)

    feats: Dict[str, float] = {}

    tee = desc_hub["TEE"]
    for name in _TEE_GLOBAL:
        feats[name] = _scalar(tee[name], "median")
    for name in _TEE_TIMEVARYING:
        for stat, suffix in _STAT_SUFFIXES.items():
            feats[f"{name}_{suffix}"] = _scalar(tee[name], stat)

    stftpow = desc_hub["STFTpow"]
    for name in _STFTPOW_DESCRIPTORS:
        for stat, suffix in _STAT_SUFFIXES.items():
            feats[f"{name}_{suffix}"] = _scalar(stftpow[name], stat)

    harmonic = desc_hub["Harmonic"]
    for name in _HARMONIC_DESCRIPTORS:
        # ttb's own key is misspelled "OddEveRatio"; raw_timbre_features.df.pickle uses the
        # correctly-spelled "OddEvenRatio" -- normalize on output.
        out_name = "OddEvenRatio" if name == "OddEveRatio" else name
        for stat, suffix in _STAT_SUFFIXES.items():
            feats[f"{out_name}_{suffix}"] = _scalar(harmonic[name], stat)

    return feats
