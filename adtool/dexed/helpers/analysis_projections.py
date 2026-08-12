"""Projection functions for the built-in analysis modules (comparison_1d, comparison_2d,
space_coverage -- see adtool/docs/VISUALIZATION.md and analysis_metrics/analysis_run).

A projection receives the list of loaded DiscoverySet objects (one per dataset the UI
was given: the primary run plus every comparison dataset) and returns one array per
dataset, in the same order, plus an optional trailing list of per-dimension labels shared
by all of them. This is the built-in-tool equivalent of the ad hoc scripts used
elsewhere in this investigation (coverage_diversity.py, analyse_theta.py) -- same
underlying arrays, but wired for interactive comparison in the viewer instead of a
one-off script run.
"""
from typing import Any

import numpy as np

N_VST_PARAMS = 155
FIXED_INDICES = [0, 1, 2, 3, 13, 44, 66, 88, 110, 132, 154]
LEARNABLE_INDICES = [i for i in range(N_VST_PARAMS) if i not in FIXED_INDICES]

# Mirrors maps.DexedStatistics.Z_FEATURE_NAMES verbatim. Not imported directly: that module
# pulls in the full DawDreamer/timbral-extraction chain at import time, far too heavy just to
# get 38 label strings for every analysis run triggered from the viewer.
Z_FEATURE_NAMES = [
    "ac_hardness", "ac_depth", "ac_brightness", "ac_roughness", "ac_warmth", "ac_boominess",
    "tt_SpecCent_IQR", "tt_SpecCrest_med", "tt_SpecCrest_IQR", "tt_SpecDecr_med", "tt_SpecDecr_IQR",
    "tt_SpecFlat_med", "tt_SpecFlat_IQR", "tt_SpecKurt_med", "tt_SpecKurt_IQR", "tt_SpecRollOff_med",
    "tt_SpecRollOff_IQR", "tt_SpecSpread_med", "tt_SpecSpread_IQR", "tt_SpecVar_med", "tt_SpecVar_IQR",
    "tt_F0_med", "tt_F0_IQR", "tt_HarmErg_IQR", "tt_InHarm_med", "tt_InHarm_IQR", "tt_OddEvenRatio_med",
    "tt_OddEvenRatio_IQR", "tt_AmpMod", "tt_Att", "tt_Dec", "tt_DecSlope", "tt_EffDur", "tt_FreqMod",
    "tt_LAT", "tt_RMSEnv_med", "tt_RMSEnv_IQR", "tt_TempCent",
]

_PARAM_NAMES_CACHE = None


def _learnable_param_names() -> list:
    global _PARAM_NAMES_CACHE
    if _PARAM_NAMES_CACHE is None:
        import pathlib
        import sqlite3
        import pandas as pd
        db = pathlib.Path.home() / "projects" / "spinvae2" / "synth" / "dexed_presets.sqlite"
        conn = sqlite3.connect(str(db))
        all_names = pd.read_sql_query("SELECT * FROM param ORDER BY index_param", conn)["name"].to_list()
        conn.close()
        _PARAM_NAMES_CACHE = [all_names[i] for i in LEARNABLE_INDICES]
    return _PARAM_NAMES_CACHE


def z_projection(datasets: list, config: dict) -> tuple:
    """Identity projection onto the 38-D behaviour space Z (the 'output' key already
    loaded by the viewer for every discovery) -- per-descriptor distributions/scatter."""
    arrays = [dataset.outputs for dataset in datasets]
    return tuple(arrays) + (Z_FEATURE_NAMES,)


def theta_projection(datasets: list, config: dict) -> tuple:
    """Projection onto the 144 learnable Dexed parameters (Theta) -- lets the viewer
    compare parameter-space distributions directly, the question one level upstream of Z."""
    arrays = []
    for dataset in datasets:
        rows = []
        for payload in dataset.payloads:
            preset = payload["params"]["dynamic_params"]["preset"]
            values = [v for _, v in preset]
            rows.append([values[i] for i in LEARNABLE_INDICES])
        arrays.append(np.asarray(rows, dtype=float))
    return tuple(arrays) + (_learnable_param_names(),)
