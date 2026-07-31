import os
import sys
from copy import deepcopy
from typing import Dict

import numpy as np

from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.leaf.locators.locators import BlobLocator
from adtool.wrappers.BoxProjector import BoxProjector
from adtool.examples.dexed.systems.Dexed import DexedSimulation

SPINVAE2_ROOT = os.path.expanduser("~/projects/spinvae2")
if SPINVAE2_ROOT not in sys.path:
    sys.path.insert(0, SPINVAE2_ROOT)

from utils.timbral_models.Extractor import timbral_extractor  # noqa: E402
from utils.ttb_extractor import compute_tt_features  # noqa: E402

# Exact 6 ACTM + 32 TT selection used throughout spinvae2 (TimbreFeatures' correlation-based
# selection, cf. notebooks/dexed_audit_TT_ACTM.ipynb section 4). Keeping the same names/order
# means this Z is directly comparable to the human-preset dataset's behaviour space.
SELECTED_AC = ['hardness', 'depth', 'brightness', 'roughness', 'warmth', 'boominess']
SELECTED_TT = [
    'SpecCent_IQR', 'SpecCrest_med', 'SpecCrest_IQR', 'SpecDecr_med', 'SpecDecr_IQR',
    'SpecFlat_med', 'SpecFlat_IQR', 'SpecKurt_med', 'SpecKurt_IQR', 'SpecRollOff_med',
    'SpecRollOff_IQR', 'SpecSpread_med', 'SpecSpread_IQR', 'SpecVar_med', 'SpecVar_IQR',
    'F0_med', 'F0_IQR', 'HarmErg_IQR', 'InHarm_med', 'InHarm_IQR', 'OddEvenRatio_med',
    'OddEvenRatio_IQR', 'AmpMod', 'Att', 'Dec', 'DecSlope', 'EffDur', 'FreqMod', 'LAT',
    'RMSEnv_med', 'RMSEnv_IQR', 'TempCent',
]
Z_FEATURE_NAMES = [f"ac_{n}" for n in SELECTED_AC] + [f"tt_{n}" for n in SELECTED_TT]
N_Z_FEATURES = len(Z_FEATURE_NAMES)  # 38


class DexedStatistics(Leaf):
    """ Z = the exact 38-feature canonical SpinVAE2 behaviour space (6 ACTM + 32 TT), computed
    directly in-process with no MATLAB dependency:
      - ACTM (7 raw features, 6 selected) via utils.timbral_models -- always was MATLAB-free.
      - TT (32 selected features) via spinvae2/utils/ttb_extractor.py, a Python port of MATLAB
        TimbreToolbox (geoffroypeeters/ttb) validated against real MATLAB ground truth on 300
        dataset presets (see notebooks/dexed_audit_TT_ACTM.ipynb, ttb validation section).
        ~13/32 TT features match MATLAB almost exactly (temporal-envelope descriptors and
        several spectral ones); a handful (F0, InHarm, HarmErg, OddEvenRatio) have a known,
        structural algorithmic difference from the specific MATLAB toolbox version used to
        generate the human-preset dataset (different harmonic-partial tracking) -- not a
        portage bug, see that notebook for the full validation and caveats.

    Using the full 38D space (instead of ACTM alone) makes Z directly comparable to the human
    dataset's coverage analysis, at the cost of extra compute per evaluation (TT adds ~3-5s on
    top of ACTM's ~1.5s -- budget IMGEP run sizes accordingly; this is no longer "free"). """

    def __init__(
        self,
        system: DexedSimulation,
        premap_key: str = "output",
        postmap_key: str = "output",
    ):
        super().__init__()
        self.locator = BlobLocator()
        self.premap_key = premap_key
        self.postmap_key = postmap_key
        self.sample_rate = system.output_Fs

        self.projector = BoxProjector(premap_key=self.postmap_key)

    def map(self, input: Dict) -> Dict:
        intermed_dict = deepcopy(input)

        array = np.array(intermed_dict[self.premap_key], dtype=np.float32)
        raw_output_key = "raw_" + self.premap_key
        intermed_dict[raw_output_key] = array
        del intermed_dict[self.premap_key]

        embedding = self._calc_static_statistics(array)

        intermed_dict[self.postmap_key] = embedding
        intermed_dict = self.projector.map(intermed_dict)
        return intermed_dict

    def sample(self):
        shape = self.projector.tensor_shape
        projection = np.zeros(shape)
        projection[np.random.randint(0, shape[0])] = 1
        return projection

    def _calc_static_statistics(self, array: np.ndarray) -> np.ndarray:
        peak = float(np.abs(array).max())
        if peak < 1e-4:
            # Silent/near-silent render (common with random raw Dexed presets, unlike the human
            # dataset which is never this quiet) -- both ACTM and ttb are unreliable below this
            # amplitude; fall back to a zero embedding instead of crashing the loop.
            return np.zeros(N_Z_FEATURES, dtype=np.float32)
        # Peak-normalize before feature extraction (both ACTM and ttb): raw random presets span
        # a much wider loudness range than the human dataset. This deliberately differs from
        # ttb's MATLAB-matching default of never normalizing internally (see ttb_extractor.py) --
        # that default is correct for reproducing the human dataset's exact values, but IMGEP
        # explores far quieter/louder presets than any human preset, where un-normalized
        # analysis becomes numerically unreliable.
        normalized = array * (0.99 / peak)

        ac_feats = timbral_extractor(normalized, fs=self.sample_rate, exclude_reverb=True)
        tt_feats = compute_tt_features(normalized, self.sample_rate)

        values = [ac_feats[n] for n in SELECTED_AC] + [tt_feats[n] for n in SELECTED_TT]
        return np.array(values, dtype=np.float32)
