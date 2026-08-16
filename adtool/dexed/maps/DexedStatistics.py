import os
import sys
from copy import deepcopy
from typing import Dict

import numpy as np

from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.leaf.locators.locators import BlobLocator
from adtool.wrappers.BoxProjector import BoxProjector
from systems.Dexed import DexedSimulation

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
        silence_sink: bool = False,
        goal_sampling: str = "onehot",
        note_like_goal_box: bool = False,
    ):
        super().__init__()
        self.locator = BlobLocator()
        self.premap_key = premap_key
        self.postmap_key = postmap_key
        self.sample_rate = system.output_Fs
        # See sample(). Default kept at the historical value so that existing runs remain
        # reproducible; "box" is what every other adtool behaviour map does and is what new
        # experiments should use.
        self.goal_sampling = goal_sampling
        # When True, only note-like discoveries widen the goal box (see maps/notelike.py).
        # Rationale: goals are drawn uniformly inside that box, and the box is stretched by
        # whatever lands in it -- measured at up to 1e17 x the valid range on the fragile
        # harmonic descriptors. Goals then point almost exclusively at coordinates no real
        # sound occupies. Restricting the box to sounds that behave like notes keeps the
        # goals inside reachable territory.
        self.note_like_goal_box = note_like_goal_box
        # See _calc_static_statistics: default False keeps the original behaviour (every
        # silent/near-silent render collapses to the exact same all-zeros embedding). Set True
        # to scatter each one to a distinct, far-away point instead -- see investigation
        # notebook section 16/17 for why the shared zero point turned into a "silence attractor"
        # for IMGEP's 1-NN search.
        self.silence_sink = silence_sink

        self.projector = BoxProjector(premap_key=self.postmap_key)

    def map(self, input: Dict) -> Dict:
        intermed_dict = deepcopy(input)

        array = np.array(intermed_dict[self.premap_key], dtype=np.float32)
        raw_output_key = "raw_" + self.premap_key
        intermed_dict[raw_output_key] = array
        del intermed_dict[self.premap_key]

        embedding = self._calc_static_statistics(array, silence_sink=self.silence_sink)

        intermed_dict[self.postmap_key] = embedding
        if self.note_like_goal_box and self.projector.low is not None:
            from maps.notelike import is_note_like_one
            if not is_note_like_one(embedding):
                # Keep the projector's clamping/shape handling, but do not let this
                # discovery move the bounds the goals are drawn from.
                low, high = self.projector.low.copy(), self.projector.high.copy()
                intermed_dict = self.projector.map(intermed_dict)
                self.projector.low, self.projector.high = low, high
                return intermed_dict
        intermed_dict = self.projector.map(intermed_dict)
        return intermed_dict

    def sample(self):
        """Draws the goal that IMGEPExplorer searches the archive for.

        'box' is what every other adtool behaviour map implements (MeanBehaviorMap,
        IdentityBehaviorMap: `return self.projector.sample()`) -- a uniform draw inside the
        bounding box of the behaviours observed so far.

        'onehot' reproduces what this class did from its initial import until 2026-08-14: a
        one-hot vector, i.e. one of only 38 possible goals for an entire run -- the canonical
        basis vectors of the RAW descriptor space, whose axes span orders of magnitude
        (Hz-scale spectral features next to 0-1 ratios). With 38 fixed targets the 1-NN search
        returns the same few archive entries over and over, which is a candidate explanation
        for the collapse observed exactly at the bootstrap boundary and for the coverage
        plateau. Kept only so the earlier runs stay reproducible.
        """
        if self.goal_sampling == "box":
            if self.projector.low is None or self.projector.high is None:
                shape = self.projector.tensor_shape
                return np.zeros(shape if shape is not None else (N_Z_FEATURES,))
            return self.projector.sample()
        shape = self.projector.tensor_shape
        projection = np.zeros(shape)
        projection[np.random.randint(0, shape[0])] = 1
        return projection

    def _calc_static_statistics(self, array: np.ndarray, silence_sink: bool = False) -> np.ndarray:
        peak = float(np.abs(array).max())
        if peak < 1e-4:
            # Silent/near-silent render (common with random raw Dexed presets, unlike the human
            # dataset which is never this quiet) -- both ACTM and ttb are unreliable below this
            # amplitude; fall back to a zero embedding instead of crashing the loop.
            if not silence_sink:
                return np.zeros(N_Z_FEATURES, dtype=np.float32)
            # silence_sink=True: every silent render used to collapse onto the exact same
            # all-zeros point -- literally identical regardless of which of many different
            # silent presets produced it. Investigation notebook section 16 found this turns
            # into a "silence attractor": that single shared point becomes an outsized target
            # for IMGEP's 1-NN search (many discoveries piled on it -> high odds of being the
            # nearest match to any goal sampled nearby), and locally mutating an already-silent
            # preset is disproportionately likely to still be silent, so the search gets stuck
            # revisiting/re-mutating that same spot instead of exploring.
            #
            # Fix: send each silent render to a distinct point far outside the normally-explored
            # region instead. A flat +1e6 offset (not a relative/multiplicative one) is "far" in
            # every dimension regardless of that dimension's natural physical scale (Hz-scale
            # spectral features vs 0-1-scale ratios alike), and a per-preset deterministic jitter
            # (seeded from the raw audio itself, so silence-sink points are still reproducible)
            # keeps different silent presets from re-colliding onto one new shared point, which
            # would just recreate the same problem one location over.
            jitter_rng = np.random.default_rng(abs(hash(array.tobytes())) % (2**32))
            return (1e6 + jitter_rng.normal(0.0, 1.0, N_Z_FEATURES)).astype(np.float32)
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
        embedding = np.array(values, dtype=np.float32)
        # A handful of tt_* features (e.g. OddEvenRatio, InHarm) can divide by a near-zero
        # denominator on degenerate-but-not-silent IMGEP presets (unlike real/human presets,
        # which never hit this) -- utils.timbrefeatures.TimbreFeatures handles this offline via
        # per-column median imputation fit on the whole human dataset, which isn't available for
        # a single online discovery here. Falling back to 0.0 keeps every saved discovery finite
        # (checkpoints, the viewer's 2D projection, and analysis modules all assume that), at the
        # cost of losing the affected feature's value for that one discovery.
        return np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
