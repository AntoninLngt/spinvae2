import os
import sys
from copy import deepcopy
from typing import Dict, List

import numpy as np

from adtool.utils.leaf.Leaf import Leaf
from adtool.examples.dexed.systems.Dexed import DexedSimulation

SPINVAE2_ROOT = os.path.expanduser("~/projects/spinvae2")
if SPINVAE2_ROOT not in sys.path:
    sys.path.insert(0, SPINVAE2_ROOT)

from synth.dexed import Dexed  # noqa: E402


class DexedParameterMap(Leaf):
    """ Theta = Dexed's 155 raw VST parameters, as a list of (idx, value) tuples with values
    in [0, 1] (Dexed's own preset format -- see synth.dexed.Dexed.assign_preset).

    sample() reuses spinvae2's Dexed.get_random_preset(): a properly quantized random preset
    (per-parameter cardinality via get_param_cardinality), NOT naive uniform noise.

    mutate() reuses spinvae2's Dexed.get_similar_preset(): the same data-augmentation function
    used to generate the dataset's preset "variations" (data/dexeddataset.py), so IMGEP
    mutations stay consistent with what the rest of the project already considers "a similar
    preset". """

    N_VST_PARAMS = 155
    # Non-learnable indices, mirroring DexedDataset's default config (constant_filter_and_tune_params=True,
    # constant_middle_C=True, all 6 operators enabled): filter/tune (0-3), middle C (13),
    # the 6 operator on/off switches (44, 66, 88, 110, 132, 154) -- these are forced to fixed
    # values by _apply_defaults() below rather than explored by IMGEP. 155 - 11 = 144 learnable.
    FIXED_INDICES = [0, 1, 2, 3, 13, 44, 66, 88, 110, 132, 154]
    FIXED_DEFAULTS = {0: 1.0, 1: 0.0, 2: 1.0, 3: 0.5, 13: 0.5,
                       44: 1.0, 66: 1.0, 88: 1.0, 110: 1.0, 132: 1.0, 154: 1.0}

    def __init__(
        self,
        system: DexedSimulation,
        premap_key: str = "params",
        seed: int = 0,
        **config_decorator_kwargs,
    ):
        super().__init__()
        self.premap_key = premap_key
        self.seed = seed
        self._rng_counter = 0

        # Throwaway Dexed instance, used only for its preset-generation helpers (get_random_preset).
        # It is NOT used for rendering -- DexedSimulation owns the (heavier) rendering engine.
        self._dexed_helper = Dexed(output_Fs=16000)
        self.learnable_indices = [i for i in range(self.N_VST_PARAMS) if i not in self.FIXED_INDICES]

    def map(self, input: Dict, override_existing: bool = True) -> Dict:
        intermed_dict = deepcopy(input)
        if (override_existing and self.premap_key in intermed_dict) or (
            self.premap_key not in intermed_dict
        ):
            intermed_dict[self.premap_key] = self.sample()
        return intermed_dict

    def sample(self) -> Dict:
        self._rng_counter += 1
        preset = self._dexed_helper.get_random_preset(seed=self.seed + self._rng_counter)
        preset = self._apply_defaults(preset)
        return {"dynamic_params": {"preset": preset}}

    def mutate(self, parameter_dict: Dict) -> Dict:
        intermed_dict = deepcopy(parameter_dict)
        preset_list = intermed_dict["dynamic_params"]["preset"]

        preset_array = np.array([v for _, v in preset_list], dtype=float)
        self._rng_counter += 1
        # variation>=2 in get_similar_preset() applies random noise to (most of) the learnable
        # parameters -- variation=1 would only change the algorithm, variation=0 is a no-op.
        mutated_array = Dexed.get_similar_preset(
            preset_array, variation=2, learnable_indices=self.learnable_indices,
            random_seed=self.seed + self._rng_counter,
        )
        mutated_preset = [(i, float(v)) for i, v in enumerate(mutated_array)]
        mutated_preset = self._apply_defaults(mutated_preset)

        intermed_dict["dynamic_params"]["preset"] = mutated_preset
        return intermed_dict

    def _apply_defaults(self, preset: List) -> List:
        """ Forces the fixed (non-learnable) parameters to spinvae2's default values, so IMGEP
        only ever explores the 144 learnable parameters -- mirrors
        Dexed.set_default_general_filter_and_tune_params() + set_all_oscillators_on(). """
        return [(i, self.FIXED_DEFAULTS.get(i, v)) for i, v in preset]
