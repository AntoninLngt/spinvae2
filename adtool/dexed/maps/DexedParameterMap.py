import os
import pickle
import sys
from copy import deepcopy
from typing import Dict, List

import numpy as np

from adtool.utils.leaf.Leaf import Leaf
from systems.Dexed import DexedSimulation

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
        noise_scale: float = 1.0,
        op_mode_mutation_prob: float = 0.0,
        reflect_boundary: bool = False,
        algorithm_mutation_prob: float = 0.0,
        big_jump_prob: float = 0.0,
        big_jump_scale: float = 8.0,
        **config_decorator_kwargs,
    ):
        super().__init__()
        self.premap_key = premap_key
        self.seed = seed
        # Multiplies mutate()'s continuous-parameter noise magnitude -- see
        # Dexed.get_similar_preset()'s docstring: `variation` itself is NOT an intensity knob
        # (it only seeds which pseudo-random variation is drawn), so this is the actual lever
        # for how far a mutation moves. Default 1.0 matches the original, uncalibrated-for-IMGEP
        # behaviour (see notebook section 14).
        self.noise_scale = noise_scale
        # Per-operator probability of flipping the ratio/fixed OP mode switch, normally never
        # mutated at all -- see get_similar_preset()'s docstring. Default 0.0 keeps the
        # original behaviour; investigation notebook section 7.1 tests whether this frozen
        # structural parameter is a bottleneck for behavioural diversity.
        self.op_mode_mutation_prob = op_mode_mutation_prob
        # Reflects instead of clips at the [0,1] boundary -- notebook section 3.4 found clipping
        # caps effective displacement well below what noise_scale predicts. Default False keeps
        # the original behaviour.
        self.reflect_boundary = reflect_boundary
        # Probability of a fully unconstrained algorithm jump, on top of change_algorithm_to_similar.
        # Default 0.0 keeps the original behaviour.
        self.algorithm_mutation_prob = algorithm_mutation_prob
        # Per-mutation probability of a large basin-hopping-style jump (noise_scale x
        # big_jump_scale for that one call) mixed into otherwise-local mutation -- see
        # get_similar_preset()'s docstring. Default 0.0 keeps the original behaviour.
        self.big_jump_prob = big_jump_prob
        self.big_jump_scale = big_jump_scale
        self._rng_counter = 0
        self.learnable_indices = [i for i in range(self.N_VST_PARAMS) if i not in self.FIXED_INDICES]
        self._dexed_helper_instance = None

    @property
    def _dexed_helper(self) -> Dexed:
        # Throwaway Dexed instance, used only for its preset-generation helpers (get_random_preset).
        # It is NOT used for rendering -- DexedSimulation owns the (heavier) rendering engine.
        # Lazily built rather than eagerly loaded in __init__: adtool restores checkpoints via
        # `cls.__new__(cls)` + `__dict__.update(state)`, bypassing __init__ entirely, and this
        # wraps a native DawDreamer object that cannot be pickled -- see
        # DexedSimulation.dexed_engine/checkpoint_state() for the same constraint.
        if self._dexed_helper_instance is None:
            self._dexed_helper_instance = Dexed(output_Fs=16000)
        return self._dexed_helper_instance

    def checkpoint_state(self) -> Dict:
        # See DexedSimulation.checkpoint_state() -- same native-object constraint, same
        # defensive approach (drop whatever fails to pickle rather than a hardcoded key).
        state = {}
        for key, value in self.__dict__.items():
            try:
                pickle.dumps(value)
            except Exception:
                value = None
            state[key] = value
        return state

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
            random_seed=self.seed + self._rng_counter, noise_scale=self.noise_scale,
            op_mode_mutation_prob=self.op_mode_mutation_prob,
            reflect_boundary=getattr(self, "reflect_boundary", False),
            algorithm_mutation_prob=getattr(self, "algorithm_mutation_prob", 0.0),
            big_jump_prob=getattr(self, "big_jump_prob", 0.0),
            big_jump_scale=getattr(self, "big_jump_scale", 8.0),
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
