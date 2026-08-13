import json
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
        random_reset_prob: float = 0.0,
        explore_indices: List[int] = None,
        base_preset_seed: int = 0,
        base_preset_mode: str = "random",
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
        # NRAB-style balance (Morel et al.): per-mutation-call probability of ignoring the
        # nearest-neighbour parent entirely and drawing a fully fresh random preset instead (the
        # same global draw the bootstrap phase uses), rather than a small local perturbation of
        # it. Unlike big_jump_prob (still centered on the parent, just larger), this is a
        # genuinely unconstrained global resample -- the bootstrap-vs-goal-directed mixture
        # continues throughout the run instead of stopping hard at equil_time/bootstrap_size.
        # Default 0.0 keeps the original behaviour; their reported optimum is p=0.5.
        self.random_reset_prob = random_reset_prob
        self._rng_counter = 0
        self.learnable_indices = [i for i in range(self.N_VST_PARAMS) if i not in self.FIXED_INDICES]
        # Restricts exploration to a subset of the 144 learnable parameters: every other
        # parameter is frozen to a fixed base preset for the whole run, so the search moves in a
        # low-dimensional subspace instead of all 144 dimensions at once. Default None keeps the
        # original behaviour (all 144 explored).
        #
        # Motivation: every diversity result so far was obtained in the full 144-D space, where
        # local mutation may simply be lost. Shrinking the problem to a handful of the
        # timbrally-decisive FM parameters (algorithm, feedback, per-operator output levels and
        # frequency ratios) tests whether goal-directed exploration works *at all* on this
        # synthesiser, before blaming the operator; the subspace can then be grown back.
        self.explore_indices = list(explore_indices) if explore_indices is not None else None
        if self.explore_indices is not None:
            invalid = [i for i in self.explore_indices if i in self.FIXED_INDICES]
            if invalid:
                raise ValueError(
                    f"explore_indices contains non-learnable (structurally fixed) indices: {invalid}"
                )
        # The frozen parameters need plausible values, not zeros: a preset with every envelope
        # and output level at zero is silent regardless of what the explored subset does.
        # "random" (default): one seeded draw of Dexed.get_random_preset -- reproducible, but a
        # single draw can land unluckily (e.g. an operator's own EG_LEVEL_1 at exactly 0.0 mutes
        # it regardless of its explored OUTPUT LEVEL, observed on runs/restricted/L1_core14 with
        # base_preset_seed=0, where OP2 and OP5 both had EG_LEVEL_1==0.0).
        # "human_median": the real human preset closest (L2, on the 144 learnable dims) to the
        # coordinate-wise median of the full 30,145-preset human database -- guaranteed to be a
        # normal, audible, working sound (no operator pinned to a degenerate extreme), unlike a
        # single synthetic random draw. Precomputed once by scripts/find_median_human_preset.py
        # into configs/restricted/human_median_base_preset.json (UID 4661, "STR JB 01").
        if base_preset_mode not in ("random", "human_median"):
            raise ValueError(f"base_preset_mode must be 'random' or 'human_median', got {base_preset_mode!r}")
        self.base_preset_seed = base_preset_seed
        self.base_preset_mode = base_preset_mode
        self._base_preset_values = None
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

    @property
    def _base_preset(self) -> List[float]:
        """ Values the non-explored parameters are frozen to when explore_indices is set.
        Built lazily (and rebuilt after a checkpoint restore, which bypasses __init__ -- same
        constraint as _dexed_helper). """
        if self._base_preset_values is None:
            if self.base_preset_mode == "human_median":
                json_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "configs", "restricted", "human_median_base_preset.json",
                )
                with open(json_path) as f:
                    full_155 = json.load(f)["full_155_values"]
                preset = list(enumerate(full_155))
            else:
                preset = self._dexed_helper.get_random_preset(seed=self.base_preset_seed)
            self._base_preset_values = [float(v) for _, v in self._apply_defaults(preset)]
        return self._base_preset_values

    def _apply_explore_restriction(self, preset: List) -> List:
        """ Freezes every learnable parameter outside explore_indices to the base preset. """
        if self.explore_indices is None:
            return preset
        explored = set(self.explore_indices)
        base = self._base_preset
        return [(i, v if (i in explored or i in self.FIXED_INDICES) else base[i])
                for i, v in preset]

    def sample(self) -> Dict:
        self._rng_counter += 1
        preset = self._dexed_helper.get_random_preset(seed=self.seed + self._rng_counter)
        preset = self._apply_defaults(preset)
        preset = self._apply_explore_restriction(preset)
        return {"dynamic_params": {"preset": preset}}

    def mutate(self, parameter_dict: Dict) -> Dict:
        self._rng_counter += 1
        random_reset_prob = getattr(self, "random_reset_prob", 0.0)
        if random_reset_prob > 0.0:
            decision_rng = np.random.default_rng(self.seed + 555555555 + self._rng_counter)
            if decision_rng.random() < random_reset_prob:
                return self.sample()

        intermed_dict = deepcopy(parameter_dict)
        preset_list = intermed_dict["dynamic_params"]["preset"]

        preset_array = np.array([v for _, v in preset_list], dtype=float)
        # With explore_indices set, only that subset is perturbed; the restriction is also
        # re-applied after the call, since get_similar_preset() may touch the algorithm index
        # independently of the indices it is given.
        mutable_indices = (self.explore_indices if self.explore_indices is not None
                           else self.learnable_indices)
        # variation>=2 in get_similar_preset() applies random noise to (most of) the learnable
        # parameters -- variation=1 would only change the algorithm, variation=0 is a no-op.
        mutated_array = Dexed.get_similar_preset(
            preset_array, variation=2, learnable_indices=mutable_indices,
            random_seed=self.seed + self._rng_counter, noise_scale=self.noise_scale,
            op_mode_mutation_prob=self.op_mode_mutation_prob,
            reflect_boundary=getattr(self, "reflect_boundary", False),
            algorithm_mutation_prob=getattr(self, "algorithm_mutation_prob", 0.0),
            big_jump_prob=getattr(self, "big_jump_prob", 0.0),
            big_jump_scale=getattr(self, "big_jump_scale", 8.0),
        )
        mutated_preset = [(i, float(v)) for i, v in enumerate(mutated_array)]
        mutated_preset = self._apply_defaults(mutated_preset)
        mutated_preset = self._apply_explore_restriction(mutated_preset)

        intermed_dict["dynamic_params"]["preset"] = mutated_preset
        return intermed_dict

    def _apply_defaults(self, preset: List) -> List:
        """ Forces the fixed (non-learnable) parameters to spinvae2's default values, so IMGEP
        only ever explores the 144 learnable parameters -- mirrors
        Dexed.set_default_general_filter_and_tune_params() + set_all_oscillators_on(). """
        return [(i, self.FIXED_DEFAULTS.get(i, v)) for i, v in preset]
