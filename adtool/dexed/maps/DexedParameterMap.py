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
    # Standard DX7 carrier table: which operators reach the output for each of the 32
    # algorithms (1-indexed). Modulators only shape carriers -- an operator at full output
    # level contributes nothing audible unless it is a carrier, which is why total operator
    # energy is identical between silent and audible discoveries while carrier level is not.
    CARRIERS = {
        1: [1, 3], 2: [1, 3], 3: [1, 4], 4: [1, 4], 5: [1, 3, 5], 6: [1, 3, 5],
        7: [1, 3], 8: [1, 3], 9: [1, 3], 10: [1, 4], 11: [1, 4], 12: [1, 3],
        13: [1, 3], 14: [1, 3], 15: [1, 3], 16: [1], 17: [1], 18: [1],
        19: [1, 4, 5], 20: [1, 2, 4], 21: [1, 2, 4, 5], 22: [1, 3, 4, 5],
        23: [1, 2, 4, 5], 24: [1, 2, 3, 4, 5], 25: [1, 2, 3, 4, 5], 26: [1, 2, 4],
        27: [1, 2, 4], 28: [1, 3, 6], 29: [1, 2, 3, 5], 30: [1, 2, 3, 5],
        31: [1, 2, 3, 4, 5], 32: [1, 2, 3, 4, 5, 6],
    }
    ALGORITHM_IDX = 4
    OP_OUTPUT_LEVEL_IDX = [23 + 22 * op + 8 for op in range(6)]  # OP1..OP6 OUTPUT LEVEL
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
        carrier_floor: float = 0.0,
        logit_mutation: bool = False,
        logit_sigma: float = 0.75,
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
        # Guarantees at least one CARRIER keeps an output level above this value. Silence is
        # governed by the carriers' output level, not by total operator energy: measured on
        # random sampling, max-carrier level in [0.6,1.0] gives 3% silence, [0.4,0.6] gives 25%,
        # [0.2,0.4] gives 53%. Local mutation regresses every parameter toward the middle of
        # [0,1], parking carriers at ~0.53 (the danger zone) where uniform sampling keeps them
        # at ~0.85. Default 0.0 disables the constraint.
        self.carrier_floor = carrier_floor
        # Mutates in logit space (noise added to log(p/(1-p)), then mapped back through the
        # sigmoid) instead of adding Gaussian noise directly in [0,1] and clipping. Clipped
        # Gaussian walks concentrate toward 0.5; in logit space the step is relative to the
        # current position, so a parameter near an extreme keeps making proportionally small
        # moves instead of being dragged to the centre. Default False keeps the original path.
        self.logit_mutation = logit_mutation
        self.logit_sigma = logit_sigma
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
        if self.carrier_floor > 0.0:
            preset = self._apply_carrier_floor(preset)
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
        if self.logit_mutation:
            decision_rng = np.random.default_rng(self.seed + 777777777 + self._rng_counter)
            mutated_array = self._logit_mutate(preset_array, decision_rng)
            mutated_preset = [(i, float(v)) for i, v in enumerate(mutated_array)]
            mutated_preset = self._apply_defaults(mutated_preset)
            mutated_preset = self._apply_explore_restriction(mutated_preset)
            if self.carrier_floor > 0.0:
                mutated_preset = self._apply_carrier_floor(mutated_preset)
            intermed_dict["dynamic_params"]["preset"] = mutated_preset
            return intermed_dict

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
        if self.carrier_floor > 0.0:
            mutated_preset = self._apply_carrier_floor(mutated_preset)

        intermed_dict["dynamic_params"]["preset"] = mutated_preset
        return intermed_dict

    def _logit_mutate(self, preset_array: np.ndarray, rng) -> np.ndarray:
        """Adds Gaussian noise in logit space, then maps back through the sigmoid.

        A clipped Gaussian walk in [0,1] has a stationary distribution concentrated around 0.5,
        which is exactly the failure mode measured here: carrier output levels park at ~0.53
        (25-50% silence) where uniform sampling keeps them at ~0.85 (3% silence). In logit space
        the boundaries are at +/-infinity, so the walk has no centre to collapse onto and the
        step size is relative to the current value.
        """
        idx = np.array(self.explore_indices if self.explore_indices is not None
                        else self.learnable_indices, dtype=int)
        p = np.clip(preset_array[idx], 1e-4, 1 - 1e-4)
        logit = np.log(p / (1.0 - p))
        logit = logit + rng.normal(0.0, self.logit_sigma * self.noise_scale, size=logit.shape)
        out = preset_array.copy()
        out[idx] = 1.0 / (1.0 + np.exp(-logit))
        return out

    def _apply_carrier_floor(self, preset: List) -> List:
        """Guarantees at least one carrier stays audible.

        Which operators are carriers depends on the algorithm, so this is resolved per preset.
        Only the single loudest carrier is raised (to carrier_floor), leaving the timbre
        otherwise untouched -- the goal is to keep the preset out of the silent region, not to
        force it loud.
        """
        values = dict(preset)
        algo_norm = values.get(self.ALGORITHM_IDX, 0.0)
        algo = int(round(1 + float(algo_norm) * 31))
        carriers = self.CARRIERS.get(max(1, min(32, algo)), [1])
        carrier_idx = [self.OP_OUTPUT_LEVEL_IDX[op - 1] for op in carriers]
        levels = [(i, float(values.get(i, 0.0))) for i in carrier_idx]
        if not levels:
            return preset
        best_i, best_v = max(levels, key=lambda t: t[1])
        if best_v >= self.carrier_floor:
            return preset
        return [(i, self.carrier_floor if i == best_i else v) for i, v in preset]

    def _apply_defaults(self, preset: List) -> List:
        """ Forces the fixed (non-learnable) parameters to spinvae2's default values, so IMGEP
        only ever explores the 144 learnable parameters -- mirrors
        Dexed.set_default_general_filter_and_tune_params() + set_all_oscillators_on(). """
        return [(i, self.FIXED_DEFAULTS.get(i, v)) for i, v in preset]
