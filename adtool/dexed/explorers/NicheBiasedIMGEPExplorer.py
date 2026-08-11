"""IMGEP explorer variant that biases *parent* selection (not goal sampling) by a
per-niche curiosity counter, following Jonsson, Erdem, Fasciani & Glette, "Quality-Diversity
Search in Sound Generation" (2026) section 2.3: their MAP-Elites gives each niche a
decreasing curiosity score that lowers how often an already-exploited niche is picked again
as a reproduction source.

adtool's own CuriosityIMGEPExplorer/novelty_weight (see investigation notebook section 11)
biases which *goal* gets sampled -- and ends up re-sampling goals near already-visited
history points, i.e. re-refinement rather than escape. This mechanism instead biases which of
the k nearest history points (to an unchanged, uniformly-sampled goal) is picked as the parent
to mutate, penalizing niches that keep reproducing without adding new cells. This is a
project-local subclass; adtool's IMGEPExplorerInstance/SpecificMutator/HistoryStore are not
modified (SpecificMutator stays a pure delegator, see notebook section 11 conclusion).

Default `niche_curiosity_bias=False` keeps stock IMGEPExplorerInstance behaviour (single
1-nearest-neighbour parent, same as every previous run in the investigation) bit-for-bit.
"""
import os
import pickle
import sys
from typing import Any, Dict

import numpy as np
from pydantic import Field

from adtool.explorers.IMGEPExplorer import (
    IMGEPConfig,
    IMGEPExplorerInstance,
)
from adtool.systems import System
from adtool.utils.expose_config.expose_config import expose
from adtool.utils.factory import instantiate_object

DEXED_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # adtool/dexed/
if DEXED_ROOT not in sys.path:
    sys.path.insert(0, DEXED_ROOT)

DEFAULT_GRID_FIT_PATH = os.path.join(DEXED_ROOT, "figures", "grid_fit.pickle")


class GridNicheTracker:
    """Same fixed PCA-4D / 8-bins-per-dim grid as scripts/coverage_curve_v3.py (frozen fit
    loaded from `grid_fit_path` so every run maps Z onto identical cells, comparable across
    seeds/configs), plus one decreasing curiosity counter per cell."""

    def __init__(self, grid_fit_path: str, start: float = 10.0, decay: float = 1.0, floor: float = 1.0):
        with open(grid_fit_path, "rb") as f:
            fit = pickle.load(f)
        self.mu = fit["mu"]
        self.sigma = fit["sigma"]
        self.pca = fit["pca"]
        self.lo = fit["lo"]
        self.hi = fit["hi"]
        self.n_bins = fit["n_bins"]
        self.start = start
        self.decay = decay
        self.floor = floor
        self.counts: Dict[tuple, float] = {}

    def cell_of(self, z: np.ndarray) -> tuple:
        z = np.asarray(z, dtype=float).reshape(1, -1)
        z_pca = self.pca.transform((z - self.mu) / self.sigma)[0]
        z_pca = np.clip(z_pca, self.lo, self.hi)
        cell = []
        for d in range(z_pca.shape[0]):
            edges = np.linspace(self.lo[d], self.hi[d], self.n_bins + 1)
            idx = int(np.clip(np.digitize(z_pca[d], edges[1:-1]), 0, self.n_bins - 1))
            cell.append(idx)
        return tuple(cell)

    def curiosity(self, cell: tuple) -> float:
        return self.counts.get(cell, self.start)

    def observe(self, cell: tuple) -> None:
        # Cully & Demiris / Jonsson et al. logic: a cell hit for the first time is a genuine
        # new discovery, left at full curiosity; a cell hit again (already occupied) means this
        # niche is being re-exploited without adding coverage, so it's decremented.
        if cell in self.counts:
            self.counts[cell] = max(self.floor, self.counts[cell] - self.decay)
        else:
            self.counts[cell] = self.start


class NicheBiasedIMGEPExplorerInstance(IMGEPExplorerInstance):
    def __init__(
        self,
        *args,
        niche_curiosity_bias: bool = False,
        niche_curiosity_k: int = 5,
        niche_curiosity_start: float = 10.0,
        niche_curiosity_decay: float = 1.0,
        niche_curiosity_min: float = 1.0,
        grid_fit_path: str = DEFAULT_GRID_FIT_PATH,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.niche_curiosity_bias = niche_curiosity_bias
        self.niche_curiosity_k = niche_curiosity_k
        self.niche_curiosity_start = niche_curiosity_start
        self.niche_curiosity_decay = niche_curiosity_decay
        self.niche_curiosity_min = niche_curiosity_min
        self.grid_fit_path = grid_fit_path
        self._niche_tracker = None
        self._niche_counts_checkpoint = None

    @property
    def niche_tracker(self):
        # Lazily built (and rebuilt after a checkpoint restore, which bypasses __init__ --
        # see checkpoint_state below): restores per-cell counts from the last checkpoint if any,
        # so a resumed run keeps penalizing already-exploited niches instead of resetting them.
        if not self.niche_curiosity_bias:
            return None
        if self._niche_tracker is None:
            self._niche_tracker = GridNicheTracker(
                self.grid_fit_path,
                start=self.niche_curiosity_start,
                decay=self.niche_curiosity_decay,
                floor=self.niche_curiosity_min,
            )
            if getattr(self, "_niche_counts_checkpoint", None):
                self._niche_tracker.counts.update(self._niche_counts_checkpoint)
        return self._niche_tracker

    def map(self, system_output) -> Dict:
        # Identical to IMGEPExplorerInstance.map(), except for the niche-counter update
        # inserted right after history.record() -- every discovery's own cell is observed here,
        # independently of whether/how it later gets picked as a parent.
        target, goal_targeting = self._extract_external_controls(system_output)

        if isinstance(system_output, list):
            new_trial_data = self.mean_var_goal(system_output)
        else:
            new_trial_data = self.observe_results(system_output)

        trial_data_reset = self.history.record(new_trial_data)

        tracker = self.niche_tracker
        if tracker is not None and self.premap_key in trial_data_reset:
            feature = np.asarray(trial_data_reset[self.premap_key], dtype=float).reshape(-1)
            if feature.size > 0 and np.all(np.isfinite(feature)):
                tracker.observe(tracker.cell_of(feature))

        if self.timestep < self.equil_time:
            trial_data_reset = self.parameter_map.map(trial_data_reset, override_existing=True)
            trial_data_reset["equil"] = 1
        else:
            params_trial = self.suggest_trial(
                history_lookback_length=self.history_lookback_length,
                goal=target,
                goal_targeting=goal_targeting,
            )
            trial_data_reset[self.postmap_key] = params_trial
            trial_data_reset = self.parameter_map.map(trial_data_reset, override_existing=False)
            trial_data_reset["equil"] = 0

        self.timestep += 1
        return trial_data_reset

    def _vector_search_for_goal(self, goal: np.ndarray, history_lookback_length: int) -> Dict:
        tracker = self.niche_tracker
        if tracker is None:
            return super()._vector_search_for_goal(goal, history_lookback_length)

        matches = self.history.nearest(
            np.asarray(goal, dtype=float),
            k=self.niche_curiosity_k,
            history_lookback_length=history_lookback_length,
        )
        if not matches:
            match = self.history.random(history_lookback_length=history_lookback_length)
            return match.payload if match else self.parameter_map.sample()

        weights = np.array(
            [tracker.curiosity(tracker.cell_of(m.feature)) for m in matches],
            dtype=float,
        )
        weights = weights / weights.sum()
        # Uses the global np.random stream (driven by --seed), the same stream that already
        # drives goal sampling -- see investigation notebook section 9 on the two independent
        # RNG streams; keeping parent selection on this stream keeps a run fully reproducible
        # from a single --seed.
        chosen = np.random.choice(len(matches), p=weights)
        return matches[chosen].payload

    def checkpoint_state(self) -> Dict:
        # Base class already drops "history" (persisted separately, see IMGEPExplorerInstance).
        # _niche_tracker wraps a fitted sklearn PCA object read from grid_fit_path, not worth
        # pickling into every checkpoint -- only the per-cell counts (the actual mutable state)
        # need to survive a checkpoint/resume, so they're saved/restored separately below.
        state = super().checkpoint_state()
        state["_niche_tracker"] = None
        state["_niche_counts_checkpoint"] = (
            dict(self._niche_tracker.counts) if self._niche_tracker is not None else None
        )
        return state


class NicheBiasedIMGEPConfig(IMGEPConfig):
    niche_curiosity_bias: bool = Field(False)
    niche_curiosity_k: int = Field(5, ge=1, le=50)
    niche_curiosity_start: float = Field(10.0, gt=0.0)
    niche_curiosity_decay: float = Field(1.0, ge=0.0)
    niche_curiosity_min: float = Field(1.0, gt=0.0)
    grid_fit_path: str = Field(DEFAULT_GRID_FIT_PATH)


@expose
class NicheBiasedIMGEPExplorer:
    config = NicheBiasedIMGEPConfig

    discovery_spec = ["params", "output", "raw_output", "rendered_outputs"]

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, system) -> "NicheBiasedIMGEPExplorerInstance":
        behavior_map = self.make_behavior_map(system)
        param_map = self.make_parameter_map(system)
        mutator = self.make_mutator()
        explorer = NicheBiasedIMGEPExplorerInstance(
            parameter_map=param_map,
            behavior_map=behavior_map,
            equil_time=self.config.equil_time,
            mutator=mutator,
            niche_curiosity_bias=self.config.niche_curiosity_bias,
            niche_curiosity_k=self.config.niche_curiosity_k,
            niche_curiosity_start=self.config.niche_curiosity_start,
            niche_curiosity_decay=self.config.niche_curiosity_decay,
            niche_curiosity_min=self.config.niche_curiosity_min,
            grid_fit_path=self.config.grid_fit_path,
        )
        return explorer

    def make_behavior_map(self, system: System):
        return instantiate_object(self.config.behavior_map, system, object_name="behavior map")

    def make_parameter_map(self, system: System):
        return instantiate_object(self.config.parameter_map, system, object_name="parameter map")

    def make_mutator(self):
        return instantiate_object(self.config.mutator, object_name="mutator")
