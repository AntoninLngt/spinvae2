"""Theta = SPINVAE-2's 256-D latent vector, for latent-space IMGEP (study 2).

The genome carried through the exploration loop is z; every sample()/mutate() call decodes
it into a raw 155-parameter Dexed preset through the trained preset decoder, so the existing
DexedSimulation (rendering) and DexedStatistics (38-D behaviour space) are reused unchanged.
The output dict carries both: {"dynamic_params": {"z": [...], "preset": [(i, v), ...]}}.

Design choices, one per study-2 hypothesis:
- sample() draws from the EMPIRICAL latent distribution (precomputed bank of validation-set
  encodings, scripts/compute_latent_banks.py), not from N(0, I): the no-KL ablation variants
  are not matched to the prior, so prior sampling would be out-of-distribution for them and
  the 4-variant comparison would be confounded from the start.
- mutate() adds Gaussian noise scaled per-dimension by the bank's std, so one sigma value
  means "the same fraction of the space's spread" across all 4 variants regardless of how
  differently their latent scales are calibrated.
"""
import os
import pickle
import sys
from copy import deepcopy
from typing import Dict

import numpy as np

from adtool.utils.leaf.Leaf import Leaf

SPINVAE2_ROOT = os.path.expanduser("~/projects/spinvae2")
if SPINVAE2_ROOT not in sys.path:
    sys.path.insert(0, SPINVAE2_ROOT)

# Stage-2 checkpoints live on the NFS-shared home, NOT under /data or /data2: those are
# LOCAL disks on each host (bonsho vs gottan), so a training run done on the GPU machine is
# invisible from the exploration machine. `checkpoints_shared/` is inside the repo on
# /net/home, which every host mounts -- keep it that way when retraining.
LOGS_ROOT = os.path.expanduser("~/projects/spinvae2/checkpoints_shared/stage2_ablation")
ZBANK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "configs", "latent")


class DexedLatentMap(Leaf):

    DIM_Z = 256

    def __init__(
        self,
        system=None,
        premap_key: str = "params",
        seed: int = 0,
        variant: str = "avec_tout",
        sigma: float = 0.1,
        sample_mode: str = "gaussian",
        sample_bank_noise: float = 0.05,
        **config_decorator_kwargs,
    ):
        super().__init__()
        self.premap_key = premap_key
        self.seed = seed
        # One of the 4 stage-2 runs: avec_tout / sans_timbre / sans_KL / sans_les_deux.
        # Selects both the decoder checkpoint and the matching latent bank.
        self.variant = variant
        # Mutation step, as a fraction of the empirical per-dimension std (see module docstring).
        self.sigma = sigma
        # "gaussian": z ~ mean + std * eps (empirical Gaussian fit of the bank) -- the latent
        #   analogue of study 1's uniform random draws;
        # "bank": a random bank vector (the encoding of a real human preset) + small noise --
        #   a human-anchored bootstrap, closer to "start from plausible sounds".
        if sample_mode not in ("gaussian", "bank"):
            raise ValueError(f"sample_mode must be 'gaussian' or 'bank', got {sample_mode!r}")
        self.sample_mode = sample_mode
        self.sample_bank_noise = sample_bank_noise
        self._rng_counter = 0
        # Heavy/native members are built lazily: adtool restores checkpoints via cls.__new__ +
        # __dict__.update, bypassing __init__, and a torch model doesn't survive that pickle
        # round-trip -- same constraint as DexedParameterMap._dexed_helper.
        self._model = None
        self._dataset = None
        self._zbank = None

    # ------------------------------------------------------------------ lazy resources
    def _load_zbank(self):
        if self._zbank is None:
            path = os.path.join(ZBANK_DIR, f"{self.variant}_zbank.npz")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{path} not found -- run scripts/compute_latent_banks.py first")
            self._zbank = np.load(path)
        return self._zbank

    def _load_model(self):
        if self._model is None:
            import torch
            import data.build
            from model.hierarchicalvae import HierarchicalVAE
            model_dir = os.path.join(LOGS_ROOT, self.variant)
            with open(os.path.join(model_dir, "config.pickle"), "rb") as f:
                cfgs = pickle.load(f)
            model_config, train_config = cfgs["model"], cfgs["train"]
            train_config.verbosity = 0
            # The pickled config freezes the paths of the TRAINING host (gottan: /data/...),
            # but /data and /data2 are local disks, so those paths do not resolve on the
            # exploration host (bonsho: /data2/...). Re-point them at whatever the current
            # host declares, otherwise dataset loading dies on a missing constraints file.
            import utils.config_confidential as _cc
            for _cfg in (model_config, train_config):
                if getattr(_cfg, "data_root_path", None):
                    _cfg.data_root_path = _cc.data_root_path
                if getattr(_cfg, "logs_root_dir", None):
                    _cfg.logs_root_dir = _cc.logs_root_dir
            # The dataset instance is needed by Preset2d for learnable-tensor -> raw-VST
            # conversion (indexes helper + cardinalities); built once, CPU-only.
            self._dataset = data.build.get_dataset(model_config, train_config)
            m = HierarchicalVAE(model_config, train_config, self._dataset.preset_indexes_helper)
            m.load_checkpoints(os.path.join(model_dir, "checkpoint.tar"),
                               map_location=torch.device("cpu"))
            m.eval()
            self._model = m
        return self._model

    def checkpoint_state(self) -> Dict:
        # Same defensive approach as DexedParameterMap: drop whatever fails to pickle.
        state = {}
        for key, value in self.__dict__.items():
            try:
                pickle.dumps(value)
            except Exception:
                value = None
            state[key] = value
        return state

    # ------------------------------------------------------------------ decode
    def _decode(self, z: np.ndarray):
        """256-D latent vector -> Dexed preset as a list of (idx, value in [0,1]) tuples."""
        import torch
        from data.preset2d import Preset2d
        m = self._load_model()
        with torch.no_grad():
            z_t = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
            decoder_out = m.decoder.preset_decoder(z_t, u_target=None)
            v_out = decoder_out[0]
        preset2d = Preset2d(self._dataset, learnable_tensor_preset=v_out[0])
        raw = preset2d.to_raw()  # full 155 raw VST values, fixed params at dataset defaults
        return [(i, float(v)) for i, v in enumerate(raw)]

    def _pack(self, z: np.ndarray) -> Dict:
        return {"dynamic_params": {"z": [float(v) for v in z], "preset": self._decode(z)}}

    # ------------------------------------------------------------------ adtool interface
    def map(self, input: Dict, override_existing: bool = True) -> Dict:
        intermed_dict = deepcopy(input)
        if (override_existing and self.premap_key in intermed_dict) or (
            self.premap_key not in intermed_dict
        ):
            intermed_dict[self.premap_key] = self.sample()
        return intermed_dict

    def sample(self) -> Dict:
        self._rng_counter += 1
        rng = np.random.default_rng(self.seed + self._rng_counter)
        bank = self._load_zbank()
        if self.sample_mode == "bank":
            z = bank["z_mu"][rng.integers(len(bank["z_mu"]))].astype(float)
            z = z + self.sample_bank_noise * bank["std"] * rng.standard_normal(self.DIM_Z)
        else:  # "gaussian"
            z = bank["mean"] + bank["std"] * rng.standard_normal(self.DIM_Z)
        return self._pack(z)

    def mutate(self, parameter_dict: Dict) -> Dict:
        self._rng_counter += 1
        rng = np.random.default_rng(self.seed + 999999999 + self._rng_counter)
        intermed_dict = deepcopy(parameter_dict)
        z = np.asarray(intermed_dict["dynamic_params"]["z"], dtype=float)
        bank = self._load_zbank()
        z = z + self.sigma * bank["std"] * rng.standard_normal(self.DIM_Z)
        intermed_dict["dynamic_params"] = self._pack(z)["dynamic_params"]
        return intermed_dict
