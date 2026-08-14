"""Precomputes, for each stage-2 SPINVAE-2 variant, a bank of latent encodings of the
validation split, plus their per-dimension mean/std. DexedLatentMap consumes these banks:
sample() needs an empirical latent distribution (the no-KL variants are NOT matched to
N(0,I), so sampling the prior would be out-of-distribution for them), and mutate() scales
its noise by the per-dimension std so that one sigma value is comparable across variants.

Usage (from adtool/dexed/):
    python scripts/compute_latent_banks.py                  # CPU, all 4 variants
    python scripts/compute_latent_banks.py --device cuda    # tries a soft GPU lock first
    python scripts/compute_latent_banks.py --variants avec_tout sans_KL

Output: configs/latent/<variant>_zbank.npz  with keys z_mu (N,256), mean (256,), std (256,),
        uids (N,) -- the preset UIDs, so bank vectors can be traced back to human presets.
"""
import argparse
import os
import pathlib
import pickle
import sys

import numpy as np

SPINVAE2_ROOT = os.path.expanduser("~/projects/spinvae2")
if SPINVAE2_ROOT not in sys.path:
    sys.path.insert(0, SPINVAE2_ROOT)

LOGS_ROOT = pathlib.Path("/data/anasynth_nonbp/longeot/spinvae2_logs/stage2_ablation")
OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "configs" / "latent"
ALL_VARIANTS = ["avec_tout", "sans_timbre", "sans_KL", "sans_les_deux"]


def get_device(requested: str):
    import torch
    if requested == "cuda":
        try:
            import manage_gpus as gpl
            gpu_id = gpl.get_gpu_lock(gpu_device_id=-1, soft=True)
            # The cluster's sitecustomize.py wipes CUDA_VISIBLE_DEVICES unless PYTHON_GPU_LOCK=1
            # was set before interpreter startup, so masking must happen through torch directly.
            torch.cuda.set_device(gpu_id if torch.cuda.device_count() > 1 else 0)
            print(f"soft-locked GPU {gpu_id}")
            return f"cuda:{gpu_id if torch.cuda.device_count() > 1 else 0}"
        except Exception as e:
            print(f"no GPU available ({e}), falling back to CPU")
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=ALL_VARIANTS)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    import torch
    import data.build
    from model.hierarchicalvae import HierarchicalVAE

    device = get_device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Dataset + dataloader are identical across variants (same splits/config): build once
    # from the first variant's stored config.
    with open(LOGS_ROOT / args.variants[0] / "config.pickle", "rb") as f:
        cfgs = pickle.load(f)
    model_config, train_config = cfgs["model"], cfgs["train"]
    train_config.verbosity = 0
    dataset = data.build.get_dataset(model_config, train_config)
    dataloaders, _ = data.build.get_split_dataloaders(train_config, dataset, num_workers=2)
    val_loader = dataloaders["validation"]

    for variant in args.variants:
        model_dir = LOGS_ROOT / variant
        print(f"=== {variant} ===")
        with open(model_dir / "config.pickle", "rb") as f:
            cfgs = pickle.load(f)
        m = HierarchicalVAE(cfgs["model"], cfgs["train"], dataset.preset_indexes_helper)
        m.load_checkpoints(model_dir / "checkpoint.tar", map_location=torch.device(device))
        m.to(device)
        m.eval()

        z_mu_all, uids_all = [], []
        with torch.no_grad():
            for x_in, v_in, uid, notes, labels, attributes in val_loader:
                out = m.parse_outputs(m(x_in.to(device), v_in.to(device), uid, notes, pass_index=0))
                z_mu_all.append(out.z_mu.cpu().numpy())
                uids_all.append(uid.numpy())
        z_mu = np.concatenate(z_mu_all, axis=0)
        uids = np.concatenate(uids_all, axis=0)
        std = z_mu.std(axis=0)
        std[std == 0] = 1.0

        out_path = OUT_DIR / f"{variant}_zbank.npz"
        np.savez(out_path, z_mu=z_mu, mean=z_mu.mean(axis=0), std=std, uids=uids)
        print(f"saved {out_path}  z_mu={z_mu.shape}  mean|std over dims: "
              f"std range [{std.min():.3f}, {std.max():.3f}]")
        del m
        if device != "cpu":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
