"""Shared utilities for coverage and diversity analysis over Dexed IMGEP discoveries (Z=38D).

Two things live here, both meant to be reused across the human/random/IMGEP comparison and
Study 2 later:

- Plausible (not theoretical) per-dimension bounds for the Z observation space, derived by
  combining several discovery sets and trimming to a percentile range rather than raw min/max
  -- most tt_* features have no clean theoretical bound, and raw min/max is fragile to the
  occasional degenerate discovery (e.g. the NaN/Inf -> 0.0 fallback in DexedStatistics).
- The Vendi Score (Friedman & Dieng, 2022): a reference-free diversity metric, VS = exp(H(p))
  where p are the eigenvalues of a normalized similarity matrix K/n (trace 1). Cosine similarity
  is used here, on z-scored features (raw Z has wildly different per-feature scales -- e.g. F0
  in Hz vs. ratios in [0,1] -- which would dominate a cosine similarity computed on raw values).
"""

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


def load_z_matrix(discoveries_dir) -> np.ndarray:
    """Load every discovery.json's "output" (Z, 38D) under discoveries_dir into an (N, 38) array."""
    files = sorted(Path(discoveries_dir).rglob("discovery.json"))
    if not files:
        raise ValueError(f"No discoveries found in {discoveries_dir}")
    rows = []
    for f in files:
        payload = json.load(f.open())
        rows.append(payload["output"])
    return np.asarray(rows, dtype=float)


def compute_plausible_bounds(
    z_datasets: Sequence[np.ndarray],
    names: Sequence[str],
    lower_pct: float = 0.5,
    upper_pct: float = 99.5,
) -> Dict[str, Tuple[float, float]]:
    """Per-dimension [lower_pct, upper_pct] percentile bounds, computed on the union of all
    given datasets (e.g. random + human + IMGEP together) -- this is the canonical Z
    observation-space definition to reuse for grid_coverage() below, and for Study 2 later so
    that different runs stay comparable against the same fixed boundaries."""
    combined = np.concatenate(z_datasets, axis=0)
    if combined.shape[1] != len(names):
        raise ValueError(f"{combined.shape[1]} dims vs {len(names)} names")
    n_nan = np.isnan(combined).sum()
    if n_nan:
        print(f"compute_plausible_bounds: {n_nan} NaN values found across the combined "
              f"dataset ({combined.shape[0]} rows) -- ignored via nanpercentile, not imputed.")
    lo = np.nanpercentile(combined, lower_pct, axis=0)
    hi = np.nanpercentile(combined, upper_pct, axis=0)
    return {name: (float(l), float(h)) for name, l, h in zip(names, lo, hi)}


def grid_coverage(
    Z: np.ndarray,
    bounds: Dict[str, Tuple[float, float]],
    names: Sequence[str],
    dimensions: Sequence[int] = None,
    bins_per_dim: int = 10,
) -> float:
    """Fraction of occupied grid cells within `bounds`, on the given subset of dimensions
    (a full 38D grid is intractable -- pick a handful of dimensions, e.g. via PCA loadings or
    domain knowledge, or call this repeatedly on 2D/3D slices)."""
    dimensions = list(range(len(names))) if dimensions is None else list(dimensions)
    sub = Z[:, dimensions]
    edges = [
        np.linspace(bounds[names[d]][0], bounds[names[d]][1], bins_per_dim + 1)
        for d in dimensions
    ]
    clipped = np.clip(sub, [e[0] for e in edges], [e[-1] for e in edges])
    idx = np.stack(
        [np.clip(np.digitize(clipped[:, i], edges[i][1:-1]), 0, bins_per_dim - 1)
         for i in range(len(dimensions))],
        axis=1,
    )
    occupied = len(set(map(tuple, idx)))
    total = bins_per_dim ** len(dimensions)
    return occupied / total


def goal_reaching_competence(
    Z: np.ndarray,
    goals: np.ndarray,
    bounds: Dict[str, Tuple[float, float]],
    names: Sequence[str],
) -> Dict[str, float]:
    """How well does a set of discoveries answer the request "give me a sound with behaviour
    g"? For each goal, reports the distance to the nearest reached behaviour. Lower is better.

    This is what the diversity measures do not capture: Vendi Score and grid_coverage both
    reward raw spread, whereas an IMGEP optimises for reaching the goals it sets itself. A set
    can be very spread out (random FM programming produces many far-apart degenerate sounds)
    while still leaving the relevant part of the goal space poorly covered, and vice versa.

    `goals` must be supplied in the same raw units as `Z`, and choosing them well is the whole
    difficulty. Do NOT sample them uniformly inside `bounds`: in 38 dimensions the expected
    distance between two uniform points in the unit box is sqrt(d/6) ~ 2.5, and the nearest of
    a few thousand archive points sits at essentially that same distance -- the measure
    saturates and every method scores alike (verified empirically: a 4x spread in Vendi Score
    collapsed to a 4% spread here). The reachable-sound manifold occupies a far lower-dimensional
    subset of the box, so uniform goals are mostly physically unrealisable. Use real behaviours
    instead, e.g. a held-out slice of the human preset corpus, which asks the well-posed
    question: can this exploration method reach musically relevant targets?

    Coordinates are min-max normalised over `bounds` and clipped to [0, 1], so each descriptor
    contributes equally regardless of physical unit. Clipping matters: `bounds` are percentile
    based, so a handful of points fall far outside and would otherwise dominate the mean
    (observed: mean 2151 against a p90 of 1.47). The median and p90 are reported as the robust
    summaries; divide by sqrt(len(names)) to read a distance as a fraction of the box diagonal.
    """
    lo = np.array([bounds[n][0] for n in names], dtype=float)
    hi = np.array([bounds[n][1] for n in names], dtype=float)
    span = np.where(hi > lo, hi - lo, 1.0)

    def normalise(M):
        M = np.asarray(M, dtype=float)
        if np.isnan(M).any():
            raise ValueError("goal_reaching_competence received NaN rows -- drop them first.")
        return np.clip((M - lo) / span, 0.0, 1.0)

    dists, _ = cKDTree(normalise(Z)).query(normalise(goals), k=1)
    return {
        "mean_distance_to_goal": float(dists.mean()),
        "median_distance_to_goal": float(np.median(dists)),
        "p90_distance_to_goal": float(np.percentile(dists, 90)),
    }


def vendi_score(X: np.ndarray, kernel: str = "cosine", rbf_sigma: float = None) -> float:
    """Reference-free diversity: VS = exp(-sum(p * log(p))), p = eigenvalues of K/n, K a
    similarity matrix with k(x, x) = 1. O(n^3) (eigendecomposition) -- subsample large datasets
    to a comparable n before calling this, both for tractability and for a fair comparison
    (Vendi Score, like most diversity measures, is sensitive to sample size)."""
    X = np.asarray(X, dtype=float)
    if np.isnan(X).any():
        raise ValueError(
            "vendi_score received NaN rows -- drop them first (X[~np.isnan(X).any(axis=1)]), "
            "do not impute: a handful of NaN rows is expected in the human dataset (~3/30145) "
            "and negligible to exclude, but silently imputing would distort the diversity measure."
        )
    n = X.shape[0]
    if kernel == "cosine":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Xn = X / norms
        K = Xn @ Xn.T
    elif kernel == "rbf":
        if rbf_sigma is None:
            raise ValueError("rbf_sigma is required for kernel='rbf'")
        sq_dists = np.sum(X**2, axis=1, keepdims=True) + np.sum(X**2, axis=1) - 2 * X @ X.T
        K = np.exp(-np.clip(sq_dists, 0, None) / (2 * rbf_sigma**2))
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    eigvals = np.linalg.eigvalsh(K / n)
    eigvals = eigvals[eigvals > 1e-12]
    eigvals = eigvals / eigvals.sum()  # numerical safety net; trace(K/n) is already 1
    entropy = -np.sum(eigvals * np.log(eigvals))
    return float(np.exp(entropy))
