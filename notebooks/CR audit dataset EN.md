# Regenerated Dexed dataset audit — report

Antonin Longeot — full dataset (30,145 presets), DawDreamer migration + TT/ACTM computation completed.

## 1. Context

Migration of audio rendering (RenderMan → DawDreamer) completed, full dataset regenerated: 120,580 wav +
spectrograms. Computation of timbre descriptors (TimbreToolbox + AudioCommons Timbral Models) completed
on the 30,145 presets, with automatic correlation-based selection exactly reproducing the SpinVAE2 paper's
method: **6 ACTM features + 32 TT features = 38 features** (`a ∈ R^38`), identical to the paper.

## 2. Coverage of the behaviour space (Z, 38 features)

PCA: 43.6% explained variance on 2 components — confirms that the space is intrinsically
multidimensional, not summarizable by 2 axes.

![PCA of Z, coloured by ac_brightness](pca_z_brightness.png)

The brightness gradient is very clean and continuous on the PCA — the main structure of the Z space
is well captured by the first two components.

UMAP (preserves local neighbourhood structure, unlike PCA):

![UMAP of Z, coloured by ac_brightness](umap_z_brightness.png)

Globally connected and richly textured structure, with a **clearly detached outgrowth**

(on the right of the figure) — a group of presets very different from the rest, investigated in
detail in section 5 below (spoiler: it turns out to be a computation artefact, not an acoustic
phenomenon).

Comparison of two descriptors (semantic `ac_brightness` vs. purely spectral `tt_SpecCent_IQR`) on
the same embedding:

![UMAP of Z, double colouring](umap_z_double_coloring.png)

Both features organize spatially in a very coherent way, including on the detached group —
which confirms that the observed structure is not an artefact of a single feature.

**Quantified coverage (50×50 grid on the UMAP embedding): 671/2500 cells occupied = 26.8%**,
mean nearest-neighbour distance = 0.0228.

**Quantified coverage directly in native 38D** (bypassing any 2D projection, which is not an
isometry): mean nearest-neighbour distance = 1.12 (5-NN = 1.80), versus 9.71 (5-NN = 10.21) for a
uniform random cloud of the same size and bounding box — a ratio of **0.116** (0.176 for 5-NN). The
human dataset therefore occupies the behaviour space in a **much more concentrated/clustered** way
than a uniform draw, quantitatively confirming what the UMAP already suggested visually, without
depending on any projection.

## 3. Comparison with the raw parameter space (Θ, 155 dimensions)

Same analysis performed directly on the 155 raw DX7 parameters, to compare the structure of the
parameter space to that of the behaviour space.

![PCA of Θ, coloured by ac_brightness](pca_theta.png)

Sharp contrast with the PCA of Z: the brightness gradient is here **completely mixed**, with no
visible spatial organization — two presets with close parameters can sound very different,
and vice versa.

![UMAP of Θ, coloured by ac_brightness](umap_theta.png)

Structure also very different from that of Z: a dense central core surrounded by many small
isolated clusters and scattered points, rather than a connected continuum.

**Θ coverage: 513/2500 cells = 20.5%**, mean nearest-neighbour distance = 0.0141.

| | Z (behaviour, 38D) | Θ (parameters, 155D) |
|---|---|---|
| Grid occupancy (50×50) | 26.8% | 20.5% |
| Mean nearest-neighbour distance | 0.0228 | 0.0141 |

The lower NN distance for Θ despite lower grid occupancy is probably explained by a bimodal
structure (very dense core + sparse periphery), whereas Z is more uniformly distributed. These
structural differences between Θ and Z concretely illustrate the non-linearity of the Θ→Z mapping
that the internship aims to exploit.

## 4. Categorising presets by instrument

To interpret the structures observed on the Z maps (sections 2, 5, 6), all 30,145 presets were
tagged with an instrument family, in three steps.

### 4.1 Keyword-based categorisation

Words are extracted from preset names (e.g. `BRITE.FULL` → `BRITE`, `FULL`), then matched against
27 categories (PIANO, BASS, BRASS, STRINGS, SYNTH, ORGAN, GUITAR, BELL, FLUTE, CLAV, VOICE, LEAD,
RHODES, HARP, PERC, SYNTH_VINTAGE, ENSEMBLE, PAD, SAX, DRUM, HIHAT, MARIMBA, TOM, SNARE, CLAP,
BLOCK, KICK) via keyword lists (with short-word disambiguation: `STR`, `TOM`, `BS`, `ENS`... only
match as whole tokens, to avoid false positives like BOMBS, BOTTOM, LAMBDA). This first pass covers
instruments well (PIANO 2106, BASS 1954, BRASS 1687, STRINGS 1590...) but leaves **48.1% of presets
uncategorised** (`OTHER`, 14,499/30,145) — overly creative names, unlisted abbreviations, or pure
sound-design.

### 4.2 Visualisation on the behaviour map (Z)

![UMAP of Z, coloured by category (keywords only)](umap_z_category.png)

Despite the high proportion of `OTHER` (light grey), the identified categories already organise
into coherent regions rather than a random mix — a sign that the categorisation, though
incomplete, captures a genuine timbral structure.

![PCA of Z, coloured by category](pca_z_category.png)

![Dominant category per UMAP grid cell](umap_z_category_grid.png)

The dominant-category-per-cell map (60×60 grid) confirms the same region-based organisation, with
fairly sharp boundaries between neighbouring families.

### 4.3 Resolving the `OTHER` bucket via k-NN

A **k-NN classifier (k=5, distance-weighted), trained directly in native 38D space** on the 15,646
already-categorised presets, predicts the category of the 14,499 `OTHER` presets, on the premise
that acoustically close presets in Z likely belong to the same instrument family. **Accuracy
validated at 64.2% via 5-fold cross-validation** (on already-labelled presets, so an honest
measure, not an optimistic figure). Final result: **27 categories, 0 remaining `OTHER` preset**.

![UMAP of Z, final categories (keywords + k-NN)](umap_z_category_knn_final.png)

| Category | n | | Category | n |
|---|---|---|---|---|
| PIANO | 3929 | | VOICE | 1084 |
| BASS | 3443 | | LEAD | 671 |
| STRINGS | 3426 | | PERC | 671 |
| BRASS | 3259 | | DRUM | 636 |
| SYNTH | 2069 | | RHODES | 630 |
| BELL | 1612 | | HARP | 587 |
| GUITAR | 1583 | | SYNTH_VINTAGE | 556 |
| ORGAN | 1490 | | ENSEMBLE | 504 |
| FLUTE | 1208 | | PAD | 459 |
| CLAV | 1110 | | SAX / HIHAT / MARIMBA / TOM / SNARE / CLAP / BLOCK / KICK | 279 → 35 |

This final categorisation (`point_zone_display` in the notebook) is then used as a reference to
quantify K-means sub-cluster purity (section 6) and to identify the composition of the percussive
peninsula.

## 5. Investigating the detached group: a computation artefact, not an acoustic phenomenon

The detached group spotted on the Z UMAP (section 2) was first isolated "by hand" using hard-coded
bounds on the embedding — fragile, since it depends on UMAP's `random_state`. It was re-identified
robustly and reproducibly with **HDBSCAN run directly in native 38D space** (not on the 2D
projection): **398 presets**, a compact cluster (internal radius 0.45 for a distance of 11.8 from
the global centre — a genuine separated cluster, not a sparse filament).

![Detached cluster isolated by HDBSCAN (native 38D)](hdbscan_detached_cluster.png)

**Root cause identified**: the `tt_F0_med` feature (median fundamental frequency) is **exactly 0.0
for all 398 presets, without a single exception** (checked on the raw value, before log+z-score
normalisation). This is not a shared acoustic property but a **fallback artefact of TimbreToolbox's
F0 extractor**, which silently fails on non-tonal/inharmonic sounds (representative names:
CYMPLOSION, CRASH CYMB, BOOM-GONG, SPACE NOS2/3, REFS WHISL, EXPLOSION — essentially percussive/noisy
sound-design presets, consistent with 72% of them falling into the `OTHER` keyword category). This
constant value creates an artificial "gap" in the 38-dimensional Z space, separating these presets
from everything else — **a methodological bias in the feature pipeline, not a genuine timbral
phenomenon worth exploring further**. Note for later: exclude or handle non-tonal presets separately
before any analysis that relies heavily on `tt_F0_med`.

## 6. Sub-structure of the main body (K-means, native 38D)

The "main body" of the UMAP (27,056 presets, what remains after removing the detached group) is not
homogeneous either. HDBSCAN fails to find fine structure there (72% noise) — expected, since HDBSCAN
looks for islands separated by empty space, not a way to partition a dense continuum. **K-means
(k=12, native 38D)** handles this case better (every point assigned, no leftover noise):

![K-means sub-clusters of the main body](kmeans_main_subclusters.png)

**Purity score per sub-cluster** (share of the majority category, via the keyword+k-NN
categorisation from section 4, cross-validated at 64.2% accuracy):

| Sub-cluster | n | Dominant category | Purity |
|---|---|---|---|
| 1 | 2035 | STRINGS | **56.6%** |
| 10 | 2140 | BASS | 51.3% |
| 7 | 3887 | PIANO | 46.3% |
| 0 | 326 | DRUM | 30.4% |
| 3 | 605 | DRUM | 28.6% |
| 9 | 1511 | FLUTE | 26.7% |
| 2 | 2190 | BRASS | 24.4% |
| 11 | 2166 | PIANO | 22.7% |
| 6 | 5355 | BRASS | 21.1% |
| 5 | 3834 | PIANO | 20.0% |
| 4 | 329 | DRUM | 14.0% |
| 8 | 2678 | GUITAR | 13.3% |

The STRINGS/PAD cluster stands out clearly as the most homogeneous of the whole partition — a 38D
timbral signature visibly more separable than piano/bass/brass, which overlap heavily with each
other (13-25% purity only).

**Non-tonal/percussive "peninsula" within the main body** (sub-clusters 0, 3, 4 combined, 1260
presets — not to be confused with the detached group in section 5):

![Percussive peninsula](peninsula_percussive.png)

Composition dominated by DRUM (318), BASS (120), PERC (116), TOM (89), MARIMBA (82)... Unlike the
detached group, this is **not** a `tt_F0_med` artefact — the most distinctive features here are
temporal/envelope-related (`tt_SpecDecr_IQR` +3.5σ, `tt_DecSlope` -3.4σ, `tt_RMSEnv_IQR`/`_med`
-3.2σ), consistent with a genuine signature of short, transient percussive sounds (fast attack,
sharp decay) rather than an extraction defect.

## 7. Replacing MATLAB with `ttb` (Python port of TimbreToolbox)

### 7.1 Motivation

TimbreToolbox (TT) previously depended on MATLAB, only available on `bonsho` (not `gottan`),
with ~20-40s startup per instance -- prohibitive for any iterative loop (IMGEP) and
cumbersome for the main pipeline (MATLAB zombie processes, hangs already encountered).
[`geoffroypeeters/ttb`](https://github.com/geoffroypeeters/ttb) is a Python reimplementation
of TT by Peeters himself -- a natural candidate to drop the MATLAB dependency.

### 7.2 Bugs found and fixed in `ttb`

The port contained 4 real bugs, fixed (see `~/projects/ttb_peeters/peeTimbreToolbox.py`):
1. **Crash** on high-F0 presets (harmonic bin indices out of bounds) -- fixed with a clip.
2. **`RMSEnv` incorrectly defined**: the port stored the raw Hilbert envelope instead of the
   actual windowed RMS (23.2ms window/2.9ms hop) that MATLAB computes -- fixed, `RMSEnv` went
   from r=0.75 to **r=1.000**.
3. **Per-file amplitude normalization** absent from MATLAB but present in the port -- removed.
   This was the root cause of most discrepancies on absolute-amplitude descriptors (`RMSEnv`,
   `HarmErg`).
4. **Wrong STFT frequency axis** (`sr_hz` never multiplied in + wrong FFT size, 512 instead of
   the real config's 1024) -- fixed. Brought `SpecCent_IQR` from r=0.79 to **r=0.997** on the
   validation sample.

### 7.3 Feature-by-feature validation (30,145 presets, full dataset)

Full regeneration of the 32 TT features via `ttb` (`utils/ttb_extractor.py` +
`utils/regenerate_tt_features_ttb.py`, ~82 min, 32 workers, 2 failures on totally silent
presets) and direct comparison against MATLAB values (see
`notebooks/dexed_compare_matlab_vs_ttb.ipynb`):

| Group | Features | Correlation |
|---|---|---|
| Near-perfect | `EffDur`, `RMSEnv_med/IQR`, `TempCent`, `Dec`, `Att`, `LAT`, `DecSlope` | r ≥ 0.995 |
| Good | `SpecCent_IQR`, `SpecVar_med/IQR`, `SpecRollOff_med/IQR`, `F0_med/IQR`, `AmpMod`, `SpecSpread_med/IQR`, `SpecFlat_med`, `SpecCrest_med` | 0.7–0.96 |
| Weak | `HarmErg_IQR`, `InHarm_med/IQR`, `SpecDecr_IQR`, `SpecKurt_med/IQR`, `OddEvenRatio_med/IQR` | 0.01–0.48 |

**11/32 features at r>0.9, 20/32 at r>0.7.**

![MATLAB vs ttb comparison: best (top) and worst (bottom) matches](feature_comparison_scatter.png)

Each point is a preset (30,145 total), X axis = MATLAB value, Y axis = `ttb` value. Top row:
near-perfect diagonal alignment (`EffDur`, `RMSEnv_med`, `TempCent`, `RMSEnv_IQR`); bottom row:
much more scattered clouds (`SpecDecr_IQR`, `SpecKurt_med`, `OddEvenRatio_med/IQR`) --
visually consistent with the weak r values.

The weak group corresponds to descriptors derived from harmonic partial tracking -- root
cause identified: the installed MATLAB toolbox (`VincentPerreault0/timbretoolbox`) uses a
**per-frame-variable** inharmonicity coefficient with a corrected objective function, whereas
`ttb` implements the **original 2011 JASA paper's** algorithm (constant coefficient,
uncorrected objective) -- a documented, structural algorithmic difference, not a portage bug.

### 7.4 Structural comparison (before/after)

Despite these feature-level discrepancies, global coverage metrics stay close:

| Metric | MATLAB | ttb |
|---|---|---|
| PCA explained variance (2 comp.) | 46.0% | 47.6% |
| UMAP grid occupancy (50×50) | 26.8% | 22.2% |
| Native 38D ratio NN / uniform-random | 0.116 | 0.107 |
| k-NN categorisation (CV accuracy) | 64.2% | 64.3% |
| HDBSCAN -- detached islands | 1 (n=398) | **2** (n=514 + n=460) |

**PCA of Z, coloured by `ac_brightness`** -- MATLAB on the left, `ttb` on the right:

<table><tr>
<td><img src="pca_z_brightness.png" width="420"/></td>
<td><img src="pca_z_brightness_ttb.png" width="420"/></td>
</tr></table>

Same brightness gradient, same overall structure -- PCA is not noticeably affected by the TT
toolbox change.

**UMAP of Z, coloured by `ac_brightness`** -- MATLAB on the left, `ttb` on the right:

<table><tr>
<td><img src="umap_z_brightness.png" width="420"/></td>
<td><img src="umap_z_brightness_ttb.png" width="420"/></td>
</tr></table>

Very similar overall structure in both cases -- same general organisation of the point cloud,
same dense/sparse regions.

**HDBSCAN, detached group(s)** -- MATLAB (1 island) on the left, `ttb` (2 islands) on the right:

<table><tr>
<td><img src="hdbscan_detached_cluster.png" width="420"/></td>
<td><img src="hdbscan_detached_cluster_ttb.png" width="420"/></td>
</tr></table>

This is the most visible structural difference: `ttb` splits into two what was a single group
with MATLAB -- two percussive sub-families (pure/noisy percussion -- snare, clap, cymbal --
vs semi-melodic percussion -- tom, marimba, xylophone) that MATLAB did not distinguish. The
same `tt_F0_med` artefact is confirmed on both islands (floor at 45.0 Hz instead of MATLAB's
0.0 -- different fallback value, same underlying phenomenon).

**K-means sub-clustering of the main body** -- MATLAB on the left, `ttb` on the right:

<table><tr>
<td><img src="kmeans_main_subclusters.png" width="420"/></td>
<td><img src="kmeans_main_subclusters_ttb.png" width="420"/></td>
</tr></table>

Broadly comparable partitioning (main blob split into per-instrument regions), but not
comparable sub-cluster by sub-cluster: K-means numbers clusters arbitrarily from one run to
the next, so "sub-cluster 3" on MATLAB has no relation to "sub-cluster 3" on `ttb` (see
section 6 and the `ttb` notebook). The percussive residue identified within the main body
(section 6 for MATLAB, `ttb` notebook section 14.4) is deliberately not illustrated here: on
MATLAB it was a genuine peninsula connected to the detached group (1260 presets), while on
`ttb` it is a much smaller isolated residue (902 presets) that no longer connects to the 2
islands -- the two are no longer comparable under the same "peninsula" label. Full detail in
`notebooks/dexed_audit_TT_ACTM_ttb.ipynb`.

### 7.5 Usage recommendation

`ttb` is reliable for any structural/global-coverage analysis (replaces MATLAB with no
notable loss). For fine-grained analysis specifically relying on `F0`/`InHarm`/`HarmErg`/
`OddEvenRatio`/`SpecKurt`, stay cautious or keep using the existing MATLAB values.

## 8. adtool (IMGEP) integration -- current state

The 3 adtool building blocks (`DexedParameterMap`, `DexedSimulation`, `DexedStatistics`) are
built, tested end-to-end, and **`DexedStatistics` now computes the full 38D Z space** (6 ACTM
+ 32 TT via `ttb`, not just ACTM) -- directly comparable to the human baseline above. Measured
cost: ~10-13s/evaluation in 38D (vs ~1.5s for ACTM alone) -- the IMGEP loop is no longer
"free", budget accordingly for larger experiments. A first 150-iteration experiment in 38D has
been run (`adtool/examples/dexed/run_150_38D/`). Exploration notebook:
`adtool/examples/dexed/dexed_adtool_overview.ipynb`.

## 9. Next steps

- Rigorously compare IMGEP exploration coverage to the human 38D baseline above (grid
  occupancy, mean NN distance, same methodology) -- now directly possible, same feature space
  on both sides.
- Address the `tt_F0_med` bias (section 5, also confirmed on `ttb` in section 7.4) before any
  further analysis relying on it: exclude or isolate non-tonal presets, or compute a more
  robust F0 for these cases.
- Listen to a sample of the percussive peninsula (section 6) and of the 2 `ttb` islands
  (section 7.4) to confirm/refine the temporal/envelope interpretation.
- Run a larger-scale IMGEP experiment (budget the ~10-13s/evaluation cost in 38D, cf. section
  8) to get a statistically robust coverage comparison.
