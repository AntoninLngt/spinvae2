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

(on the right of the figure) — a group of presets very different from the rest, to be investigated
(listening to a few examples is planned).

Comparison of two descriptors (semantic `ac_brightness` vs. purely spectral `tt_SpecCent_IQR`) on
the same embedding:

![UMAP of Z, double colouring](umap_z_double_coloring.png)

Both features organize spatially in a very coherent way, including on the detached group —
which confirms that the observed structure is not an artefact of a single feature.

**Quantified coverage (50×50 grid on the UMAP embedding): 671/2500 cells occupied = 26.8%**,
mean nearest-neighbour distance = 0.0228.

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

## 4. Next steps

- Investigate the detached group of presets on the Z UMAP map (listening, UID identification).
- Quantify coverage directly in the 38-dimensional space (not only on the UMAP embedding,
  which is not an isometry).
- First adtool integration tests (IMGEP exploration) on raw Dexed parameters, under discussion.
