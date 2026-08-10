"""Identity projection for adtool's offline analysis modules (Comparison1D/2D, SpaceCoverage).

DexedStatistics already saves the full 38D canonical Z (6 ACTM + 32 TT, raw/unnormalized) as
each discovery's "output" -- adtool's DiscoverySet loader stacks that into `dataset.outputs`
for every discovery set (see adtool.user_tools.analysis_metrics.shared.discovery), so no actual
dimensionality reduction is needed here: this projection is just a passthrough that also
attaches the feature names for axis labels.

Kept in this file (not in maps/DexedStatistics.py) because analysis-config "projection" entries
must be free functions with signature (datasets, config) -> (*arrays, labels), not methods.
"""

from maps.DexedStatistics import Z_FEATURE_NAMES


def identity_z_projection(datasets, config):
    values = [dataset.outputs for dataset in datasets]
    return (*values, Z_FEATURE_NAMES)
