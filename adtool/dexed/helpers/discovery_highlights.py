"""Visualization-side highlights for the IMGEP investigation (see VISUALIZATION.md):
surfaces the three diagnostics the notebook/report already rely on -- silence, the
silence_sink relocation marker, and bootstrap vs. goal-directed phase -- directly in the
2D discovery map, instead of only in offline notebook analysis.
"""
from typing import Any

import numpy as np

from adtool.user_tools.visu.highlights import (
    DiscoveryHighlightField,
    DiscoveryHighlightProvider,
    DiscoveryHighlightRule,
)

# Matches DexedStatistics._calc_static_statistics' silence threshold.
SILENCE_PEAK_THRESHOLD = 1e-4
# Matches the silence_sink offset (see DexedStatistics / coverage_curve_v3.py's drop_sink).
SILENCE_SINK_THRESHOLD = 1e4


class DexedDiscoveryHighlights(DiscoveryHighlightProvider):
    def fields(self) -> list[DiscoveryHighlightField]:
        return [
            DiscoveryHighlightField(
                field_id="peak_amplitude",
                label="Peak amplitude",
                value_type="number",
                min=0,
                max=1,
            ),
            DiscoveryHighlightField(
                field_id="is_silent",
                label="Quasi-silent (peak < 1e-4)",
                value_type="number",
                min=0,
                max=1,
            ),
            DiscoveryHighlightField(
                field_id="is_silence_sink",
                label="Silence-sink relocated point",
                value_type="number",
                min=0,
                max=1,
            ),
            DiscoveryHighlightField(
                field_id="is_bootstrap",
                label="Bootstrap (vs. goal-directed)",
                value_type="number",
                min=0,
                max=1,
            ),
        ]

    def rules(self) -> list[DiscoveryHighlightRule]:
        return [
            DiscoveryHighlightRule(
                rule_id="silent",
                label="Quasi-silent discoveries",
                field_id="is_silent",
                clauses=[{"lower": 1, "upper": 1}],
                enabled_by_default=True,
            ),
            DiscoveryHighlightRule(
                rule_id="silence_sink",
                label="Silence-sink relocated points",
                field_id="is_silence_sink",
                clauses=[{"lower": 1, "upper": 1}],
            ),
            DiscoveryHighlightRule(
                rule_id="bootstrap",
                label="Bootstrap-phase discoveries",
                field_id="is_bootstrap",
                clauses=[{"lower": 1, "upper": 1}],
            ),
        ]

    def compute_filters(self, discovery_payload: dict[str, Any]) -> dict[str, Any]:
        raw = np.asarray(discovery_payload.get("raw_output", []), dtype=float)
        peak = float(np.abs(raw).max()) if raw.size else 0.0

        output = np.asarray(discovery_payload.get("output", []), dtype=float)
        is_sink = bool(output.size) and bool((np.abs(output) > SILENCE_SINK_THRESHOLD).any())

        return {
            "peak_amplitude": min(peak, 1.0),
            "is_silent": 1.0 if (peak < SILENCE_PEAK_THRESHOLD or is_sink) else 0.0,
            "is_silence_sink": 1.0 if is_sink else 0.0,
            "is_bootstrap": 1.0 if int(discovery_payload.get("equil", 0)) == 1 else 0.0,
        }
