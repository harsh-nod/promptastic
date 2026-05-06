"""Metric target definitions and loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._types import MetricTarget


def _target(
    name: str,
    direction: str,
    *,
    minimum: float = 0.0,
    ideal: float = 0.0,
    maximum: float = float("inf"),
    weight: float = 1.0,
    category: str = "",
) -> MetricTarget:
    return MetricTarget(
        name=name,
        direction=direction,
        minimum=minimum,
        ideal=ideal,
        maximum=maximum,
        weight=weight,
        category=category,
    )


# ---------------------------------------------------------------------------
# Default metric targets
# ---------------------------------------------------------------------------

# Category A: Attention Routing
ATTENTION_TARGETS: list[MetricTarget] = [
    _target(
        "context_bleed_ratio", "below",
        ideal=1.0, maximum=3.0, weight=1.2, category="attention",
    ),
    _target(
        "density_cv", "below",
        ideal=0.5, maximum=2.0, weight=1.0, category="attention",
    ),
]

# Category B: Processing Dynamics
DYNAMICS_TARGETS: list[MetricTarget] = [
    # These are region-specific -- applied dynamically per region
]

# Category C: Causal Importance
CAUSAL_TARGETS: list[MetricTarget] = [
    # Applied dynamically per region when patching data is available
]

# Category D: Head Specialization
HEAD_TARGETS: list[MetricTarget] = [
    # Applied dynamically per region when per-head data is available
]


def default_targets() -> list[MetricTarget]:
    """Return the full set of default metric targets.

    Region-specific targets (retention, peak layer, causal importance) are
    generated dynamically at extraction time based on discovered regions.
    These are the non-region-specific defaults.
    """
    return list(ATTENTION_TARGETS)


def make_region_targets(
    region: str,
    *,
    is_rules: bool = False,
    is_examples: bool = False,
    has_patching: bool = False,
    has_per_head: bool = False,
) -> list[MetricTarget]:
    """Generate region-specific metric targets.

    Different region types get different default thresholds:
    - Rules/directive regions: should be absorbed early, high retention
    - Example regions: moderate absorption, lower retention acceptable
    - Other regions: basic attention targets only
    """
    targets: list[MetricTarget] = []

    # Terminal attention (all regions should have some)
    min_attn = 0.05 if is_rules else 0.02
    targets.append(_target(
        f"terminal_attention_{region}", "above",
        minimum=0.0, ideal=min_attn, weight=1.2, category="attention",
    ))

    # Retention ratio
    if is_rules:
        targets.append(_target(
            f"retention_ratio_{region}", "above",
            minimum=0.0, ideal=0.3, weight=1.0, category="dynamics",
        ))
        # Rules should peak early (first 20% of layers)
        targets.append(_target(
            f"peak_layer_frac_{region}", "below",
            ideal=0.10, maximum=0.30, weight=1.0, category="dynamics",
        ))
    elif is_examples:
        targets.append(_target(
            f"retention_ratio_{region}", "above",
            minimum=0.0, ideal=0.1, weight=0.8, category="dynamics",
        ))
        targets.append(_target(
            f"peak_layer_frac_{region}", "below",
            ideal=0.15, maximum=0.40, weight=0.8, category="dynamics",
        ))

    # Causal importance (when patching data available)
    if has_patching:
        ideal_kl = 0.1 if is_rules else 0.05
        targets.append(_target(
            f"causal_importance_{region}", "above",
            minimum=0.0, ideal=ideal_kl, weight=1.5, category="causal",
        ))

    # Head specialization (when per-head data available)
    if has_per_head:
        targets.append(_target(
            f"head_variance_{region}", "above",
            minimum=0.0, ideal=0.001, weight=0.8, category="head",
        ))

    return targets


def build_targets_for_regions(
    regions: list[str],
    *,
    rules_regions: set[str] | None = None,
    example_regions: set[str] | None = None,
    has_patching: bool = False,
    has_per_head: bool = False,
) -> list[MetricTarget]:
    """Build a complete target list for a set of discovered regions.

    Combines default non-region targets with per-region targets.
    """
    if rules_regions is None:
        rules_regions = {"rules", "directive", "entity_rules", "passage_rules"}
    if example_regions is None:
        example_regions = {"examples", "few_shot", "demonstrations"}

    targets = default_targets()

    for region in regions:
        is_rules = region in rules_regions
        is_examples = region in example_regions
        targets.extend(make_region_targets(
            region,
            is_rules=is_rules,
            is_examples=is_examples,
            has_patching=has_patching,
            has_per_head=has_per_head,
        ))

    return targets


def load_targets_from_file(path: str) -> list[MetricTarget]:
    """Load custom metric targets from a JSON file.

    Expected format: list of dicts with keys matching MetricTarget fields.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    targets: list[MetricTarget] = []
    for entry in data:
        targets.append(MetricTarget(
            name=entry["name"],
            direction=entry["direction"],
            minimum=entry.get("minimum", 0.0),
            ideal=entry.get("ideal", 0.0),
            maximum=entry.get("maximum", float("inf")),
            weight=entry.get("weight", 1.0),
            category=entry.get("category", ""),
        ))
    return targets


def targets_to_dict(targets: list[MetricTarget]) -> dict[str, MetricTarget]:
    """Convert a target list to a name-keyed dict for lookup."""
    return {t.name: t for t in targets}
