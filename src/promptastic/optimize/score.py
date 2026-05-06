"""Scoring functions for prompt optimization.

Maps raw metric values to satisfaction scores and computes a weighted
composite score across all metrics.
"""

from __future__ import annotations

from math import isnan

from ._types import MetricScore, MetricTarget, OptimizationScore


def metric_satisfaction(value: float, target: MetricTarget) -> float:
    """Map a raw metric value to a [0, 1] satisfaction score.

    - ``"above"``: 0 at *minimum*, 1 at *ideal* or above
    - ``"below"``: 1 at *ideal* or below, 0 at *maximum*
    - ``"range"``: 1 inside [ideal, maximum], tapering to 0 outside
    """
    if isnan(value):
        return 0.0

    if target.direction == "above":
        if value >= target.ideal:
            return 1.0
        if value <= target.minimum:
            return 0.0
        span = target.ideal - target.minimum
        if span <= 0:
            return 1.0 if value >= target.ideal else 0.0
        return (value - target.minimum) / span

    if target.direction == "below":
        if value <= target.ideal:
            return 1.0
        if value >= target.maximum:
            return 0.0
        span = target.maximum - target.ideal
        if span <= 0:
            return 1.0 if value <= target.ideal else 0.0
        return (target.maximum - value) / span

    if target.direction == "range":
        # Inside [ideal, maximum] -> 1.0
        # Below ideal, taper to 0 at minimum
        # Above maximum, taper to 0 at maximum + (maximum - ideal)
        if target.ideal <= value <= target.maximum:
            return 1.0
        if value < target.ideal:
            span = target.ideal - target.minimum
            if span <= 0:
                return 0.0
            return max(0.0, (value - target.minimum) / span)
        # value > target.maximum
        upper_span = target.maximum - target.ideal
        if upper_span <= 0:
            return 0.0
        return max(0.0, 1.0 - (value - target.maximum) / upper_span)

    return 0.0


def composite_score(
    metric_values: dict[str, float],
    targets: dict[str, MetricTarget],
    weights: dict[str, float] | None = None,
) -> OptimizationScore:
    """Compute a weighted composite score from metric values and targets.

    Only metrics that appear in *both* ``metric_values`` and ``targets``
    are scored.  Weights default to each target's own ``weight`` field.
    """
    per_metric: dict[str, MetricScore] = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for name, target in targets.items():
        if name not in metric_values:
            continue
        value = metric_values[name]
        sat = metric_satisfaction(value, target)
        w = (weights or {}).get(name, target.weight)
        per_metric[name] = MetricScore(
            value=value, satisfaction=sat, weight=w, target=target,
        )
        weighted_sum += sat * w
        total_weight += w

    total = weighted_sum / total_weight if total_weight > 0 else 0.0
    num_satisfied = sum(1 for m in per_metric.values() if m.satisfaction >= 0.9)

    return OptimizationScore(
        total=total,
        per_metric=per_metric,
        num_satisfied=num_satisfied,
        num_total=len(per_metric),
    )
