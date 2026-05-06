"""Tests for promptastic.optimize.score -- satisfaction and composite scoring."""

from math import nan

import pytest

from promptastic.optimize._types import MetricTarget
from promptastic.optimize.score import composite_score, metric_satisfaction


# ---------------------------------------------------------------
# metric_satisfaction
# ---------------------------------------------------------------


class TestMetricSatisfactionAbove:
    def test_at_ideal(self):
        t = MetricTarget(name="x", direction="above", minimum=0.0, ideal=1.0)
        assert metric_satisfaction(1.0, t) == 1.0

    def test_above_ideal(self):
        t = MetricTarget(name="x", direction="above", minimum=0.0, ideal=1.0)
        assert metric_satisfaction(2.0, t) == 1.0

    def test_at_minimum(self):
        t = MetricTarget(name="x", direction="above", minimum=0.0, ideal=1.0)
        assert metric_satisfaction(0.0, t) == 0.0

    def test_below_minimum(self):
        t = MetricTarget(name="x", direction="above", minimum=0.0, ideal=1.0)
        assert metric_satisfaction(-0.5, t) == 0.0

    def test_midpoint(self):
        t = MetricTarget(name="x", direction="above", minimum=0.0, ideal=1.0)
        assert abs(metric_satisfaction(0.5, t) - 0.5) < 1e-9

    def test_quarter(self):
        t = MetricTarget(name="x", direction="above", minimum=0.0, ideal=1.0)
        assert abs(metric_satisfaction(0.25, t) - 0.25) < 1e-9


class TestMetricSatisfactionBelow:
    def test_at_ideal(self):
        t = MetricTarget(name="x", direction="below", ideal=1.0, maximum=3.0)
        assert metric_satisfaction(1.0, t) == 1.0

    def test_below_ideal(self):
        t = MetricTarget(name="x", direction="below", ideal=1.0, maximum=3.0)
        assert metric_satisfaction(0.5, t) == 1.0

    def test_at_maximum(self):
        t = MetricTarget(name="x", direction="below", ideal=1.0, maximum=3.0)
        assert metric_satisfaction(3.0, t) == 0.0

    def test_above_maximum(self):
        t = MetricTarget(name="x", direction="below", ideal=1.0, maximum=3.0)
        assert metric_satisfaction(5.0, t) == 0.0

    def test_midpoint(self):
        t = MetricTarget(name="x", direction="below", ideal=1.0, maximum=3.0)
        assert abs(metric_satisfaction(2.0, t) - 0.5) < 1e-9


class TestMetricSatisfactionRange:
    def test_inside_range(self):
        t = MetricTarget(name="x", direction="range", minimum=0.0, ideal=0.3, maximum=0.7)
        assert metric_satisfaction(0.5, t) == 1.0

    def test_at_ideal_boundary(self):
        t = MetricTarget(name="x", direction="range", minimum=0.0, ideal=0.3, maximum=0.7)
        assert metric_satisfaction(0.3, t) == 1.0

    def test_at_maximum_boundary(self):
        t = MetricTarget(name="x", direction="range", minimum=0.0, ideal=0.3, maximum=0.7)
        assert metric_satisfaction(0.7, t) == 1.0

    def test_below_ideal(self):
        t = MetricTarget(name="x", direction="range", minimum=0.0, ideal=0.4, maximum=0.6)
        # Halfway between minimum and ideal
        assert abs(metric_satisfaction(0.2, t) - 0.5) < 1e-9

    def test_at_minimum(self):
        t = MetricTarget(name="x", direction="range", minimum=0.0, ideal=0.4, maximum=0.6)
        assert metric_satisfaction(0.0, t) == 0.0


class TestMetricSatisfactionEdgeCases:
    def test_nan_value(self):
        t = MetricTarget(name="x", direction="above", minimum=0.0, ideal=1.0)
        assert metric_satisfaction(nan, t) == 0.0

    def test_zero_span_above(self):
        t = MetricTarget(name="x", direction="above", minimum=0.5, ideal=0.5)
        assert metric_satisfaction(0.5, t) == 1.0
        assert metric_satisfaction(0.4, t) == 0.0

    def test_zero_span_below(self):
        t = MetricTarget(name="x", direction="below", ideal=1.0, maximum=1.0)
        assert metric_satisfaction(1.0, t) == 1.0
        assert metric_satisfaction(1.5, t) == 0.0


# ---------------------------------------------------------------
# composite_score
# ---------------------------------------------------------------


def test_composite_score_all_satisfied():
    targets = {
        "a": MetricTarget(name="a", direction="above", minimum=0.0, ideal=1.0, weight=1.0),
        "b": MetricTarget(name="b", direction="below", ideal=1.0, maximum=3.0, weight=1.0),
    }
    values = {"a": 1.5, "b": 0.5}
    score = composite_score(values, targets)
    assert score.total == 1.0
    assert score.num_satisfied == 2
    assert score.num_total == 2


def test_composite_score_none_satisfied():
    targets = {
        "a": MetricTarget(name="a", direction="above", minimum=0.0, ideal=1.0, weight=1.0),
        "b": MetricTarget(name="b", direction="below", ideal=1.0, maximum=3.0, weight=1.0),
    }
    values = {"a": 0.0, "b": 3.0}
    score = composite_score(values, targets)
    assert score.total == 0.0
    assert score.num_satisfied == 0


def test_composite_score_weighted():
    targets = {
        "a": MetricTarget(name="a", direction="above", minimum=0.0, ideal=1.0, weight=2.0),
        "b": MetricTarget(name="b", direction="above", minimum=0.0, ideal=1.0, weight=1.0),
    }
    # a satisfied (weight 2), b at 0 (weight 1)
    values = {"a": 1.0, "b": 0.0}
    score = composite_score(values, targets)
    expected = (1.0 * 2.0 + 0.0 * 1.0) / 3.0
    assert abs(score.total - expected) < 1e-9


def test_composite_score_missing_metrics_ignored():
    targets = {
        "a": MetricTarget(name="a", direction="above", minimum=0.0, ideal=1.0, weight=1.0),
        "b": MetricTarget(name="b", direction="above", minimum=0.0, ideal=1.0, weight=1.0),
    }
    values = {"a": 1.0}  # b not in values
    score = composite_score(values, targets)
    assert score.total == 1.0
    assert score.num_total == 1


def test_composite_score_custom_weights():
    targets = {
        "a": MetricTarget(name="a", direction="above", minimum=0.0, ideal=1.0, weight=1.0),
    }
    values = {"a": 0.5}
    score = composite_score(values, targets, weights={"a": 3.0})
    # Weight 3.0 overrides target's 1.0
    assert score.per_metric["a"].weight == 3.0


def test_composite_score_monotonic():
    """Improving a metric value should improve the score."""
    targets = {
        "a": MetricTarget(name="a", direction="above", minimum=0.0, ideal=1.0, weight=1.0),
    }
    score_low = composite_score({"a": 0.3}, targets)
    score_high = composite_score({"a": 0.7}, targets)
    assert score_high.total > score_low.total


def test_composite_score_empty():
    score = composite_score({}, {})
    assert score.total == 0.0
    assert score.num_total == 0
