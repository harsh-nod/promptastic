"""Tests for promptastic.analysis.head_analysis -- per-head analysis."""

from math import log2

import numpy as np

from promptastic.analysis.head_analysis import (
    find_specialist_heads,
    head_specialization,
    head_variance_by_region,
)


# ---------------------------------------------------------------
# head_specialization
# ---------------------------------------------------------------


def test_head_specialization_finds_dominant_region():
    """When one region has higher attention, it should be the dominant one."""
    num_layers = 2
    num_heads = 2
    region_names = ["rules", "examples"]

    per_head_data = {
        "rules": np.array([[0.9, 0.1], [0.3, 0.8]]),      # (layers, heads)
        "examples": np.array([[0.1, 0.9], [0.7, 0.2]]),
    }

    result = head_specialization(per_head_data, region_names, num_layers, num_heads)
    layers = result["per_layer"]
    assert len(layers) == 2

    # Layer 0, Head 0: rules=0.9 > examples=0.1 -> dominant=rules
    assert layers[0]["heads"][0]["dominant_region"] == "rules"
    # Layer 0, Head 1: examples=0.9 > rules=0.1 -> dominant=examples
    assert layers[0]["heads"][1]["dominant_region"] == "examples"


def test_head_specialization_entropy():
    """A head attending equally to two regions should have entropy=1.0 (log2(2))."""
    num_layers = 1
    num_heads = 1
    region_names = ["a", "b"]

    per_head_data = {
        "a": np.array([[0.5]]),
        "b": np.array([[0.5]]),
    }

    result = head_specialization(per_head_data, region_names, num_layers, num_heads)
    head = result["per_layer"][0]["heads"][0]
    assert abs(head["entropy"] - 1.0) < 1e-9  # log2(2) = 1.0


def test_head_specialization_zero_total():
    """When all weights are zero, dominant should be 'none'."""
    per_head_data = {
        "a": np.array([[0.0]]),
        "b": np.array([[0.0]]),
    }
    result = head_specialization(per_head_data, ["a", "b"], 1, 1)
    head = result["per_layer"][0]["heads"][0]
    assert head["dominant_region"] == "none"
    assert head["entropy"] == 0.0


def test_head_specialization_single_region_dominant():
    """A head with all weight on one region should have entropy=0."""
    per_head_data = {
        "a": np.array([[1.0]]),
        "b": np.array([[0.0]]),
    }
    result = head_specialization(per_head_data, ["a", "b"], 1, 1)
    head = result["per_layer"][0]["heads"][0]
    assert head["dominant_region"] == "a"
    assert abs(head["entropy"] - 0.0) < 1e-9


# ---------------------------------------------------------------
# head_variance_by_region
# ---------------------------------------------------------------


def test_head_variance_by_region_basic():
    num_layers = 2
    num_heads = 4
    per_head_data = {
        "rules": np.array([
            [0.1, 0.2, 0.3, 0.4],  # layer 0
            [0.5, 0.5, 0.5, 0.5],  # layer 1
        ]),
    }
    result = head_variance_by_region(per_head_data, ["rules"], num_layers, num_heads)
    assert "rules" in result
    info = result["rules"]
    # Layer 0 has variance, layer 1 has zero variance
    assert info["per_layer_variance"][1] == 0.0
    assert info["per_layer_variance"][0] > 0.0
    assert info["max_variance_layer"] == 0


def test_head_variance_missing_region():
    per_head_data = {}
    result = head_variance_by_region(per_head_data, ["missing"], 2, 4)
    assert result["missing"]["mean_variance"] == 0.0


def test_head_variance_uniform_heads():
    """All heads identical -> zero variance."""
    arr = np.ones((3, 4)) * 0.25
    per_head_data = {"r": arr}
    result = head_variance_by_region(per_head_data, ["r"], 3, 4)
    assert abs(result["r"]["mean_variance"]) < 1e-12


# ---------------------------------------------------------------
# find_specialist_heads
# ---------------------------------------------------------------


def test_find_specialist_heads_with_threshold():
    """A head concentrated on one region has entropy 0 and should be found."""
    per_head_data = {
        "rules": np.array([[1.0, 0.0]]),     # layer 0: head 0 all on rules
        "examples": np.array([[0.0, 1.0]]),   # layer 0: head 1 all on examples
    }
    specialists = find_specialist_heads(
        per_head_data, ["rules", "examples"], 1, 2, entropy_threshold=1.0
    )
    assert len(specialists) == 2
    # Both should have entropy 0
    for s in specialists:
        assert s["entropy"] == 0.0


def test_find_specialist_heads_excludes_high_entropy():
    """Uniformly distributed heads should NOT be found as specialists."""
    per_head_data = {
        "a": np.array([[0.5]]),
        "b": np.array([[0.5]]),
    }
    # entropy=1.0, threshold=0.5 -> should be excluded
    specialists = find_specialist_heads(
        per_head_data, ["a", "b"], 1, 1, entropy_threshold=0.5
    )
    assert len(specialists) == 0


def test_find_specialist_heads_sorted_by_entropy():
    """Results should be sorted by entropy ascending."""
    per_head_data = {
        "a": np.array([[1.0, 0.7]]),
        "b": np.array([[0.0, 0.3]]),
    }
    specialists = find_specialist_heads(
        per_head_data, ["a", "b"], 1, 2, entropy_threshold=2.0
    )
    if len(specialists) >= 2:
        assert specialists[0]["entropy"] <= specialists[1]["entropy"]


def test_find_specialist_heads_empty_when_all_zero():
    per_head_data = {
        "a": np.array([[0.0]]),
        "b": np.array([[0.0]]),
    }
    specialists = find_specialist_heads(
        per_head_data, ["a", "b"], 1, 1, entropy_threshold=2.0
    )
    assert len(specialists) == 0
