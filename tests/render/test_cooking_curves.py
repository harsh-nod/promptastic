"""Tests for promptastic.render.cooking_curves -- trajectory computation."""

import numpy as np

from promptastic.render.cooking_curves import compute_region_trajectories


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _make_region_map(regions):
    """Build a region_map dict from a list of (name, tok_start, tok_end)."""
    return {
        name: {"tok_start": s, "tok_end": e}
        for name, s, e in regions
    }


def _make_layer_weights(num_layers, total_tokens, fill=0.0):
    """Build layer_weights dict: {layer_idx: np.ndarray of shape (total_tokens,)}.

    All values default to *fill*.
    """
    return {
        li: np.full(total_tokens, fill, dtype=np.float64)
        for li in range(num_layers)
    }


# ---------------------------------------------------------------
# compute_region_trajectories
# ---------------------------------------------------------------


def test_compute_region_trajectories_basic():
    region_map = _make_region_map([("rules", 0, 4), ("examples", 4, 8)])
    num_layers = 3
    total_tokens = 8
    layer_weights = _make_layer_weights(num_layers, total_tokens)

    # Set rules tokens to specific values per layer
    for li in range(num_layers):
        layer_weights[li][0:4] = float(li) * 0.1

    trajectories = compute_region_trajectories(
        region_map, layer_weights, ["rules", "examples"]
    )

    assert "rules" in trajectories
    assert "examples" in trajectories
    assert len(trajectories["rules"]) == num_layers

    # Rules: layer 0 -> mean(0,0,0,0)=0, layer 1 -> mean(0.1)*4/4=0.1
    assert abs(trajectories["rules"][0] - 0.0) < 1e-9
    assert abs(trajectories["rules"][1] - 0.1) < 1e-9
    assert abs(trajectories["rules"][2] - 0.2) < 1e-9


def test_compute_region_trajectories_correct_means():
    """Verify that the mean is computed over the correct token range."""
    region_map = _make_region_map([("r", 2, 6)])
    layer_weights = {
        0: np.array([0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.0, 0.0]),
    }
    trajectories = compute_region_trajectories(region_map, layer_weights, ["r"])
    expected = np.mean([0.1, 0.2, 0.3, 0.4])
    assert abs(trajectories["r"][0] - expected) < 1e-9


def test_compute_region_trajectories_empty_regions():
    region_map = _make_region_map([("empty", 5, 5)])  # zero-length region
    layer_weights = {0: np.ones(10)}
    trajectories = compute_region_trajectories(
        region_map, layer_weights, ["empty"]
    )
    # Zero-length region (tok_end <= tok_start) should be skipped
    assert "empty" not in trajectories


def test_compute_region_trajectories_missing_region():
    region_map = _make_region_map([("rules", 0, 4)])
    layer_weights = {0: np.ones(8)}
    trajectories = compute_region_trajectories(
        region_map, layer_weights, ["nonexistent"]
    )
    assert "nonexistent" not in trajectories


def test_compute_region_trajectories_different_token_ranges():
    """Regions with different token spans should have independent trajectories."""
    region_map = _make_region_map([("small", 0, 2), ("large", 2, 10)])
    total_tokens = 10
    layer_weights = {
        0: np.array([1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    }
    trajectories = compute_region_trajectories(
        region_map, layer_weights, ["small", "large"]
    )
    # small: mean(1.0, 1.0) = 1.0
    assert abs(trajectories["small"][0] - 1.0) < 1e-9
    # large: mean(0.1 * 8) = 0.1
    assert abs(trajectories["large"][0] - 0.1) < 1e-9


def test_compute_region_trajectories_multiple_layers():
    region_map = _make_region_map([("r", 0, 4)])
    layer_weights = {
        0: np.array([0.1, 0.1, 0.1, 0.1]),
        1: np.array([0.5, 0.5, 0.5, 0.5]),
        2: np.array([0.3, 0.3, 0.3, 0.3]),
    }
    trajectories = compute_region_trajectories(region_map, layer_weights, ["r"])
    assert len(trajectories["r"]) == 3
    assert abs(trajectories["r"][0] - 0.1) < 1e-9
    assert abs(trajectories["r"][1] - 0.5) < 1e-9
    assert abs(trajectories["r"][2] - 0.3) < 1e-9


def test_compute_region_trajectories_layers_are_sorted():
    """Even if layer_weights keys are out of order, layers should be sorted."""
    region_map = _make_region_map([("r", 0, 2)])
    layer_weights = {
        2: np.array([0.3, 0.3]),
        0: np.array([0.1, 0.1]),
        1: np.array([0.2, 0.2]),
    }
    trajectories = compute_region_trajectories(region_map, layer_weights, ["r"])
    # Should be sorted: [0.1, 0.2, 0.3]
    assert abs(trajectories["r"][0] - 0.1) < 1e-9
    assert abs(trajectories["r"][1] - 0.2) < 1e-9
    assert abs(trajectories["r"][2] - 0.3) < 1e-9


def test_compute_region_trajectories_weights_shorter_than_region():
    """If weight array is shorter than region's tok_end, skip that layer."""
    region_map = _make_region_map([("r", 0, 10)])
    layer_weights = {
        0: np.ones(5),   # shorter than tok_end=10
        1: np.ones(10),  # exact fit
    }
    trajectories = compute_region_trajectories(region_map, layer_weights, ["r"])
    # Layer 0: e(10) > len(w)(5) -> should give 0
    assert trajectories["r"][0] == 0.0
    # Layer 1: fits -> should give 1.0
    assert abs(trajectories["r"][1] - 1.0) < 1e-9
