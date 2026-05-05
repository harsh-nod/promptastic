"""Tests for promptastic.analysis.metrics -- core metric computations."""

from math import isnan, nan

import numpy as np
import pytest

from promptastic.analysis.metrics import (
    avg_final_layers,
    compute_per_token_density,
    compute_region_ratio,
    cooking_curve_stats,
    phase_mean,
    safe_mean,
    safe_median,
    safe_std,
)


# ---------------------------------------------------------------
# Helpers -- build minimal attention data dicts
# ---------------------------------------------------------------


def _make_attention_data(region_values_per_layer):
    """Build an attention data dict from {region: [val_L0, val_L1, ...]}."""
    all_regions = list(region_values_per_layer.keys())
    num_layers = len(next(iter(region_values_per_layer.values())))
    per_layer = []
    for li in range(num_layers):
        region_means = {}
        for r in all_regions:
            region_means[r] = region_values_per_layer[r][li]
        per_layer.append({"layer": li, "per_region_mean": region_means})

    return {"terminal": {"per_layer": per_layer}}


def _make_sample(region_values_per_layer):
    """Wrap attention data in a sample-like dict."""
    return {"attention": _make_attention_data(region_values_per_layer)}


# ---------------------------------------------------------------
# avg_final_layers
# ---------------------------------------------------------------


def test_avg_final_layers_basic():
    data = _make_attention_data({
        "rules": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    })
    # FINAL_LAYERS=4, so average of [0.5, 0.6, 0.7, 0.8]
    result = avg_final_layers(data, "terminal", "rules", n_layers=4)
    expected = (0.5 + 0.6 + 0.7 + 0.8) / 4
    assert abs(result - expected) < 1e-9


def test_avg_final_layers_custom_n():
    data = _make_attention_data({
        "rules": [0.1, 0.2, 0.3, 0.4],
    })
    # Last 2 layers: [0.3, 0.4]
    result = avg_final_layers(data, "terminal", "rules", n_layers=2)
    expected = (0.3 + 0.4) / 2
    assert abs(result - expected) < 1e-9


def test_avg_final_layers_missing_position():
    data = _make_attention_data({"rules": [0.1]})
    result = avg_final_layers(data, "nonexistent_pos", "rules")
    assert isnan(result)


def test_avg_final_layers_missing_region():
    data = _make_attention_data({"rules": [0.1, 0.2, 0.3, 0.4]})
    result = avg_final_layers(data, "terminal", "nonexistent_region")
    assert isnan(result)


# ---------------------------------------------------------------
# compute_region_ratio
# ---------------------------------------------------------------


def test_compute_region_ratio_basic():
    samples = [
        _make_sample({"a": [0.0, 0.0, 0.0, 0.0, 0.4, 0.4, 0.4, 0.4],
                       "b": [0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2]}),
    ]
    ratios = compute_region_ratio(samples, "a", "b", "terminal", n_layers=4)
    assert len(ratios) == 1
    assert abs(ratios[0] - 2.0) < 1e-9


def test_compute_region_ratio_skips_zero_denominator():
    samples = [
        _make_sample({"a": [0.0, 0.0, 0.0, 0.0, 0.4, 0.4, 0.4, 0.4],
                       "b": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}),
    ]
    ratios = compute_region_ratio(samples, "a", "b", "terminal", n_layers=4)
    assert len(ratios) == 0  # skipped because b=0


# ---------------------------------------------------------------
# compute_per_token_density
# ---------------------------------------------------------------


def test_per_token_density_basic():
    result = compute_per_token_density(0.5, 10)
    assert abs(result - 0.05) < 1e-9


def test_per_token_density_zero_tokens():
    result = compute_per_token_density(0.5, 0)
    assert isnan(result)


def test_per_token_density_negative_tokens():
    result = compute_per_token_density(0.5, -1)
    assert isnan(result)


def test_per_token_density_nan_attention():
    result = compute_per_token_density(nan, 10)
    assert isnan(result)


# ---------------------------------------------------------------
# cooking_curve_stats
# ---------------------------------------------------------------


def test_cooking_curve_stats_normal():
    curve = np.array([0.0, 0.1, 0.5, 0.3, 0.2])
    stats = cooking_curve_stats(curve)
    assert stats["peak_layer"] == 2
    assert abs(stats["peak_value"] - 0.5) < 1e-9
    assert abs(stats["terminal_value"] - 0.2) < 1e-9
    assert abs(stats["retention_ratio"] - 0.4) < 1e-9


def test_cooking_curve_stats_empty():
    curve = np.array([])
    stats = cooking_curve_stats(curve)
    assert stats["peak_layer"] == -1
    assert stats["peak_value"] == 0.0


def test_cooking_curve_stats_all_zero():
    curve = np.zeros(10)
    stats = cooking_curve_stats(curve)
    assert stats["peak_layer"] == -1
    assert stats["retention_ratio"] == 0.0


def test_cooking_curve_stats_single_peak():
    curve = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    stats = cooking_curve_stats(curve)
    assert stats["peak_layer"] == 2
    assert stats["terminal_value"] == 0.0
    assert stats["retention_ratio"] == 0.0


def test_cooking_curve_stats_monotonic_increase():
    curve = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    stats = cooking_curve_stats(curve)
    assert stats["peak_layer"] == 4
    assert abs(stats["retention_ratio"] - 1.0) < 1e-9


# ---------------------------------------------------------------
# phase_mean
# ---------------------------------------------------------------


def test_phase_mean_basic():
    curve = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = phase_mean(curve, 1, 3)  # inclusive [1,3] -> mean(2,3,4)
    expected = (2.0 + 3.0 + 4.0) / 3
    assert abs(result - expected) < 1e-9


def test_phase_mean_clamps_end():
    curve = np.array([1.0, 2.0, 3.0])
    result = phase_mean(curve, 0, 100)  # clamps to len-1=2
    expected = (1.0 + 2.0 + 3.0) / 3
    assert abs(result - expected) < 1e-9


def test_phase_mean_invalid_range():
    curve = np.array([1.0, 2.0, 3.0])
    result = phase_mean(curve, 5, 3)  # start > end
    assert isnan(result)


def test_phase_mean_negative_start():
    curve = np.array([1.0, 2.0, 3.0])
    result = phase_mean(curve, -1, 2)
    assert isnan(result)


def test_phase_mean_all_nan():
    curve = np.array([nan, nan, nan])
    result = phase_mean(curve, 0, 2)
    assert isnan(result)


def test_phase_mean_partial_nan():
    curve = np.array([1.0, nan, 3.0])
    result = phase_mean(curve, 0, 2)
    expected = (1.0 + 3.0) / 2  # NaN skipped
    assert abs(result - expected) < 1e-9


# ---------------------------------------------------------------
# safe_mean
# ---------------------------------------------------------------


def test_safe_mean_basic():
    result = safe_mean([1.0, 2.0, 3.0])
    assert abs(result - 2.0) < 1e-9


def test_safe_mean_with_nan():
    result = safe_mean([1.0, nan, 3.0])
    expected = (1.0 + 3.0) / 2
    assert abs(result - expected) < 1e-9


def test_safe_mean_all_nan():
    result = safe_mean([nan, nan])
    assert isnan(result)


def test_safe_mean_empty():
    result = safe_mean([])
    assert isnan(result)


# ---------------------------------------------------------------
# safe_median
# ---------------------------------------------------------------


def test_safe_median_basic():
    result = safe_median([1.0, 2.0, 3.0])
    assert abs(result - 2.0) < 1e-9


def test_safe_median_with_nan():
    result = safe_median([1.0, nan, 5.0])
    expected = (1.0 + 5.0) / 2  # median of [1.0, 5.0]
    assert abs(result - expected) < 1e-9


def test_safe_median_all_nan():
    result = safe_median([nan, nan])
    assert isnan(result)


def test_safe_median_empty():
    result = safe_median([])
    assert isnan(result)


# ---------------------------------------------------------------
# safe_std
# ---------------------------------------------------------------


def test_safe_std_basic():
    result = safe_std([1.0, 2.0, 3.0])
    expected = float(np.std([1.0, 2.0, 3.0], ddof=1))
    assert abs(result - expected) < 1e-9


def test_safe_std_with_nan():
    result = safe_std([1.0, nan, 3.0])
    expected = float(np.std([1.0, 3.0], ddof=1))
    assert abs(result - expected) < 1e-9


def test_safe_std_single_value():
    """ddof=1 needs at least 2 values; single value should return NaN."""
    result = safe_std([5.0])
    assert isnan(result)


def test_safe_std_all_nan():
    result = safe_std([nan, nan])
    assert isnan(result)


def test_safe_std_empty():
    result = safe_std([])
    assert isnan(result)
