"""Tests for promptastic.optimize.extract -- metric extraction."""

from math import isnan

import numpy as np
import pytest

from promptastic.optimize.extract import (
    extract_all_metrics,
    extract_attention_metrics,
    extract_causal_metrics,
    extract_dynamics_metrics,
)


# ---------------------------------------------------------------
# Helpers
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


def _make_per_token_attention(region_map, num_layers, num_tokens):
    """Build per_token_attention data for cooking curve extraction."""
    per_layer = []
    for li in range(num_layers):
        # Uniform weights
        weights = [1.0 / num_tokens] * num_tokens
        per_layer.append({"layer": li, "weights": weights})
    return {"terminal": {"per_layer": per_layer}}


def _make_sample(
    region_values_per_layer,
    region_map=None,
    per_token_attention=None,
    patching=None,
):
    """Build a synthetic sample dict."""
    sample = {
        "attention": _make_attention_data(region_values_per_layer),
    }
    if region_map:
        sample["region_map"] = region_map
    if per_token_attention:
        sample["per_token_attention"] = per_token_attention
    if patching:
        sample["patching"] = patching
    return sample


# ---------------------------------------------------------------
# extract_attention_metrics
# ---------------------------------------------------------------


def test_extract_attention_terminal():
    samples = [
        _make_sample({"rules": [0.1, 0.2, 0.3, 0.4], "examples": [0.4, 0.3, 0.2, 0.1]}),
    ]
    metrics = extract_attention_metrics(samples, ["rules", "examples"])
    assert "terminal_attention_rules" in metrics
    assert "terminal_attention_examples" in metrics
    # Last 4 layers average
    assert abs(metrics["terminal_attention_rules"] - 0.25) < 1e-9


def test_extract_attention_context_bleed():
    samples = [
        _make_sample({
            "conversation_turns": [0.0, 0.0, 0.5, 0.6],
            "current_message": [0.0, 0.0, 0.1, 0.1],
        }),
    ]
    metrics = extract_attention_metrics(
        samples, ["conversation_turns", "current_message"],
    )
    assert "context_bleed_ratio" in metrics
    # ratio = mean(conv_turns final 4) / mean(current_message final 4)
    # = 0.275 / 0.05 = 5.5
    assert metrics["context_bleed_ratio"] > 5.0


def test_extract_attention_density_cv():
    samples = [
        _make_sample(
            {"rules": [0.1, 0.2, 0.3, 0.4], "examples": [0.1, 0.2, 0.3, 0.4]},
            region_map={
                "rules": {"tok_start": 0, "tok_end": 10, "n_tokens": 10},
                "examples": {"tok_start": 10, "tok_end": 20, "n_tokens": 10},
            },
        ),
    ]
    metrics = extract_attention_metrics(samples, ["rules", "examples"])
    # Same attention, same tokens -> CV should be 0
    if "density_cv" in metrics:
        assert metrics["density_cv"] < 0.01


def test_extract_attention_density_cv_imbalanced():
    samples = [
        _make_sample(
            {"rules": [0.1, 0.1, 0.1, 0.1], "examples": [0.5, 0.5, 0.5, 0.5]},
            region_map={
                "rules": {"tok_start": 0, "tok_end": 10, "n_tokens": 10},
                "examples": {"tok_start": 10, "tok_end": 20, "n_tokens": 10},
            },
        ),
    ]
    metrics = extract_attention_metrics(samples, ["rules", "examples"])
    assert "density_cv" in metrics
    assert metrics["density_cv"] > 0.5


# ---------------------------------------------------------------
# extract_dynamics_metrics
# ---------------------------------------------------------------


def test_extract_dynamics_no_per_token():
    """Without per_token_attention, dynamics metrics should be empty."""
    samples = [
        _make_sample({"rules": [0.1, 0.2, 0.3, 0.4]}),
    ]
    metrics = extract_dynamics_metrics(samples, ["rules"], num_layers=4)
    # compute_region_attention_per_layer needs per_token_attention + region_map
    # Without it, should still not crash
    assert isinstance(metrics, dict)


def test_extract_dynamics_with_per_token():
    region_map = {
        "rules": {"tok_start": 0, "tok_end": 5, "n_tokens": 5},
    }
    per_token = _make_per_token_attention(region_map, num_layers=8, num_tokens=10)
    samples = [{
        "attention": _make_attention_data({"rules": [0.1] * 8}),
        "region_map": region_map,
        "per_token_attention": per_token,
    }]
    metrics = extract_dynamics_metrics(samples, ["rules"], num_layers=8)
    assert "peak_layer_frac_rules" in metrics
    assert "retention_ratio_rules" in metrics
    assert "peak_value_rules" in metrics


# ---------------------------------------------------------------
# extract_causal_metrics
# ---------------------------------------------------------------


def test_extract_causal_with_patching():
    patching_results = [
        {"region": "rules", "layer": 0, "kl_divergence": 0.5, "logit_diff": 0.1,
         "top_token_change": "a->b", "baseline_top_token": "a"},
        {"region": "rules", "layer": 4, "kl_divergence": 0.3, "logit_diff": 0.05,
         "top_token_change": "", "baseline_top_token": "a"},
        {"region": "examples", "layer": 0, "kl_divergence": 0.1, "logit_diff": 0.01,
         "top_token_change": "", "baseline_top_token": "a"},
    ]
    samples = [
        _make_sample(
            {"rules": [0.1] * 8, "examples": [0.1] * 8},
            patching={"results": patching_results},
        ),
    ]
    metrics = extract_causal_metrics(samples, ["rules", "examples"], num_layers=8)
    assert "causal_importance_rules" in metrics
    assert metrics["causal_importance_rules"] > metrics.get("causal_importance_examples", 0)


def test_extract_causal_no_patching():
    samples = [_make_sample({"rules": [0.1] * 8})]
    metrics = extract_causal_metrics(samples, ["rules"], num_layers=8)
    assert metrics == {}


# ---------------------------------------------------------------
# extract_all_metrics
# ---------------------------------------------------------------


def test_extract_all_basic():
    samples = [
        _make_sample({"rules": [0.1, 0.2, 0.3, 0.4], "examples": [0.4, 0.3, 0.2, 0.1]}),
    ]
    metrics = extract_all_metrics(
        samples, num_layers=4, regions=["rules", "examples"],
    )
    assert "terminal_attention_rules" in metrics
    assert "terminal_attention_examples" in metrics


def test_extract_all_auto_regions():
    samples = [{
        "attention": _make_attention_data({"rules": [0.1], "examples": [0.2]}),
        "region_map": {
            "rules": {"tok_start": 0, "tok_end": 5, "n_tokens": 5},
            "examples": {"tok_start": 5, "tok_end": 10, "n_tokens": 5},
            "system_prompt": {"tok_start": 0, "tok_end": 10, "n_tokens": 10},
        },
    }]
    metrics = extract_all_metrics(samples, num_layers=1)
    # system_prompt should be skipped
    assert "terminal_attention_rules" in metrics
    assert "terminal_attention_system_prompt" not in metrics
