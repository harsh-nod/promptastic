"""Tests for promptastic.analysis.causal -- causal analysis from patching."""

from math import isnan

import numpy as np

from promptastic.analysis.causal import (
    attention_vs_causal_correlation,
    causal_importance_score,
    find_critical_layers,
    rank_regions_by_causal_importance,
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _make_patching_results(region_layer_kl):
    """Build a list of PatchingResult dicts.

    region_layer_kl: list of (region, layer, kl_divergence) tuples.
    """
    results = []
    for region, layer, kl in region_layer_kl:
        results.append({
            "region": region,
            "layer": layer,
            "kl_divergence": kl,
            "logit_diff": kl * 0.5,  # dummy
            "top_token_change": "a",
            "baseline_top_token": "b",
        })
    return results


# ---------------------------------------------------------------
# causal_importance_score
# ---------------------------------------------------------------


def test_causal_importance_score_averages():
    results = _make_patching_results([
        ("rules", 0, 0.2),
        ("rules", 1, 0.4),
        ("rules", 2, 0.6),
    ])
    score = causal_importance_score(results, "rules")
    expected = (0.2 + 0.4 + 0.6) / 3
    assert abs(score - expected) < 1e-9


def test_causal_importance_score_filters_region():
    results = _make_patching_results([
        ("rules", 0, 0.5),
        ("examples", 0, 0.1),
        ("rules", 1, 0.3),
    ])
    score = causal_importance_score(results, "rules")
    expected = (0.5 + 0.3) / 2
    assert abs(score - expected) < 1e-9


def test_causal_importance_score_no_match():
    results = _make_patching_results([("rules", 0, 0.5)])
    score = causal_importance_score(results, "nonexistent")
    assert isnan(score)


def test_causal_importance_score_custom_metric():
    results = _make_patching_results([
        ("rules", 0, 0.5),
        ("rules", 1, 0.3),
    ])
    score = causal_importance_score(results, "rules", metric="logit_diff")
    expected = (0.25 + 0.15) / 2  # logit_diff = kl * 0.5
    assert abs(score - expected) < 1e-9


# ---------------------------------------------------------------
# rank_regions_by_causal_importance
# ---------------------------------------------------------------


def test_rank_regions_descending():
    results = _make_patching_results([
        ("rules", 0, 0.5),
        ("rules", 1, 0.5),
        ("examples", 0, 0.1),
        ("examples", 1, 0.1),
        ("format", 0, 0.8),
        ("format", 1, 0.8),
    ])
    ranking = rank_regions_by_causal_importance(results)
    names = [r[0] for r in ranking]
    # format (0.8) > rules (0.5) > examples (0.1)
    assert names[0] == "format"
    assert names[-1] == "examples"


def test_rank_regions_empty_input():
    ranking = rank_regions_by_causal_importance([])
    assert ranking == []


# ---------------------------------------------------------------
# find_critical_layers
# ---------------------------------------------------------------


def test_find_critical_layers_basic():
    results = _make_patching_results([
        ("rules", 0, 0.05),
        ("rules", 1, 0.15),
        ("rules", 2, 0.25),
        ("rules", 3, 0.03),
    ])
    critical = find_critical_layers(results, "rules", threshold=0.1)
    assert 1 in critical
    assert 2 in critical
    assert 0 not in critical
    assert 3 not in critical


def test_find_critical_layers_none_above_threshold():
    results = _make_patching_results([
        ("rules", 0, 0.01),
        ("rules", 1, 0.02),
    ])
    critical = find_critical_layers(results, "rules", threshold=0.1)
    assert critical == []


def test_find_critical_layers_sorted():
    results = _make_patching_results([
        ("rules", 5, 0.5),
        ("rules", 2, 0.5),
        ("rules", 8, 0.5),
    ])
    critical = find_critical_layers(results, "rules", threshold=0.1)
    assert critical == sorted(critical)


# ---------------------------------------------------------------
# attention_vs_causal_correlation
# ---------------------------------------------------------------


def test_correlation_perfectly_correlated():
    """Attention and patching that grow together should correlate near +1."""
    num_layers = 8
    attention_data = {
        "terminal": {
            "per_layer": [
                {"layer": i, "per_region_mean": {"rules": float(i) / num_layers}}
                for i in range(num_layers)
            ],
        },
    }
    patching_results = [
        {"region": "rules", "layer": i, "kl_divergence": float(i) / num_layers}
        for i in range(num_layers)
    ]
    corr = attention_vs_causal_correlation(
        attention_data, patching_results, "rules", "terminal", num_layers
    )
    assert corr > 0.99


def test_correlation_uncorrelated():
    """Constant attention should have zero std, yielding NaN."""
    num_layers = 8
    attention_data = {
        "terminal": {
            "per_layer": [
                {"layer": i, "per_region_mean": {"rules": 0.5}}
                for i in range(num_layers)
            ],
        },
    }
    patching_results = [
        {"region": "rules", "layer": i, "kl_divergence": float(i)}
        for i in range(num_layers)
    ]
    corr = attention_vs_causal_correlation(
        attention_data, patching_results, "rules", "terminal", num_layers
    )
    # Zero variance in attention -> NaN
    assert isnan(corr)


def test_correlation_negatively_correlated():
    """Attention decreasing while patching increases -> negative correlation."""
    num_layers = 8
    attention_data = {
        "terminal": {
            "per_layer": [
                {"layer": i, "per_region_mean": {"rules": float(num_layers - i)}}
                for i in range(num_layers)
            ],
        },
    }
    patching_results = [
        {"region": "rules", "layer": i, "kl_divergence": float(i)}
        for i in range(num_layers)
    ]
    corr = attention_vs_causal_correlation(
        attention_data, patching_results, "rules", "terminal", num_layers
    )
    assert corr < -0.99
