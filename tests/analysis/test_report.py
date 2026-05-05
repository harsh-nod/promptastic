"""Tests for promptastic.analysis.report -- report generation helpers."""

import numpy as np

from promptastic.analysis.report import (
    _classify_story,
    compute_context_bleed,
    compute_cooking_table,
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _make_sample_with_per_token(region_trajectories, num_layers=8):
    """Build a sample dict with per_token_attention and region_map.

    region_trajectories: dict of region_name -> list of per-layer values
    """
    region_map = {}
    per_layer = []

    # Assign each region a consecutive token span of length 4
    tok_cursor = 0
    for rname in region_trajectories:
        region_map[rname] = {
            "tok_start": tok_cursor,
            "tok_end": tok_cursor + 4,
            "n_tokens": 4,
        }
        tok_cursor += 4

    total_tokens = tok_cursor

    for li in range(num_layers):
        weights = [0.0] * total_tokens
        for rname, traj in region_trajectories.items():
            info = region_map[rname]
            val = traj[li] if li < len(traj) else 0.0
            for t in range(info["tok_start"], info["tok_end"]):
                weights[t] = val
        per_layer.append({"layer": li, "weights": weights})

    return {
        "region_map": region_map,
        "per_token_attention": {
            "terminal": {"per_layer": per_layer},
        },
    }


# ---------------------------------------------------------------
# _classify_story
# ---------------------------------------------------------------


def test_classify_story_immediate_peak_strong_fade():
    result = _classify_story(peak_layer=0, ratio=15.0, num_layers=64)
    assert result == "Immediate peak, strong fade"


def test_classify_story_immediate_peak():
    result = _classify_story(peak_layer=0, ratio=5.0, num_layers=64)
    assert result == "Immediate peak"


def test_classify_story_early_read():
    result = _classify_story(peak_layer=0, ratio=1.5, num_layers=64)
    assert result == "Early read"


def test_classify_story_absorbed():
    # frac = 5/63 ~ 0.079 -> "Absorbed by L5"
    result = _classify_story(peak_layer=5, ratio=2.0, num_layers=64)
    assert "Absorbed" in result


def test_classify_story_absorption_phase():
    # frac = 10/63 ~ 0.159
    result = _classify_story(peak_layer=10, ratio=2.0, num_layers=64)
    assert result == "Absorption phase"


def test_classify_story_deep_compression():
    # frac = 20/63 ~ 0.317
    result = _classify_story(peak_layer=20, ratio=2.0, num_layers=64)
    assert result == "Deep compression"


def test_classify_story_mid_phase_peak():
    # frac = 40/63 ~ 0.635
    result = _classify_story(peak_layer=40, ratio=2.0, num_layers=64)
    assert result == "Mid-phase peak"


def test_classify_story_output_prep():
    # frac = 50/63 ~ 0.794
    result = _classify_story(peak_layer=50, ratio=2.0, num_layers=64)
    assert result == "Output prep phase"


def test_classify_story_late_bloomer():
    # frac = 60/63 ~ 0.952, ratio >= 2
    result = _classify_story(peak_layer=60, ratio=5.0, num_layers=64)
    assert result == "Late bloomer"


def test_classify_story_latest_bloomer():
    # frac = 60/63 ~ 0.952, ratio < 2
    result = _classify_story(peak_layer=60, ratio=1.5, num_layers=64)
    assert result == "Latest bloomer"


def test_classify_story_single_layer():
    result = _classify_story(peak_layer=0, ratio=1.0, num_layers=1)
    assert result == "Single layer"


def test_classify_story_returns_string():
    """All valid combinations should return a non-empty string."""
    for peak in range(0, 64, 8):
        for ratio in (0.5, 1.5, 5.0, 15.0):
            result = _classify_story(peak, ratio, 64)
            assert isinstance(result, str)
            assert len(result) > 0


# ---------------------------------------------------------------
# compute_cooking_table
# ---------------------------------------------------------------


def test_compute_cooking_table_returns_extended_stats():
    num_layers = 8
    traj = [0.1, 0.5, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15]
    samples = [_make_sample_with_per_token({"rules": traj}, num_layers)]
    table = compute_cooking_table(samples, ["rules"], num_layers=num_layers)

    assert "rules" in table
    stats = table["rules"]
    assert "peak_layer" in stats
    assert "peak_value" in stats
    assert "terminal_value" in stats
    assert "retention_ratio" in stats
    assert "peak_terminal_ratio" in stats
    assert "story" in stats
    assert "n_samples" in stats
    assert "trajectory" in stats
    assert isinstance(stats["trajectory"], list)
    assert len(stats["trajectory"]) == num_layers


def test_compute_cooking_table_correct_peak():
    num_layers = 4
    traj = [0.1, 0.9, 0.3, 0.2]
    samples = [_make_sample_with_per_token({"r": traj}, num_layers)]
    table = compute_cooking_table(samples, ["r"], num_layers=num_layers)
    assert table["r"]["peak_layer"] == 1


def test_compute_cooking_table_multiple_regions():
    num_layers = 4
    samples = [
        _make_sample_with_per_token(
            {
                "rules": [0.5, 0.3, 0.2, 0.1],
                "examples": [0.1, 0.2, 0.3, 0.5],
            },
            num_layers,
        ),
    ]
    table = compute_cooking_table(
        samples, ["rules", "examples"], num_layers=num_layers
    )
    assert "rules" in table
    assert "examples" in table
    assert table["rules"]["peak_layer"] == 0
    assert table["examples"]["peak_layer"] == 3


def test_compute_cooking_table_empty_region():
    """Region not in the sample should not appear in the table."""
    num_layers = 4
    samples = [_make_sample_with_per_token({"rules": [0.5, 0.3, 0.2, 0.1]}, num_layers)]
    table = compute_cooking_table(samples, ["nonexistent"], num_layers=num_layers)
    assert "nonexistent" not in table


# ---------------------------------------------------------------
# compute_context_bleed
# ---------------------------------------------------------------


def test_compute_context_bleed_basic():
    num_layers = 4
    samples = [
        _make_sample_with_per_token(
            {
                "conversation_turns": [0.1, 0.2, 0.3, 0.6],
                "current_message": [0.1, 0.1, 0.1, 0.3],
            },
            num_layers,
        ),
    ]
    result = compute_context_bleed(samples, num_layers=num_layers)
    assert "mean_ratio" in result
    assert "median_ratio" in result
    assert "pct_above_2x" in result
    assert "conv_turns_mean" in result
    assert "current_message_mean" in result
    assert "n_samples" in result


def test_compute_context_bleed_ratio():
    """Ratio should be conv_terminal / curr_terminal."""
    num_layers = 4
    # conv terminal = 0.8, curr terminal = 0.4 -> ratio = 2.0
    samples = [
        _make_sample_with_per_token(
            {
                "conversation_turns": [0.1, 0.2, 0.3, 0.8],
                "current_message": [0.1, 0.1, 0.1, 0.4],
            },
            num_layers,
        ),
    ]
    result = compute_context_bleed(samples, num_layers=num_layers)
    assert abs(result["mean_ratio"] - 2.0) < 1e-6


def test_compute_context_bleed_no_current_message():
    """When current_message terminal is zero, that sample should be skipped."""
    num_layers = 4
    samples = [
        _make_sample_with_per_token(
            {
                "conversation_turns": [0.1, 0.2, 0.3, 0.4],
                "current_message": [0.0, 0.0, 0.0, 0.0],
            },
            num_layers,
        ),
    ]
    result = compute_context_bleed(samples, num_layers=num_layers)
    assert result["n_samples"] == 0
    assert result["mean_ratio"] == 0.0
