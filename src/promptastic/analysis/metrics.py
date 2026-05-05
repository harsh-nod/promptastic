"""Core metric computations for mechanistic interpretability analysis.

Provides functions for terminal attention averaging, region ratios,
per-token density, cooking curve statistics, per-head attention,
MLP delta trajectories, gradient attribution, and NaN-safe aggregations.
"""

from __future__ import annotations

from math import isnan
from typing import Any

import numpy as np

from ..constants import FINAL_LAYERS
from .._types import CookingStats


def avg_final_layers(
    attention_data: dict[str, Any],
    position: str,
    region: str,
    n_layers: int = FINAL_LAYERS,
) -> float:
    """Mean attention to a region over the last n_layers at a query position.

    Extracts per_region_mean values from the final n_layers entries in
    the per_layer list for the given position.

    Returns NaN if the position or region is absent.
    """
    pos_block = attention_data.get(position)
    if pos_block is None:
        return float("nan")
    layer_entries = pos_block.get("per_layer", [])
    collected: list[float] = []
    for entry in layer_entries[-n_layers:]:
        region_means = entry.get("per_region_mean", {})
        if region in region_means:
            collected.append(region_means[region])
    if not collected:
        return float("nan")
    return sum(collected) / len(collected)


def compute_region_ratio(
    samples: list[dict[str, Any]],
    region_a: str,
    region_b: str,
    position: str = "terminal",
    n_layers: int = FINAL_LAYERS,
) -> list[float]:
    """Per-sample ratio of region_a attention to region_b attention.

    Skips samples where region_b attention is zero or either value is NaN.
    Useful for context bleed detection (conv_turns / current_message).
    """
    ratios: list[float] = []
    for sample in samples:
        attn = sample.get("attention", sample)
        val_a = avg_final_layers(attn, position, region_a, n_layers)
        val_b = avg_final_layers(attn, position, region_b, n_layers)
        if not isnan(val_a) and not isnan(val_b) and val_b > 0:
            ratios.append(val_a / val_b)
    return ratios


def compute_per_token_density(attention: float, n_tokens: int) -> float:
    """Attention divided by token count for fair cross-region comparison.

    Returns NaN when n_tokens is non-positive or attention is NaN.
    """
    if n_tokens <= 0 or isnan(attention):
        return float("nan")
    return attention / n_tokens


def compute_region_attention_per_layer(
    sample: dict[str, Any],
    region_name: str,
    position: str = "terminal",
    num_layers: int = 64,
) -> np.ndarray:
    """Build a (num_layers,) array of mean attention for a region per layer.

    Uses per_token_attention data and region boundaries to compute the
    mean weight across the region's token span at each layer.
    Returns a zero array if the region or data is missing.
    """
    region_map = sample.get("region_map", {})
    region_info = region_map.get(region_name)
    if region_info is None:
        return np.zeros(num_layers)

    tok_start = region_info["tok_start"]
    tok_end = region_info["tok_end"]
    span_len = region_info.get("n_tokens", tok_end - tok_start)
    if span_len <= 0:
        return np.zeros(num_layers)

    layer_list = (
        sample.get("per_token_attention", {})
        .get(position, {})
        .get("per_layer", [])
    )
    if not layer_list:
        return np.zeros(num_layers)

    result = np.zeros(num_layers)
    for entry in layer_list:
        layer_idx = entry["layer"]
        weights = entry["weights"]
        if layer_idx < num_layers and tok_end <= len(weights):
            result[layer_idx] = float(np.mean(weights[tok_start:tok_end]))
    return result


def cooking_curve_stats(curve: np.ndarray) -> CookingStats:
    """Summarize a cooking curve: peak location, peak value, terminal value.

    retention_ratio is terminal_value / peak_value. Returns zeros
    for empty or all-zero curves.
    """
    if len(curve) == 0 or not np.any(curve > 0):
        return {
            "peak_layer": -1,
            "peak_value": 0.0,
            "terminal_value": 0.0,
            "retention_ratio": 0.0,
        }
    peak_idx = int(np.argmax(curve))
    peak_val = float(curve[peak_idx])
    term_val = float(curve[-1])
    retention = term_val / peak_val if peak_val > 0 else 0.0
    return {
        "peak_layer": peak_idx,
        "peak_value": peak_val,
        "terminal_value": term_val,
        "retention_ratio": retention,
    }


def phase_mean(curve: np.ndarray, phase_start: int, phase_end: int) -> float:
    """Mean attention within a layer range [phase_start, phase_end] inclusive.

    Clamps phase_end to the curve length. Returns NaN for invalid ranges
    or entirely NaN segments.
    """
    phase_end = min(phase_end, len(curve) - 1)
    if phase_start > phase_end or phase_start < 0:
        return float("nan")
    segment = curve[phase_start : phase_end + 1]
    valid = segment[~np.isnan(segment)]
    if len(valid) == 0:
        return float("nan")
    return float(np.mean(valid))


def safe_mean(values: list[float]) -> float:
    """NaN-safe arithmetic mean."""
    clean = [v for v in values if not isnan(v)]
    if not clean:
        return float("nan")
    return float(np.mean(clean))


def safe_median(values: list[float]) -> float:
    """NaN-safe median."""
    clean = sorted(v for v in values if not isnan(v))
    if not clean:
        return float("nan")
    return float(np.median(clean))


def safe_std(values: list[float]) -> float:
    """NaN-safe standard deviation with ddof=1."""
    clean = [v for v in values if not isnan(v)]
    if len(clean) < 2:
        return float("nan")
    return float(np.std(clean, ddof=1))


# ---------------------------------------------------------------------------
# Extended metrics: per-head, MLP delta, gradient trajectory
# ---------------------------------------------------------------------------


def compute_per_head_attention(
    sample: dict[str, Any],
    region_name: str,
    position: str,
    num_layers: int,
    num_heads: int,
) -> np.ndarray:
    """Build a (num_layers, num_heads) array of per-head attention for a region.

    Reads from sample["per_head_attention"][position]["per_layer"], where
    each entry contains a "heads" list of per-head weight arrays.
    Returns zeros if data is missing.
    """
    region_map = sample.get("region_map", {})
    region_info = region_map.get(region_name)
    if region_info is None:
        return np.zeros((num_layers, num_heads))

    tok_start = region_info["tok_start"]
    tok_end = region_info["tok_end"]
    span_len = region_info.get("n_tokens", tok_end - tok_start)
    if span_len <= 0:
        return np.zeros((num_layers, num_heads))

    head_data = (
        sample.get("per_head_attention", {})
        .get(position, {})
        .get("per_layer", [])
    )
    if not head_data:
        return np.zeros((num_layers, num_heads))

    result = np.zeros((num_layers, num_heads))
    for entry in head_data:
        layer_idx = entry.get("layer", -1)
        heads = entry.get("heads", [])
        if layer_idx < 0 or layer_idx >= num_layers:
            continue
        for head_idx, head_weights in enumerate(heads):
            if head_idx >= num_heads:
                break
            if tok_end <= len(head_weights):
                result[layer_idx, head_idx] = float(
                    np.mean(head_weights[tok_start:tok_end])
                )
    return result


def compute_mlp_delta_trajectory(
    sample: dict[str, Any],
    region_name: str,
    position: str,
    num_layers: int,
) -> np.ndarray:
    """Build a (num_layers,) trajectory of MLP contribution norms.

    Reads from sample["mlp_deltas"][position]["per_layer"], where
    each entry has a "delta_norm" for the specified region. This measures
    how much the MLP sub-layer at each layer alters the residual stream
    representation of the region.
    """
    mlp_layers = (
        sample.get("mlp_deltas", {})
        .get(position, {})
        .get("per_layer", [])
    )
    if not mlp_layers:
        return np.zeros(num_layers)

    result = np.zeros(num_layers)
    for entry in mlp_layers:
        layer_idx = entry.get("layer", -1)
        region_deltas = entry.get("region_deltas", {})
        if layer_idx < 0 or layer_idx >= num_layers:
            continue
        if region_name in region_deltas:
            result[layer_idx] = float(region_deltas[region_name])
    return result


def compute_gradient_trajectory(
    sample: dict[str, Any],
    region_name: str,
    position: str,
    num_layers: int,
) -> np.ndarray:
    """Build a (num_layers,) trajectory of gradient attribution scores.

    Reads from sample["gradients"][position]["per_layer"], where each
    entry provides a per-region gradient norm. Higher values indicate
    layers where the region's tokens have more influence on the output
    via gradient flow.
    """
    grad_layers = (
        sample.get("gradients", {})
        .get(position, {})
        .get("per_layer", [])
    )
    if not grad_layers:
        return np.zeros(num_layers)

    result = np.zeros(num_layers)
    for entry in grad_layers:
        layer_idx = entry.get("layer", -1)
        region_grads = entry.get("region_norms", {})
        if layer_idx < 0 or layer_idx >= num_layers:
            continue
        if region_name in region_grads:
            result[layer_idx] = float(region_grads[region_name])
    return result
