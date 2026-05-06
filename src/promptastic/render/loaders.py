"""Unified data loading from MI result JSON files.

Each public loader extracts the specific shape required by its matching
renderer.  All share the same JSON validation helpers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .._types import PatchingResult, RegionInfo
from ._shared import parse_layer_spec


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _load_result(path: str) -> dict[str, Any]:
    """Read and parse a result JSON, returning the top-level dict."""
    with open(path) as fh:
        data: dict[str, Any] = json.load(fh)
    return data


def _check_per_token(data: dict[str, Any]) -> None:
    """Abort if per-token attention data is absent."""
    if "per_token_attention" not in data:
        print("ERROR: result JSON has no per_token_attention. "
              "Re-run the engine without --no-per-token.")
        sys.exit(1)
    if "token_labels" not in data:
        print("ERROR: result JSON has no token_labels. "
              "Re-run the engine with the latest version.")
        sys.exit(1)


def _check_position(container: dict[str, Any], position: str) -> None:
    """Abort if the requested position key is missing."""
    if position not in container:
        print(f"ERROR: position '{position}' not found. "
              f"Available: {list(container.keys())}")
        sys.exit(1)


def _piece_boundaries(region_map: dict[str, RegionInfo]) -> dict[str, RegionInfo]:
    """Extract the top-level piece names (system_prompt, user_message,
    response) from the region map."""
    pieces: dict[str, RegionInfo] = {}
    for name in ("system_prompt", "user_message", "response"):
        if name in region_map:
            pieces[name] = region_map[name]
    return pieces


# ============================================================================
# HEATMAP DATA
# ============================================================================

def load_heatmap_data(
    path: str,
    position: str,
    layer_spec: str,
) -> tuple[list[str], np.ndarray, dict[str, RegionInfo], dict[str, RegionInfo]]:
    """Load per-token attention averaged across the requested layers.

    Returns ``(token_labels, weights_1d, region_map, piece_boundaries)``.
    """
    data = _load_result(path)
    _check_per_token(data)

    labels: list[str] = data["token_labels"]
    regions: dict[str, RegionInfo] = data["region_map"]
    per_token = data["per_token_attention"]
    _check_position(per_token, position)

    entries = per_token[position]["per_layer"]
    max_layer = max(e["layer"] for e in entries) + 1 if entries else 64
    wanted = set(parse_layer_spec(layer_spec, max_layer))

    arrays: list[np.ndarray] = []
    for entry in entries:
        if entry["layer"] in wanted:
            arrays.append(np.asarray(entry["weights"], dtype=np.float64))

    if not arrays:
        print(f"ERROR: no data for layers {sorted(wanted)}")
        sys.exit(1)

    weights = np.mean(arrays, axis=0)
    return labels, weights, regions, _piece_boundaries(regions)


# ============================================================================
# COOKING CURVE DATA
# ============================================================================

def load_cooking_data(
    path: str,
    position: str,
) -> tuple[dict[str, RegionInfo], dict[int, np.ndarray], list[str]]:
    """Load per-layer attention weights plus the region map.

    Returns ``(region_map, {layer: weights_1d}, token_labels)``.
    """
    data = _load_result(path)
    _check_per_token(data)

    per_token = data["per_token_attention"]
    _check_position(per_token, position)

    regions: dict[str, RegionInfo] = data["region_map"]
    labels: list[str] = data.get("token_labels", [])

    lw: dict[int, np.ndarray] = {}
    for entry in per_token[position]["per_layer"]:
        lw[entry["layer"]] = np.asarray(entry["weights"], dtype=np.float64)

    return regions, lw, labels


# ============================================================================
# ALL-LAYER DATA (for layer GIFs)
# ============================================================================

def load_all_layers(
    path: str,
    position: str,
) -> tuple[list[str], dict[int, np.ndarray], dict[str, RegionInfo], dict[str, RegionInfo]]:
    """Load every layer's per-token attention.

    Returns ``(token_labels, {layer: weights}, region_map, piece_boundaries)``.
    """
    data = _load_result(path)
    _check_per_token(data)

    per_token = data["per_token_attention"]
    _check_position(per_token, position)

    labels: list[str] = data["token_labels"]
    regions: dict[str, RegionInfo] = data["region_map"]

    lw: dict[int, np.ndarray] = {}
    for entry in per_token[position]["per_layer"]:
        lw[entry["layer"]] = np.asarray(entry["weights"], dtype=np.float64)

    return labels, lw, regions, _piece_boundaries(regions)


# ============================================================================
# VARIANT CURVES (aggregate mode)
# ============================================================================

def load_variant_curves(
    base_path: Path,
    dirname: str,
) -> dict[str, np.ndarray]:
    """Load all ``sample_*.json`` files in *dirname* and stack per-region
    attention into ``{region: (n_samples, n_layers)}`` arrays.

    Uses the ``attention.terminal.per_layer`` path (not per-token).
    """
    folder = base_path / dirname
    collected: dict[str, list[list[float]]] = {}

    for fp in sorted(folder.glob("sample_*.json")):
        with open(fp) as fh:
            sample = json.load(fh)

        rm = sample["region_map"]
        layers = sample["attention"]["terminal"]["per_layer"]

        for region in rm:
            if region not in collected:
                collected[region] = []
            curve = [ld["per_region_mean"].get(region, 0.0) for ld in layers]
            collected[region].append(curve)

    return {r: np.asarray(rows) for r, rows in collected.items()}


# ============================================================================
# PATCHING DATA
# ============================================================================

def load_patching_data(path: str) -> list[PatchingResult]:
    """Load causal patching results from a result JSON.

    Returns a list of ``PatchingResult`` dicts.
    """
    data = _load_result(path)
    if "patching" not in data:
        print("ERROR: result JSON has no patching data. "
              "Re-run the engine with --capture patching.")
        sys.exit(1)

    patching = data["patching"]
    entries = patching.get("results", patching) if isinstance(patching, dict) else patching
    results: list[PatchingResult] = []
    for entry in entries:
        results.append(PatchingResult(
            region=entry["region"],
            layer=entry["layer"],
            kl_divergence=entry.get("kl_divergence", 0.0),
            logit_diff=entry.get("logit_diff", 0.0),
            top_token_change=entry.get("top_token_change", ""),
            baseline_top_token=entry.get("baseline_top_token", ""),
        ))
    return results


# ============================================================================
# PER-HEAD DATA
# ============================================================================

def load_per_head_data(
    path: str,
    position: str,
) -> dict[str, Any]:
    """Load per-head attention breakdown.

    Returns a dict with ``per_layer`` being a list of
    ``{layer, heads: [{head_idx, region_weights, ...}]}`` entries.
    """
    data = _load_result(path)
    if "per_head_attention" not in data:
        print("ERROR: result JSON has no per_head_attention. "
              "Re-run with --capture per_head.")
        sys.exit(1)

    container = data["per_head_attention"]
    _check_position(container, position)
    return container[position]


# ============================================================================
# MLP DATA
# ============================================================================

def load_mlp_data(
    path: str,
    position: str,
) -> dict[str, Any]:
    """Load per-layer MLP delta norms.

    Returns a dict with ``per_layer`` as a list of ``{layer, ...}`` entries.
    """
    data = _load_result(path)
    if "mlp" not in data:
        print("ERROR: result JSON has no mlp data. "
              "Re-run with --capture mlp.")
        sys.exit(1)

    container = data["mlp"]
    _check_position(container, position)
    return container[position]


# ============================================================================
# SAE DATA
# ============================================================================

def load_sae_data(
    path: str,
    position: str,
) -> dict[str, Any]:
    """Load SAE feature activations.

    Returns a dict with ``per_layer`` as a list of
    ``{layer, features: [{feature_idx, activation, label}]}`` entries.
    """
    data = _load_result(path)
    if "sae" not in data:
        print("ERROR: result JSON has no sae data. "
              "Re-run with --capture sae.")
        sys.exit(1)

    container = data["sae"]
    _check_position(container, position)
    return container[position]


# ============================================================================
# GENERATION TIMELINE DATA
# ============================================================================

def load_generation_data(path: str) -> dict[str, Any]:
    """Load generation step data (autoregressive attention tracking).

    Returns a dict with a ``steps`` key containing per-step records.
    """
    data = _load_result(path)
    if "generation" not in data:
        print("ERROR: result JSON has no generation data. "
              "Re-run with --capture generation.")
        sys.exit(1)
    return data["generation"]


# ============================================================================
# GRADIENT DATA
# ============================================================================

def load_gradient_data(
    path: str,
    position: str,
) -> dict[str, Any]:
    """Load per-layer gradient attribution data.

    Returns a dict with ``per_layer`` entries containing gradient norms.
    """
    data = _load_result(path)
    if "gradients" not in data:
        print("ERROR: result JSON has no gradient data. "
              "Re-run with --capture gradients.")
        sys.exit(1)

    container = data["gradients"]
    _check_position(container, position)
    return container[position]
