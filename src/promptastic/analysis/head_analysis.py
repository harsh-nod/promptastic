#!/usr/bin/env python3
"""Per-head attention analysis for mechanistic interpretability.

Identifies specialist heads, computes head-level entropy over regions,
and measures inter-head variance to reveal which heads focus on
specific prompt regions.

Usage:
    python -m promptastic.analysis.head_analysis \\
        --base-dir ./data/results \\
        --variant results_baseline \\
        --num-heads 32
"""

from __future__ import annotations

import argparse
import json
from math import log2, isnan
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import FINAL_LAYERS, SKIP_REGIONS
from .formatting import fmt, print_header, print_subheader
from .metrics import compute_per_head_attention, safe_mean


def head_specialization(
    per_head_data: np.ndarray,
    region_names: list[str],
    num_layers: int,
    num_heads: int,
) -> dict[str, Any]:
    """Compute per-head specialization across regions.

    Args:
        per_head_data: dict mapping region_name -> (num_layers, num_heads) array
                       of mean attention values.
        region_names: list of region names (keys into per_head_data).
        num_layers: number of layers.
        num_heads: number of heads per layer.

    Returns:
        Dict with 'per_layer' key mapping to a list of per-layer dicts.
        Each per-layer dict has 'layer', plus a 'heads' list where each
        head entry has 'dominant_region', 'dominant_weight', and 'entropy'.
    """
    # per_head_data is actually a dict[str, np.ndarray] despite the type hint
    per_layer_results: list[dict[str, Any]] = []

    for layer_idx in range(num_layers):
        head_entries: list[dict[str, Any]] = []
        for head_idx in range(num_heads):
            weights: dict[str, float] = {}
            for region in region_names:
                arr = per_head_data.get(region)  # type: ignore[union-attr]
                if arr is not None and layer_idx < arr.shape[0] and head_idx < arr.shape[1]:
                    weights[region] = float(arr[layer_idx, head_idx])
                else:
                    weights[region] = 0.0

            total = sum(weights.values())
            if total > 0:
                dominant = max(weights, key=weights.get)  # type: ignore[arg-type]
                dominant_weight = weights[dominant]
                # Shannon entropy over normalized distribution
                probs = [w / total for w in weights.values() if w > 0]
                entropy = -sum(p * log2(p) for p in probs) if probs else 0.0
            else:
                dominant = "none"
                dominant_weight = 0.0
                entropy = 0.0

            head_entries.append({
                "head_idx": head_idx,
                "dominant_region": dominant,
                "dominant_weight": dominant_weight,
                "entropy": entropy,
            })

        per_layer_results.append({
            "layer": layer_idx,
            "heads": head_entries,
        })

    return {"per_layer": per_layer_results}


def head_variance_by_region(
    per_head_data: dict[str, np.ndarray],
    region_names: list[str],
    num_layers: int,
    num_heads: int,
) -> dict[str, dict[str, Any]]:
    """Compute per-region variance of attention across heads.

    Returns a dict mapping region name to:
      - mean_variance: mean variance across layers
      - per_layer_variance: list of per-layer variance values
      - max_variance_layer: layer with the highest inter-head variance
    """
    result: dict[str, dict[str, Any]] = {}

    for region in region_names:
        arr = per_head_data.get(region)
        if arr is None:
            result[region] = {
                "mean_variance": 0.0,
                "per_layer_variance": [0.0] * num_layers,
                "max_variance_layer": 0,
            }
            continue

        per_layer_var = np.var(arr, axis=1)  # variance across heads
        mean_var = float(np.mean(per_layer_var))
        max_layer = int(np.argmax(per_layer_var))

        result[region] = {
            "mean_variance": mean_var,
            "per_layer_variance": per_layer_var.tolist(),
            "max_variance_layer": max_layer,
        }

    return result


def find_specialist_heads(
    per_head_data: dict[str, np.ndarray],
    region_names: list[str],
    num_layers: int,
    num_heads: int,
    entropy_threshold: float = 1.0,
) -> list[dict[str, Any]]:
    """Find heads with low entropy (high specialization) across regions.

    A specialist head is one that concentrates attention on a small number
    of regions, indicated by Shannon entropy below entropy_threshold.

    Returns a list of dicts with layer, head_idx, dominant_region,
    dominant_weight, and entropy, sorted by entropy ascending.
    """
    specialists: list[dict[str, Any]] = []

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            weights: dict[str, float] = {}
            for region in region_names:
                arr = per_head_data.get(region)
                if arr is not None and layer_idx < arr.shape[0] and head_idx < arr.shape[1]:
                    weights[region] = float(arr[layer_idx, head_idx])
                else:
                    weights[region] = 0.0

            total = sum(weights.values())
            if total <= 0:
                continue

            probs = [w / total for w in weights.values() if w > 0]
            entropy = -sum(p * log2(p) for p in probs)

            if entropy < entropy_threshold:
                dominant = max(weights, key=weights.get)  # type: ignore[arg-type]
                specialists.append({
                    "layer": layer_idx,
                    "head_idx": head_idx,
                    "dominant_region": dominant,
                    "dominant_weight": weights[dominant],
                    "entropy": entropy,
                })

    specialists.sort(key=lambda x: x["entropy"])
    return specialists


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for head analysis CLI."""
    parser = argparse.ArgumentParser(
        description="Per-head attention analysis",
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Base directory containing result directories",
    )
    parser.add_argument(
        "--variant",
        required=True,
        help="Variant directory name (relative to base-dir)",
    )
    parser.add_argument(
        "--position",
        default="terminal",
        help="Query position (default: terminal)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=32,
        help="Number of attention heads per layer (default: 32)",
    )
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=1.0,
        help="Entropy threshold for specialist head detection (default: 1.0)",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    variant_dir = base_dir / args.variant
    if not variant_dir.exists():
        print(f"Directory not found: {variant_dir}")
        return

    # Load samples
    samples: list[dict[str, Any]] = []
    for filepath in sorted(variant_dir.glob("sample_*.json")):
        with open(filepath) as fh:
            samples.append(json.load(fh))

    if not samples:
        print(f"No samples found in {variant_dir}")
        return

    # Discover regions and layers
    region_map = samples[0].get("region_map", {})
    region_names = sorted(r for r in region_map if r not in SKIP_REGIONS)

    num_layers = 64
    for sample in samples:
        attn = sample.get("attention", {})
        for pos_data in attn.values():
            entries = pos_data.get("per_layer", [])
            if entries:
                num_layers = max(e["layer"] for e in entries) + 1
                break

    print(
        f"  Loaded {len(samples)} samples, "
        f"{len(region_names)} regions, "
        f"{num_layers} layers, "
        f"{args.num_heads} heads"
    )

    # Build aggregated per-head data across samples
    region_arrays: dict[str, list[np.ndarray]] = {r: [] for r in region_names}
    for sample in samples:
        for region in region_names:
            arr = compute_per_head_attention(
                sample, region, args.position, num_layers, args.num_heads
            )
            if np.any(arr > 0):
                region_arrays[region].append(arr)

    avg_per_head: dict[str, np.ndarray] = {}
    for region in region_names:
        arrs = region_arrays[region]
        if arrs:
            avg_per_head[region] = np.mean(arrs, axis=0)
        else:
            avg_per_head[region] = np.zeros((num_layers, args.num_heads))

    # Head variance by region
    print_header("Head Variance by Region")
    variance_info = head_variance_by_region(
        avg_per_head, region_names, num_layers, args.num_heads
    )
    print(f"  {'Region':<22} {'Mean Var':>12} {'Max Var Layer':>14}")
    print(f"  {'---':<22} {'---':>12} {'---':>14}")
    for region in region_names:
        info = variance_info[region]
        print(
            f"  {region:<22} {fmt(info['mean_variance'], 12)} "
            f"{'L' + str(info['max_variance_layer']):>14}"
        )

    # Specialist heads
    print_header("Specialist Heads")
    specialists = find_specialist_heads(
        avg_per_head,
        region_names,
        num_layers,
        args.num_heads,
        args.entropy_threshold,
    )
    if specialists:
        print(
            f"  {'Layer':>5} {'Head':>5} {'Region':<22} "
            f"{'Weight':>9} {'Entropy':>9}"
        )
        print(
            f"  {'---':>5} {'---':>5} {'---':<22} "
            f"{'---':>9} {'---':>9}"
        )
        for spec in specialists[:30]:
            print(
                f"  L{spec['layer']:>3} H{spec['head_idx']:>3} "
                f"{spec['dominant_region']:<22} "
                f"{fmt(spec['dominant_weight'])} {fmt(spec['entropy'])}"
            )
        if len(specialists) > 30:
            print(f"  ... and {len(specialists) - 30} more")
    else:
        print("  No specialist heads found below entropy threshold.")

    print(f"\n  Total specialist heads: {len(specialists)}")


if __name__ == "__main__":
    main()
