#!/usr/bin/env python3
"""Causal analysis from activation patching results.

Provides scoring, ranking, critical layer detection, and correlation
between attention-based and causal importance measures.

Usage:
    python -m promptastic.analysis.causal \\
        --base-dir ./data/results \\
        --variant results_baseline \\
        --metric kl_divergence
"""

from __future__ import annotations

import argparse
import json
from math import isnan
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import analysis_phases
from .formatting import fmt, print_header, print_subheader
from .metrics import avg_final_layers, safe_mean


def causal_importance_score(
    patching_results: list[dict[str, Any]],
    region: str,
    metric: str = "kl_divergence",
) -> float:
    """Average effect across all patched layers for a region.

    patching_results is a flat list of PatchingResult dicts.
    Returns the mean of the chosen metric for entries matching the region.
    Returns NaN if no matching entries exist.
    """
    values: list[float] = []
    for entry in patching_results:
        if entry.get("region") != region:
            continue
        val = entry.get(metric, float("nan"))
        if not isnan(val):
            values.append(val)
    return safe_mean(values)


def causal_importance_by_phase(
    patching_results: list[dict[str, Any]],
    region: str,
    num_layers: int,
    metric: str = "kl_divergence",
) -> dict[str, float]:
    """Break down causal importance by analysis phase.

    Returns a dict mapping phase name to mean metric value for layers
    within that phase.
    """
    phases = analysis_phases(num_layers)
    phase_scores: dict[str, float] = {}

    for phase_name, (start, end) in phases.items():
        vals: list[float] = []
        for entry in patching_results:
            if entry.get("region") != region:
                continue
            layer = entry.get("layer", -1)
            if start <= layer <= end:
                val = entry.get(metric, float("nan"))
                if not isnan(val):
                    vals.append(val)
        phase_scores[phase_name] = safe_mean(vals)

    return phase_scores


def rank_regions_by_causal_importance(
    patching_results: list[dict[str, Any]],
    metric: str = "kl_divergence",
) -> list[tuple[str, float]]:
    """Rank all regions by their mean causal importance, descending.

    Returns a list of (region_name, score) tuples sorted from highest
    to lowest. Regions with NaN scores are excluded.
    """
    region_set: set[str] = set()
    for entry in patching_results:
        r = entry.get("region")
        if r is not None:
            region_set.add(r)

    scored: list[tuple[str, float]] = []
    for region in region_set:
        score = causal_importance_score(patching_results, region, metric)
        if not isnan(score):
            scored.append((region, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def find_critical_layers(
    patching_results: list[dict[str, Any]],
    region: str,
    threshold: float = 0.1,
) -> list[int]:
    """Identify layers where patching the region exceeds the threshold.

    Returns a sorted list of layer indices where kl_divergence
    exceeds the given threshold.
    """
    critical: list[int] = []
    # Aggregate by layer (mean across samples if duplicated)
    layer_vals: dict[int, list[float]] = {}
    for entry in patching_results:
        if entry.get("region") != region:
            continue
        layer = entry.get("layer", -1)
        val = entry.get("kl_divergence", float("nan"))
        if layer >= 0 and not isnan(val):
            layer_vals.setdefault(layer, []).append(val)

    for layer, vals in layer_vals.items():
        if safe_mean(vals) > threshold:
            critical.append(layer)

    return sorted(critical)


def attention_vs_causal_correlation(
    attention_data: dict[str, Any],
    patching_results: list[dict[str, Any]],
    region: str,
    position: str,
    num_layers: int,
) -> float:
    """Pearson correlation between per-layer attention and causal effect.

    Builds a per-layer attention vector from attention_data and a per-layer
    patching effect vector, then computes their correlation. This reveals
    whether high-attention layers are also causally important.

    Returns NaN if either vector has zero variance or insufficient data.
    """
    # Build attention per-layer vector
    attn_vec = np.zeros(num_layers)
    pos_block = attention_data.get(position)
    if pos_block is not None:
        for entry in pos_block.get("per_layer", []):
            layer_idx = entry.get("layer", -1)
            region_means = entry.get("per_region_mean", {})
            if 0 <= layer_idx < num_layers and region in region_means:
                attn_vec[layer_idx] = region_means[region]

    # Build patching per-layer vector
    patch_vec = np.zeros(num_layers)
    layer_vals: dict[int, list[float]] = {}
    for entry in patching_results:
        if entry.get("region") != region:
            continue
        layer = entry.get("layer", -1)
        kl = entry.get("kl_divergence", float("nan"))
        if 0 <= layer < num_layers and not isnan(kl):
            layer_vals.setdefault(layer, []).append(kl)

    for layer, vals in layer_vals.items():
        patch_vec[layer] = safe_mean(vals)

    # Pearson correlation
    if np.std(attn_vec) == 0 or np.std(patch_vec) == 0:
        return float("nan")

    corr_matrix = np.corrcoef(attn_vec, patch_vec)
    return float(corr_matrix[0, 1])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for causal analysis CLI."""
    parser = argparse.ArgumentParser(
        description="Causal analysis from activation patching results",
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
        "--metric",
        default="kl_divergence",
        help="Patching metric to analyze (default: kl_divergence)",
    )
    parser.add_argument(
        "--position",
        default="terminal",
        help="Query position (default: terminal)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Threshold for critical layer detection (default: 0.1)",
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

    # Collect all patching entries
    all_patching: list[dict[str, Any]] = []
    for sample in samples:
        all_patching.extend(sample.get("patching", []))

    if not all_patching:
        print("No patching data found in samples.")
        return

    # Detect layers
    num_layers = 64
    for sample in samples:
        attn = sample.get("attention", {})
        for pos_data in attn.values():
            entries = pos_data.get("per_layer", [])
            if entries:
                num_layers = max(e["layer"] for e in entries) + 1
                break

    print(f"  Loaded {len(samples)} samples, {len(all_patching)} patching entries")

    # Ranking
    print_header("Region Ranking by Causal Importance")
    ranking = rank_regions_by_causal_importance(all_patching, args.metric)
    print(f"  {'Region':<22} {'Score':>12}")
    print(f"  {'---':<22} {'---':>12}")
    for region, score in ranking:
        print(f"  {region:<22} {fmt(score, 12)}")

    # Per-phase breakdown for top regions
    for region, _ in ranking[:5]:
        print_subheader(f"{region} -- by phase")
        by_phase = causal_importance_by_phase(
            all_patching, region, num_layers, args.metric
        )
        for phase_name, score in by_phase.items():
            print(f"    {phase_name:<22} {fmt(score)}")

    # Critical layers
    print_header("Critical Layers")
    for region, _ in ranking:
        critical = find_critical_layers(
            all_patching, region, args.threshold
        )
        if critical:
            layer_str = ", ".join(f"L{l}" for l in critical)
            print(f"  {region:<22} {layer_str}")

    # Attention vs causal correlation
    print_header("Attention vs Causal Correlation")
    print(f"  {'Region':<22} {'Pearson r':>12}")
    print(f"  {'---':<22} {'---':>12}")
    for region, _ in ranking:
        # Average correlation across samples
        corrs: list[float] = []
        for sample in samples:
            attn_data = sample.get("attention", {})
            sample_patching = sample.get("patching", [])
            r = attention_vs_causal_correlation(
                attn_data, sample_patching, region, args.position, num_layers
            )
            if not isnan(r):
                corrs.append(r)
        mean_corr = safe_mean(corrs)
        corr_str = fmt(mean_corr) if not isnan(mean_corr) else "N/A".rjust(12)
        print(f"  {region:<22} {corr_str}")


if __name__ == "__main__":
    main()
