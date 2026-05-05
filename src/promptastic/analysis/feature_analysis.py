#!/usr/bin/env python3
"""SAE (Sparse Autoencoder) feature analysis.

Identifies top-activating SAE features for prompt regions and computes
feature overlap between region pairs using Jaccard similarity.

Usage:
    python -m promptastic.analysis.feature_analysis \\
        --base-dir ./data/results \\
        --variant results_baseline \\
        --top-k 10
"""

from __future__ import annotations

import argparse
import json
from math import isnan
from pathlib import Path
from typing import Any

from ..constants import SKIP_REGIONS
from .formatting import fmt, print_header
from .metrics import safe_mean


def top_features_for_region(
    sae_data: dict[str, Any],
    region: str,
    position: str,
    num_layers: int,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Find the top-k SAE features for a region by mean activation.

    sae_data is structured as:
      sae_data[position]["per_layer"] -> list of {layer, region_features: {region: [SAEFeatureActivation]}}

    Returns a list of dicts with feature_idx, mean_activation, label,
    and layers (list of layer indices where it activates).
    Sorted by mean_activation descending.
    """
    feature_totals: dict[int, list[float]] = {}
    feature_labels: dict[int, str] = {}
    feature_layers: dict[int, set[int]] = {}

    layer_entries = sae_data.get(position, {}).get("per_layer", [])

    for entry in layer_entries:
        layer_idx = entry.get("layer", -1)
        region_features = entry.get("region_features", {})
        activations = region_features.get(region, [])

        for feat in activations:
            fidx = feat.get("feature_idx", -1)
            act = feat.get("activation", 0.0)
            label = feat.get("label", "")

            if fidx < 0 or isnan(act):
                continue

            feature_totals.setdefault(fidx, []).append(act)
            if label:
                feature_labels[fidx] = label
            feature_layers.setdefault(fidx, set()).add(layer_idx)

    # Rank by mean activation
    ranked: list[dict[str, Any]] = []
    for fidx, vals in feature_totals.items():
        ranked.append({
            "feature_idx": fidx,
            "mean_activation": safe_mean(vals),
            "label": feature_labels.get(fidx, ""),
            "layers": sorted(feature_layers.get(fidx, set())),
            "n_activations": len(vals),
        })

    ranked.sort(key=lambda x: x["mean_activation"], reverse=True)
    return ranked[:top_k]


def feature_overlap(
    sae_data: dict[str, Any],
    region_a: str,
    region_b: str,
    position: str,
    top_k: int = 20,
) -> float:
    """Jaccard similarity of top-k feature sets between two regions.

    Finds the top_k features for each region, then computes
    |intersection| / |union|. Returns 0.0 if both sets are empty.
    """
    # Reuse top_features_for_region to get feature index sets
    # We need num_layers but it is only used for iteration, so pass a large value
    # and let the function handle whatever is in the data
    feats_a = top_features_for_region(sae_data, region_a, position, 256, top_k)
    feats_b = top_features_for_region(sae_data, region_b, position, 256, top_k)

    set_a = {f["feature_idx"] for f in feats_a}
    set_b = {f["feature_idx"] for f in feats_b}

    if not set_a and not set_b:
        return 0.0

    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for SAE feature analysis CLI."""
    parser = argparse.ArgumentParser(
        description="SAE feature analysis for prompt regions",
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
        "--top-k",
        type=int,
        default=10,
        help="Number of top features to show per region (default: 10)",
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

    # Discover regions
    region_map = samples[0].get("region_map", {})
    region_names = sorted(r for r in region_map if r not in SKIP_REGIONS)

    # Check for SAE data
    has_sae = any(s.get("sae_features") for s in samples)
    if not has_sae:
        print("No SAE feature data found in samples.")
        return

    print(f"  Loaded {len(samples)} samples, {len(region_names)} regions")

    # Top features per region (aggregate across samples)
    for region in region_names:
        print_header(f"Top {args.top_k} SAE Features: {region}")

        # Merge SAE data across samples
        merged_layers: dict[int, dict[str, list[dict]]] = {}
        for sample in samples:
            sae = sample.get("sae_features", {})
            layer_entries = sae.get(args.position, {}).get("per_layer", [])
            for entry in layer_entries:
                layer_idx = entry.get("layer", -1)
                region_features = entry.get("region_features", {})
                feats = region_features.get(region, [])
                if layer_idx not in merged_layers:
                    merged_layers[layer_idx] = {}
                merged_layers[layer_idx].setdefault(region, []).extend(feats)

        # Rebuild as single sae_data structure
        merged_sae: dict[str, Any] = {
            args.position: {
                "per_layer": [
                    {
                        "layer": layer_idx,
                        "region_features": region_feats,
                    }
                    for layer_idx, region_feats in sorted(
                        merged_layers.items()
                    )
                ]
            }
        }

        top = top_features_for_region(
            merged_sae, region, args.position, 256, args.top_k
        )

        if not top:
            print("  No features found.")
            continue

        print(
            f"  {'Rank':>4} {'Feature':>8} {'Mean Act':>10} "
            f"{'Layers':>8} {'Label'}"
        )
        print(
            f"  {'---':>4} {'---':>8} {'---':>10} {'---':>8} {'---'}"
        )
        for i, feat in enumerate(top, 1):
            layer_str = f"{len(feat['layers'])}L"
            print(
                f"  {i:>4} F{feat['feature_idx']:>6} "
                f"{fmt(feat['mean_activation'], 10)} "
                f"{layer_str:>8} {feat['label']}"
            )

    # Feature overlap matrix
    if len(region_names) >= 2:
        print_header("Feature Overlap (Jaccard Similarity)")

        # Merge all SAE data for overlap computation
        all_merged_layers: dict[int, dict[str, list[dict]]] = {}
        for sample in samples:
            sae = sample.get("sae_features", {})
            for entry in sae.get(args.position, {}).get("per_layer", []):
                layer_idx = entry.get("layer", -1)
                for rname, feats in entry.get("region_features", {}).items():
                    if layer_idx not in all_merged_layers:
                        all_merged_layers[layer_idx] = {}
                    all_merged_layers[layer_idx].setdefault(
                        rname, []
                    ).extend(feats)

        all_merged_sae: dict[str, Any] = {
            args.position: {
                "per_layer": [
                    {"layer": l, "region_features": rf}
                    for l, rf in sorted(all_merged_layers.items())
                ]
            }
        }

        # Print upper triangle of overlap matrix
        max_name = max(len(r) for r in region_names)
        header = f"  {'':>{max_name}}"
        for r in region_names:
            header += f" {r[:8]:>8}"
        print(header)

        for i, r_a in enumerate(region_names):
            row = f"  {r_a:>{max_name}}"
            for j, r_b in enumerate(region_names):
                if j < i:
                    row += f" {'':>8}"
                elif j == i:
                    row += f" {'1.000':>8}"
                else:
                    overlap = feature_overlap(
                        all_merged_sae, r_a, r_b, args.position, args.top_k
                    )
                    row += f" {overlap:>8.3f}"
            print(row)


if __name__ == "__main__":
    main()
