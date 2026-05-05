#!/usr/bin/env python3
"""N-variant comparison with auto-discovered regions and extended metrics.

Loads sample JSON files from variant directories, auto-discovers regions
and layer counts, and prints comparison tables for terminal attention,
per-token density, region ratios, logit lens trajectories, per-head
variance, gradient attribution, and patching summaries.

Usage:
    python -m promptastic.analysis.compare \\
        --base-dir ./data/results \\
        --variants results_baseline:Baseline results_v2:V2 \\
        --ratio conversation_turns:current_message \\
        --by-seed
"""

from __future__ import annotations

import argparse
import json
from math import isnan
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import FINAL_LAYERS, SKIP_REGIONS
from .formatting import delta_str, fmt, pct, print_header, print_subheader
from .metrics import (
    avg_final_layers,
    compute_per_head_attention,
    compute_gradient_trajectory,
    compute_region_ratio,
    safe_mean,
    safe_median,
    safe_std,
)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_variant(dirpath: Path) -> list[dict[str, Any]]:
    """Load all sample_*.json files from a variant directory, sorted."""
    samples: list[dict[str, Any]] = []
    for filepath in sorted(dirpath.glob("sample_*.json")):
        with open(filepath) as fh:
            samples.append(json.load(fh))
    return samples


def _auto_discover_regions(samples: list[dict[str, Any]]) -> list[str]:
    """Extract sorted region names from the first sample, skipping containers."""
    if not samples:
        return []
    region_map = samples[0].get("region_map", {})
    return sorted(name for name in region_map if name not in SKIP_REGIONS)


def _detect_num_layers(samples: list[dict[str, Any]]) -> int:
    """Infer layer count from the attention data in the first available sample."""
    for sample in samples:
        attn = sample.get("attention", {})
        for pos_data in attn.values():
            layer_entries = pos_data.get("per_layer", [])
            if layer_entries:
                return max(entry["layer"] for entry in layer_entries) + 1
    return 64


def _extract_seed(case_id: str) -> str:
    """Parse seed suffix from a case_id like 'sample_01_b' -> 'seed_b'."""
    parts = case_id.rsplit("_", 1)
    if len(parts) == 2 and len(parts[1]) == 1 and parts[1].isalpha():
        return f"seed_{parts[1]}"
    return "seed_default"


# ---------------------------------------------------------------------------
# Standard tables
# ---------------------------------------------------------------------------


def table_terminal_attention(
    variants: dict[str, list[dict[str, Any]]],
    regions: list[str],
    position: str,
    num_layers: int = 64,
) -> None:
    """Print per-region terminal attention with deltas from first variant."""
    layer_lo = num_layers - FINAL_LAYERS
    layer_hi = num_layers - 1
    print_header(
        f"Per-Region Terminal Attention ({position}, L{layer_lo}-{layer_hi} avg)"
    )

    labels = list(variants.keys())
    baseline_label = labels[0]
    baseline_means: dict[str, float] = {}

    # Header row
    row = f"  {'Region':<22}"
    for label in labels:
        row += f" {label:>12}"
    if len(labels) > 1:
        row += " |"
        for label in labels[1:]:
            row += f" {'d_' + label:>9}"
    print(row)

    # Separator
    sep = f"  {'---':^22}"
    for _ in labels:
        sep += f" {'---':^12}"
    if len(labels) > 1:
        sep += " |"
        for _ in labels[1:]:
            sep += f" {'---':^9}"
    print(sep)

    for region in regions:
        row = f"  {region:<22}"
        for label in labels:
            samples = variants[label]
            vals = [
                avg_final_layers(s.get("attention", s), position, region)
                for s in samples
            ]
            mean_val = safe_mean(vals)
            if label == baseline_label:
                baseline_means[region] = mean_val
            row += f" {pct(mean_val, 12)}"

        if len(labels) > 1:
            row += " |"
            for label in labels[1:]:
                vals = [
                    avg_final_layers(
                        s.get("attention", s), position, region
                    )
                    for s in variants[label]
                ]
                mean_val = safe_mean(vals)
                ref = baseline_means.get(region, float("nan"))
                row += f" {delta_str(mean_val, ref)}"
        print(row)


def table_region_ratios(
    variants: dict[str, list[dict[str, Any]]],
    ratio_pairs: list[tuple[str, str]],
    position: str,
    num_layers: int = 64,
) -> None:
    """Print ratio statistics for each region pair across variants."""
    layer_lo = num_layers - FINAL_LAYERS
    layer_hi = num_layers - 1
    for region_a, region_b in ratio_pairs:
        print_header(
            f"Region Ratio: {region_a} / {region_b} "
            f"({position}, L{layer_lo}-{layer_hi} avg)"
        )
        print(f"  Lower = less {region_a} relative to {region_b}.\n")

        print(
            f"  {'Variant':<14} {'n':>4} {'Mean':>9} {'Median':>9} "
            f"{'Std':>9} {'Min':>9} {'Max':>9}"
        )
        print(
            f"  {'---':<14} {'---':>4} {'---':>9} {'---':>9} "
            f"{'---':>9} {'---':>9} {'---':>9}"
        )

        for label, samples in variants.items():
            ratios = compute_region_ratio(
                samples, region_a, region_b, position
            )
            valid = [r for r in ratios if not isnan(r)]
            min_val = min(valid) if valid else float("nan")
            max_val = max(valid) if valid else float("nan")
            print(
                f"  {label:<14} {len(valid):>4} "
                f"{fmt(safe_mean(ratios))} {fmt(safe_median(ratios))} "
                f"{fmt(safe_std(ratios))} "
                f"{fmt(min_val)} {fmt(max_val)}"
            )


def table_per_token_density(
    variants: dict[str, list[dict[str, Any]]],
    regions: list[str],
    position: str,
    num_layers: int = 64,
) -> None:
    """Print per-token attention density (attention / n_tokens * 1000)."""
    layer_lo = num_layers - FINAL_LAYERS
    layer_hi = num_layers - 1
    print_header(
        f"Per-Token Attention Density ({position}, L{layer_lo}-{layer_hi} avg)"
    )
    print("  Density = attention_weight / n_tokens * 1000. Higher = more per token.\n")

    header = f"  {'Region':<22}"
    for label in variants:
        header += f" {label:>12}"
    print(header)
    print(f"  {'---':<22}" + "".join(f" {'---':>12}" for _ in variants))

    for region in regions:
        row = f"  {region:<22}"
        for label, samples in variants.items():
            densities: list[float] = []
            for s in samples:
                attn_val = avg_final_layers(
                    s.get("attention", s), position, region
                )
                rm = s.get("region_map", {}).get(region, {})
                n_tok = rm.get(
                    "n_tokens", rm.get("tok_end", 0) - rm.get("tok_start", 0)
                )
                if n_tok > 0 and not isnan(attn_val):
                    densities.append(attn_val / n_tok * 1000)
            row += f" {fmt(safe_mean(densities), 12)}"
        print(row)


def table_logit_lens(
    variants: dict[str, list[dict[str, Any]]],
    tracked_tokens: list[str],
    position: str,
    num_layers: int,
) -> None:
    """Print logit lens rank trajectories for tracked tokens."""
    if not tracked_tokens:
        return

    probe_layers = sorted(
        set(
            l
            for l in [0, 16, 32, 48, 56, num_layers - 4, num_layers - 1]
            if 0 <= l < num_layers
        )
    )

    for token_str in tracked_tokens:
        print_header(
            f"Logit Lens: '{token_str}' Rank Trajectory at {position}"
        )
        print("  Rank of token across layers. Lower = stronger prediction.\n")

        header = f"  {'Variant':<14}"
        for layer in probe_layers:
            header += f" {'L' + str(layer):>6}"
        print(header)
        print(
            f"  {'---':<14}"
            + "".join(f" {'---':>6}" for _ in probe_layers)
        )

        for label, samples in variants.items():
            collected: dict[int, list[float]] = {l: [] for l in probe_layers}
            for sample in samples:
                ll_entries = sample.get("logit_lens", {}).get(position, [])
                for entry in ll_entries:
                    if entry["layer"] in collected:
                        tracked = entry.get("tracked", {})
                        tok_info = tracked.get(token_str, {})
                        rank = tok_info.get("rank", float("nan"))
                        collected[entry["layer"]].append(rank)

            row = f"  {label:<14}"
            for layer in probe_layers:
                vals = collected[layer]
                avg = sum(vals) / len(vals) if vals else float("nan")
                if isnan(avg):
                    row += f" {'N/A':>6}"
                else:
                    row += f" {avg:6.0f}"
            print(row)


def table_by_seed(
    variants: dict[str, list[dict[str, Any]]],
    regions: list[str],
    position: str,
) -> None:
    """Print per-seed breakdown of terminal attention for multi-seed variants."""
    for label, samples in variants.items():
        seeds = sorted(
            set(
                _extract_seed(s["metadata"]["case_id"]) for s in samples
            )
        )
        if len(seeds) <= 1:
            continue

        print_header(f"{label} -- Per-Seed Terminal Attention")

        header = f"  {'Region':<22}"
        for seed in seeds:
            header += f" {seed:>12}"
        header += f" {'All':>12}"
        print(header)
        print(
            f"  {'---':<22}"
            + "".join(f" {'---':>12}" for _ in seeds)
            + f" {'---':>12}"
        )

        for region in regions:
            row = f"  {region:<22}"
            for seed in seeds:
                seed_samples = [
                    s
                    for s in samples
                    if _extract_seed(s["metadata"]["case_id"]) == seed
                ]
                vals = [
                    avg_final_layers(
                        s.get("attention", s), position, region
                    )
                    for s in seed_samples
                ]
                row += f" {pct(safe_mean(vals), 12)}"
            all_vals = [
                avg_final_layers(s.get("attention", s), position, region)
                for s in samples
            ]
            row += f" {pct(safe_mean(all_vals), 12)}"
            print(row)


# ---------------------------------------------------------------------------
# Extended tables: patching, per-head variance, gradient attribution
# ---------------------------------------------------------------------------


def table_patching_summary(
    variants: dict[str, list[dict[str, Any]]],
    regions: list[str],
    metric: str = "kl_divergence",
) -> None:
    """Print mean causal importance per region from activation patching data.

    Reads sample["patching"] which is a list of PatchingResult dicts.
    For each region, averages the chosen metric across layers and samples.
    """
    print_header(f"Patching Importance ({metric})")
    print("  Mean effect of zeroing region activations. Higher = more causal.\n")

    header = f"  {'Region':<22}"
    for label in variants:
        header += f" {label:>12}"
    print(header)
    print(f"  {'---':<22}" + "".join(f" {'---':>12}" for _ in variants))

    for region in regions:
        row = f"  {region:<22}"
        for label, samples in variants.items():
            all_effects: list[float] = []
            for sample in samples:
                patching_entries = sample.get("patching", [])
                region_entries = [
                    e for e in patching_entries if e.get("region") == region
                ]
                for entry in region_entries:
                    val = entry.get(metric, float("nan"))
                    if not isnan(val):
                        all_effects.append(val)
            row += f" {fmt(safe_mean(all_effects), 12)}"
        print(row)


def table_per_head_variance(
    variants: dict[str, list[dict[str, Any]]],
    regions: list[str],
    position: str,
    num_layers: int,
    num_heads: int,
) -> None:
    """Print variance of attention across heads for each region.

    High variance indicates head specialization; low variance means
    all heads attend similarly.
    """
    print_header(f"Per-Head Attention Variance ({position})")
    print("  Variance across heads (mean over layers and samples).\n")

    header = f"  {'Region':<22}"
    for label in variants:
        header += f" {label:>12}"
    print(header)
    print(f"  {'---':<22}" + "".join(f" {'---':>12}" for _ in variants))

    for region in regions:
        row = f"  {region:<22}"
        for label, samples in variants.items():
            sample_variances: list[float] = []
            for sample in samples:
                head_arr = compute_per_head_attention(
                    sample, region, position, num_layers, num_heads
                )
                # Variance across the head axis, then mean across layers
                per_layer_var = np.var(head_arr, axis=1)
                sample_variances.append(float(np.mean(per_layer_var)))
            row += f" {fmt(safe_mean(sample_variances), 12)}"
        print(row)


def table_gradient_attribution(
    variants: dict[str, list[dict[str, Any]]],
    regions: list[str],
    position: str,
    num_layers: int,
) -> None:
    """Print mean gradient attribution norm for each region.

    Summarizes gradient flow from output back through each region.
    """
    print_header(f"Gradient Attribution ({position})")
    print("  Mean gradient norm across layers and samples.\n")

    header = f"  {'Region':<22}"
    for label in variants:
        header += f" {label:>12}"
    print(header)
    print(f"  {'---':<22}" + "".join(f" {'---':>12}" for _ in variants))

    for region in regions:
        row = f"  {region:<22}"
        for label, samples in variants.items():
            norms: list[float] = []
            for sample in samples:
                traj = compute_gradient_trajectory(
                    sample, region, position, num_layers
                )
                mean_norm = float(np.mean(traj))
                if mean_norm > 0:
                    norms.append(mean_norm)
            row += f" {fmt(safe_mean(norms), 12)}"
        print(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the compare module CLI."""
    parser = argparse.ArgumentParser(
        description="N-variant MI comparison with auto-discovered regions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Base directory containing variant result directories",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        required=True,
        help="Variant specs as dirname:label (e.g., results_baseline:Baseline)",
    )
    parser.add_argument(
        "--position",
        default="terminal",
        help="Query position to analyze (default: terminal)",
    )
    parser.add_argument(
        "--ratio",
        nargs="*",
        default=[],
        help="Region ratio pairs as region_a:region_b",
    )
    parser.add_argument(
        "--logit-lens-tokens",
        nargs="*",
        default=[],
        help="Tokens to track in logit lens trajectory",
    )
    parser.add_argument(
        "--by-seed",
        action="store_true",
        help="Show per-seed breakdown for stability analysis",
    )
    parser.add_argument(
        "--metrics",
        default="all",
        help=(
            "Comma-separated metrics: terminal,density,ratios,logit_lens,"
            "patching,head_variance,gradient,all"
        ),
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=32,
        help="Number of attention heads per layer (default: 32)",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    requested = set(args.metrics.split(","))
    show_all = "all" in requested

    # Parse and load variants
    variants: dict[str, list[dict[str, Any]]] = {}
    for spec in args.variants:
        if ":" in spec:
            dirname, label = spec.split(":", 1)
        else:
            dirname = spec
            label = spec.replace("results_", "")
        dirpath = base_dir / dirname
        if not dirpath.exists():
            print(f"  SKIP {label}: {dirpath} not found")
            continue
        samples = _load_variant(dirpath)
        variants[label] = samples
        print(f"  Loaded {label}: {len(samples)} samples from {dirpath}")

    if not variants:
        print("No variant data found.")
        return

    first_samples = next(iter(variants.values()))
    regions = _auto_discover_regions(first_samples)
    num_layers = _detect_num_layers(first_samples)
    print(f"  Discovered {len(regions)} regions, {num_layers} layers")

    # Parse ratio pairs
    ratio_pairs: list[tuple[str, str]] = []
    for r in args.ratio:
        parts = r.split(":")
        if len(parts) == 2:
            ratio_pairs.append((parts[0], parts[1]))

    # Print requested tables
    if show_all or "terminal" in requested:
        table_terminal_attention(variants, regions, args.position, num_layers)

    if show_all or "density" in requested:
        table_per_token_density(variants, regions, args.position, num_layers)

    if show_all or "ratios" in requested:
        if ratio_pairs:
            table_region_ratios(
                variants, ratio_pairs, args.position, num_layers
            )

    if show_all or "logit_lens" in requested:
        table_logit_lens(
            variants, args.logit_lens_tokens, args.position, num_layers
        )

    if show_all or "patching" in requested:
        table_patching_summary(variants, regions)

    if show_all or "head_variance" in requested:
        table_per_head_variance(
            variants, regions, args.position, num_layers, args.num_heads
        )

    if show_all or "gradient" in requested:
        table_gradient_attribution(
            variants, regions, args.position, num_layers
        )

    if args.by_seed:
        table_by_seed(variants, regions, args.position)


if __name__ == "__main__":
    main()
