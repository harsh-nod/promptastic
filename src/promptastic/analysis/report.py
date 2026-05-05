#!/usr/bin/env python3
"""Markdown experiment report generator with baseline delta analysis.

Produces per-experiment markdown reports containing context bleed ratios,
cooking curve tables, delta-from-baseline sections, and extended causal
importance, head specialization, and attention-vs-gradient comparisons.

Usage:
    python -m promptastic.analysis.report \\
        --base-dir ./data/results \\
        --experiments baseline:Baseline:results_baseline v2:V2:results_v2 \\
        --output-dir ./reports
"""

from __future__ import annotations

import argparse
import json
from math import isnan
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import FINAL_LAYERS, SKIP_REGIONS, analysis_phases
from .._types import ContextBleedResult, ExtendedCookingStats
from .metrics import (
    compute_gradient_trajectory,
    compute_per_head_attention,
    compute_region_attention_per_layer,
    cooking_curve_stats,
    safe_mean,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_samples(dirpath: Path) -> list[dict[str, Any]]:
    """Load all sample_*.json files from a directory."""
    samples: list[dict[str, Any]] = []
    for filepath in sorted(dirpath.glob("sample_*.json")):
        with open(filepath) as fh:
            samples.append(json.load(fh))
    return samples


def _auto_regions(samples: list[dict[str, Any]]) -> list[str]:
    """Sorted region names from first sample, excluding containers."""
    if not samples:
        return []
    rm = samples[0].get("region_map", {})
    return sorted(name for name in rm if name not in SKIP_REGIONS)


def _detect_num_layers(samples: list[dict[str, Any]]) -> int:
    """Infer layer count from per_token_attention data."""
    for s in samples:
        pt = s.get("per_token_attention", {})
        for pos_data in pt.values():
            entries = pos_data.get("per_layer", [])
            if entries:
                return max(e["layer"] for e in entries) + 1
    return 64


def _classify_story(
    peak_layer: int, ratio: float, num_layers: int = 64
) -> str:
    """Generate a narrative label from cooking curve shape.

    peak_layer: where the curve peaks
    ratio: peak_value / terminal_value
    num_layers: total layer count for fractional scaling
    """
    if num_layers <= 1:
        return "Single layer"
    frac = peak_layer / (num_layers - 1)

    if frac <= 0.04:
        if ratio > 10:
            return "Immediate peak, strong fade"
        if ratio > 3:
            return "Immediate peak"
        return "Early read"
    if frac <= 0.12:
        return f"Absorbed by L{peak_layer}"
    if frac <= 0.20:
        return "Absorption phase"
    if frac <= 0.50:
        return "Deep compression"
    if frac <= 0.75:
        return "Mid-phase peak"
    if frac <= 0.88:
        return "Output prep phase"
    if ratio < 2:
        return "Latest bloomer"
    return "Late bloomer"


# ---------------------------------------------------------------------------
# Core report computations
# ---------------------------------------------------------------------------


def compute_cooking_table(
    samples: list[dict[str, Any]],
    regions: list[str],
    position: str = "terminal",
    num_layers: int = 64,
) -> dict[str, ExtendedCookingStats]:
    """Compute cooking curve statistics for every region across samples.

    Averages per-layer trajectories across samples, then computes peak,
    terminal, ratio, and narrative classification.
    """
    table: dict[str, ExtendedCookingStats] = {}

    for region_name in regions:
        trajectories: list[np.ndarray] = []
        for sample in samples:
            traj = compute_region_attention_per_layer(
                sample, region_name, position, num_layers
            )
            if np.any(traj > 0):
                trajectories.append(traj)

        if not trajectories:
            continue

        avg_traj = np.mean(trajectories, axis=0)
        stats = cooking_curve_stats(avg_traj)

        pt_ratio = (
            stats["peak_value"] / stats["terminal_value"]
            if stats["terminal_value"] > 0
            else float("inf")
        )
        story = _classify_story(stats["peak_layer"], pt_ratio, num_layers)

        table[region_name] = {
            **stats,
            "peak_terminal_ratio": pt_ratio,
            "story": story,
            "n_samples": len(trajectories),
            "trajectory": avg_traj.tolist(),
        }

    return table


def compute_context_bleed(
    samples: list[dict[str, Any]],
    conv_region: str = "conversation_turns",
    curr_region: str = "current_message",
    position: str = "terminal",
    num_layers: int = 64,
) -> ContextBleedResult:
    """Measure how much conversation history dominates over current message.

    Computes terminal-layer ratio of conv_region / curr_region attention
    across all samples.
    """
    ratios: list[float] = []
    conv_vals: list[float] = []
    curr_vals: list[float] = []

    for sample in samples:
        conv_traj = compute_region_attention_per_layer(
            sample, conv_region, position, num_layers
        )
        curr_traj = compute_region_attention_per_layer(
            sample, curr_region, position, num_layers
        )
        conv_term = float(conv_traj[-1])
        curr_term = float(curr_traj[-1])

        if curr_term > 0:
            ratios.append(conv_term / curr_term)
        conv_vals.append(conv_term)
        curr_vals.append(curr_term)

    return {
        "mean_ratio": float(np.mean(ratios)) if ratios else 0.0,
        "median_ratio": float(np.median(ratios)) if ratios else 0.0,
        "pct_above_2x": (
            float(np.mean([r > 2 for r in ratios]) * 100) if ratios else 0.0
        ),
        "conv_turns_mean": float(np.mean(conv_vals)) if conv_vals else 0.0,
        "current_message_mean": float(np.mean(curr_vals)) if curr_vals else 0.0,
        "n_samples": len(ratios),
    }


# ---------------------------------------------------------------------------
# Extended report sections
# ---------------------------------------------------------------------------


def _section_causal_importance(
    samples: list[dict[str, Any]],
    regions: list[str],
    num_layers: int,
) -> list[str]:
    """Build markdown lines for a Causal Importance section from patching data."""
    lines: list[str] = []
    has_patching = any(s.get("patching") for s in samples)
    if not has_patching:
        return lines

    lines.append("## Causal Importance (Activation Patching)")
    lines.append("")
    lines.append(
        "| Region | Mean KL Divergence | Mean Logit Diff | Critical Layers |"
    )
    lines.append("|--------|-------------------|-----------------|-----------------|")

    phases = analysis_phases(num_layers)

    for region in regions:
        kl_vals: list[float] = []
        ld_vals: list[float] = []
        layer_effects: dict[int, list[float]] = {}

        for sample in samples:
            for entry in sample.get("patching", []):
                if entry.get("region") != region:
                    continue
                kl = entry.get("kl_divergence", float("nan"))
                ld = entry.get("logit_diff", float("nan"))
                layer = entry.get("layer", -1)
                if not isnan(kl):
                    kl_vals.append(kl)
                if not isnan(ld):
                    ld_vals.append(ld)
                if layer >= 0 and not isnan(kl):
                    layer_effects.setdefault(layer, []).append(kl)

        if not kl_vals:
            continue

        mean_kl = safe_mean(kl_vals)
        mean_ld = safe_mean(ld_vals)

        # Find layers where mean KL > overall_mean * 2
        threshold = mean_kl * 2
        critical = sorted(
            l
            for l, vals in layer_effects.items()
            if safe_mean(vals) > threshold
        )
        critical_str = ", ".join(f"L{l}" for l in critical[:6]) or "none"

        lines.append(
            f"| {region} | {mean_kl:.4f} | {mean_ld:.4f} | {critical_str} |"
        )

    lines.append("")
    return lines


def _section_head_specialization(
    samples: list[dict[str, Any]],
    regions: list[str],
    num_layers: int,
    num_heads: int = 32,
) -> list[str]:
    """Build markdown lines for a Head Specialization section."""
    lines: list[str] = []
    has_head_data = any(s.get("per_head_attention") for s in samples)
    if not has_head_data:
        return lines

    lines.append("## Head Specialization")
    lines.append("")
    lines.append(
        "| Region | Mean Head Variance | Max Variance Layer | Specialist Heads |"
    )
    lines.append("|--------|-------------------|-------------------|-----------------|")

    for region in regions:
        all_var_per_layer: list[np.ndarray] = []
        for sample in samples:
            head_arr = compute_per_head_attention(
                sample, region, "terminal", num_layers, num_heads
            )
            if np.any(head_arr > 0):
                all_var_per_layer.append(np.var(head_arr, axis=1))

        if not all_var_per_layer:
            continue

        avg_var = np.mean(all_var_per_layer, axis=0)
        mean_var = float(np.mean(avg_var))
        max_var_layer = int(np.argmax(avg_var))

        # Count heads with high individual contribution at terminal layers
        specialist_count = 0
        for sample in samples:
            head_arr = compute_per_head_attention(
                sample, region, "terminal", num_layers, num_heads
            )
            terminal_heads = head_arr[-FINAL_LAYERS:, :]
            if terminal_heads.size > 0:
                head_means = np.mean(terminal_heads, axis=0)
                overall_mean = np.mean(head_means)
                if overall_mean > 0:
                    specialist_count += int(
                        np.sum(head_means > overall_mean * 2)
                    )

        avg_specialists = (
            specialist_count / len(samples) if samples else 0
        )

        lines.append(
            f"| {region} | {mean_var:.6f} | L{max_var_layer:02d} | "
            f"~{avg_specialists:.0f} |"
        )

    lines.append("")
    return lines


def _section_attention_vs_gradient(
    samples: list[dict[str, Any]],
    regions: list[str],
    num_layers: int,
) -> list[str]:
    """Build markdown lines comparing attention trajectories to gradient norms."""
    lines: list[str] = []
    has_grads = any(s.get("gradients") for s in samples)
    if not has_grads:
        return lines

    lines.append("## Attention vs Gradient Attribution")
    lines.append("")
    lines.append(
        "| Region | Attn Peak Layer | Grad Peak Layer | "
        "Layer Gap | Correlation |"
    )
    lines.append(
        "|--------|----------------|----------------|----------|------------|"
    )

    for region in regions:
        attn_trajs: list[np.ndarray] = []
        grad_trajs: list[np.ndarray] = []
        for sample in samples:
            attn_t = compute_region_attention_per_layer(
                sample, region, "terminal", num_layers
            )
            grad_t = compute_gradient_trajectory(
                sample, region, "terminal", num_layers
            )
            if np.any(attn_t > 0):
                attn_trajs.append(attn_t)
            if np.any(grad_t > 0):
                grad_trajs.append(grad_t)

        if not attn_trajs or not grad_trajs:
            continue

        avg_attn = np.mean(attn_trajs, axis=0)
        avg_grad = np.mean(grad_trajs, axis=0)

        attn_peak = int(np.argmax(avg_attn))
        grad_peak = int(np.argmax(avg_grad))
        gap = abs(attn_peak - grad_peak)

        # Pearson correlation between the two trajectories
        if np.std(avg_attn) > 0 and np.std(avg_grad) > 0:
            corr = float(np.corrcoef(avg_attn, avg_grad)[0, 1])
        else:
            corr = float("nan")

        corr_str = f"{corr:.3f}" if not isnan(corr) else "N/A"
        lines.append(
            f"| {region} | L{attn_peak:02d} | L{grad_peak:02d} | "
            f"{gap} | {corr_str} |"
        )

    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def write_experiment_report(
    exp_key: str,
    exp_label: str,
    samples: list[dict[str, Any]],
    regions: list[str],
    baseline_stats: dict[str, ExtendedCookingStats],
    baseline_bleed: dict[str, Any],
    num_layers: int,
    output_dir: Path,
) -> Path:
    """Write a full experiment report to a markdown file.

    Includes context bleed, cooking curves, delta from baseline,
    causal importance, head specialization, and attention vs gradient.
    """
    cooking = compute_cooking_table(samples, regions, num_layers=num_layers)
    bleed = compute_context_bleed(samples, num_layers=num_layers)

    report_path = output_dir / f"{exp_key}.md"
    lines: list[str] = []

    # Title
    lines.append(f"# {exp_label}")
    lines.append("")
    lines.append(f"**Samples analyzed**: {len(samples)}")
    lines.append("")

    # Context Bleed
    lines.append("## Context Bleed")
    lines.append("")
    lines.append("| Metric | Value | Baseline | Delta |")
    lines.append("|--------|-------|----------|-------|")

    if baseline_bleed:
        bl_mean = baseline_bleed.get("mean_ratio", 0)
        bl_median = baseline_bleed.get("median_ratio", 0)
        bl_pct = baseline_bleed.get("pct_above_2x", 0)

        d_mean = bleed["mean_ratio"] - bl_mean
        d_mean_pct = (d_mean / bl_mean * 100) if bl_mean > 0 else 0
        lines.append(
            f"| Mean conv/curr ratio | {bleed['mean_ratio']:.2f}x | "
            f"{bl_mean:.2f}x | {d_mean:+.2f}x ({d_mean_pct:+.1f}%) |"
        )
        lines.append(
            f"| Median | {bleed['median_ratio']:.2f}x | "
            f"{bl_median:.2f}x | "
            f"{bleed['median_ratio'] - bl_median:+.2f}x |"
        )
        lines.append(
            f"| Samples >2x | {bleed['pct_above_2x']:.0f}% | "
            f"{bl_pct:.0f}% | "
            f"{bleed['pct_above_2x'] - bl_pct:+.0f}pp |"
        )
    else:
        lines.append(
            f"| Mean conv/curr ratio | {bleed['mean_ratio']:.2f}x | -- | -- |"
        )
    lines.append("")

    # Cooking Curves
    lines.append("## Region Cooking Curves")
    lines.append("")
    lines.append(
        "| Region | Peak Layer | Peak Attn | Terminal Attn "
        "| Peak/Terminal | Story |"
    )
    lines.append(
        "|--------|-----------|-----------|-------------|"
        "--------------|-------|"
    )
    for region_name in regions:
        if region_name not in cooking:
            continue
        s = cooking[region_name]
        lines.append(
            f"| {region_name} | L{s['peak_layer']:02d} | "
            f"{s['peak_value']:.6f} | {s['terminal_value']:.6f} | "
            f"{s['peak_terminal_ratio']:.1f}x | {s['story']} |"
        )
    lines.append("")

    # Delta from Baseline
    if exp_key != "baseline" and baseline_stats:
        lines.append("## Delta from Baseline")
        lines.append("")
        lines.append(
            "| Region | Peak Layer Delta | Terminal Attn Delta "
            "| Interpretation |"
        )
        lines.append(
            "|--------|-----------------|--------------------"
            "|----------------|"
        )

        for region_name in regions:
            if region_name not in cooking:
                if region_name in baseline_stats:
                    lines.append(
                        f"| {region_name} | -- | -- | **REMOVED** |"
                    )
                continue
            if region_name not in baseline_stats:
                lines.append(
                    f"| {region_name} | -- | -- | **NEW** |"
                )
                continue

            bl = baseline_stats[region_name]
            ex = cooking[region_name]
            layer_delta = ex["peak_layer"] - bl["peak_layer"]
            term_delta_pct = (
                (ex["terminal_value"] - bl["terminal_value"])
                / bl["terminal_value"]
                * 100
                if bl["terminal_value"] > 0
                else 0.0
            )

            layer_str = f"+{layer_delta}" if layer_delta > 0 else str(layer_delta)
            if abs(term_delta_pct) < 10:
                interp = "Stable"
            elif term_delta_pct > 0:
                interp = "Gained"
            else:
                interp = "Lost"

            lines.append(
                f"| {region_name} | {layer_str} | "
                f"{term_delta_pct:+.1f}% | {interp} |"
            )
        lines.append("")

    # Extended sections (only emitted when data is present)
    lines.extend(
        _section_causal_importance(samples, regions, num_layers)
    )
    lines.extend(
        _section_head_specialization(samples, regions, num_layers)
    )
    lines.extend(
        _section_attention_vs_gradient(samples, regions, num_layers)
    )

    # Raw trajectories
    lines.append("## Raw Trajectories (JSON)")
    lines.append("")
    lines.append("<details>")
    lines.append("<summary>Click to expand</summary>")
    lines.append("")
    lines.append("```json")
    traj_data: dict[str, Any] = {}
    for region_name, stats in cooking.items():
        traj_data[region_name] = {
            "peak_layer": stats["peak_layer"],
            "peak_value": stats["peak_value"],
            "terminal_value": stats["terminal_value"],
            "trajectory": [round(v, 8) for v in stats["trajectory"]],
        }
    lines.append(json.dumps(traj_data, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("</details>")

    report_path.write_text("\n".join(lines))
    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the report module CLI."""
    parser = argparse.ArgumentParser(
        description="Generate markdown experiment reports",
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Base directory containing experiment result directories",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Experiment specs as key:label:dirname triples",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for reports (default: base-dir/reports)",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = (
        Path(args.output_dir) if args.output_dir else base_dir / "reports"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse experiment specs
    experiments: list[dict[str, str]] = []
    for spec in args.experiments:
        parts = spec.split(":")
        if len(parts) >= 3:
            experiments.append(
                {"key": parts[0], "label": parts[1], "dirname": parts[2]}
            )
        elif len(parts) == 2:
            experiments.append(
                {"key": parts[0], "label": parts[1], "dirname": parts[0]}
            )
        else:
            experiments.append(
                {"key": parts[0], "label": parts[0], "dirname": parts[0]}
            )

    # Load and analyze all experiments
    all_cooking: dict[str, dict[str, ExtendedCookingStats]] = {}
    all_bleed: dict[str, ContextBleedResult] = {}
    all_samples: dict[str, list[dict[str, Any]]] = {}

    for exp in experiments:
        exp_path = base_dir / exp["dirname"]
        samples = _load_samples(exp_path)
        if not samples:
            print(f"SKIP {exp['key']}: no samples in {exp_path}")
            continue

        all_samples[exp["key"]] = samples
        regions = _auto_regions(samples)
        num_layers = _detect_num_layers(samples)

        cooking = compute_cooking_table(
            samples, regions, num_layers=num_layers
        )
        bleed = compute_context_bleed(samples, num_layers=num_layers)

        all_cooking[exp["key"]] = cooking
        all_bleed[exp["key"]] = bleed
        print(
            f"  Analyzed {exp['key']}: {len(samples)} samples, "
            f"{len(regions)} regions"
        )

    if not all_samples:
        print("No experiment data found.")
        return

    # First experiment is the baseline
    baseline_key = experiments[0]["key"]
    baseline_stats = all_cooking.get(baseline_key, {})
    baseline_bleed: dict[str, Any] = all_bleed.get(baseline_key, {})

    # Write individual reports
    for exp in experiments:
        if exp["key"] not in all_samples:
            continue
        samples = all_samples[exp["key"]]
        regions = _auto_regions(samples)
        num_layers = _detect_num_layers(samples)

        report_path = write_experiment_report(
            exp["key"],
            exp["label"],
            samples,
            regions,
            baseline_stats,
            baseline_bleed,
            num_layers,
            output_dir,
        )
        print(f"  Written: {report_path}")

    print(f"\nAll reports in: {output_dir}")


if __name__ == "__main__":
    main()
