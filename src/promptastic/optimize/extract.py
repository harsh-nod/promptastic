"""Metric extraction from analysis results.

Converts raw ``analyze_case()`` result dicts into a flat
``dict[str, float]`` of optimisation-relevant metrics by calling into
the existing ``analysis`` functions.
"""

from __future__ import annotations

from math import isnan
from typing import Any

import numpy as np

from ..analysis.metrics import (
    avg_final_layers,
    compute_per_head_attention,
    compute_per_token_density,
    compute_region_attention_per_layer,
    compute_region_ratio,
    cooking_curve_stats,
    safe_mean,
)
from ..constants import SKIP_REGIONS


def _usable_regions(results: list[dict[str, Any]]) -> list[str]:
    """Collect region names present in results, excluding containers."""
    names: set[str] = set()
    for r in results:
        for name in r.get("region_map", {}):
            if name not in SKIP_REGIONS:
                names.add(name)
    return sorted(names)


def extract_attention_metrics(
    results: list[dict[str, Any]],
    regions: list[str],
    position: str = "terminal",
) -> dict[str, float]:
    """Category A: attention routing efficiency metrics."""
    metrics: dict[str, float] = {}

    # Per-region terminal attention
    for region in regions:
        vals = [
            avg_final_layers(r.get("attention", {}), position, region)
            for r in results
        ]
        metrics[f"terminal_attention_{region}"] = safe_mean(vals)

    # Context bleed ratio (only if both regions exist)
    ratios = compute_region_ratio(
        results, "conversation_turns", "current_message", position,
    )
    if ratios:
        metrics["context_bleed_ratio"] = safe_mean(ratios)

    # Per-token density coefficient of variation
    densities: dict[str, float] = {}
    for region in regions:
        region_densities: list[float] = []
        for r in results:
            attn = avg_final_layers(r.get("attention", {}), position, region)
            rm = r.get("region_map", {}).get(region, {})
            n_tok = rm.get("n_tokens", rm.get("tok_end", 0) - rm.get("tok_start", 0))
            if n_tok > 0 and not isnan(attn):
                region_densities.append(compute_per_token_density(attn, n_tok))
        d = safe_mean(region_densities)
        if not isnan(d):
            densities[region] = d

    valid = [v for v in densities.values() if v > 0]
    if len(valid) >= 2:
        metrics["density_cv"] = float(np.std(valid) / np.mean(valid))

    return metrics


def extract_dynamics_metrics(
    results: list[dict[str, Any]],
    regions: list[str],
    num_layers: int,
    position: str = "terminal",
) -> dict[str, float]:
    """Category B: processing dynamics (cooking curve) metrics."""
    metrics: dict[str, float] = {}

    for region in regions:
        curves: list[np.ndarray] = []
        for r in results:
            c = compute_region_attention_per_layer(r, region, position, num_layers)
            if np.any(c > 0):
                curves.append(c)

        if not curves:
            continue

        mean_curve = np.mean(curves, axis=0)
        stats = cooking_curve_stats(mean_curve)

        metrics[f"peak_layer_frac_{region}"] = (
            stats["peak_layer"] / max(num_layers - 1, 1)
        )
        metrics[f"retention_ratio_{region}"] = stats["retention_ratio"]
        metrics[f"peak_value_{region}"] = stats["peak_value"]

    return metrics


def extract_causal_metrics(
    results: list[dict[str, Any]],
    regions: list[str],
    num_layers: int,
    position: str = "terminal",
) -> dict[str, float]:
    """Category C: causal importance metrics (requires patching data)."""
    from ..analysis.causal import (
        attention_vs_causal_correlation,
        causal_importance_score,
        find_critical_layers,
    )

    metrics: dict[str, float] = {}

    # Collect all patching entries
    all_patching: list[dict[str, Any]] = []
    for r in results:
        patching = r.get("patching", {})
        if isinstance(patching, dict):
            all_patching.extend(patching.get("results", []))
        elif isinstance(patching, list):
            all_patching.extend(patching)

    if not all_patching:
        return metrics

    for region in regions:
        score = causal_importance_score(all_patching, region)
        if not isnan(score):
            metrics[f"causal_importance_{region}"] = score

        critical = find_critical_layers(all_patching, region)
        metrics[f"critical_layer_spread_{region}"] = float(len(critical))

    # Attention-causal correlation
    for region in regions:
        corrs: list[float] = []
        for r in results:
            attn_data = r.get("attention", {})
            patching = r.get("patching", {})
            entries = (
                patching.get("results", [])
                if isinstance(patching, dict)
                else patching
            )
            c = attention_vs_causal_correlation(
                attn_data, entries, region, position, num_layers,
            )
            if not isnan(c):
                corrs.append(c)
        if corrs:
            metrics[f"attn_causal_corr_{region}"] = safe_mean(corrs)

    return metrics


def extract_head_metrics(
    results: list[dict[str, Any]],
    regions: list[str],
    num_layers: int,
    num_heads: int,
    position: str = "terminal",
) -> dict[str, float]:
    """Category D: head specialization metrics (requires per-head data)."""
    from ..analysis.head_analysis import (
        find_specialist_heads,
        head_variance_by_region,
    )

    metrics: dict[str, float] = {}

    # Build aggregated per-head data
    region_arrays: dict[str, list[np.ndarray]] = {r: [] for r in regions}
    for r in results:
        for region in regions:
            arr = compute_per_head_attention(
                r, region, position, num_layers, num_heads,
            )
            if np.any(arr > 0):
                region_arrays[region].append(arr)

    avg_per_head: dict[str, np.ndarray] = {}
    for region in regions:
        arrs = region_arrays[region]
        if arrs:
            avg_per_head[region] = np.mean(arrs, axis=0)
        else:
            avg_per_head[region] = np.zeros((num_layers, num_heads))

    if not any(np.any(v > 0) for v in avg_per_head.values()):
        return metrics

    # Head variance per region
    variance_info = head_variance_by_region(
        avg_per_head, regions, num_layers, num_heads,
    )
    for region in regions:
        metrics[f"head_variance_{region}"] = variance_info[region]["mean_variance"]

    # Specialist head count per region
    specialists = find_specialist_heads(
        avg_per_head, regions, num_layers, num_heads,
    )
    counts: dict[str, int] = {r: 0 for r in regions}
    for spec in specialists:
        dom = spec.get("dominant_region", "")
        if dom in counts:
            counts[dom] += 1
    for region in regions:
        metrics[f"specialist_head_count_{region}"] = float(counts[region])

    return metrics


def extract_all_metrics(
    results: list[dict[str, Any]],
    num_layers: int,
    num_heads: int = 32,
    position: str = "terminal",
    regions: list[str] | None = None,
    has_patching: bool = False,
    has_per_head: bool = False,
) -> dict[str, float]:
    """Extract all optimization metrics from analysis results.

    Calls per-category extractors and merges into a single dict.
    """
    if regions is None:
        regions = _usable_regions(results)

    metrics: dict[str, float] = {}
    metrics.update(extract_attention_metrics(results, regions, position))
    metrics.update(extract_dynamics_metrics(results, regions, num_layers, position))

    if has_patching:
        metrics.update(extract_causal_metrics(results, regions, num_layers, position))

    if has_per_head:
        metrics.update(extract_head_metrics(
            results, regions, num_layers, num_heads, position,
        ))

    return metrics
