#!/usr/bin/env python3
"""SAE feature visualization dashboard.

Multi-panel image:
1. Top-K features bar chart (sorted by activation)
2. Feature activation trajectories across layers
3. Feature co-occurrence matrix (features x features)

Usage:
    python -m promptastic.render.feature_dashboard --result r.json
    python -m promptastic.render.feature_dashboard --result r.json --top-k 15
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ..constants import display_phases
from ._shared import (
    BG_COLOR,
    GRID_COLOR,
    TEXT_COLOR,
    TEXT_DIM,
    REGION_PALETTE,
    get_colormap,
    get_font,
    text_color_for_bg,
)
from .loaders import load_sae_data


# ============================================================================
# HELPERS
# ============================================================================

def _collect_feature_info(
    sae_data: dict[str, Any],
    num_layers: int,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    """Extract the top-K features and their per-layer activation trajectories.

    Returns:
        top_features: list of {feature_idx, label, total_activation}
        trajectories: feature_idx -> (num_layers,) activation array
    """
    entries = sae_data.get("per_layer", [])

    # Accumulate total activation per feature across layers
    feature_total: dict[int, float] = {}
    feature_labels: dict[int, str] = {}
    feature_by_layer: dict[int, dict[int, float]] = {}

    for entry in entries:
        layer = entry["layer"]
        for feat in entry.get("features", []):
            fid = feat["feature_idx"]
            act = feat.get("activation", 0.0)
            feature_total[fid] = feature_total.get(fid, 0.0) + act
            if fid not in feature_labels:
                feature_labels[fid] = feat.get("label", f"F{fid}")
            if fid not in feature_by_layer:
                feature_by_layer[fid] = {}
            feature_by_layer[fid][layer] = act

    # Rank by total activation
    ranked = sorted(feature_total.items(), key=lambda p: p[1], reverse=True)
    top_ids = [fid for fid, _ in ranked[:top_k]]

    top_features = [
        {
            "feature_idx": fid,
            "label": feature_labels.get(fid, f"F{fid}"),
            "total_activation": feature_total[fid],
        }
        for fid in top_ids
    ]

    trajectories: dict[int, np.ndarray] = {}
    for fid in top_ids:
        arr = np.zeros(num_layers, dtype=np.float64)
        layer_map = feature_by_layer.get(fid, {})
        for l, v in layer_map.items():
            if 0 <= l < num_layers:
                arr[l] = v
        trajectories[fid] = arr

    return top_features, trajectories


# ============================================================================
# RENDERER
# ============================================================================

def render_feature_dashboard(
    sae_data: dict[str, Any],
    region_names: list[str],
    num_layers: int,
    top_k: int = 10,
    width: int = 1800,
    height: int = 1200,
) -> Image.Image:
    """Render a multi-panel SAE feature dashboard.

    Panels:
    - Top bar chart: features ranked by total activation
    - Trajectories: feature activation across layers (like cooking curves)
    - Co-occurrence: feature x feature correlation matrix
    """
    ft_title = get_font(14)
    ft_label = get_font(11)
    ft_tick = get_font(10)
    ft_small = get_font(9)

    top_features, trajectories = _collect_feature_info(sae_data, num_layers, top_k)
    n_feat = len(top_features)

    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    draw.text((20, 10), "SAE Feature Dashboard", fill=TEXT_COLOR, font=ft_title)
    draw.text((20, 30),
              f"Top {n_feat} features by total activation across {num_layers} layers",
              fill=TEXT_DIM, font=ft_tick)

    if n_feat == 0:
        draw.text((width // 3, height // 2),
                  "No SAE feature data available",
                  fill=TEXT_COLOR, font=ft_label)
        return img

    # ---- PANEL 1: Bar chart (top portion) ----
    panel1_top = 55
    panel1_h = 260
    bar_ml, bar_mr = 160, 40
    bar_w = width - bar_ml - bar_mr
    max_act = max(f["total_activation"] for f in top_features)
    max_act = max(max_act, 1e-12)
    bar_h = max(1, (panel1_h - 20) // max(n_feat, 1))

    draw.text((bar_ml, panel1_top - 2), "Total Activation",
              fill=TEXT_COLOR, font=ft_label)

    for i, feat in enumerate(top_features):
        y = panel1_top + 18 + i * bar_h
        bw = int(feat["total_activation"] / max_act * bar_w)
        col = REGION_PALETTE[i % len(REGION_PALETTE)]

        draw.rectangle([bar_ml, y + 1, bar_ml + bw, y + bar_h - 2], fill=col)
        # Feature label
        lbl = feat["label"][:20]
        draw.text((4, y + 1), lbl, fill=col, font=ft_small)
        # Value
        draw.text((bar_ml + bw + 4, y + 1),
                  f"{feat['total_activation']:.3f}",
                  fill=TEXT_DIM, font=ft_small)

    # ---- PANEL 2: Trajectory curves (middle portion) ----
    panel2_top = panel1_top + panel1_h + 30
    panel2_h = 340
    curve_ml, curve_mr = 90, 40
    curve_legend = 200
    cw = width - curve_ml - curve_mr - curve_legend
    ch = panel2_h - 50

    draw.text((curve_ml, panel2_top), "Feature Trajectories",
              fill=TEXT_COLOR, font=ft_label)

    cy0 = panel2_top + 22
    cy1 = cy0 + ch

    # Y range
    all_vals = np.concatenate(list(trajectories.values())) if trajectories else np.array([0.0])
    y_hi = float(np.max(all_vals)) * 1.12
    y_hi = max(y_hi, 1e-12)

    # Grid
    for l in range(0, num_layers, max(1, num_layers // 8)):
        gx = curve_ml + int(l / max(1, num_layers - 1) * cw)
        draw.line([(gx, cy0), (gx, cy1)], fill=GRID_COLOR, width=1)
        draw.text((gx - 5, cy1 + 4), f"L{l}", fill=TEXT_DIM, font=ft_tick)

    draw.rectangle([curve_ml, cy0, curve_ml + cw, cy1], outline=(80, 80, 92))

    # Phase annotations
    for plabel, ps, pe in display_phases(num_layers):
        pe_c = min(pe, num_layers - 1)
        pxs = curve_ml + int(ps / max(1, num_layers - 1) * cw)
        pxe = curve_ml + int(pe_c / max(1, num_layers - 1) * cw)
        mid = (pxs + pxe) // 2
        draw.text((mid - len(plabel) * 3, cy0 - 12), plabel,
                  fill=(90, 90, 105), font=ft_small)

    # Plot
    for i, feat in enumerate(top_features):
        fid = feat["feature_idx"]
        arr = trajectories[fid]
        col = REGION_PALETTE[i % len(REGION_PALETTE)]

        pts: list[tuple[int, int]] = []
        for l in range(num_layers):
            x = curve_ml + int(l / max(1, num_layers - 1) * cw)
            y = cy1 - int(arr[l] / y_hi * ch)
            pts.append((x, max(cy0, min(cy1, y))))

        for j in range(len(pts) - 1):
            draw.line([pts[j], pts[j + 1]], fill=col, width=2)

        # Peak dot
        pk = int(np.argmax(arr))
        ppx, ppy = pts[pk]
        draw.ellipse([ppx - 2, ppy - 2, ppx + 2, ppy + 2], fill=col)

    # Trajectory legend
    leg_x = curve_ml + cw + 18
    leg_y = cy0 + 4
    for i, feat in enumerate(top_features):
        col = REGION_PALETTE[i % len(REGION_PALETTE)]
        draw.rectangle([leg_x, leg_y + 2, leg_x + 8, leg_y + 10], fill=col)
        draw.text((leg_x + 12, leg_y), feat["label"][:16],
                  fill=col, font=ft_small)
        leg_y += 13

    # ---- PANEL 3: Co-occurrence matrix (bottom portion) ----
    panel3_top = panel2_top + panel2_h + 30
    available_h = height - panel3_top - 30

    if n_feat >= 2 and available_h > 80:
        draw.text((90, panel3_top), "Feature Co-occurrence",
                  fill=TEXT_COLOR, font=ft_label)

        # Build co-occurrence: correlate layer trajectories
        mat = np.zeros((n_feat, n_feat), dtype=np.float64)
        feat_ids = [f["feature_idx"] for f in top_features]
        arrs = [trajectories[fid] for fid in feat_ids]

        for a in range(n_feat):
            for b in range(n_feat):
                if a == b:
                    mat[a, b] = 1.0
                else:
                    va, vb = arrs[a], arrs[b]
                    na = np.linalg.norm(va)
                    nb = np.linalg.norm(vb)
                    if na > 0 and nb > 0:
                        mat[a, b] = float(np.dot(va, vb) / (na * nb))

        # Draw matrix
        mat_ml = 160
        cell_sz = min(max(1, available_h // max(n_feat, 1)),
                      max(1, (width - mat_ml - 60) // max(n_feat, 1)),
                      28)

        lut = get_colormap("coolwarm")
        mat_y0 = panel3_top + 20

        for a in range(n_feat):
            for b in range(n_feat):
                val = (mat[a, b] + 1.0) / 2.0  # map [-1, 1] to [0, 1]
                idx = min(255, max(0, int(val * 255)))
                c = (int(lut[idx][0]), int(lut[idx][1]), int(lut[idx][2]))
                x0 = mat_ml + b * cell_sz
                y0 = mat_y0 + a * cell_sz
                draw.rectangle([x0, y0, x0 + cell_sz - 1, y0 + cell_sz - 1],
                               fill=c)

        # Row labels
        for a in range(n_feat):
            y = mat_y0 + a * cell_sz + cell_sz // 2 - 4
            draw.text((4, y), top_features[a]["label"][:18],
                      fill=TEXT_DIM, font=ft_small)

    return img


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render SAE feature dashboard",
    )
    parser.add_argument("--result", required=True, help="Path to result JSON")
    parser.add_argument("--output", default=None, help="Output PNG path")
    parser.add_argument("--position", default="terminal")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top features to show")
    args = parser.parse_args()

    if args.output is None:
        rp = Path(args.result)
        args.output = str(rp.parent / f"features_{rp.stem}.png")

    print(f"Loading SAE data: {args.result}")
    sae_data = load_sae_data(args.result, args.position)

    entries = sae_data.get("per_layer", [])
    num_layers = (max(e["layer"] for e in entries) + 1) if entries else 64

    # Region names from the result (not needed for SAE viz, pass empty)
    region_names: list[str] = []

    print(f"  {num_layers} layers, top-k={args.top_k}")

    img = render_feature_dashboard(
        sae_data, region_names, num_layers,
        top_k=args.top_k,
    )
    img.save(args.output)
    print(f"Saved: {args.output} ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
