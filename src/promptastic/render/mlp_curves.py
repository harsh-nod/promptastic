#!/usr/bin/env python3
"""MLP contribution trajectories across layers.

Renders cooking-curve-style plots of MLP delta norms per region, showing
how MLP transformations contribute to each region's representation at
every layer.

Usage:
    python -m promptastic.render.mlp_curves --result r.json
    python -m promptastic.render.mlp_curves --result r.json --normalize per-region
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ..constants import SKIP_REGIONS, display_phases
from ._shared import (
    BG_COLOR,
    GRID_COLOR,
    AXIS_COLOR,
    TEXT_COLOR,
    TEXT_DIM,
    REGION_PALETTE,
    get_font,
)
from .loaders import load_mlp_data


# ============================================================================
# CHART LAYOUT
# ============================================================================

_ML = 90
_MR = 36
_MT = 68
_MB = 55
_LEGEND_W = 210
_LW = 2


# ============================================================================
# TICK HELPER
# ============================================================================

def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(max(raw, 1e-30)))
    r = raw / mag
    if r <= 1.5:
        step = mag
    elif r <= 3.5:
        step = 2 * mag
    elif r <= 7.5:
        step = 5 * mag
    else:
        step = 10 * mag
    start = math.floor(lo / step) * step
    out: list[float] = []
    v = start
    while v <= hi + step * 0.01:
        if v >= lo - step * 0.01:
            out.append(v)
        v += step
    return out


# ============================================================================
# RENDERER
# ============================================================================

def render_mlp_curves(
    mlp_data: dict[str, Any],
    region_names: list[str],
    num_layers: int,
    width: int = 1400,
    height: int = 700,
    normalize_mode: str = "raw",
) -> Image.Image:
    """Render MLP delta-norm trajectories for each region.

    Parameters
    ----------
    mlp_data : dict
        Contains ``per_layer`` list with ``{layer, delta_norm, ...}`` or
        per-region delta norms under ``per_region_delta``.
    region_names : list of str
        Which regions to plot.
    """
    ft_title = get_font(14)
    ft_label = get_font(11)
    ft_tick = get_font(10)
    ft_legend = get_font(10)

    entries = mlp_data.get("per_layer", [])

    # Build trajectories: region -> (n_layers,) array of delta norms
    trajectories: dict[str, np.ndarray] = {}
    for rname in region_names:
        curve = np.zeros(num_layers, dtype=np.float64)
        for entry in entries:
            l = entry["layer"]
            if 0 <= l < num_layers:
                per_region = entry.get("per_region_delta", {})
                curve[l] = per_region.get(rname, 0.0)
        if np.any(curve > 0):
            trajectories[rname] = curve

    if not trajectories:
        # Fallback: if no per-region data, plot global delta_norm
        global_curve = np.zeros(num_layers, dtype=np.float64)
        for entry in entries:
            l = entry["layer"]
            if 0 <= l < num_layers:
                global_curve[l] = entry.get("delta_norm", 0.0)
        if np.any(global_curve > 0):
            trajectories["global_mlp"] = global_curve

    # Chart area
    cx0 = _ML
    cy0 = _MT
    cx1 = width - _MR - _LEGEND_W
    cy1 = height - _MB
    cw = cx1 - cx0
    ch = cy1 - cy0

    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    draw.text((cx0, 14), "MLP Contribution Trajectories",
              fill=TEXT_COLOR, font=ft_title)
    mode_tag = " [per-region normalized]" if normalize_mode == "per-region" else ""
    draw.text((cx0, 34),
              f"Delta norm by region across layers{mode_tag}",
              fill=TEXT_DIM, font=ft_tick)

    names = list(trajectories.keys())
    if not names:
        draw.text((cx0 + cw // 3, cy0 + ch // 2),
                  "No MLP data available", fill=TEXT_COLOR, font=ft_label)
        return img

    # Prepare plot data
    curves: dict[str, np.ndarray] = {}
    for nm, traj in trajectories.items():
        if normalize_mode == "per-region":
            pk = traj.max()
            curves[nm] = traj / pk if pk > 0 else traj.copy()
        else:
            curves[nm] = traj.copy()

    all_v = np.concatenate(list(curves.values()))
    y_lo = 0.0
    y_hi = float(np.max(all_v)) * 1.12

    # Ticks
    xt = list(range(0, num_layers, max(1, num_layers // 8)))
    if (num_layers - 1) not in xt:
        xt.append(num_layers - 1)
    yt = _ticks(y_lo, y_hi, 6)

    def px(layer: int, val: float) -> tuple[int, int]:
        x = cx0 + int(layer / max(1, num_layers - 1) * cw)
        y = cy1 - int((val - y_lo) / max(1e-15, y_hi - y_lo) * ch)
        return x, y

    # Grid
    for l in xt:
        gx, _ = px(l, 0)
        draw.line([(gx, cy0), (gx, cy1)], fill=GRID_COLOR, width=1)
        draw.text((gx - 5, cy1 + 5), f"L{l}", fill=TEXT_DIM, font=ft_tick)

    for val in yt:
        _, gy = px(0, val)
        if cy0 <= gy <= cy1:
            draw.line([(cx0, gy), (cx1, gy)], fill=GRID_COLOR, width=1)
            txt = f"{val:.1f}" if normalize_mode == "per-region" else (
                f"{val:.1e}" if val != 0 else "0")
            draw.text((cx0 - 65, gy - 6), txt, fill=TEXT_DIM, font=ft_tick)

    draw.rectangle([cx0, cy0, cx1, cy1], outline=AXIS_COLOR)
    draw.text((cx0 + cw // 2 - 16, cy1 + 22), "Layer",
              fill=TEXT_COLOR, font=ft_label)

    # Phase annotations
    for plabel, ps, pe in display_phases(num_layers):
        pe_c = min(pe, num_layers - 1)
        pxs, _ = px(ps, 0)
        pxe, _ = px(pe_c, 0)
        mid = (pxs + pxe) // 2
        bb = ft_tick.getbbox(plabel)
        tw = bb[2] - bb[0]
        draw.text((mid - tw // 2, cy0 - 14), plabel,
                  fill=(95, 95, 108), font=ft_tick)
        if ps > 0:
            draw.line([(pxs, cy0), (pxs, cy0 + 5)],
                      fill=(75, 75, 88), width=1)

    # Draw curves
    for idx, nm in enumerate(names):
        c = curves[nm]
        col = REGION_PALETTE[idx % len(REGION_PALETTE)]

        pts = [px(l, c[l]) for l in range(num_layers)]
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=col, width=_LW)

        peak_l = int(np.argmax(c))
        ppx, ppy = px(peak_l, c[peak_l])
        draw.ellipse([ppx - 3, ppy - 3, ppx + 3, ppy + 3], fill=col)

    # Legend
    leg_x = cx1 + 18
    leg_y = cy0 + 4
    draw.text((leg_x, leg_y), "Regions", fill=TEXT_COLOR, font=ft_label)
    leg_y += 18

    ordered = sorted(names, key=lambda n: int(np.argmax(curves[n])))
    for nm in ordered:
        idx = names.index(nm)
        col = REGION_PALETTE[idx % len(REGION_PALETTE)]
        pk_l = int(np.argmax(trajectories.get(nm, curves[nm])))
        draw.rectangle([leg_x, leg_y + 2, leg_x + 10, leg_y + 12], fill=col)
        draw.text((leg_x + 14, leg_y), f"{nm[:18]} (L{pk_l})",
                  fill=col, font=ft_legend)
        leg_y += 15

    draw.text((cx0, height - 16),
              "Dot marks peak delta-norm layer for each region.",
              fill=TEXT_DIM, font=ft_tick)

    return img


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render MLP contribution trajectories",
    )
    parser.add_argument("--result", required=True, help="Path to result JSON")
    parser.add_argument("--output", default=None, help="Output PNG path")
    parser.add_argument("--position", default="terminal")
    parser.add_argument("--normalize", choices=["raw", "per-region"],
                        default="raw")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=700)
    args = parser.parse_args()

    if args.output is None:
        rp = Path(args.result)
        args.output = str(rp.parent / f"mlp_{rp.stem}_{args.position}.png")

    print(f"Loading MLP data: {args.result}")
    mlp_data = load_mlp_data(args.result, args.position)

    entries = mlp_data.get("per_layer", [])
    num_layers = (max(e["layer"] for e in entries) + 1) if entries else 64

    # Discover regions from per_region_delta
    region_set: set[str] = set()
    for entry in entries:
        region_set.update(entry.get("per_region_delta", {}).keys())
    region_names = sorted(r for r in region_set if r not in SKIP_REGIONS)

    print(f"  {num_layers} layers, {len(region_names)} regions")

    img = render_mlp_curves(
        mlp_data, region_names, num_layers,
        width=args.width, height=args.height,
        normalize_mode=args.normalize,
    )
    img.save(args.output)
    print(f"Saved: {args.output} ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
