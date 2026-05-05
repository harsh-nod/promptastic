#!/usr/bin/env python3
"""Per-region attention trajectory ("cooking curves") across all layers.

Each named region's mean per-token attention is plotted as a curve from
layer 0 to the final layer.  Phase annotations, peak markers, and an
ordered legend provide interpretive context.

Usage:
    python -m promptastic.render.cooking_curves --result sample_01.json
    python -m promptastic.render.cooking_curves --result sample_01.json --normalize per-region
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .._types import RegionInfo
from ..constants import SKIP_REGIONS, DEFAULT_DISPLAY_REGIONS, display_phases
from ._shared import (
    BG_COLOR,
    GRID_COLOR,
    AXIS_COLOR,
    TEXT_COLOR,
    TEXT_DIM,
    REGION_PALETTE,
    get_font,
)
from .loaders import load_cooking_data


# ============================================================================
# CHART LAYOUT
# ============================================================================

_MARGIN_LEFT = 90
_MARGIN_TOP = 68
_MARGIN_RIGHT = 36
_MARGIN_BOTTOM = 55
_LEGEND_W = 210
_CURVE_WIDTH = 2


# ============================================================================
# TRAJECTORY COMPUTATION
# ============================================================================

def compute_region_trajectories(
    region_map: dict[str, RegionInfo],
    layer_weights: dict[int, np.ndarray],
    regions: list[str],
) -> dict[str, np.ndarray]:
    """Compute per-token mean attention for each region at every layer.

    Parameters
    ----------
    region_map : dict
        Maps region name to ``{tok_start, tok_end, ...}``.
    layer_weights : dict
        Maps layer index to a 1-D weight array over all tokens.
    regions : list of str
        Region names to include.

    Returns
    -------
    dict mapping region name to a ``(n_layers,)`` array.
    """
    sorted_layers = sorted(layer_weights.keys())
    n = len(sorted_layers)
    trajectories: dict[str, np.ndarray] = {}

    for rname in regions:
        if rname not in region_map:
            continue
        info = region_map[rname]
        s, e = info["tok_start"], info["tok_end"]
        if e <= s:
            continue

        curve = np.zeros(n, dtype=np.float64)
        for li, layer in enumerate(sorted_layers):
            w = layer_weights[layer]
            if e <= len(w):
                curve[li] = float(np.mean(w[s:e]))
        trajectories[rname] = curve

    return trajectories


# ============================================================================
# TICK GENERATION
# ============================================================================

def _readable_ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    """Generate round tick values in [lo, hi]."""
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / target
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

    first = math.floor(lo / step) * step
    ticks: list[float] = []
    v = first
    while v <= hi + step * 0.01:
        if v >= lo - step * 0.01:
            ticks.append(v)
        v += step
    return ticks


# ============================================================================
# RENDERER
# ============================================================================

def render_cooking_curves(
    trajectories: dict[str, np.ndarray],
    position: str,
    result_path: str,
    width: int = 1400,
    height: int = 700,
    normalize_mode: str = "raw",
    highlight: list[str] | None = None,
) -> Image.Image:
    """Render cooking curves as a PIL Image.

    Parameters
    ----------
    trajectories : dict
        Region name to ``(n_layers,)`` array of mean per-token attention.
    normalize_mode : str
        ``"raw"`` keeps original magnitudes; ``"per-region"`` scales each
        curve to [0, 1] based on its own peak.
    highlight : list of str, optional
        If given, only these regions are drawn at full brightness.
    """
    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    ft_title = get_font(14)
    ft_label = get_font(11)
    ft_tick = get_font(10)
    ft_legend = get_font(10)

    # Chart area
    cx0 = _MARGIN_LEFT
    cy0 = _MARGIN_TOP
    cx1 = width - _MARGIN_RIGHT - _LEGEND_W
    cy1 = height - _MARGIN_BOTTOM
    cw = cx1 - cx0
    ch = cy1 - cy0

    # Title
    stem = Path(result_path).stem
    title = f"Attention Cooking Curves -- {stem} ({position})"
    if normalize_mode == "per-region":
        title += "  [per-region normalized]"
    draw.text((cx0, 14), title, fill=TEXT_COLOR, font=ft_title)
    draw.text((cx0, 34), "Mean attention per token by region across layers",
              fill=TEXT_DIM, font=ft_tick)

    # Prepare plot data
    names = list(trajectories.keys())
    n_layers = len(next(iter(trajectories.values())))

    curves: dict[str, np.ndarray] = {}
    for nm, traj in trajectories.items():
        if normalize_mode == "per-region":
            pk = traj.max()
            curves[nm] = traj / pk if pk > 0 else traj.copy()
        else:
            curves[nm] = traj.copy()

    # Y-axis range
    all_v = np.concatenate(list(curves.values()))
    y_lo = 0.0
    y_hi = float(np.max(all_v)) * 1.12

    # Ticks
    xt = list(range(0, n_layers, max(1, n_layers // 8)))
    if (n_layers - 1) not in xt:
        xt.append(n_layers - 1)
    yt = _readable_ticks(y_lo, y_hi, 6)

    def px(layer: int, val: float) -> tuple[int, int]:
        x = cx0 + int(layer / max(1, n_layers - 1) * cw)
        y = cy1 - int((val - y_lo) / max(1e-15, y_hi - y_lo) * ch)
        return x, y

    # Grid
    for layer in xt:
        gx, _ = px(layer, 0)
        draw.line([(gx, cy0), (gx, cy1)], fill=GRID_COLOR, width=1)
        draw.text((gx - 5, cy1 + 5), f"L{layer}", fill=TEXT_DIM, font=ft_tick)

    for val in yt:
        _, gy = px(0, val)
        if cy0 <= gy <= cy1:
            draw.line([(cx0, gy), (cx1, gy)], fill=GRID_COLOR, width=1)
            txt = f"{val:.1f}" if normalize_mode == "per-region" else (
                f"{val:.1e}" if val != 0 else "0")
            draw.text((cx0 - 65, gy - 6), txt, fill=TEXT_DIM, font=ft_tick)

    # Border
    draw.rectangle([cx0, cy0, cx1, cy1], outline=AXIS_COLOR)

    # X label
    draw.text((cx0 + cw // 2 - 16, cy1 + 22), "Layer",
              fill=TEXT_COLOR, font=ft_label)

    # Phase annotations
    for phase_label, ps, pe in display_phases(n_layers):
        pe_c = min(pe, n_layers - 1)
        pxs, _ = px(ps, 0)
        pxe, _ = px(pe_c, 0)
        mid = (pxs + pxe) // 2
        bb = ft_tick.getbbox(phase_label)
        tw = bb[2] - bb[0]
        draw.text((mid - tw // 2, cy0 - 14), phase_label,
                  fill=(95, 95, 108), font=ft_tick)
        if ps > 0:
            draw.line([(pxs, cy0), (pxs, cy0 + 5)], fill=(75, 75, 88), width=1)

    # Draw curves
    use_dim = highlight is not None

    for idx, nm in enumerate(names):
        c = curves[nm]
        col = REGION_PALETTE[idx % len(REGION_PALETTE)]

        if use_dim and nm not in (highlight or []):
            col = (col[0] // 4, col[1] // 4, col[2] // 4)
            lw = 1
        else:
            lw = _CURVE_WIDTH

        pts = [px(l, c[l]) for l in range(n_layers)]
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=col, width=lw)

        # Peak dot
        peak_l = int(np.argmax(c))
        ppx, ppy = px(peak_l, c[peak_l])
        if not use_dim or nm in (highlight or []):
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
        if use_dim and nm not in (highlight or []):
            col = (col[0] // 4, col[1] // 4, col[2] // 4)

        pk_l = int(np.argmax(trajectories[nm]))
        short = nm[:18]
        draw.rectangle([leg_x, leg_y + 2, leg_x + 10, leg_y + 12], fill=col)
        draw.text((leg_x + 14, leg_y), f"{short} (L{pk_l})",
                  fill=col, font=ft_legend)
        leg_y += 15

    # Footer
    draw.text((cx0, height - 16),
              "Dot marks each region's peak layer.",
              fill=TEXT_DIM, font=ft_tick)

    return img


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render per-region attention cooking curves",
    )
    parser.add_argument("--result", required=True, help="Path to result JSON")
    parser.add_argument("--output", default=None, help="Output PNG path")
    parser.add_argument("--position", default="terminal",
                        help="Query position (default: terminal)")
    parser.add_argument("--width", type=int, default=1400, help="Image width")
    parser.add_argument("--height", type=int, default=700, help="Image height")
    parser.add_argument("--normalize", choices=["raw", "per-region"],
                        default="raw",
                        help="Normalization mode (default: raw)")
    parser.add_argument("--regions", default=None,
                        help="Comma-separated region names to plot")
    parser.add_argument("--highlight", default=None,
                        help="Comma-separated region names to highlight")
    args = parser.parse_args()

    if args.output is None:
        rp = Path(args.result)
        args.output = str(rp.parent / f"cooking_{rp.stem}_{args.position}.png")

    print(f"Loading: {args.result}")
    region_map, layer_weights, token_labels = load_cooking_data(
        args.result, args.position,
    )
    print(f"  {len(token_labels)} tokens, {len(layer_weights)} layers, "
          f"{len(region_map)} regions")

    if args.regions:
        regions = [r.strip() for r in args.regions.split(",")]
    else:
        regions = [r for r in region_map if r not in SKIP_REGIONS]

        def _order(name: str) -> int:
            try:
                return DEFAULT_DISPLAY_REGIONS.index(name)
            except ValueError:
                return 999
        regions.sort(key=_order)

    trajectories = compute_region_trajectories(region_map, layer_weights, regions)
    print(f"  Plotting {len(trajectories)} regions")

    for nm, crv in sorted(trajectories.items(),
                          key=lambda p: int(np.argmax(p[1]))):
        pk = int(np.argmax(crv))
        pv = float(crv.max())
        tv = float(np.mean(crv[-4:]))
        ratio = pv / tv if tv > 0 else float("inf")
        print(f"    {nm:25s}  peak L{pk:02d} ({pv:.6f})  "
              f"terminal ({tv:.6f})  ratio {ratio:.1f}x")

    hl = ([h.strip() for h in args.highlight.split(",")]
          if args.highlight else None)

    img = render_cooking_curves(
        trajectories,
        position=args.position,
        result_path=args.result,
        width=args.width,
        height=args.height,
        normalize_mode=args.normalize,
        highlight=hl,
    )
    img.save(args.output)
    print(f"Saved: {args.output} ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
