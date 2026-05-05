#!/usr/bin/env python3
"""Cross-step attention timeline for autoregressive generation.

X axis = generation step, Y axis = per-region attention share.
Below the chart, the generated text is displayed token by token.

Usage:
    python -m promptastic.render.generation_timeline --result r.json
    python -m promptastic.render.generation_timeline --result r.json --width 1600
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ._shared import (
    BG_COLOR,
    GRID_COLOR,
    AXIS_COLOR,
    TEXT_COLOR,
    TEXT_DIM,
    REGION_COLORS,
    REGION_PALETTE,
    get_font,
)
from .loaders import load_generation_data


# ============================================================================
# LAYOUT
# ============================================================================

_ML = 80
_MR = 200
_MT = 65
_MB = 100  # space for generated text ribbon


# ============================================================================
# RENDERER
# ============================================================================

def render_generation_timeline(
    generation_data: dict[str, Any],
    region_names: list[str],
    width: int = 1400,
    height: int = 700,
) -> Image.Image:
    """Render a stacked-area timeline of per-region attention across
    generation steps, with the generated text displayed beneath.

    Parameters
    ----------
    generation_data : dict
        Contains ``steps`` list with per-step ``{step, generated_token,
        region_attention: {region: float}, ...}`` entries.
    region_names : list of str
        Regions to track (defines stacking order).
    """
    ft_title = get_font(14)
    ft_label = get_font(11)
    ft_tick = get_font(10)
    ft_small = get_font(9)
    ft_token = get_font(11)

    steps = generation_data.get("steps", [])
    n_steps = len(steps)

    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    draw.text((20, 10), "Generation Attention Timeline",
              fill=TEXT_COLOR, font=ft_title)
    draw.text((20, 30),
              f"{n_steps} generation steps, {len(region_names)} tracked regions",
              fill=TEXT_DIM, font=ft_tick)

    if n_steps == 0:
        draw.text((width // 3, height // 2),
                  "No generation data available",
                  fill=TEXT_COLOR, font=ft_label)
        return img

    # Chart area
    cx0, cy0 = _ML, _MT
    cx1 = width - _MR
    cy1 = height - _MB
    cw = cx1 - cx0
    ch = cy1 - cy0

    # Build per-step per-region values
    step_data = np.zeros((n_steps, len(region_names)), dtype=np.float64)
    gen_tokens: list[str] = []

    for si, step in enumerate(steps):
        ra = step.get("region_attention", {})
        for ri, rn in enumerate(region_names):
            step_data[si, ri] = ra.get(rn, 0.0)
        gen_tokens.append(step.get("generated_token", "?"))

    # Normalize each step to sum to 1 for stacked area
    row_sums = step_data.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    stacked = step_data / row_sums

    # Assign colors
    region_col: dict[str, tuple[int, int, int]] = {}
    for ri, rn in enumerate(region_names):
        region_col[rn] = REGION_COLORS.get(
            rn, REGION_PALETTE[ri % len(REGION_PALETTE)])

    # Grid
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        gy = cy1 - int(frac * ch)
        draw.line([(cx0, gy), (cx1, gy)], fill=GRID_COLOR, width=1)
        draw.text((cx0 - 38, gy - 5), f"{frac:.0%}",
                  fill=TEXT_DIM, font=ft_tick)

    # X ticks
    tick_stride = max(1, n_steps // 10)
    for s in range(0, n_steps, tick_stride):
        gx = cx0 + int(s / max(1, n_steps - 1) * cw) if n_steps > 1 else cx0
        draw.line([(gx, cy0), (gx, cy1)], fill=GRID_COLOR, width=1)
        draw.text((gx - 3, cy1 + 4), str(s), fill=TEXT_DIM, font=ft_tick)

    draw.rectangle([cx0, cy0, cx1, cy1], outline=AXIS_COLOR)
    draw.text((cx0 + cw // 2 - 12, cy1 + 18), "Step",
              fill=TEXT_COLOR, font=ft_label)

    # Draw stacked area (bottom to top)
    # We accumulate bottom boundaries for each region
    bottoms = np.zeros(n_steps, dtype=np.float64)

    for ri in range(len(region_names) - 1, -1, -1):
        rn = region_names[ri]
        col = region_col[rn]
        tops = bottoms + stacked[:, ri]

        # Build polygon
        upper: list[tuple[int, int]] = []
        lower: list[tuple[int, int]] = []
        for s in range(n_steps):
            x = cx0 + int(s / max(1, n_steps - 1) * cw) if n_steps > 1 else cx0
            yu = cy1 - int(tops[s] * ch)
            yl = cy1 - int(bottoms[s] * ch)
            upper.append((x, max(cy0, min(cy1, yu))))
            lower.append((x, max(cy0, min(cy1, yl))))

        polygon = upper + lower[::-1]
        if len(polygon) >= 3:
            # Use RGBA overlay for semi-transparency
            overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            od.polygon(polygon, fill=(*col, 160))
            img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
            draw = ImageDraw.Draw(img)

        # Mean line on top
        for s in range(n_steps - 1):
            x0 = cx0 + int(s / max(1, n_steps - 1) * cw) if n_steps > 1 else cx0
            x1_pt = cx0 + int((s + 1) / max(1, n_steps - 1) * cw) if n_steps > 1 else cx0
            yu0 = cy1 - int(tops[s] * ch)
            yu1 = cy1 - int(tops[s + 1] * ch)
            draw.line([(x0, yu0), (x1_pt, yu1)], fill=col, width=1)

        bottoms = tops.copy()

    # Generated text ribbon below chart
    ribbon_y = cy1 + 36
    draw.text((_ML, ribbon_y - 14), "Generated:", fill=TEXT_DIM, font=ft_small)
    tx = _ML
    for si, tok in enumerate(gen_tokens):
        display = tok.replace("\n", "\\n")
        bb = ft_token.getbbox(display)
        tw = bb[2] - bb[0] + 4
        if tx + tw > width - 20:
            ribbon_y += 16
            tx = _ML
        draw.text((tx, ribbon_y), display, fill=TEXT_COLOR, font=ft_token)
        tx += tw

    # Legend
    leg_x = cx1 + 12
    leg_y = cy0 + 4
    draw.text((leg_x, leg_y), "Regions", fill=TEXT_COLOR, font=ft_label)
    leg_y += 16
    for rn in region_names:
        col = region_col[rn]
        draw.rectangle([(leg_x, leg_y + 1), (leg_x + 10, leg_y + 11)], fill=col)
        draw.text((leg_x + 14, leg_y), rn[:16], fill=col, font=ft_small)
        leg_y += 14

    return img


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render generation attention timeline",
    )
    parser.add_argument("--result", required=True, help="Path to result JSON")
    parser.add_argument("--output", default=None, help="Output PNG path")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=700)
    args = parser.parse_args()

    if args.output is None:
        rp = Path(args.result)
        args.output = str(rp.parent / f"generation_{rp.stem}.png")

    print(f"Loading generation data: {args.result}")
    gen_data = load_generation_data(args.result)

    steps = gen_data.get("steps", [])
    print(f"  {len(steps)} generation steps")

    # Discover regions
    region_set: set[str] = set()
    for step in steps:
        region_set.update(step.get("region_attention", {}).keys())
    region_names = sorted(region_set)

    print(f"  {len(region_names)} regions tracked")

    img = render_generation_timeline(
        gen_data, region_names,
        width=args.width, height=args.height,
    )
    img.save(args.output)
    print(f"Saved: {args.output} ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
