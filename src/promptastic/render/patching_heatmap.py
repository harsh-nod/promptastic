#!/usr/bin/env python3
"""Region x Layer causal importance heatmap.

Renders a grid with regions on the Y axis, layers on the X axis, and cells
colored by the effect magnitude of patching each (region, layer) pair.

Usage:
    python -m promptastic.render.patching_heatmap --result patching_result.json
    python -m promptastic.render.patching_heatmap --result r.json --metric logit_diff
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .._types import PatchingResult
from ..constants import display_phases
from ._shared import (
    BG_COLOR,
    TEXT_COLOR,
    TEXT_DIM,
    get_colormap,
    get_font,
    text_color_for_bg,
    _COLORMAP_BUILDERS,
)
from .loaders import load_patching_data


# ============================================================================
# RENDERER
# ============================================================================

_MARGIN_L = 180
_MARGIN_R = 30
_MARGIN_T = 70
_MARGIN_B = 55
_CELL_PAD = 1


def render_patching_heatmap(
    patching_data: list[PatchingResult],
    region_names: list[str],
    num_layers: int,
    metric: str = "kl_divergence",
    width: int = 1400,
    height: int = 800,
    colormap_name: str = "coolwarm",
) -> Image.Image:
    """Render a region-by-layer patching heatmap.

    Parameters
    ----------
    patching_data : list of PatchingResult
        Each entry has ``region``, ``layer``, and at least one metric field.
    region_names : list of str
        Ordered region names to display on the Y axis.
    num_layers : int
        Total number of layers in the model.
    metric : str
        Which metric field to visualize (``kl_divergence`` or ``logit_diff``).
    """
    ft_title = get_font(14)
    ft_label = get_font(11)
    ft_tick = get_font(10)
    ft_cell = get_font(9)

    lut = get_colormap(colormap_name)

    # Build value grid
    r_idx = {r: i for i, r in enumerate(region_names)}
    n_regions = len(region_names)
    grid = np.zeros((n_regions, num_layers), dtype=np.float64)

    for entry in patching_data:
        rn = entry["region"]
        ly = entry["layer"]
        if rn in r_idx and 0 <= ly < num_layers:
            grid[r_idx[rn], ly] = entry.get(metric, 0.0)  # type: ignore[arg-type]

    # Normalize to [0, 1]
    vmin = float(np.min(grid))
    vmax = float(np.max(grid))
    span = max(vmax - vmin, 1e-12)
    normed = (grid - vmin) / span

    # Layout
    plot_w = width - _MARGIN_L - _MARGIN_R
    plot_h = height - _MARGIN_T - _MARGIN_B
    cell_w = max(1, plot_w // num_layers)
    cell_h = max(1, plot_h // n_regions)

    # Adjust image size to fit cells exactly
    actual_w = _MARGIN_L + cell_w * num_layers + _MARGIN_R
    actual_h = _MARGIN_T + cell_h * n_regions + _MARGIN_B

    img = Image.new("RGB", (actual_w, actual_h), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((_MARGIN_L, 12),
              f"Causal Patching Heatmap -- metric: {metric}",
              fill=TEXT_COLOR, font=ft_title)
    draw.text((_MARGIN_L, 32),
              f"Regions x Layers  ({n_regions} regions, {num_layers} layers)",
              fill=TEXT_DIM, font=ft_tick)

    # Phase boundary lines
    for _, ps, pe in display_phases(num_layers):
        if ps > 0:
            bx = _MARGIN_L + ps * cell_w
            draw.line([(bx, _MARGIN_T), (bx, _MARGIN_T + cell_h * n_regions)],
                      fill=(90, 90, 105), width=1)

    # Draw cells
    for ri in range(n_regions):
        for li in range(num_layers):
            x0 = _MARGIN_L + li * cell_w + _CELL_PAD
            y0 = _MARGIN_T + ri * cell_h + _CELL_PAD
            x1 = x0 + cell_w - 2 * _CELL_PAD
            y1 = y0 + cell_h - 2 * _CELL_PAD

            idx = min(255, int(normed[ri, li] * 255))
            c = (int(lut[idx][0]), int(lut[idx][1]), int(lut[idx][2]))
            draw.rectangle([x0, y0, x1, y1], fill=c)

            # Annotate high-effect cells
            if cell_w >= 18 and cell_h >= 14:
                val = grid[ri, li]
                if abs(val) > span * 0.5:
                    fg = text_color_for_bg(*c)
                    draw.text((x0 + 2, y0 + 1), f"{val:.2f}",
                              fill=fg, font=ft_cell)

    # Y-axis labels (region names)
    for ri, rn in enumerate(region_names):
        y = _MARGIN_T + ri * cell_h + cell_h // 2 - 5
        draw.text((6, y), rn[:22], fill=TEXT_COLOR, font=ft_tick)

    # X-axis labels (layers)
    tick_stride = max(1, num_layers // 12)
    for l in range(0, num_layers, tick_stride):
        x = _MARGIN_L + l * cell_w + cell_w // 2 - 4
        draw.text((x, _MARGIN_T + cell_h * n_regions + 6),
                  f"L{l}", fill=TEXT_DIM, font=ft_tick)

    # Colorbar
    cb_w, cb_h = min(200, plot_w // 3), 12
    cb_x = _MARGIN_L
    cb_y = actual_h - 22
    for i in range(cb_w):
        idx = min(255, int(i / (cb_w - 1) * 255))
        c = (int(lut[idx][0]), int(lut[idx][1]), int(lut[idx][2]))
        draw.line([(cb_x + i, cb_y), (cb_x + i, cb_y + cb_h)], fill=c)
    draw.text((cb_x - 2, cb_y - 12), f"{vmin:.2e}", fill=TEXT_DIM, font=ft_cell)
    draw.text((cb_x + cb_w - 10, cb_y - 12), f"{vmax:.2e}",
              fill=TEXT_DIM, font=ft_cell)

    return img


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render region x layer causal patching heatmap",
    )
    parser.add_argument("--result", required=True, help="Path to result JSON")
    parser.add_argument("--output", default=None, help="Output PNG path")
    parser.add_argument("--metric", default="kl_divergence",
                        choices=["kl_divergence", "logit_diff"],
                        help="Metric to visualize")
    parser.add_argument("--colormap", default="coolwarm",
                        help=f"Colormap ({', '.join(_COLORMAP_BUILDERS)})")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=800)
    args = parser.parse_args()

    if args.output is None:
        rp = Path(args.result)
        args.output = str(rp.parent / f"patching_{rp.stem}_{args.metric}.png")

    print(f"Loading patching data: {args.result}")
    data = load_patching_data(args.result)
    print(f"  {len(data)} patching entries")

    # Discover regions and layers
    regions_seen: dict[str, int] = {}
    max_layer = 0
    for entry in data:
        rn = entry["region"]
        if rn not in regions_seen:
            regions_seen[rn] = len(regions_seen)
        max_layer = max(max_layer, entry["layer"])

    region_names = sorted(regions_seen.keys(),
                          key=lambda r: regions_seen[r])
    num_layers = max_layer + 1

    print(f"  {len(region_names)} regions, {num_layers} layers")

    img = render_patching_heatmap(
        data, region_names, num_layers,
        metric=args.metric,
        width=args.width,
        height=args.height,
        colormap_name=args.colormap,
    )
    img.save(args.output)
    print(f"Saved: {args.output} ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
