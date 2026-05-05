#!/usr/bin/env python3
"""Per-head attention visualization: grid and specialization views.

*grid* mode renders a heads-by-regions grid for selected layers.
*specialization* mode shows each head's dominant region across layer phases.

Usage:
    python -m promptastic.render.head_grid --result r.json --mode grid --layers 0,16,32,48
    python -m promptastic.render.head_grid --result r.json --mode specialization
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
    REGION_COLORS,
    REGION_PALETTE,
    TEXT_COLOR,
    TEXT_DIM,
    get_colormap,
    get_font,
    parse_layer_spec,
    text_color_for_bg,
    _COLORMAP_BUILDERS,
)
from .loaders import load_per_head_data


# ============================================================================
# GRID MODE
# ============================================================================

_CELL_PAD = 1
_LABEL_W = 60
_REGION_LABEL_H = 80


def render_head_grid(
    per_head_data: dict[str, Any],
    region_names: list[str],
    num_heads: int,
    layers: list[int],
    width: int = 1800,
    colormap_name: str = "viridis",
) -> Image.Image:
    """Render a heads x regions grid for each selected layer, tiled vertically.

    Parameters
    ----------
    per_head_data : dict
        Contains ``per_layer`` list with entries ``{layer, heads: [...]}``.
    region_names : list of str
        Regions to show (columns).
    num_heads : int
        Number of attention heads.
    layers : list of int
        Which layers to render grids for.
    """
    ft_title = get_font(13)
    ft_label = get_font(10)
    ft_small = get_font(9)

    lut = get_colormap(colormap_name)

    n_regions = len(region_names)
    n_layers_shown = len(layers)

    # Build layer -> head -> region -> weight lookup
    layer_index: dict[int, list[dict[str, Any]]] = {}
    for entry in per_head_data["per_layer"]:
        layer_index[entry["layer"]] = entry.get("heads", [])

    # Compute cell size
    ml, mr, mt, mb = 70, 20, 55, 20
    grid_w = width - ml - mr
    cell_w = max(1, grid_w // n_regions)
    cell_h = max(12, min(20, 500 // max(num_heads, 1)))
    single_grid_h = cell_h * num_heads + _REGION_LABEL_H
    total_h = mt + single_grid_h * n_layers_shown + mb

    img = Image.new("RGB", (width, total_h), BG_COLOR)
    draw = ImageDraw.Draw(img)

    draw.text((ml, 10), "Per-Head Attention Grid", fill=TEXT_COLOR, font=ft_title)
    draw.text((ml, 28), f"{num_heads} heads x {n_regions} regions, "
              f"layers: {', '.join(str(l) for l in layers)}",
              fill=TEXT_DIM, font=ft_small)

    for li, layer in enumerate(layers):
        base_y = mt + li * single_grid_h
        draw.text((4, base_y + 6), f"L{layer}", fill=TEXT_COLOR, font=ft_label)

        heads_list = layer_index.get(layer, [])
        head_weights: dict[int, dict[str, float]] = {}
        for h in heads_list:
            head_weights[h.get("head_idx", 0)] = h.get("region_weights", {})

        # Find global max for this layer for normalization
        all_vals: list[float] = []
        for hi in range(num_heads):
            rw = head_weights.get(hi, {})
            for rn in region_names:
                all_vals.append(rw.get(rn, 0.0))
        vmax = max(all_vals) if all_vals else 1.0
        vmax = max(vmax, 1e-12)

        for hi in range(num_heads):
            rw = head_weights.get(hi, {})
            for ri, rn in enumerate(region_names):
                val = rw.get(rn, 0.0)
                normed = min(1.0, val / vmax)
                idx = min(255, int(normed * 255))
                c = (int(lut[idx][0]), int(lut[idx][1]), int(lut[idx][2]))

                x0 = ml + ri * cell_w + _CELL_PAD
                y0 = base_y + hi * cell_h + _CELL_PAD
                x1 = x0 + cell_w - 2 * _CELL_PAD
                y1 = y0 + cell_h - 2 * _CELL_PAD
                draw.rectangle([x0, y0, x1, y1], fill=c)

            # Head index label
            hy = base_y + hi * cell_h + cell_h // 2 - 4
            draw.text((ml - 24, hy), f"H{hi}", fill=TEXT_DIM, font=ft_small)

        # Region labels at bottom of this grid
        label_y = base_y + num_heads * cell_h + 4
        for ri, rn in enumerate(region_names):
            x = ml + ri * cell_w + 2
            short = rn[:cell_w // 6] if cell_w > 40 else rn[:5]
            draw.text((x, label_y), short, fill=TEXT_DIM, font=ft_small)

    return img


# ============================================================================
# SPECIALIZATION MODE
# ============================================================================

def render_head_specialization(
    per_head_data: dict[str, Any],
    region_names: list[str],
    num_heads: int,
    num_layers: int,
    width: int = 1400,
    height: int = 700,
) -> Image.Image:
    """Show each head's dominant region across phases of the forward pass.

    X axis = phase index, Y axis = head index.  Cell color = dominant region.
    """
    ft_title = get_font(13)
    ft_label = get_font(10)
    ft_small = get_font(9)

    phases = display_phases(num_layers)
    n_phases = len(phases)

    # Build per-layer head data index
    layer_index: dict[int, list[dict[str, Any]]] = {}
    for entry in per_head_data["per_layer"]:
        layer_index[entry["layer"]] = entry.get("heads", [])

    # Assign colors to regions
    region_col: dict[str, tuple[int, int, int]] = {}
    for ri, rn in enumerate(region_names):
        region_col[rn] = REGION_COLORS.get(rn,
                                           REGION_PALETTE[ri % len(REGION_PALETTE)])

    ml, mr, mt, mb = 55, 180, 60, 45
    pw = width - ml - mr
    ph = height - mt - mb
    cell_w = max(1, pw // n_phases)
    cell_h = max(1, ph // max(num_heads, 1))

    actual_h = mt + cell_h * num_heads + mb
    img = Image.new("RGB", (width, actual_h), BG_COLOR)
    draw = ImageDraw.Draw(img)

    draw.text((ml, 10), "Head Specialization by Phase",
              fill=TEXT_COLOR, font=ft_title)
    draw.text((ml, 28),
              f"{num_heads} heads, {n_phases} phases, "
              f"color = dominant region",
              fill=TEXT_DIM, font=ft_small)

    # For each phase, average head weights across layers in that phase
    for pi, (plabel, ps, pe) in enumerate(phases):
        x0 = ml + pi * cell_w

        # Phase column label
        draw.text((x0 + 2, mt - 14), plabel[:cell_w // 6],
                  fill=TEXT_DIM, font=ft_small)

        # Collect head weights across layers in this phase range
        head_accum: dict[int, dict[str, list[float]]] = {}
        for l in range(ps, pe + 1):
            heads = layer_index.get(l, [])
            for h in heads:
                hi = h.get("head_idx", 0)
                rw = h.get("region_weights", {})
                if hi not in head_accum:
                    head_accum[hi] = {rn: [] for rn in region_names}
                for rn in region_names:
                    head_accum[hi][rn].append(rw.get(rn, 0.0))

        for hi in range(num_heads):
            y0 = mt + hi * cell_h
            if hi in head_accum:
                means = {rn: float(np.mean(vals)) if vals else 0.0
                         for rn, vals in head_accum[hi].items()}
                dom = max(means, key=means.get)  # type: ignore[arg-type]
                col = region_col.get(dom, (100, 100, 100))
            else:
                col = (40, 40, 50)

            draw.rectangle([x0 + _CELL_PAD, y0 + _CELL_PAD,
                            x0 + cell_w - _CELL_PAD,
                            y0 + cell_h - _CELL_PAD],
                           fill=col)

    # Head labels
    for hi in range(num_heads):
        y = mt + hi * cell_h + cell_h // 2 - 4
        draw.text((ml - 28, y), f"H{hi}", fill=TEXT_DIM, font=ft_small)

    # Legend
    lx = width - 165
    ly = mt + 6
    draw.text((lx, ly - 2), "Regions", fill=TEXT_COLOR, font=ft_label)
    ly += 16
    for rn in region_names:
        col = region_col.get(rn, (100, 100, 100))
        draw.rectangle([(lx, ly + 1), (lx + 10, ly + 11)], fill=col)
        draw.text((lx + 14, ly), rn[:16], fill=col, font=ft_small)
        ly += 14

    return img


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-head attention visualization",
    )
    parser.add_argument("--result", required=True, help="Path to result JSON")
    parser.add_argument("--output", default=None, help="Output PNG path")
    parser.add_argument("--position", default="terminal")
    parser.add_argument("--layers", default="0,16,32,48",
                        help="Layers for grid mode (ignored in specialization)")
    parser.add_argument("--mode", choices=["grid", "specialization"],
                        default="grid")
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--colormap", default="viridis")
    args = parser.parse_args()

    if args.output is None:
        rp = Path(args.result)
        args.output = str(rp.parent / f"heads_{args.mode}_{rp.stem}.png")

    print(f"Loading per-head data: {args.result}")
    per_head = load_per_head_data(args.result, args.position)

    # Discover dimensions
    entries = per_head.get("per_layer", [])
    all_heads: set[int] = set()
    all_regions: set[str] = set()
    max_layer = 0
    for entry in entries:
        max_layer = max(max_layer, entry["layer"])
        for h in entry.get("heads", []):
            all_heads.add(h.get("head_idx", 0))
            all_regions.update(h.get("region_weights", {}).keys())

    num_heads = max(all_heads) + 1 if all_heads else 1
    num_layers = max_layer + 1
    region_names = sorted(all_regions)

    print(f"  {num_heads} heads, {num_layers} layers, "
          f"{len(region_names)} regions")

    if args.mode == "grid":
        layers = parse_layer_spec(args.layers, num_layers)
        img = render_head_grid(per_head, region_names, num_heads, layers,
                               width=args.width, colormap_name=args.colormap)
    else:
        img = render_head_specialization(per_head, region_names, num_heads,
                                         num_layers, width=args.width)

    img.save(args.output)
    print(f"Saved: {args.output} ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
