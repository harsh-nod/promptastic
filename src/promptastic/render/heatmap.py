#!/usr/bin/env python3
"""Per-token attention heatmap renderer.

Produces a PNG where every token is colored by its attention weight,
using rank-based normalization to handle the power-law distribution.

Usage:
    python -m promptastic.render.heatmap --result sample_01.json
    python -m promptastic.render.heatmap --result sample_01.json --mask-chatml --clip-low 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .._types import RegionInfo
from ._shared import (
    BG_COLOR,
    CHATML_COLOR,
    LEFT_MARGIN,
    RIGHT_MARGIN,
    SECTION_GAP,
    TEXT_COLOR,
    TEXT_DIM,
    colormap_lookup,
    draw_gradient_rect,
    gaussian_smooth,
    get_colormap,
    get_font,
    layout_tokens,
    normalize_weights,
    _COLORMAP_BUILDERS,
)
from .loaders import load_heatmap_data


# ============================================================================
# CONSTANTS
# ============================================================================

_TOP_PAD = 48
_LEGEND_GAP = 28
_TOKEN_PAD_X = 1


# ============================================================================
# LEGEND
# ============================================================================

def _render_legend(
    draw: ImageDraw.ImageDraw,
    lut: np.ndarray,
    y: int,
    img_width: int,
    position: str,
    layer_spec: str,
    smoothing: float,
    raw_range: tuple[float, float],
    show_regions: bool,
    n_tokens: int,
    cmap_name: str,
) -> int:
    """Draw an interpretive legend at the bottom. Returns total height used."""
    ft = get_font(11)
    ft_sm = get_font(10)
    x0 = LEFT_MARGIN
    y0 = y

    draw.line([(x0, y), (img_width - RIGHT_MARGIN, y)], fill="#404048", width=1)
    y += 8

    draw.text((x0, y), "KEY", fill="#aaaaaa", font=get_font(13))
    y += 20

    # Colorbar
    bar_w, bar_h = 280, 14
    draw.text((x0, y + 1), "Attention:", fill="#999999", font=ft)
    bx = x0 + 76
    for i in range(bar_w):
        idx = min(255, int(i / (bar_w - 1) * 255))
        c = (int(lut[idx][0]), int(lut[idx][1]), int(lut[idx][2]))
        draw.line([(bx + i, y), (bx + i, y + bar_h)], fill=c)
    draw.rectangle([bx - 1, y - 1, bx + bar_w, y + bar_h], outline="#555555")
    draw.text((bx, y + bar_h + 2), "Low", fill="#777777", font=ft_sm)
    draw.text((bx + bar_w - 22, y + bar_h + 2), "High", fill="#777777", font=ft_sm)
    draw.text((bx + bar_w + 10, y + 1),
              f"(raw: {raw_range[0]:.2e} -- {raw_range[1]:.2e}, rank-normalized)",
              fill="#666666", font=ft_sm)
    y += bar_h + 18

    # Section boundary
    draw.rectangle([x0, y + 4, x0 + 18, y + 6], fill="#ff4444")
    draw.text((x0 + 26, y),
              "Section boundary -- system_prompt / user_message / response",
              fill="#999999", font=ft)
    y += 18

    if show_regions:
        draw.rectangle([x0, y + 2, x0 + 1, y + 14], fill="#00ff88")
        draw.text((x0 + 26, y), "Region boundary -- sub-regions within sections",
                  fill="#999999", font=ft)
        y += 18

    draw.text((x0 + 2, y), "\\n", fill="#cccccc", font=ft)
    draw.text((x0 + 26, y),
              "Newline token -- fills remaining line width",
              fill="#999999", font=ft)
    y += 22

    draw.text((x0, y), "Parameters", fill="#aaaaaa", font=get_font(12))
    y += 16
    for line in [
        f"Query position: {position}",
        f"Layers averaged: {layer_spec}",
        f"Smoothing sigma: {smoothing}" if smoothing > 0 else "Smoothing: none",
        f"Colormap: {cmap_name}",
        f"Total tokens: {n_tokens}",
    ]:
        draw.text((x0 + 6, y), line, fill="#888888", font=ft_sm)
        y += 14

    y += 8
    return y - y0


# ============================================================================
# RENDERER
# ============================================================================

def render_heatmap(
    token_labels: list[str],
    weights: np.ndarray,
    region_map: dict[str, RegionInfo],
    piece_boundaries: dict[str, RegionInfo],
    width: int,
    smoothing: float,
    colormap_name: str,
    show_regions: bool,
    position: str,
    layer_spec: str,
    result_path: str,
    clip_low: float = 5.0,
    mask_chatml: bool = False,
) -> Image.Image:
    """Render a full per-token attention heatmap as a PIL Image."""
    lut = get_colormap(colormap_name)
    ft = get_font(12)
    raw_lo, raw_hi = float(weights.min()), float(weights.max())

    # Optional ChatML masking
    chatml_mask: np.ndarray | None = None
    chatml_set: set[int] = set()
    if mask_chatml:
        chatml_mask = np.array([
            not (tok.startswith("<|im_") or (tok.endswith("|>") and "im" in tok))
            for tok in token_labels
        ])
        chatml_set = {i for i, m in enumerate(chatml_mask) if not m}

    smoothed = gaussian_smooth(weights, smoothing)
    normed = normalize_weights(smoothed, clip_low=clip_low, mask=chatml_mask)
    colors = colormap_lookup(normed, lut)

    for i in chatml_set:
        if i < len(colors):
            colors[i] = CHATML_COLOR

    content_w = width - LEFT_MARGIN - RIGHT_MARGIN
    rects, body_h = layout_tokens(
        token_labels, colors, piece_boundaries, region_map,
        show_regions, ft, content_w,
    )

    rect_by_idx = {r["token_idx"]: r for r in rects}
    legend_est = 210
    total_h = _TOP_PAD + body_h + _LEGEND_GAP + legend_est

    img = Image.new("RGB", (width, int(total_h)), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Title
    stem = Path(result_path).stem
    draw.text((LEFT_MARGIN, 10),
              f"Attention Heatmap -- {stem}",
              fill="#dddddd", font=get_font(14))
    sub = f"position={position}  layers={layer_spec}"
    if smoothing > 0:
        sub += f"  smoothing={smoothing}"
    draw.text((LEFT_MARGIN, 28), sub, fill="#888888", font=get_font(10))

    # Pre-index boundaries
    piece_at: dict[int, str] = {}
    for pn, pi in piece_boundaries.items():
        piece_at[pi["tok_start"]] = pn

    region_at: dict[int, str] = {}
    if show_regions:
        skip = {"system_prompt", "user_message", "response", "chat_template"}
        for rn, ri in region_map.items():
            if rn not in skip:
                region_at[ri["tok_start"]] = rn

    # Draw tokens
    yoff = _TOP_PAD
    drawn_sections: set[int] = set()

    for rect in rects:
        rx = LEFT_MARGIN + rect["x"]
        ry = yoff + rect["y"]
        rw = rect["w"]
        rh = rect["h"]
        ti = rect["token_idx"]

        # Section separator
        if ti in piece_at and ti > 0 and ti not in drawn_sections:
            drawn_sections.add(ti)
            sy = ry - SECTION_GAP // 2
            draw.line([(LEFT_MARGIN, sy), (width - RIGHT_MARGIN, sy)],
                      fill="#ff4444", width=2)
            draw.text((LEFT_MARGIN + 4, sy - 13),
                      "> " + piece_at[ti].replace("_", " "),
                      fill="#ff6666", font=get_font(10))

        # Region marker
        if show_regions and ti in region_at:
            draw.line([(rx, ry), (rx, ry + rh)], fill="#00ff88", width=1)
            if rect["x"] < 2:
                draw.text((rx + 3, ry - 11), region_at[ti][:24],
                          fill="#00cc66", font=get_font(9))

        # Gradient background blended with neighbours
        own_c = rect["color"]
        prev = rect_by_idx.get(ti - 1)
        if prev is not None and prev["y"] == rect["y"]:
            pc = prev["color"]
            lc = ((pc[0] + own_c[0]) // 2, (pc[1] + own_c[1]) // 2,
                  (pc[2] + own_c[2]) // 2)
        else:
            lc = own_c

        nxt = rect_by_idx.get(ti + 1)
        if nxt is not None and nxt["y"] == rect["y"]:
            nc = nxt["color"]
            rc = ((nc[0] + own_c[0]) // 2, (nc[1] + own_c[1]) // 2,
                  (nc[2] + own_c[2]) // 2)
        else:
            rc = own_c

        draw_gradient_rect(draw, int(rx), int(ry), int(rw), int(rh), lc, rc)
        draw.text((rx + _TOKEN_PAD_X, ry + 2), rect["text"],
                  fill=rect["fg"], font=ft)

    # Legend
    leg_y = yoff + body_h + _LEGEND_GAP
    leg_h = _render_legend(
        draw, lut, leg_y, width,
        position, layer_spec, smoothing,
        (raw_lo, raw_hi), show_regions,
        len(token_labels), colormap_name,
    )

    final_h = leg_y + leg_h + 16
    if final_h < total_h:
        img = img.crop((0, 0, width, int(final_h)))

    return img


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render per-token attention heatmap",
    )
    parser.add_argument("--result", required=True,
                        help="Path to result JSON (must include per-token data)")
    parser.add_argument("--output", default=None, help="Output PNG path")
    parser.add_argument("--position", default="terminal",
                        help="Query position (default: terminal)")
    parser.add_argument("--layers", default="final",
                        help="Layer spec: 'final', 'all', '48', '60-63'")
    parser.add_argument("--smoothing", type=float, default=0.0,
                        help="Gaussian smoothing sigma (default: 0)")
    parser.add_argument("--colormap", default="inferno",
                        help=f"Colormap ({', '.join(_COLORMAP_BUILDERS)})")
    parser.add_argument("--width", type=int, default=1800, help="Image width")
    parser.add_argument("--clip-low", type=float, default=5.0,
                        help="Floor trim -- bottom X%% ranked tokens become black")
    parser.add_argument("--mask-chatml", action="store_true",
                        help="Exclude ChatML tokens from ranking")
    parser.add_argument("--show-regions", action="store_true",
                        help="Show region boundary markers")
    args = parser.parse_args()

    if args.output is None:
        rp = Path(args.result)
        args.output = str(rp.parent / f"heatmap_{rp.stem}_{args.position}.png")

    print(f"Loading: {args.result}")
    labels, weights, rmap, pieces = load_heatmap_data(
        args.result, args.position, args.layers,
    )
    print(f"  {len(labels)} tokens, {len(rmap)} regions")
    print(f"  Attention range: [{weights.min():.6f}, {weights.max():.6f}]")

    img = render_heatmap(
        labels, weights, rmap, pieces,
        width=args.width,
        smoothing=args.smoothing,
        colormap_name=args.colormap,
        show_regions=args.show_regions,
        position=args.position,
        layer_spec=args.layers,
        result_path=args.result,
        clip_low=args.clip_low,
        mask_chatml=args.mask_chatml,
    )
    img.save(args.output)
    print(f"Saved: {args.output} ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
