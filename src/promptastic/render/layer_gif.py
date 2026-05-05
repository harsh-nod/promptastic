#!/usr/bin/env python3
"""Animated GIF showing per-token attention sweeping through every layer.

Each frame renders the spatial heatmap for one layer.  Watching the
animation reveals forward-pass dynamics: early rule absorption, quiet
mid-layer compression, late current-message dominance.

Usage:
    python -m promptastic.render.layer_gif --result sample_01.json
    python -m promptastic.render.layer_gif --result sample_01.json --mask-chatml --fps 8
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .._types import RegionInfo
from ..constants import display_phases
from ._shared import (
    BG_COLOR,
    CHATML_COLOR,
    LEFT_MARGIN,
    LINE_HEIGHT,
    LINE_SPACING,
    RIGHT_MARGIN,
    SECTION_GAP,
    _COLORMAP_BUILDERS,
    colormap_lookup,
    draw_gradient_rect,
    gaussian_smooth,
    get_colormap,
    get_font,
    layout_tokens,
    normalize_weights,
)
from .loaders import load_all_layers


# ============================================================================
# FRAME RENDERER
# ============================================================================

_HEADER_H = 34
_TOKEN_PAD_X = 1


def render_single_layer_frame(
    token_labels: list[str],
    weights: np.ndarray,
    region_map: dict[str, RegionInfo],
    piece_boundaries: dict[str, RegionInfo],
    layer: int,
    width: int,
    smoothing: float,
    colormap_name: str,
    mask_chatml: bool,
    clip_low: float,
    position: str,
    result_path: str,
    target_height: int,
    num_layers: int = 64,
) -> Image.Image:
    """Render one layer's attention as a heatmap frame."""
    lut = get_colormap(colormap_name)
    ft = get_font(12)

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
    rects, _ = layout_tokens(
        token_labels, colors, piece_boundaries, region_map,
        False, ft, content_w,
    )
    rect_by_idx = {r["token_idx"]: r for r in rects}

    img = Image.new("RGB", (width, target_height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Header: layer number
    ft_big = get_font(18)
    draw.text((LEFT_MARGIN, 6),
              f"Layer {layer}/{num_layers - 1}",
              fill="#dddddd", font=ft_big)

    # Phase label
    ft_phase = get_font(11)
    phase_name = ""
    phase_col = "#888888"
    phase_hues = ["#6ec8be", "#888888", "#ff6b6b", "#ffd75e"]
    for pi, (plabel, ps, pe) in enumerate(display_phases(num_layers)):
        if ps <= layer <= pe:
            phase_name = plabel
            phase_col = phase_hues[min(pi, len(phase_hues) - 1)]
            break
    draw.text((LEFT_MARGIN + 155, 10), phase_name, fill=phase_col, font=ft_phase)

    # Case label
    stem = Path(result_path).stem
    draw.text((width - RIGHT_MARGIN - 190, 10),
              f"{stem}  pos={position}",
              fill="#666666", font=get_font(10))

    # Piece starts
    piece_at: dict[int, str] = {}
    for pn, pi in piece_boundaries.items():
        piece_at[pi["tok_start"]] = pn

    # Tokens
    yoff = _HEADER_H
    drawn: set[int] = set()

    for rect in rects:
        rx = LEFT_MARGIN + rect["x"]
        ry = yoff + rect["y"]
        rw = rect["w"]
        rh = rect["h"]
        ti = rect["token_idx"]

        if ti in piece_at and ti > 0 and ti not in drawn:
            drawn.add(ti)
            sy = ry - SECTION_GAP // 2
            draw.line([(LEFT_MARGIN, sy), (width - RIGHT_MARGIN, sy)],
                      fill="#ff4444", width=2)

        own = rect["color"]
        prev = rect_by_idx.get(ti - 1)
        if prev and prev["y"] == rect["y"]:
            pc = prev["color"]
            lc = ((pc[0] + own[0]) // 2, (pc[1] + own[1]) // 2,
                  (pc[2] + own[2]) // 2)
        else:
            lc = own

        nxt = rect_by_idx.get(ti + 1)
        if nxt and nxt["y"] == rect["y"]:
            nc = nxt["color"]
            rc = ((nc[0] + own[0]) // 2, (nc[1] + own[1]) // 2,
                  (nc[2] + own[2]) // 2)
        else:
            rc = own

        draw_gradient_rect(draw, int(rx), int(ry), int(rw), int(rh), lc, rc)
        draw.text((rx + _TOKEN_PAD_X, ry + 2), rect["text"],
                  fill=rect["fg"], font=ft)

    return img


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render animated layer-sweep GIF",
    )
    parser.add_argument("--result", required=True, help="Path to result JSON")
    parser.add_argument("--output", default=None, help="Output GIF path")
    parser.add_argument("--position", default="terminal",
                        help="Query position (default: terminal)")
    parser.add_argument("--fps", type=int, default=6,
                        help="Frames per second (default: 6)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Layer stride between frames (default: 1)")
    parser.add_argument("--smoothing", type=float, default=2.0,
                        help="Gaussian smoothing sigma")
    parser.add_argument("--colormap", default="inferno",
                        help=f"Colormap ({', '.join(_COLORMAP_BUILDERS)})")
    parser.add_argument("--width", type=int, default=2000, help="Frame width")
    parser.add_argument("--mask-chatml", action="store_true",
                        help="Mask ChatML tokens")
    parser.add_argument("--clip-low", type=float, default=5.0,
                        help="Floor trim percentage")
    args = parser.parse_args()

    if args.output is None:
        rp = Path(args.result)
        args.output = str(
            rp.parent / f"layersweep_{rp.stem}_{args.position}.gif")

    print(f"Loading: {args.result}")
    labels, lw, rmap, pieces = load_all_layers(args.result, args.position)

    sorted_layers = sorted(lw.keys())
    n_layers = len(sorted_layers)
    render_layers = sorted_layers[::args.stride]
    print(f"  {len(labels)} tokens, {n_layers} layers")
    print(f"  Rendering {len(render_layers)} frames (stride={args.stride})")

    # Pre-compute frame height from first layer
    lut = get_colormap(args.colormap)
    ft = get_font(12)
    cw = args.width - LEFT_MARGIN - RIGHT_MARGIN

    first_w = lw[render_layers[0]]
    sm = gaussian_smooth(first_w, args.smoothing)
    nr = normalize_weights(sm, clip_low=args.clip_low)
    cols = colormap_lookup(nr, lut)
    _, body_h = layout_tokens(labels, cols, pieces, rmap, False, ft, cw)
    frame_h = _HEADER_H + body_h + 18
    print(f"  Frame size: {args.width}x{frame_h}")

    frames: list[Image.Image] = []
    for fi, layer in enumerate(render_layers):
        frame = render_single_layer_frame(
            labels, lw[layer], rmap, pieces,
            layer=layer,
            width=args.width,
            smoothing=args.smoothing,
            colormap_name=args.colormap,
            mask_chatml=args.mask_chatml,
            clip_low=args.clip_low,
            position=args.position,
            result_path=args.result,
            target_height=frame_h,
            num_layers=n_layers,
        )
        frames.append(frame)
        print(f"  Frame {fi + 1}/{len(render_layers)}: Layer {layer}", end="\r")

    print(f"\n  Assembling GIF ({len(frames)} frames, {args.fps} fps)...")
    ms = int(1000 / args.fps)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=ms,
        loop=0,
    )
    mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Saved: {args.output} ({mb:.1f} MB, {len(frames)} frames, "
          f"{args.fps} fps)")


if __name__ == "__main__":
    main()
