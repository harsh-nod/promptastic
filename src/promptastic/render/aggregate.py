#!/usr/bin/env python3
"""Multi-sample aggregate cooking curves with confidence bands.

Averages per-region attention across all samples at every layer, optionally
overlaying multiple variants for comparison.

Usage:
    python -m promptastic.render.aggregate --base-dir ./data --variants baseline:Baseline
    python -m promptastic.render.aggregate --base-dir ./data --variants baseline:Base composite:Comp --compare
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ..constants import SKIP_REGIONS, display_phases
from ._shared import BG_COLOR, REGION_COLORS, TEXT_COLOR, TEXT_DIM, get_font
from .loaders import load_variant_curves


# ============================================================================
# DEFAULT REGIONS
# ============================================================================

_DEFAULT_REGIONS = [
    "entity_rules", "passage_rules", "expansion_rules", "complexity_rules",
    "directive", "output_format",
    "conversation_turns", "current_message", "stored_passages",
]

# Line styles for comparison overlay
_VARIANT_STYLES: list[dict[str, Any]] = [
    {"dash": None},
    {"dash": [8, 4]},
    {"dash": [3, 3]},
    {"dash": [12, 3, 3, 3]},
]


# ============================================================================
# SINGLE VARIANT
# ============================================================================

def render_single_variant(
    curves: dict[str, np.ndarray],
    dirname: str,
    regions: list[str],
    normalize: str,
    output: Path,
    width: int = 1400,
    height: int = 700,
    show_bands: bool = True,
) -> None:
    """Render aggregate cooking curves for one variant, with optional
    +/-1 std shaded bands.  Saves directly to *output*."""
    n_layers = next(iter(curves.values())).shape[1] if curves else 64
    n_samples = next(iter(curves.values())).shape[0] if curves else 0

    ml, mr, mt, mb = 75, 240, 75, 55
    pw = width - ml - mr
    ph = height - mt - mb

    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    ft = get_font(12)
    ft_sm = get_font(10)
    ft_title = get_font(15)

    draw.text((8, 6),
              f"Aggregate Cooking Curves -- {dirname}  [n={n_samples}, {normalize}]",
              fill=TEXT_COLOR, font=ft_title)
    draw.text((8, 26),
              "Mean attention per token (shaded = +/-1 std)",
              fill=TEXT_DIM, font=ft_sm)

    # Phase dividers
    for plabel, ps, pe in display_phases(n_layers):
        pe_c = min(pe, n_layers - 1)
        xs = ml + int(ps / (n_layers - 1) * pw)
        xe = ml + int(pe_c / (n_layers - 1) * pw)
        xm = (xs + xe) // 2
        draw.text((xm - len(plabel) * 3, mt - 16), plabel,
                  fill=(115, 115, 128), font=ft_sm)
        draw.line([(xs, mt - 4), (xs, mt + ph)], fill=(55, 55, 68), width=1)

    # Stats
    stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for rname in regions:
        if rname not in curves:
            continue
        arr = curves[rname]
        mu = np.mean(arr, axis=0)
        sd = np.std(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(n_layers)
        if normalize == "per-region":
            pk = float(np.max(mu))
            if pk > 0:
                sd = sd / pk
                mu = mu / pk
        stats[rname] = (mu, sd)

    # Y range
    y_ceil = 0.0
    for mu, sd in stats.values():
        y_ceil = max(y_ceil, float(np.max(mu + sd)))
    y_ceil *= 1.12

    # Horizontal grid
    for frac in np.linspace(0, y_ceil, 6):
        gy = mt + ph - int(frac / max(y_ceil, 1e-15) * ph)
        if mt <= gy <= mt + ph:
            draw.line([(ml, gy), (ml + pw, gy)], fill=(48, 48, 58), width=1)
            draw.text((ml - 34, gy - 5), f"{frac:.2f}",
                      fill=(115, 115, 128), font=ft_sm)

    # X ticks
    for l in range(0, n_layers, max(1, n_layers // 8)):
        x = ml + int(l / (n_layers - 1) * pw)
        draw.text((x - 6, mt + ph + 6), f"L{l}",
                  fill=(115, 115, 128), font=ft_sm)
    draw.text((ml + pw - 14, mt + ph + 6), f"L{n_layers - 1}",
              fill=(115, 115, 128), font=ft_sm)

    # Confidence bands
    if show_bands:
        for rname, (mu, sd) in stats.items():
            col = REGION_COLORS.get(rname, (175, 175, 175))
            band_rgba = (*col, 35)

            overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)

            upper: list[tuple[int, int]] = []
            lower: list[tuple[int, int]] = []
            for l in range(n_layers):
                x = ml + int(l / (n_layers - 1) * pw)
                hi = mu[l] + sd[l]
                lo = mu[l] - sd[l]
                yh = mt + ph - int(hi / max(y_ceil, 1e-15) * ph)
                yl = mt + ph - int(lo / max(y_ceil, 1e-15) * ph)
                upper.append((x, max(mt, min(mt + ph, yh))))
                lower.append((x, max(mt, min(mt + ph, yl))))

            polygon = upper + lower[::-1]
            if len(polygon) >= 3:
                od.polygon(polygon, fill=band_rgba)
            img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
            draw = ImageDraw.Draw(img)

    # Mean lines + legend
    leg_y = mt + 8
    for rname, (mu, sd) in stats.items():
        col = REGION_COLORS.get(rname, (175, 175, 175))
        pk_l = int(np.argmax(mu))

        pts: list[tuple[int, int]] = []
        for l in range(n_layers):
            x = ml + int(l / (n_layers - 1) * pw)
            y = mt + ph - int(mu[l] / max(y_ceil, 1e-15) * ph)
            pts.append((x, max(mt, min(mt + ph, y))))

        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=col, width=2)

        # Peak dot
        px, py = pts[pk_l]
        draw.ellipse([(px - 3, py - 3), (px + 3, py + 3)], fill=col)

        # Legend entry
        lx = width - 224
        draw.rectangle([(lx, leg_y), (lx + 10, leg_y + 10)], fill=col)
        draw.text((lx + 14, leg_y - 1), f"{rname} (L{pk_l})",
                  fill=col, font=ft_sm)
        leg_y += 15

    draw.text((ml + pw // 2 - 16, height - 18), "Layer",
              fill=TEXT_DIM, font=ft)

    img.save(str(output))
    print(f"Saved: {output} ({width}x{height})")


# ============================================================================
# COMPARISON MODE
# ============================================================================

def render_comparison(
    all_curves: dict[str, dict[str, np.ndarray]],
    regions: list[str],
    normalize: str,
    output: Path,
    width: int = 1400,
    height: int = 700,
) -> None:
    """Overlay mean curves from multiple variants."""
    n_layers = 64
    for vc in all_curves.values():
        for arr in vc.values():
            n_layers = arr.shape[1]
            break
        break

    ml, mr, mt, mb = 75, 290, 75, 55
    pw = width - ml - mr
    ph = height - mt - mb

    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    ft = get_font(12)
    ft_sm = get_font(10)
    ft_title = get_font(15)

    vnames = list(all_curves.keys())
    draw.text((8, 6),
              f"Cooking Curve Comparison -- {' vs '.join(vnames)}  [{normalize}]",
              fill=TEXT_COLOR, font=ft_title)

    style_desc = []
    labels = ["solid", "dashed", "dotted", "dash-dot"]
    for vi, vn in enumerate(vnames):
        style_desc.append(f"{labels[vi % len(labels)]}={vn}")
    draw.text((8, 26), ", ".join(style_desc), fill=TEXT_DIM, font=ft_sm)

    # Phase dividers
    for plabel, ps, pe in display_phases(n_layers):
        pe_c = min(pe, n_layers - 1)
        xs = ml + int(ps / (n_layers - 1) * pw)
        xe = ml + int(pe_c / (n_layers - 1) * pw)
        xm = (xs + xe) // 2
        draw.text((xm - len(plabel) * 3, mt - 16), plabel,
                  fill=(115, 115, 128), font=ft_sm)
        draw.line([(xs, mt - 4), (xs, mt + ph)], fill=(55, 55, 68), width=1)

    # Compute all means
    vmeans: dict[str, dict[str, np.ndarray]] = {}
    y_ceil = 0.0
    for vn, vc in all_curves.items():
        vmeans[vn] = {}
        for rname in regions:
            if rname not in vc:
                continue
            mu = np.mean(vc[rname], axis=0)
            if normalize == "per-region":
                pk = float(np.max(mu))
                if pk > 0:
                    mu = mu / pk
            vmeans[vn][rname] = mu
            y_ceil = max(y_ceil, float(np.max(mu)))
    y_ceil *= 1.12

    # Grid
    for frac in np.linspace(0, y_ceil, 6):
        gy = mt + ph - int(frac / max(y_ceil, 1e-15) * ph)
        if mt <= gy <= mt + ph:
            draw.line([(ml, gy), (ml + pw, gy)], fill=(48, 48, 58), width=1)
            draw.text((ml - 34, gy - 5), f"{frac:.2f}",
                      fill=(115, 115, 128), font=ft_sm)

    for l in range(0, n_layers, max(1, n_layers // 8)):
        x = ml + int(l / (n_layers - 1) * pw)
        draw.text((x - 6, mt + ph + 6), f"L{l}",
                  fill=(115, 115, 128), font=ft_sm)
    draw.text((ml + pw - 14, mt + ph + 6), f"L{n_layers - 1}",
              fill=(115, 115, 128), font=ft_sm)

    # Plot
    leg_y = mt + 8
    for rname in regions:
        col = REGION_COLORS.get(rname, (175, 175, 175))
        any_drawn = False

        for vi, (vn, means) in enumerate(vmeans.items()):
            if rname not in means:
                continue
            mu = means[rname]
            style = _VARIANT_STYLES[vi % len(_VARIANT_STYLES)]

            pts: list[tuple[int, int]] = []
            for l in range(n_layers):
                x = ml + int(l / (n_layers - 1) * pw)
                y = mt + ph - int(mu[l] / max(y_ceil, 1e-15) * ph)
                pts.append((x, max(mt, min(mt + ph, y))))

            if style["dash"] is None:
                for i in range(len(pts) - 1):
                    draw.line([pts[i], pts[i + 1]], fill=col, width=2)
            else:
                pattern = style["dash"]
                seg_idx = 0
                drawing = True
                counter = 0
                for i in range(len(pts) - 1):
                    if drawing:
                        draw.line([pts[i], pts[i + 1]], fill=col, width=2)
                    counter += 1
                    if counter >= pattern[seg_idx]:
                        counter = 0
                        seg_idx = (seg_idx + 1) % len(pattern)
                        drawing = not drawing

            any_drawn = True

        if any_drawn:
            lx = width - 270
            draw.rectangle([(lx, leg_y), (lx + 10, leg_y + 10)], fill=col)
            draw.text((lx + 14, leg_y - 1), rname, fill=col, font=ft_sm)
            leg_y += 15

    # Variant style legend
    leg_y += 10
    for vi, vn in enumerate(vnames):
        style = _VARIANT_STYLES[vi % len(_VARIANT_STYLES)]
        y = leg_y + vi * 15
        lx = width - 270
        if style["dash"] is None:
            draw.line([(lx, y + 5), (lx + 18, y + 5)],
                      fill=(195, 195, 195), width=2)
        else:
            seg_on = style["dash"][0]
            for s in range(0, 18, sum(style["dash"][:2])):
                draw.line([(lx + s, y + 5), (lx + s + seg_on, y + 5)],
                          fill=(195, 195, 195), width=2)
        draw.text((lx + 24, y - 1), vn, fill=(195, 195, 195), font=ft_sm)

    draw.text((ml + pw // 2 - 16, height - 18), "Layer",
              fill=TEXT_DIM, font=ft)

    img.save(str(output))
    print(f"Saved: {output} ({width}x{height})")


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate cooking curves across samples",
    )
    parser.add_argument("--base-dir", required=True,
                        help="Base directory with variant result dirs")
    parser.add_argument("--variants", nargs="+", required=True,
                        help="name:Label pairs for each variant directory")
    parser.add_argument("--normalize", choices=["raw", "per-region"],
                        default="per-region")
    parser.add_argument("--output", default=None, help="Output PNG path")
    parser.add_argument("--regions", default=None,
                        help="Comma-separated region names")
    parser.add_argument("--compare", action="store_true",
                        help="Overlay variants (default for >1 variant)")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=700)
    args = parser.parse_args()

    base = Path(args.base_dir)
    regions = args.regions.split(",") if args.regions else _DEFAULT_REGIONS

    # Parse name:Label pairs
    parsed: list[tuple[str, str]] = []
    for entry in args.variants:
        if ":" in entry:
            dirname, label = entry.split(":", 1)
        else:
            dirname = entry
            label = entry.replace("results_", "")
        parsed.append((dirname, label))

    if args.compare or len(parsed) > 1:
        all_curves: dict[str, dict[str, np.ndarray]] = {}
        for dirname, label in parsed:
            print(f"Loading {dirname}...")
            all_curves[label] = load_variant_curves(base, dirname)
            n = next(iter(all_curves[label].values())).shape[0]
            print(f"  {n} samples loaded")

        out = (Path(args.output) if args.output else
               base / f"comparison_{'_vs_'.join(l for _, l in parsed)}.png")
        render_comparison(all_curves, regions, args.normalize,
                          out, args.width, args.height)
    else:
        dirname, label = parsed[0]
        print(f"Loading {dirname}...")
        curves = load_variant_curves(base, dirname)
        n = next(iter(curves.values())).shape[0]
        print(f"  {n} samples loaded")

        out = (Path(args.output) if args.output else
               base / f"aggregate_{label}_{args.normalize}.png")
        render_single_variant(curves, label, regions, args.normalize,
                              out, args.width, args.height)


if __name__ == "__main__":
    main()
