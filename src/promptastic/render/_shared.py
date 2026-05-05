"""Visualization primitives for the promptastic render pipeline.

Provides colors, fonts, colormaps, signal processing, normalization,
token layout, and drawing helpers shared across all renderers.
"""

from __future__ import annotations

import numpy as np
from PIL import ImageDraw, ImageFont

from .._types import RegionInfo, TokenRect


# ============================================================================
# COLOR CONSTANTS (dark theme)
# ============================================================================

BG_COLOR = (18, 18, 26)
GRID_COLOR = (44, 44, 56)
AXIS_COLOR = (110, 110, 124)
TEXT_COLOR = (210, 210, 215)
TEXT_DIM = (125, 125, 138)
CHATML_COLOR = (55, 55, 55)


# ============================================================================
# REGION PALETTES
# ============================================================================

REGION_PALETTE: list[tuple[int, int, int]] = [
    (240, 100, 100),   # red
    (70, 210, 190),    # teal
    (250, 220, 95),    # amber
    (155, 145, 250),   # lavender
    (10, 220, 140),    # green
    (250, 150, 60),    # orange
    (110, 180, 250),   # blue
    (250, 110, 195),   # pink
    (175, 215, 80),    # lime
    (225, 60, 140),    # magenta
    (90, 225, 220),    # cyan
    (245, 195, 105),   # sand
    (100, 85, 225),    # purple
    (215, 245, 248),   # ice
    (245, 170, 155),   # peach
    (120, 230, 230),   # aqua
]

REGION_COLORS: dict[str, tuple[int, int, int]] = {
    "directive":           (250, 75, 75),
    "entity_rules":        (10, 215, 195),
    "passage_rules":       (195, 155, 10),
    "expansion_rules":     (135, 115, 215),
    "complexity_rules":    (10, 195, 95),
    "output_format":       (250, 155, 10),
    "conversation_turns":  (95, 155, 250),
    "current_message":     (250, 95, 195),
    "stored_passages":     (175, 195, 10),
    "task_reminders":      (145, 145, 145),
    "expansion_examples":  (195, 95, 250),
}


# ============================================================================
# LAYOUT CONSTANTS
# ============================================================================

LINE_HEIGHT = 20
LINE_SPACING = 2
SECTION_GAP = 12
LEFT_MARGIN = 16
RIGHT_MARGIN = 16


# ============================================================================
# FONT LOADING
# ============================================================================

def get_font(size: int) -> ImageFont.FreeTypeFont:
    """Load a monospace font at *size*, trying several system paths before
    falling back to the PIL built-in bitmap font."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "C:/Windows/Fonts/consola.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()  # type: ignore[return-value]


# ============================================================================
# COLORMAPS
# ============================================================================

def _lerp_lut(anchors: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Linearly interpolate a list of (index, R, G, B) anchors into a
    256-entry uint8 lookup table."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for seg in range(len(anchors) - 1):
        i0, r0, g0, b0 = anchors[seg]
        i1, r1, g1, b1 = anchors[seg + 1]
        span = i1 - i0
        for j in range(span):
            f = j / span
            lut[i0 + j] = [
                int(r0 + (r1 - r0) * f),
                int(g0 + (g1 - g0) * f),
                int(b0 + (b1 - b0) * f),
            ]
    lut[255] = [anchors[-1][1], anchors[-1][2], anchors[-1][3]]
    return lut


def _make_inferno() -> np.ndarray:
    return _lerp_lut([
        (0, 0, 0, 3), (20, 12, 7, 50), (40, 45, 12, 90),
        (60, 75, 13, 108), (80, 105, 22, 112), (100, 132, 38, 106),
        (120, 158, 58, 90), (140, 182, 78, 76), (160, 204, 104, 56),
        (180, 220, 130, 36), (200, 234, 158, 18), (220, 244, 186, 10),
        (240, 250, 218, 30), (255, 252, 254, 162),
    ])


def _make_viridis() -> np.ndarray:
    return _lerp_lut([
        (0, 68, 1, 84), (20, 72, 28, 108), (40, 68, 50, 122),
        (60, 60, 70, 133), (80, 50, 92, 138), (100, 42, 112, 142),
        (120, 34, 132, 142), (140, 28, 152, 138), (160, 38, 170, 128),
        (180, 70, 186, 110), (200, 110, 198, 90), (220, 160, 212, 60),
        (240, 210, 224, 30), (255, 253, 231, 37),
    ])


def _make_hot() -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.35:
            r, g, b = t / 0.35, 0.0, 0.0
        elif t < 0.7:
            r = 1.0
            g = (t - 0.35) / 0.35
            b = 0.0
        else:
            r, g = 1.0, 1.0
            b = (t - 0.7) / 0.3
        lut[i] = [int(r * 255), int(g * 255), int(b * 255)]
    return lut


def _make_coolwarm() -> np.ndarray:
    return _lerp_lut([
        (0, 55, 72, 190), (36, 95, 128, 220), (72, 138, 174, 236),
        (108, 182, 208, 242), (128, 228, 228, 228), (148, 238, 192, 166),
        (184, 226, 142, 102), (220, 206, 85, 55), (255, 178, 6, 40),
    ])


_COLORMAP_BUILDERS = {
    "inferno": _make_inferno,
    "viridis": _make_viridis,
    "hot": _make_hot,
    "coolwarm": _make_coolwarm,
}


def get_colormap(name: str) -> np.ndarray:
    """Return a 256x3 uint8 ndarray for the named colormap.

    Supported: inferno, viridis, hot, coolwarm.
    """
    builder = _COLORMAP_BUILDERS.get(name)
    if builder is None:
        available = ", ".join(sorted(_COLORMAP_BUILDERS))
        raise ValueError(f"Unknown colormap '{name}'. Choose from: {available}")
    return builder()


# ============================================================================
# SIGNAL PROCESSING
# ============================================================================

def gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    """Apply 1-D Gaussian smoothing via convolution.  Output length matches
    input length."""
    if sigma <= 0:
        return values.copy()
    radius = int(sigma * 3)
    k_size = 2 * radius + 1
    if k_size > len(values):
        k_size = len(values) if len(values) % 2 == 1 else max(1, len(values) - 1)
        radius = k_size // 2
    xs = np.arange(k_size) - radius
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


# ============================================================================
# NORMALIZATION
# ============================================================================

def normalize_weights(
    weights: np.ndarray,
    clip_low: float = 5.0,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Rank-based histogram equalization to [0, 1].

    Assigns each token a value proportional to its rank among all tokens,
    eliminating the power-law skew inherent in raw attention.

    Parameters
    ----------
    clip_low : float
        Bottom *clip_low* percent of ranks map to 0.
    mask : array of bool, optional
        If given, only tokens where mask==True participate in ranking.
        Masked-out tokens receive 0.
    """
    out = np.zeros(len(weights), dtype=np.float64)
    idx = np.where(mask)[0] if mask is not None else np.arange(len(weights))
    if len(idx) == 0:
        return out

    vals = weights[idx]
    order = np.argsort(vals)
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(len(order), dtype=np.float64)

    # Average ties
    sorted_vals = vals[order]
    i = 0
    while i < len(sorted_vals):
        j = i + 1
        while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j > i + 1:
            tie_rank = float(np.mean(ranks[order[i:j]]))
            ranks[order[i:j]] = tie_rank
        i = j

    n = len(ranks)
    normed = ranks / max(1, n - 1)

    if clip_low > 0:
        threshold = clip_low / 100.0
        normed = np.clip((normed - threshold) / (1.0 - threshold), 0.0, 1.0)

    out[idx] = normed
    return out


# ============================================================================
# COLORMAP HELPERS
# ============================================================================

def colormap_lookup(
    normed: np.ndarray, lut: np.ndarray,
) -> list[tuple[int, int, int]]:
    """Map an array of [0, 1] values to RGB tuples through a 256-entry LUT."""
    indices = np.clip((normed * 255).astype(int), 0, 255)
    rgb = lut[indices]
    return [(int(r), int(g), int(b)) for r, g, b in rgb]


def text_color_for_bg(r: int, g: int, b: int) -> str:
    """Return ``'white'`` or ``'black'`` (as hex) for legibility on the given
    background color."""
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "#ffffff" if lum < 140 else "#000000"


# ============================================================================
# TOKEN TEXT HELPERS
# ============================================================================

def sanitize_token(label: str) -> str:
    """Replace control characters with printable escapes, keeping the text
    readable for rendering."""
    parts: list[str] = []
    for ch in label:
        code = ord(ch)
        if ch == "\t":
            parts.append("  ")
        elif ch == "\r":
            pass
        elif code < 32 and ch != "\n":
            parts.append(f"\\x{code:02x}")
        else:
            parts.append(ch)
    return "".join(parts)


def is_newline_token(label: str) -> bool:
    """True when the token is purely a line-break character."""
    return label.strip("\r") == "\n"


# ============================================================================
# LAYER SPEC PARSING
# ============================================================================

def parse_layer_spec(spec: str, num_layers: int = 64) -> list[int]:
    """Parse a human-friendly layer specification string.

    Accepted forms:
    - ``"final"``     -- last FINAL_LAYERS layers
    - ``"all"``       -- every layer
    - ``"48"``        -- single layer
    - ``"0,16,32"``   -- comma-separated list
    - ``"60-63"``     -- inclusive range
    - ``"0,16,32,60-63"`` -- mixed
    """
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(num_layers))
    if spec == "final":
        from ..constants import FINAL_LAYERS
        start = max(0, num_layers - FINAL_LAYERS)
        return list(range(start, num_layers))

    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            layers.update(range(int(lo), int(hi) + 1))
        else:
            layers.add(int(part))
    return sorted(layers)


# ============================================================================
# TOKEN LAYOUT ENGINE
# ============================================================================

def layout_tokens(
    token_labels: list[str],
    colors: list[tuple[int, int, int]],
    piece_boundaries: dict[str, RegionInfo],
    region_map: dict[str, RegionInfo],
    show_regions: bool,
    font: ImageFont.FreeTypeFont,
    content_width: int,
) -> tuple[list[TokenRect], int]:
    """Compute pixel rectangles for every token using left-to-right flow with
    line wrapping.

    Newline tokens force a break.  Piece boundaries insert extra vertical
    space.  Returns ``(rects, total_height)``.
    """
    # Pre-index piece starts
    piece_at: dict[int, str] = {}
    for pname, pinfo in piece_boundaries.items():
        piece_at[pinfo["tok_start"]] = pname

    # Pre-index region starts (skip containers)
    region_at: dict[int, str] = {}
    if show_regions:
        _skip = {"system_prompt", "user_message", "response", "chat_template"}
        for rname, rinfo in region_map.items():
            if rname not in _skip:
                region_at[rinfo["tok_start"]] = rname

    rects: list[TokenRect] = []
    cx, cy = 0.0, 0.0
    pad_x = 1

    for i, raw in enumerate(token_labels):
        # Piece separator gap
        if i > 0 and i in piece_at:
            if cx > 0:
                cy += LINE_HEIGHT + LINE_SPACING
            cy += SECTION_GAP
            cx = 0.0

        display = sanitize_token(raw)

        if is_newline_token(raw):
            w = max(content_width - cx, float(LINE_HEIGHT))
            rects.append({
                "x": cx, "y": cy,
                "w": w, "h": LINE_HEIGHT,
                "color": colors[i],
                "fg": text_color_for_bg(*colors[i]),
                "text": "\\n",
                "token_idx": i,
            })
            cy += LINE_HEIGHT + LINE_SPACING
            cx = 0.0
            continue

        bbox = font.getbbox(display) if display else (0, 0, 4, 0)
        tw = (bbox[2] - bbox[0]) + pad_x * 2

        if cx > 0 and cx + tw > content_width:
            cy += LINE_HEIGHT + LINE_SPACING
            cx = 0.0

        rects.append({
            "x": cx, "y": cy,
            "w": tw, "h": LINE_HEIGHT,
            "color": colors[i],
            "fg": text_color_for_bg(*colors[i]),
            "text": display,
            "token_idx": i,
        })
        cx += tw

    total_h = cy + LINE_HEIGHT
    return rects, int(total_h)


# ============================================================================
# DRAWING HELPERS
# ============================================================================

def draw_gradient_rect(
    draw: ImageDraw.ImageDraw,
    x: int, y: int, w: int, h: int,
    color_left: tuple[int, int, int],
    color_right: tuple[int, int, int],
) -> None:
    """Draw a filled rectangle with a horizontal linear gradient between
    two colors."""
    w = max(1, w)
    for col in range(w):
        t = col / max(1, w - 1)
        r = int(color_left[0] + (color_right[0] - color_left[0]) * t)
        g = int(color_left[1] + (color_right[1] - color_left[1]) * t)
        b = int(color_left[2] + (color_right[2] - color_left[2]) * t)
        draw.line([(x + col, y), (x + col, y + h)], fill=(r, g, b))
