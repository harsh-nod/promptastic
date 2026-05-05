"""Table output helpers for analysis scripts.

Provides formatted string builders for floats, percentages, deltas,
and section headers used across all analysis output.
"""

from __future__ import annotations

from math import isnan


def fmt(val: float, width: int = 9) -> str:
    """Format a float with smart precision based on magnitude.

    Handles NaN, zero, and very small values with appropriate precision.
    Returns right-justified string of the requested width.
    """
    if isnan(val):
        return "N/A".rjust(width)
    if val == 0:
        return "0.0000".rjust(width)
    magnitude = abs(val)
    if magnitude < 1e-4:
        return f"{val:.2e}".rjust(width)
    if magnitude < 1e-3:
        return f"{val:.5f}".rjust(width)
    if magnitude < 1.0:
        return f"{val:.4f}".rjust(width)
    if magnitude < 100.0:
        return f"{val:.3f}".rjust(width)
    return f"{val:.1f}".rjust(width)


def pct(val: float, width: int = 8) -> str:
    """Format a value as a percentage string.

    Multiplies by 100 and appends '%'. NaN values render as 'N/A'.
    """
    if isnan(val):
        return "N/A".rjust(width)
    return f"{val * 100:.3f}%".rjust(width)


def delta_str(val: float, ref: float, width: int = 9) -> str:
    """Format the percentage change from ref to val.

    Returns '+X.Y%' or '-X.Y%'. If ref is zero or either value is NaN,
    returns 'N/A'.
    """
    if isnan(val) or isnan(ref) or ref == 0:
        return "N/A".rjust(width)
    change = (val - ref) / abs(ref) * 100
    prefix = "+" if change >= 0 else ""
    return f"{prefix}{change:.1f}%".rjust(width)


def print_header(title: str, width: int = 90) -> None:
    """Print a bordered section header with double-line borders."""
    border = "=" * width
    print(f"\n{border}")
    print(f"  {title}")
    print(border)


def print_subheader(title: str) -> None:
    """Print a lighter subsection header with dashes."""
    print(f"\n  --- {title} ---")
