"""Tests for promptastic.constants -- phase boundary scaling."""

from promptastic.constants import (
    FINAL_LAYERS,
    SKIP_REGIONS,
    display_phases,
    analysis_phases,
)


# ---------------------------------------------------------------
# display_phases
# ---------------------------------------------------------------


def test_display_phases_64_layers():
    phases = display_phases(64)
    assert len(phases) == 4
    # Each entry is (name, start, end)
    names = [p[0] for p in phases]
    assert "Rules absorbed" in names
    assert "Output formatting" in names


def test_display_phases_32_layers():
    phases = display_phases(32)
    assert len(phases) == 4
    # First phase should start at 0
    assert phases[0][1] == 0
    # Last phase should end at 31 (num_layers - 1)
    assert phases[-1][2] == 31


def test_display_phases_80_layers():
    phases = display_phases(80)
    assert len(phases) == 4
    assert phases[0][1] == 0
    assert phases[-1][2] == 79


def test_display_phases_128_layers():
    phases = display_phases(128)
    assert len(phases) == 4
    assert phases[0][1] == 0
    assert phases[-1][2] == 127


def test_display_phases_no_overlap():
    """Adjacent display phases must not overlap: next start >= prev end."""
    for num_layers in (32, 64, 80, 128):
        phases = display_phases(num_layers)
        for i in range(len(phases) - 1):
            _, _, end_cur = phases[i]
            _, start_next, _ = phases[i + 1]
            assert start_next >= end_cur, (
                f"Overlap at {num_layers} layers: phase {i} ends at {end_cur}, "
                f"phase {i+1} starts at {start_next}"
            )


def test_display_phases_cover_full_range():
    """First phase starts at 0, last phase ends at num_layers - 1."""
    for num_layers in (32, 64, 80, 128):
        phases = display_phases(num_layers)
        assert phases[0][1] == 0
        assert phases[-1][2] == num_layers - 1


# ---------------------------------------------------------------
# analysis_phases
# ---------------------------------------------------------------


def test_analysis_phases_64_layers():
    phases = analysis_phases(64)
    assert len(phases) == 5
    assert "P1_broad_read" in phases
    assert "P5_output_prep" in phases


def test_analysis_phases_32_layers():
    phases = analysis_phases(32)
    assert len(phases) == 5
    for name, (start, end) in phases.items():
        assert start >= 0
        assert end <= 31
        assert start <= end


def test_analysis_phases_80_layers():
    phases = analysis_phases(80)
    for name, (start, end) in phases.items():
        assert start >= 0
        assert end <= 79
        assert start <= end


def test_analysis_phases_128_layers():
    phases = analysis_phases(128)
    for name, (start, end) in phases.items():
        assert start >= 0
        assert end <= 127
        assert start <= end


def test_analysis_phases_boundaries_dont_overlap():
    """No two analysis phases should have overlapping interiors.

    Adjacent phases may share a boundary layer (start_next == end_cur)
    due to rounding, but one phase's interior must not extend into
    another's.
    """
    for num_layers in (32, 64, 80, 128):
        phases = analysis_phases(num_layers)
        ranges = sorted(phases.values(), key=lambda r: r[0])
        for i in range(len(ranges) - 1):
            _, end_cur = ranges[i]
            start_next, _ = ranges[i + 1]
            assert start_next >= end_cur, (
                f"Overlap at {num_layers} layers: "
                f"({end_cur}) overlaps with ({start_next})"
            )


def test_analysis_phases_last_phase_reaches_end():
    """P5_output_prep should reach num_layers - 1."""
    for num_layers in (32, 64, 80, 128):
        phases = analysis_phases(num_layers)
        assert phases["P5_output_prep"][1] == num_layers - 1


# ---------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------


def test_final_layers_is_positive():
    assert FINAL_LAYERS > 0


def test_skip_regions_contains_expected():
    assert "chat_template" in SKIP_REGIONS
    assert "system_prompt" in SKIP_REGIONS
