"""Tests for promptastic.analysis.compare -- auto-discovery helpers."""

from promptastic.analysis.compare import _auto_discover_regions, _detect_num_layers
from promptastic.constants import SKIP_REGIONS


# ---------------------------------------------------------------
# _auto_discover_regions
# ---------------------------------------------------------------


def test_auto_discover_regions_excludes_skip():
    samples = [
        {
            "region_map": {
                "rules": {"tok_start": 0, "tok_end": 10},
                "examples": {"tok_start": 10, "tok_end": 20},
                "system_prompt": {"tok_start": 0, "tok_end": 50},
                "chat_template": {"tok_start": 0, "tok_end": 5},
            },
        },
    ]
    regions = _auto_discover_regions(samples)
    assert "rules" in regions
    assert "examples" in regions
    assert "system_prompt" not in regions
    assert "chat_template" not in regions


def test_auto_discover_regions_sorted():
    samples = [
        {
            "region_map": {
                "z_region": {"tok_start": 0, "tok_end": 5},
                "a_region": {"tok_start": 5, "tok_end": 10},
                "m_region": {"tok_start": 10, "tok_end": 15},
            },
        },
    ]
    regions = _auto_discover_regions(samples)
    assert regions == sorted(regions)


def test_auto_discover_regions_empty_samples():
    assert _auto_discover_regions([]) == []


def test_auto_discover_regions_all_skip():
    samples = [
        {
            "region_map": {
                name: {"tok_start": 0, "tok_end": 5}
                for name in SKIP_REGIONS
            },
        },
    ]
    regions = _auto_discover_regions(samples)
    assert regions == []


# ---------------------------------------------------------------
# _detect_num_layers
# ---------------------------------------------------------------


def test_detect_num_layers_basic():
    samples = [
        {
            "attention": {
                "terminal": {
                    "per_layer": [
                        {"layer": 0, "per_region_mean": {}},
                        {"layer": 1, "per_region_mean": {}},
                        {"layer": 31, "per_region_mean": {}},
                    ],
                },
            },
        },
    ]
    assert _detect_num_layers(samples) == 32


def test_detect_num_layers_64():
    samples = [
        {
            "attention": {
                "terminal": {
                    "per_layer": [
                        {"layer": i, "per_region_mean": {}}
                        for i in range(64)
                    ],
                },
            },
        },
    ]
    assert _detect_num_layers(samples) == 64


def test_detect_num_layers_no_attention():
    """Should default to 64 when no attention data is found."""
    samples = [{"metadata": {"case_id": "test"}}]
    assert _detect_num_layers(samples) == 64


def test_detect_num_layers_empty():
    assert _detect_num_layers([]) == 64
