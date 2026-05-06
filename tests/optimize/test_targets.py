"""Tests for promptastic.optimize.targets -- target definitions and profiles."""

import json
import tempfile
from pathlib import Path

import pytest

from promptastic.optimize._types import MetricTarget
from promptastic.optimize.targets import (
    build_targets_for_regions,
    default_targets,
    load_targets_from_file,
    make_region_targets,
    targets_to_dict,
)


def test_default_targets_not_empty():
    targets = default_targets()
    assert len(targets) > 0


def test_default_targets_have_valid_directions():
    for t in default_targets():
        assert t.direction in ("above", "below", "range")


def test_make_region_targets_rules():
    targets = make_region_targets("rules", is_rules=True)
    names = {t.name for t in targets}
    assert "terminal_attention_rules" in names
    assert "retention_ratio_rules" in names
    assert "peak_layer_frac_rules" in names


def test_make_region_targets_rules_retention_threshold():
    targets = make_region_targets("rules", is_rules=True)
    ret = next(t for t in targets if t.name == "retention_ratio_rules")
    assert ret.ideal == 0.3
    assert ret.direction == "above"


def test_make_region_targets_examples():
    targets = make_region_targets("examples", is_examples=True)
    names = {t.name for t in targets}
    assert "retention_ratio_examples" in names
    # Lower threshold for examples
    ret = next(t for t in targets if t.name == "retention_ratio_examples")
    assert ret.ideal == 0.1


def test_make_region_targets_generic():
    targets = make_region_targets("current_message")
    names = {t.name for t in targets}
    assert "terminal_attention_current_message" in names
    # No retention or peak targets for generic regions
    assert "retention_ratio_current_message" not in names


def test_make_region_targets_with_patching():
    targets = make_region_targets("rules", is_rules=True, has_patching=True)
    names = {t.name for t in targets}
    assert "causal_importance_rules" in names


def test_make_region_targets_with_per_head():
    targets = make_region_targets("rules", is_rules=True, has_per_head=True)
    names = {t.name for t in targets}
    assert "head_variance_rules" in names


def test_build_targets_for_regions():
    regions = ["rules", "examples", "current_message"]
    targets = build_targets_for_regions(regions)
    names = {t.name for t in targets}
    # Should have global + per-region targets
    assert "context_bleed_ratio" in names
    assert "terminal_attention_rules" in names
    assert "terminal_attention_examples" in names
    assert "terminal_attention_current_message" in names
    # Rules-specific (auto-detected)
    assert "retention_ratio_rules" in names


def test_build_targets_for_regions_custom_rules_set():
    regions = ["directive", "other"]
    targets = build_targets_for_regions(
        regions, rules_regions={"directive"},
    )
    names = {t.name for t in targets}
    assert "retention_ratio_directive" in names
    assert "retention_ratio_other" not in names


def test_targets_to_dict():
    targets = [
        MetricTarget(name="a", direction="above"),
        MetricTarget(name="b", direction="below"),
    ]
    d = targets_to_dict(targets)
    assert "a" in d
    assert "b" in d
    assert d["a"].direction == "above"


def test_load_targets_from_file():
    data = [
        {"name": "custom_metric", "direction": "above", "ideal": 0.5, "weight": 2.0},
        {"name": "another", "direction": "below", "ideal": 1.0, "maximum": 3.0},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()
        targets = load_targets_from_file(f.name)

    assert len(targets) == 2
    assert targets[0].name == "custom_metric"
    assert targets[0].ideal == 0.5
    assert targets[0].weight == 2.0
    assert targets[1].maximum == 3.0
