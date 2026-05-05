"""Shared type definitions for the promptastic pipeline."""

from __future__ import annotations

from typing import TypedDict


class RegionInfo(TypedDict, total=False):
    """Token-level region boundary."""

    tok_start: int
    tok_end: int
    n_tokens: int


class CharRegionInfo(TypedDict):
    """Character-level region boundary from annotation."""

    char_start: int
    char_end: int


class CookingStats(TypedDict):
    """Summary statistics for a cooking curve."""

    peak_layer: int
    peak_value: float
    terminal_value: float
    retention_ratio: float


class ExtendedCookingStats(CookingStats, total=False):
    """Cooking stats with additional analysis fields."""

    peak_terminal_ratio: float
    story: str
    n_samples: int
    trajectory: list[float]


class ContextBleedResult(TypedDict):
    """Context bleed analysis between conversation history and current message."""

    mean_ratio: float
    median_ratio: float
    pct_above_2x: float
    conv_turns_mean: float
    current_message_mean: float
    n_samples: int


class TokenRect(TypedDict):
    """Layout rectangle for a rendered token."""

    x: float
    y: float
    w: float
    h: float
    color: tuple[int, int, int]
    fg: str
    text: str
    token_idx: int


class PerHeadStats(TypedDict, total=False):
    """Per-head attention statistics."""

    head_idx: int
    region_weights: dict[str, float]
    max_region: str
    max_weight: float
    entropy: float


class PatchingResult(TypedDict):
    """Result of patching a single region at a single layer."""

    region: str
    layer: int
    kl_divergence: float
    logit_diff: float
    top_token_change: str
    baseline_top_token: str


class MLPDelta(TypedDict, total=False):
    """MLP contribution at a single layer."""

    layer: int
    post_attn_top_tokens: list[dict]
    post_mlp_top_tokens: list[dict]
    delta_norm: float


class SAEFeatureActivation(TypedDict, total=False):
    """A single SAE feature activation."""

    feature_idx: int
    activation: float
    label: str


class GenerationStepData(TypedDict, total=False):
    """Captured data from a single generation step."""

    step: int
    generated_token: str
    generated_token_id: int
    region_attention: dict[str, float]
    logit_lens_top: list[dict]


class CaptureConfig(TypedDict, total=False):
    """Configuration for what to capture during analysis."""

    attention: bool
    per_head: bool
    residual: bool
    logit_lens: bool
    mlp: bool
    tuned_lens: bool
    sae: bool
    patching: bool
    gradients: bool
    generation: bool
    patching_method: str
    patching_layers: str
    patching_regions: list[str]
    gradient_method: str
    max_new_tokens: int
    sae_weights_path: str
    tuned_lens_path: str
