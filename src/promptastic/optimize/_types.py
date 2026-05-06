"""Type definitions for prompt optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricTarget:
    """Target specification for a single optimization metric."""

    name: str
    direction: str  # "above", "below", "range"
    weight: float = 1.0
    category: str = ""  # "attention", "dynamics", "causal", "head"
    # For "above": satisfaction is 0 at minimum, 1 at ideal
    minimum: float = 0.0
    ideal: float = 0.0
    # For "below": satisfaction is 1 at ideal, 0 at maximum
    maximum: float = float("inf")
    # For "range": satisfaction is 1 inside [ideal, maximum], tapering outside
    # uses minimum as lower taper boundary


@dataclass
class MetricScore:
    """Evaluation of a single metric against its target."""

    value: float
    satisfaction: float  # 0.0 to 1.0
    weight: float
    target: MetricTarget


@dataclass
class OptimizationScore:
    """Composite score across all metrics."""

    total: float  # weighted mean of satisfactions
    per_metric: dict[str, MetricScore]
    num_satisfied: int  # satisfaction >= 0.9
    num_total: int


@dataclass
class MutationRecord:
    """What was changed and why."""

    mutation_type: str  # "structural" or "llm_rewrite"
    operation: str  # e.g. "reorder_sections", "insert_separator"
    target_region: str
    reason: str  # diagnostic signal that triggered this
    diff_summary: str  # human-readable description


@dataclass
class IterationRecord:
    """Full record of a single optimization iteration."""

    iteration: int
    prompt_text: str
    regions: dict[str, Any]
    metrics: dict[str, float]
    score: OptimizationScore
    mutation_applied: MutationRecord | None
    forward_passes: int
    wall_time_seconds: float


@dataclass
class OptimizationResult:
    """Final result of an optimization run."""

    best_prompt: str
    best_regions: dict[str, Any]
    best_score: OptimizationScore
    best_iteration: int
    history: list[IterationRecord]
    total_forward_passes: int
    total_wall_time_seconds: float
    converged: bool
    convergence_reason: str  # "target_reached", "plateau", "budget_exhausted", "max_iterations"


@dataclass
class OptimizationConfig:
    """Configuration for an optimization run."""

    max_iterations: int = 10
    max_forward_passes: int = 500
    target_score: float = 0.85
    min_improvement: float = 0.01
    patience: int = 3
    enable_patching: bool = False
    enable_per_head: bool = False
    enable_gradients: bool = False
    mutation_strategy: str = "hybrid"  # "structural", "llm", "hybrid"
    rewrite_model: str = ""
    rewrite_api_key: str = ""
    profile: str = "general"

    # How many structural-only iterations before allowing LLM rewrites
    structural_iterations: int = 3


@dataclass
class DiagnosticIssue:
    """A single failing metric with suggested action."""

    metric_name: str
    value: float
    satisfaction: float
    target: MetricTarget
    suggested_mutation: str  # e.g. "reorder_sections", "insert_separator"
    reason: str  # human-readable explanation


@dataclass
class DiagnosticReport:
    """Collection of failing metrics and suggested actions."""

    issues: list[DiagnosticIssue]
    overall_score: float
    num_failing: int
    num_total: int
