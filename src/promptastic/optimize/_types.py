"""Type definitions for prompt optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .structure import StructuredPrompt

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

    mutation_type: str  # "structural", "llm_rewrite", or "meta_rewrite"
    operation: str  # e.g. "reorder_sections", "insert_separator", "trajectory_rewrite"
    target_region: str
    reason: str  # diagnostic signal that triggered this
    diff_summary: str  # human-readable description


@dataclass
class TrajectoryEntry:
    """Condensed view of one optimization iteration for the meta-rewriter.

    Carries only what the LLM context window needs, including optional
    response samples that ``IterationRecord`` does not store by default.
    """

    iteration: int
    prompt_text: str
    metrics: dict[str, float]
    score_total: float
    score_delta: float  # change from previous iteration (0.0 for iter 0)
    num_satisfied: int
    num_total: int
    mutation_applied: MutationRecord | None
    response_samples: list[str] = field(default_factory=list)


@dataclass
class PromptSpec:
    """A prompt that may span multiple conversation turns.

    The system_prompt is always the system message.  prefix_turns are
    additional user/assistant exchanges inserted *after* the system
    message and *before* the final user message in the chat template.
    """

    system_prompt: str
    prefix_turns: list[dict[str, str]] = field(default_factory=list)
    # Each entry: {"role": "user"|"assistant", "content": "..."}
    has_been_split: bool = False  # prevents the split mutation from firing twice
    structured_prompt: StructuredPrompt | None = None  # optional structured view

    @classmethod
    def from_string(cls, text: str) -> PromptSpec:
        """Create a single-turn spec from a plain prompt string."""
        return cls(system_prompt=text)

    def to_messages(
        self, user_message: str, response: str = "",
    ) -> list[dict[str, str]]:
        """Build the full message list for tokenization."""
        msgs: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]
        msgs.extend(self.prefix_turns)
        msgs.append({"role": "user", "content": user_message})
        if response:
            msgs.append({"role": "assistant", "content": response})
        return msgs

    def with_structured(
        self,
        structured: StructuredPrompt,
        *,
        update_text: bool = True,
    ) -> PromptSpec:
        """Return a copy with structured prompt attached."""
        system_prompt = structured.render() if update_text else self.system_prompt
        return PromptSpec(
            system_prompt=system_prompt,
            prefix_turns=list(self.prefix_turns),
            has_been_split=self.has_been_split,
            structured_prompt=structured,
        )


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
    prefix_turns: list[dict[str, str]] = field(default_factory=list)
    response_samples: list[str] = field(default_factory=list)
    fix_plan: list[str] = field(default_factory=list)
    prompt_diff: list[str] = field(default_factory=list)


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
    best_prefix_turns: list[dict[str, str]] = field(default_factory=list)


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
    mutation_strategy: str = "hybrid"  # "structural", "llm", "hybrid", "meta", "hybrid_meta"
    rewrite_model: str = ""
    rewrite_api_key: str = ""
    profile: str = "general"

    # How many structural-only iterations before allowing LLM rewrites
    structural_iterations: int = 3

    # Meta-rewriter settings (trajectory-aware N+1 prompt generation)
    meta_window_size: int = 5  # max trajectory entries in LLM context
    meta_include_responses: bool = False  # capture actual model responses
    meta_response_max_tokens: int = 64  # max tokens per response capture
    verification_enabled: bool = True
    verification_length_tolerance: float = 0.4  # +/- 40%
    use_candidate_policy: bool = True


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
