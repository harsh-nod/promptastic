"""Prompt optimization via mechanistic interpretability diagnostics.

Provides a closed-loop system that analyses how a transformer processes
a prompt, scores the internal behaviour against target metrics, and
iteratively mutates the prompt to improve those metrics.

Quick start::

    from promptastic.optimize import optimize_prompt, OptimizationConfig

    result = optimize_prompt(
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        system_prompt="...",
        region_config=region_cfg,
        conversations=conversations,
        config=OptimizationConfig(profile="general"),
    )
    print(result.best_prompt)
"""

from ._types import (
    DiagnosticIssue,
    DiagnosticReport,
    IterationRecord,
    MetricScore,
    MetricTarget,
    MutationRecord,
    OptimizationConfig,
    OptimizationResult,
    OptimizationScore,
)
from .loop import optimize_prompt

__all__ = [
    "DiagnosticIssue",
    "DiagnosticReport",
    "IterationRecord",
    "MetricScore",
    "MetricTarget",
    "MutationRecord",
    "OptimizationConfig",
    "OptimizationResult",
    "OptimizationScore",
    "optimize_prompt",
]
