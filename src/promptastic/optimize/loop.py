"""Main optimization loop and CLI entry point.

Orchestrates: analyze -> extract -> score -> diagnose -> mutate -> repeat.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .._types import CaptureConfig
from ..constants import SKIP_REGIONS
from ._types import (
    DiagnosticReport,
    IterationRecord,
    MutationRecord,
    OptimizationConfig,
    OptimizationResult,
    OptimizationScore,
)
from .diagnostics import diagnose, format_diagnostic_report
from .extract import extract_all_metrics
from .mutations import StructuralMutator
from .profiles import apply_profile
from .score import composite_score
from .targets import build_targets_for_regions, load_targets_from_file, targets_to_dict


def optimize_prompt(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    system_prompt: str,
    region_config: dict[str, Any],
    conversations: list[dict[str, Any]],
    config: OptimizationConfig | None = None,
    custom_targets_path: str | None = None,
    position: str = "terminal",
) -> OptimizationResult:
    """Run the prompt optimization loop.

    Parameters
    ----------
    model:
        Loaded HuggingFace model (stays in GPU memory across iterations).
    tokenizer:
        Corresponding tokenizer.
    adapter:
        ModelAdapter instance for the model.
    system_prompt:
        Starting system prompt text.
    region_config:
        Region configuration dict (from ``load_region_config``).
    conversations:
        List of conversation dicts for test cases.
    config:
        Optimization settings.  Uses defaults if None.
    custom_targets_path:
        Optional path to a JSON file with custom metric targets.
    position:
        Query position for metrics (default: "terminal").
    """
    from ..engine.runner import analyze_case
    from ..prep.inputs import build_test_cases

    if config is None:
        config = OptimizationConfig()

    # Build capture config
    capture_config = CaptureConfig(
        attention=True,
        residual=True,
        logit_lens=True,
        patching=config.enable_patching,
        per_head=config.enable_per_head,
        gradients=config.enable_gradients,
    )

    # Set up mutators
    mutator = StructuralMutator()
    rewriter = None
    if config.mutation_strategy in ("llm", "hybrid") and config.rewrite_model:
        from .rewriter import LLMRewriter
        rewriter = LLMRewriter(config.rewrite_model, config.rewrite_api_key)

    current_prompt = system_prompt
    history: list[IterationRecord] = []
    best_score = -1.0
    best_iteration = -1
    best_prompt = system_prompt
    best_regions: dict[str, Any] = {}
    stale_count = 0
    total_passes = 0

    for iteration in range(config.max_iterations):
        iter_start = time.time()

        # 1. Build test cases with current prompt
        test_data = build_test_cases(current_prompt, conversations, region_config)
        system_regions = test_data.get("system_regions", {})
        position_defs = test_data.get("query_positions", {})
        tracked_tokens = test_data.get("tracked_tokens", [])

        # 2. Run analysis on all cases
        case_results: list[dict[str, Any]] = []
        for case in test_data["cases"]:
            result = analyze_case(
                model=model,
                tokenizer=tokenizer,
                adapter=adapter,
                case=case,
                system_prompt=current_prompt,
                system_regions=system_regions,
                capture_config=capture_config,
                position_defs=position_defs,
                tracked_tokens=tracked_tokens,
            )
            case_results.append(result)
            total_passes += _count_forward_passes(result, capture_config)

        # 3. Discover regions and build targets
        regions = sorted(
            r for r in system_regions if r not in SKIP_REGIONS
        )

        if custom_targets_path:
            targets_list = load_targets_from_file(custom_targets_path)
        else:
            # Apply profile overrides
            profile_targets, _weight_overrides = apply_profile(config)
            targets_list = build_targets_for_regions(
                regions,
                has_patching=config.enable_patching,
                has_per_head=config.enable_per_head,
            )
            # Override with profile-specific targets
            profile_names = {t.name for t in profile_targets}
            targets_list = [
                t for t in targets_list if t.name not in profile_names
            ] + profile_targets

        targets_dict = targets_to_dict(targets_list)

        # 4. Extract metrics
        metrics = extract_all_metrics(
            case_results,
            num_layers=adapter.num_layers,
            num_heads=adapter.num_query_heads,
            position=position,
            regions=regions,
            has_patching=config.enable_patching,
            has_per_head=config.enable_per_head,
        )

        # 5. Score
        score = composite_score(metrics, targets_dict)

        # 6. Record iteration
        wall_time = time.time() - iter_start
        record = IterationRecord(
            iteration=iteration,
            prompt_text=current_prompt,
            regions=system_regions,
            metrics=metrics,
            score=score,
            mutation_applied=None,
            forward_passes=total_passes,
            wall_time_seconds=wall_time,
        )
        history.append(record)

        _print_iteration_summary(iteration, score, wall_time)

        # 7. Check convergence
        if score.total > best_score + config.min_improvement:
            best_score = score.total
            best_iteration = iteration
            best_prompt = current_prompt
            best_regions = system_regions
            stale_count = 0
        else:
            stale_count += 1

        if score.total >= config.target_score:
            return _build_result(
                history, best_prompt, best_regions, True, "target_reached",
            )
        if stale_count >= config.patience:
            return _build_result(
                history, best_prompt, best_regions, True, "plateau",
            )
        if total_passes >= config.max_forward_passes:
            return _build_result(
                history, best_prompt, best_regions, False, "budget_exhausted",
            )

        # 8. Diagnose and mutate
        report = diagnose(score)
        if not report.issues:
            return _build_result(
                history, best_prompt, best_regions, True, "all_satisfied",
            )

        mutation_applied: MutationRecord | None = None
        use_structural = (
            config.mutation_strategy == "structural"
            or (
                config.mutation_strategy == "hybrid"
                and iteration < config.structural_iterations
            )
        )

        if use_structural:
            new_prompt, mutation_applied = mutator.apply_best_mutation(
                current_prompt, region_config, report,
            )
        elif rewriter is not None:
            new_prompt, mutation_applied = rewriter.apply_best_rewrite(
                current_prompt, region_config, report,
            )
        else:
            # No LLM rewriter available, fall back to structural
            new_prompt, mutation_applied = mutator.apply_best_mutation(
                current_prompt, region_config, report,
            )

        record.mutation_applied = mutation_applied
        if mutation_applied:
            _print_mutation(mutation_applied)

        # Only update prompt if mutation was applied
        if mutation_applied is not None and new_prompt != current_prompt:
            current_prompt = new_prompt
        else:
            # No mutation possible, stop
            return _build_result(
                history, best_prompt, best_regions, False, "no_mutations_available",
            )

    return _build_result(
        history, best_prompt, best_regions, False, "max_iterations",
    )


def _count_forward_passes(
    result: dict[str, Any],
    capture_config: CaptureConfig,
) -> int:
    """Estimate the number of forward passes for a single case."""
    count = 1  # baseline pass
    if capture_config.get("patching"):
        patching = result.get("patching", {})
        if isinstance(patching, dict):
            count += len(patching.get("results", []))
    return count


def _build_result(
    history: list[IterationRecord],
    best_prompt: str,
    best_regions: dict[str, Any],
    converged: bool,
    reason: str,
) -> OptimizationResult:
    """Construct the final result from history."""
    total_passes = history[-1].forward_passes if history else 0
    total_time = sum(r.wall_time_seconds for r in history)
    best_iter = max(history, key=lambda r: r.score.total) if history else None

    return OptimizationResult(
        best_prompt=best_prompt,
        best_regions=best_regions,
        best_score=best_iter.score if best_iter else OptimizationScore(
            total=0.0, per_metric={}, num_satisfied=0, num_total=0,
        ),
        best_iteration=best_iter.iteration if best_iter else 0,
        history=history,
        total_forward_passes=total_passes,
        total_wall_time_seconds=total_time,
        converged=converged,
        convergence_reason=reason,
    )


def _print_iteration_summary(
    iteration: int,
    score: OptimizationScore,
    wall_time: float,
) -> None:
    print(
        f"  [iter {iteration}] score={score.total:.3f} "
        f"({score.num_satisfied}/{score.num_total} satisfied) "
        f"[{wall_time:.1f}s]"
    )


def _print_mutation(mutation: MutationRecord) -> None:
    print(
        f"    -> {mutation.operation} on {mutation.target_region}: "
        f"{mutation.diff_summary}"
    )


def _save_results(
    result: OptimizationResult,
    output_dir: Path,
) -> None:
    """Write optimization outputs to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Best prompt
    (output_dir / "prompt.txt").write_text(result.best_prompt, encoding="utf-8")

    # History log
    log = {
        "converged": result.converged,
        "convergence_reason": result.convergence_reason,
        "best_iteration": result.best_iteration,
        "best_score": result.best_score.total,
        "total_forward_passes": result.total_forward_passes,
        "total_wall_time_seconds": result.total_wall_time_seconds,
        "iterations": [
            {
                "iteration": rec.iteration,
                "score": rec.score.total,
                "num_satisfied": rec.score.num_satisfied,
                "num_total": rec.score.num_total,
                "metrics": rec.metrics,
                "mutation": (
                    {
                        "type": rec.mutation_applied.mutation_type,
                        "operation": rec.mutation_applied.operation,
                        "target_region": rec.mutation_applied.target_region,
                        "reason": rec.mutation_applied.reason,
                        "diff_summary": rec.mutation_applied.diff_summary,
                    }
                    if rec.mutation_applied
                    else None
                ),
                "wall_time_seconds": rec.wall_time_seconds,
            }
            for rec in result.history
        ],
    }
    (output_dir / "optimization_log.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8",
    )

    # Markdown report
    lines = [
        "# Prompt Optimization Report",
        "",
        f"**Converged:** {result.converged} ({result.convergence_reason})",
        f"**Best iteration:** {result.best_iteration}",
        f"**Best score:** {result.best_score.total:.3f} "
        f"({result.best_score.num_satisfied}/{result.best_score.num_total} satisfied)",
        f"**Total forward passes:** {result.total_forward_passes}",
        f"**Total wall time:** {result.total_wall_time_seconds:.1f}s",
        "",
        "## Score Progression",
        "",
        "| Iteration | Score | Satisfied | Mutation |",
        "|-----------|-------|-----------|----------|",
    ]
    for rec in result.history:
        mutation_str = (
            f"{rec.mutation_applied.operation} ({rec.mutation_applied.target_region})"
            if rec.mutation_applied
            else "-"
        )
        lines.append(
            f"| {rec.iteration} | {rec.score.total:.3f} | "
            f"{rec.score.num_satisfied}/{rec.score.num_total} | {mutation_str} |"
        )

    lines.extend([
        "",
        "## Best Prompt Metrics",
        "",
    ])
    best_rec = max(result.history, key=lambda r: r.score.total)
    for name, value in sorted(best_rec.metrics.items()):
        score_info = best_rec.score.per_metric.get(name)
        sat_str = f" (sat={score_info.satisfaction:.2f})" if score_info else ""
        lines.append(f"- **{name}**: {value:.4f}{sat_str}")

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for prompt optimization."""
    parser = argparse.ArgumentParser(
        description="Optimize a system prompt using MI diagnostics",
    )
    parser.add_argument(
        "--prompt", required=True,
        help="Path to the system prompt text file",
    )
    parser.add_argument(
        "--regions", required=True,
        help="Path to the region configuration JSON",
    )
    parser.add_argument(
        "--conversations", required=True,
        help="Path to the conversations JSON",
    )
    parser.add_argument(
        "--model-path", required=True,
        help="HuggingFace model path or name",
    )
    parser.add_argument(
        "--output", default="optimized",
        help="Output directory (default: optimized)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=10,
        help="Maximum optimization iterations (default: 10)",
    )
    parser.add_argument(
        "--target-score", type=float, default=0.85,
        help="Target composite score to stop at (default: 0.85)",
    )
    parser.add_argument(
        "--patience", type=int, default=3,
        help="Iterations without improvement before stopping (default: 3)",
    )
    parser.add_argument(
        "--patching", action="store_true",
        help="Enable activation patching (slower, more informative)",
    )
    parser.add_argument(
        "--per-head", action="store_true",
        help="Enable per-head analysis",
    )
    parser.add_argument(
        "--gradients", action="store_true",
        help="Enable gradient attribution",
    )
    parser.add_argument(
        "--strategy", default="hybrid",
        choices=["structural", "llm", "hybrid"],
        help="Mutation strategy (default: hybrid)",
    )
    parser.add_argument(
        "--rewrite-model", default="",
        help="Model for LLM rewrites (e.g. claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--rewrite-api-key", default="",
        help="API key for rewrite model",
    )
    parser.add_argument(
        "--profile", default="general",
        help="Optimization profile (general, anti_bleed, maximize_rules_adherence, balanced_attention)",
    )
    parser.add_argument(
        "--targets", default=None,
        help="Path to custom targets JSON file",
    )
    parser.add_argument(
        "--multi-gpu", action="store_true",
        help="Enable multi-GPU with accelerate",
    )

    args = parser.parse_args()

    # Load inputs
    prompt_text = Path(args.prompt).read_text(encoding="utf-8")

    from ..prep.regions import load_region_config
    region_config = load_region_config(args.regions)

    with open(args.conversations) as f:
        conversations = json.load(f)

    # Load model
    print(f"Loading model from {args.model_path}...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ..engine.model_adapter import ModelAdapter

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    adapter = ModelAdapter.from_model(model, tokenizer)
    print(f"  {adapter.num_layers} layers, {adapter.num_query_heads} heads")

    # Build config
    config = OptimizationConfig(
        max_iterations=args.max_iterations,
        target_score=args.target_score,
        patience=args.patience,
        enable_patching=args.patching,
        enable_per_head=args.per_head,
        enable_gradients=args.gradients,
        mutation_strategy=args.strategy,
        rewrite_model=args.rewrite_model,
        rewrite_api_key=args.rewrite_api_key,
        profile=args.profile,
    )

    # Run optimization
    print(f"\nStarting optimization (profile={config.profile}, strategy={config.mutation_strategy})")
    result = optimize_prompt(
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        system_prompt=prompt_text,
        region_config=region_config,
        conversations=conversations,
        config=config,
        custom_targets_path=args.targets,
    )

    # Save outputs
    output_dir = Path(args.output)
    _save_results(result, output_dir)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Optimization complete: {result.convergence_reason}")
    print(f"Best score: {result.best_score.total:.3f} at iteration {result.best_iteration}")
    print(f"Total forward passes: {result.total_forward_passes}")
    print(f"Total time: {result.total_wall_time_seconds:.1f}s")
    print(f"\nOutputs saved to {output_dir}/")
    print(f"  prompt.txt              -- optimized prompt")
    print(f"  optimization_log.json   -- iteration history")
    print(f"  report.md               -- summary report")


if __name__ == "__main__":
    main()
