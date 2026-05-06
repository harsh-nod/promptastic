"""Tests for promptastic.optimize.loop -- convergence and result building."""

from promptastic.optimize._types import (
    IterationRecord,
    MutationRecord,
    OptimizationScore,
)
from promptastic.optimize.loop import _build_result, _count_forward_passes


# ---------------------------------------------------------------
# _count_forward_passes
# ---------------------------------------------------------------


def test_count_forward_passes_baseline():
    result = {"metadata": {}}
    config = {"attention": True}
    assert _count_forward_passes(result, config) == 1


def test_count_forward_passes_with_patching():
    result = {
        "patching": {
            "results": [
                {"region": "r1", "layer": 0},
                {"region": "r1", "layer": 4},
                {"region": "r2", "layer": 0},
            ],
        },
    }
    config = {"patching": True}
    # 1 baseline + 3 patching entries
    assert _count_forward_passes(result, config) == 4


# ---------------------------------------------------------------
# _build_result
# ---------------------------------------------------------------


def _make_record(iteration, score_total):
    score = OptimizationScore(
        total=score_total, per_metric={}, num_satisfied=0, num_total=5,
    )
    return IterationRecord(
        iteration=iteration,
        prompt_text=f"prompt_{iteration}",
        regions={},
        metrics={},
        score=score,
        mutation_applied=None,
        forward_passes=iteration + 1,
        wall_time_seconds=1.0,
    )


def test_build_result_target_reached():
    history = [_make_record(0, 0.5), _make_record(1, 0.9)]
    result = _build_result(history, "best", {}, True, "target_reached")
    assert result.converged is True
    assert result.convergence_reason == "target_reached"
    assert result.best_prompt == "best"


def test_build_result_picks_best_score():
    history = [_make_record(0, 0.3), _make_record(1, 0.8), _make_record(2, 0.6)]
    result = _build_result(history, "best", {}, False, "max_iterations")
    assert result.best_score.total == 0.8
    assert result.best_iteration == 1


def test_build_result_total_time():
    history = [_make_record(0, 0.5), _make_record(1, 0.6)]
    result = _build_result(history, "best", {}, True, "plateau")
    assert result.total_wall_time_seconds == 2.0


def test_build_result_empty_history():
    result = _build_result([], "prompt", {}, False, "max_iterations")
    assert result.total_forward_passes == 0
    assert result.best_score.total == 0.0


def test_build_result_convergence_reasons():
    for reason in ("target_reached", "plateau", "budget_exhausted", "max_iterations"):
        history = [_make_record(0, 0.5)]
        result = _build_result(history, "p", {}, True, reason)
        assert result.convergence_reason == reason
