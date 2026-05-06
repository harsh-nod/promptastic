"""Main analysis orchestrator.

Composes capture modes, hooks, and post-processing into a single
analysis pipeline. Replaces the monolithic run_analysis.py from TeaLeaves
with a modular system that supports attention, per-head, MLP, patching,
gradients, tuned lens, SAE, and generation tracking.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .._types import CaptureConfig, RegionInfo
from ..constants import (
    DEFAULT_CAPTURES,
    DEFAULT_GRADIENT_METHOD,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_PATCHING_METHOD,
    FINAL_LAYERS,
    PATCHING_LAYER_STRIDE,
)
from .captures import get_active_modes, validate_compatibility
from .hooks import AttentionCache, GradientCache, MLPCache, ResidualCache
from .model_adapter import ModelAdapter
from .tokenization import (
    build_chat_tokens,
    build_full_region_map,
    resolve_query_positions,
)


def compute_logit_lens(
    residual_cache: ResidualCache,
    norm: torch.nn.Module,
    lm_head: torch.nn.Module,
    query_positions: dict[str, int],
    num_layers: int,
    tokenizer: Any,
    top_k: int = 50,
    tracked_token_ids: dict[str, int] | None = None,
) -> dict[str, list[dict]]:
    """Project residual stream through final norm + lm_head at each layer.

    Returns per-position, per-layer top-k tokens and tracked token ranks.
    """
    results: dict[str, list[dict]] = {}
    device = next(lm_head.parameters()).device

    for pos_name in query_positions:
        layers_data = []
        for layer_idx in range(num_layers):
            h = residual_cache.get(layer_idx, pos_name)
            if h is None:
                continue

            h_dev = h.unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                h_normed = norm(h_dev)
                lm_dtype = next(lm_head.parameters()).dtype
                logits = lm_head(h_normed.to(lm_dtype)).squeeze().float()
                probs = F.softmax(logits, dim=-1)

            top_vals, top_ids = torch.topk(logits, min(top_k, logits.shape[-1]))
            top_tokens = []
            for i in range(top_vals.shape[0]):
                tid = top_ids[i].item()
                top_tokens.append({
                    "token": tokenizer.decode([tid]),
                    "token_id": tid,
                    "logit": top_vals[i].item(),
                    "prob": probs[tid].item(),
                    "rank": i + 1,
                })

            tracked = {}
            if tracked_token_ids:
                for tok_str, tid in tracked_token_ids.items():
                    tok_logit = logits[tid].item()
                    tok_prob = probs[tid].item()
                    tok_rank = int((logits > logits[tid]).sum().item()) + 1
                    tracked[tok_str] = {
                        "token_id": tid,
                        "logit": tok_logit,
                        "prob": tok_prob,
                        "rank": tok_rank,
                    }

            layers_data.append({
                "layer": layer_idx,
                "top_k": top_tokens,
                "tracked": tracked,
            })

        results[pos_name] = layers_data

    return results


def aggregate_attention(
    attn_cache: AttentionCache,
    query_positions: dict[str, int],
    region_map: dict[str, RegionInfo],
    num_layers: int,
    num_heads: int,
) -> dict[str, dict]:
    """Convert per-head attention into per-region means and head specialization."""
    results: dict[str, dict] = {}

    for pos_name in query_positions:
        per_layer = []
        for layer_idx in range(num_layers):
            region_data = attn_cache.get(layer_idx, pos_name)
            if region_data is None:
                continue

            per_region_mean: dict[str, float] = {}
            head_max_region: list[dict] = []

            for rname, head_weights in region_data.items():
                if isinstance(head_weights, list):
                    per_region_mean[rname] = (
                        sum(head_weights) / len(head_weights)
                        if head_weights
                        else 0.0
                    )
                else:
                    per_region_mean[rname] = float(head_weights)

            # Per-head max region
            for head_idx in range(num_heads):
                best_region = ""
                best_weight = -1.0
                for rname, head_weights in region_data.items():
                    if isinstance(head_weights, list) and head_idx < len(head_weights):
                        w = head_weights[head_idx]
                        if w > best_weight:
                            best_weight = w
                            best_region = rname
                if best_region:
                    head_max_region.append({
                        "head": head_idx,
                        "region": best_region,
                        "weight": best_weight,
                    })

            per_layer.append({
                "layer": layer_idx,
                "per_region_mean": per_region_mean,
                "head_max_region": head_max_region,
            })

        results[pos_name] = {"per_layer": per_layer}

    return results


def aggregate_per_head_attention(
    attn_cache: AttentionCache,
    query_positions: dict[str, int],
    num_layers: int,
) -> dict[str, dict]:
    """Collect per-head attention without averaging."""
    results: dict[str, dict] = {}

    for pos_name in query_positions:
        per_layer = []
        for layer_idx in range(num_layers):
            head_data = attn_cache.get_per_head(layer_idx, pos_name)
            if head_data is None:
                continue
            per_layer.append({
                "layer": layer_idx,
                "per_region_per_head": head_data,
            })
        results[pos_name] = {"per_layer": per_layer}

    return results


def collect_per_token_attention(
    attn_cache: AttentionCache,
    query_positions: dict[str, int],
    num_layers: int,
) -> dict[str, dict]:
    """Extract per-token weights for heatmap visualization."""
    results: dict[str, dict] = {}

    for pos_name in query_positions:
        per_layer = []
        for layer_idx in range(num_layers):
            weights = attn_cache.get_per_token(layer_idx, pos_name)
            if weights is None:
                continue
            per_layer.append({"layer": layer_idx, "weights": weights})
        results[pos_name] = {"per_layer": per_layer}

    return results


def aggregate_mlp_data(
    mlp_cache: MLPCache,
    query_positions: dict[str, int],
    region_map: dict[str, RegionInfo],
    num_layers: int,
    norm: torch.nn.Module,
    lm_head: torch.nn.Module,
    tokenizer: Any,
    top_k: int = 10,
) -> dict[str, dict]:
    """Collect MLP deltas and project through logit lens."""
    results: dict[str, dict] = {}
    device = next(lm_head.parameters()).device

    for pos_name in query_positions:
        per_layer = []
        for layer_idx in range(num_layers):
            mlp_out = mlp_cache.get_output(layer_idx, pos_name)
            if mlp_out is None:
                continue

            delta_norm = float(mlp_out.norm().item())

            # Project MLP delta through logit lens to see what it promotes
            mlp_dev = mlp_out.unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                mlp_normed = norm(mlp_dev)
                lm_dtype = next(lm_head.parameters()).dtype
                mlp_logits = lm_head(mlp_normed.to(lm_dtype)).squeeze().float()

            top_vals, top_ids = torch.topk(mlp_logits, min(top_k, mlp_logits.shape[-1]))
            promoted_tokens = []
            for i in range(top_vals.shape[0]):
                tid = top_ids[i].item()
                promoted_tokens.append({
                    "token": tokenizer.decode([tid]),
                    "token_id": tid,
                    "logit": top_vals[i].item(),
                })

            per_layer.append({
                "layer": layer_idx,
                "delta_norm": delta_norm,
                "mlp_top_promoted_tokens": promoted_tokens,
            })

        results[pos_name] = {"per_layer": per_layer}

    return results


def _resolve_tracked_token_ids(
    tokenizer: Any,
    tracked_tokens: list[str],
) -> dict[str, int]:
    """Convert tracked token strings to token IDs."""
    mapping = {}
    for tok_str in tracked_tokens:
        ids = tokenizer.encode(tok_str, add_special_tokens=False)
        if ids:
            mapping[tok_str] = ids[0]
    return mapping


def _replace_accelerate_hooks(model: Any) -> None:
    """Strip accelerate's AlignDevicesHook and replace with stateless transfers.

    Accelerate's hooks maintain state across forward passes, causing OOM
    on the second case. We replace them with simple device-transfer hooks
    that have no persistent state.
    """
    hook_records: list[tuple[torch.nn.Module, torch.device]] = []

    for name, module in model.named_modules():
        hook = getattr(module, "_hf_hook", None)
        if hook is None:
            continue
        exec_device = getattr(hook, "execution_device", None)
        if exec_device is not None:
            hook_records.append((module, torch.device(exec_device)))

    # Strip all accelerate hooks
    for module, _ in hook_records:
        if hasattr(module, "_hf_hook"):
            delattr(module, "_hf_hook")
        # Remove all existing forward hooks
        module._forward_pre_hooks.clear()

    # Register stateless device transfer hooks
    for module, target_device in hook_records:

        def _make_pre_hook(dev: torch.device):
            def _pre_hook(mod, args):
                return tuple(
                    a.to(dev) if isinstance(a, torch.Tensor) else a for a in args
                )

            return _pre_hook

        module.register_forward_pre_hook(_make_pre_hook(target_device))


def analyze_case(
    model: Any,
    tokenizer: Any,
    adapter: ModelAdapter,
    case: dict[str, Any],
    system_prompt: str,
    system_regions: dict[str, Any],
    capture_config: CaptureConfig,
    top_k: int = 50,
    position_defs: dict[str, Any] | None = None,
    tracked_tokens: list[str] | None = None,
    prefix_turns: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Run full MI analysis on a single test case.

    Orchestrates: tokenization -> hook registration -> forward pass ->
    post-processing -> optional multi-pass modes (patching, generation).

    Parameters
    ----------
    prefix_turns:
        Optional list of user/assistant message dicts to insert between
        the system prompt and the final user message.  Used when the
        optimizer has split a monolithic prompt into multi-turn format.
    """
    case_start = time.time()
    case_id = case.get("id", "unknown")

    # 1. Build token sequence and region map
    if prefix_turns:
        from .tokenization import build_chat_tokens_multi

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(prefix_turns)
        messages.append({"role": "user", "content": case["user_message"]})
        resp = case.get("response", "")
        if resp:
            messages.append({"role": "assistant", "content": resp})

        token_ids, piece_boundaries = build_chat_tokens_multi(
            tokenizer, messages,
        )
    else:
        token_ids, piece_boundaries = build_chat_tokens(
            tokenizer, system_prompt, case["user_message"], case.get("response", "")
        )

    region_map = build_full_region_map(
        tokenizer,
        token_ids,
        piece_boundaries,
        system_prompt,
        case["user_message"],
        case.get("response", ""),
        system_regions,
        case.get("user_regions", {}),
        case.get("response_regions", {}),
    )

    # 2. Resolve query positions
    query_positions = resolve_query_positions(
        tokenizer, token_ids, piece_boundaries, position_defs or {}
    )

    # 3. Resolve tracked token IDs
    tracked_token_ids = (
        _resolve_tracked_token_ids(tokenizer, tracked_tokens)
        if tracked_tokens
        else {}
    )

    # 4. Determine active capture modes
    active_modes = get_active_modes(capture_config)
    warnings = validate_compatibility(active_modes)
    for w in warnings:
        print(f"  Warning: {w}")

    needs_grad = any(m.requires_grad for m in active_modes)
    mode_names = {m.name for m in active_modes}

    # 5. Set up caches
    residual_cache = ResidualCache()
    attn_cache = AttentionCache(
        per_head="per_head" in mode_names,
        capture_per_token=True,
    )
    mlp_cache = MLPCache() if "mlp" in mode_names else None
    grad_cache = GradientCache() if "gradients" in mode_names else None

    # 6. Register hooks
    hooks = []
    device = next(model.parameters()).device

    if "residual" in mode_names or "logit_lens" in mode_names or "tuned_lens" in mode_names or "sae" in mode_names:
        for layer_idx, layer_mod in adapter.get_layer_modules():
            h = layer_mod.register_forward_hook(
                residual_cache.make_hook(layer_idx, query_positions)
            )
            hooks.append(h)

    if "attention" in mode_names or "per_head" in mode_names:
        for layer_idx, attn_mod in adapter.get_attention_modules():
            h = attn_mod.register_forward_hook(
                attn_cache.make_hook(layer_idx, query_positions, region_map)
            )
            hooks.append(h)

    if mlp_cache is not None:
        for layer_idx, mlp_mod in adapter.get_mlp_modules():
            h_in = mlp_mod.register_forward_pre_hook(
                mlp_cache.make_input_hook(layer_idx, query_positions)
            )
            h_out = mlp_mod.register_forward_hook(
                mlp_cache.make_output_hook(layer_idx, query_positions)
            )
            hooks.extend([h_in, h_out])

    if grad_cache is not None:
        for layer_idx, layer_mod in adapter.get_layer_modules():
            h = layer_mod.register_forward_hook(
                grad_cache.make_hook(layer_idx, query_positions)
            )
            hooks.append(h)

    # 7. Forward pass
    input_ids = torch.tensor([token_ids], device=device)
    attention_mask = torch.ones(1, len(token_ids), device=device, dtype=torch.long)

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    ctx = torch.enable_grad() if needs_grad else torch.no_grad()
    with ctx:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
        )

    forward_time = time.time() - case_start

    # 8. Gradient computation (if requested)
    gradient_results = None
    if "gradients" in mode_names and grad_cache is not None:
        from .gradients import GradientAnalyzer

        grad_method = capture_config.get("gradient_method", DEFAULT_GRADIENT_METHOD)
        analyzer = GradientAnalyzer(model, adapter, method=grad_method)
        terminal_pos = query_positions.get("terminal")
        if terminal_pos is not None:
            # Target: top token at terminal position
            final_logits = outputs.logits[0, terminal_pos, :]
            target_tid = int(final_logits.argmax().item())
            gradient_results = analyzer.run(
                input_ids, attention_mask, target_tid,
                query_positions, region_map, tokenizer,
            )

    # 9. Remove single-pass hooks
    for h in hooks:
        h.remove()
    hooks.clear()

    # 10. Post-processing
    result: dict[str, Any] = {}

    # Token labels
    token_labels = [tokenizer.decode([tid]) for tid in token_ids]
    result["token_labels"] = token_labels
    result["region_map"] = region_map
    result["query_positions"] = query_positions

    # Logit lens
    if "logit_lens" in mode_names:
        result["logit_lens"] = compute_logit_lens(
            residual_cache,
            adapter.get_norm(),
            adapter.get_lm_head(),
            query_positions,
            adapter.num_layers,
            tokenizer,
            top_k=top_k,
            tracked_token_ids=tracked_token_ids,
        )

    # Tuned lens
    if "tuned_lens" in mode_names:
        tuned_path = capture_config.get("tuned_lens_path")
        if tuned_path:
            from .tuned_lens import TunedLens

            tl = TunedLens.load(tuned_path)
            result["tuned_lens"] = tl.project(
                residual_cache,
                adapter.get_norm(),
                adapter.get_lm_head(),
                query_positions,
                adapter.num_layers,
                tokenizer,
                top_k=top_k,
                tracked_token_ids=tracked_token_ids,
            )

    # SAE
    if "sae" in mode_names:
        sae_path = capture_config.get("sae_weights_path")
        if sae_path:
            from .sae import SAEAnalyzer, SAEDecoder

            sae = SAEDecoder(sae_path)
            analyzer = SAEAnalyzer(sae)
            result["sae"] = analyzer.analyze(
                residual_cache, query_positions, adapter.num_layers
            )

    # Attention
    if "attention" in mode_names:
        result["attention"] = aggregate_attention(
            attn_cache, query_positions, region_map,
            adapter.num_layers, adapter.num_query_heads,
        )

    # Per-head attention
    if "per_head" in mode_names:
        result["per_head_attention"] = aggregate_per_head_attention(
            attn_cache, query_positions, adapter.num_layers,
        )

    # Per-token attention
    result["per_token_attention"] = collect_per_token_attention(
        attn_cache, query_positions, adapter.num_layers,
    )

    # MLP
    if "mlp" in mode_names and mlp_cache is not None:
        result["mlp"] = aggregate_mlp_data(
            mlp_cache, query_positions, region_map,
            adapter.num_layers, adapter.get_norm(),
            adapter.get_lm_head(), tokenizer,
        )

    # Gradients
    if gradient_results is not None:
        result["gradients"] = gradient_results

    # 11. Multi-pass modes
    if "patching" in mode_names:
        from .patching import PatchingEngine

        method = capture_config.get("patching_method", DEFAULT_PATCHING_METHOD)
        engine = PatchingEngine(model, adapter, tokenizer, method=method)

        # Determine which layers to patch
        layers_spec = capture_config.get("patching_layers", "")
        if layers_spec:
            target_layers = [int(x) for x in layers_spec.split(",")]
        else:
            target_layers = list(range(0, adapter.num_layers, PATCHING_LAYER_STRIDE))

        # Filter regions to patch
        patch_regions = capture_config.get("patching_regions")
        rm = region_map
        if patch_regions:
            rm = {k: v for k, v in region_map.items() if k in patch_regions}

        patching_results = engine.run_full_sweep(
            input_ids, attention_mask, query_positions,
            rm, target_layers, tokenizer,
        )
        result["patching"] = {
            "method": method,
            "baseline_top_token": tokenizer.decode(
                [int(outputs.logits[0, -1, :].argmax().item())]
            ),
            "results": [
                {
                    "region": r.get("region", r["region"]),
                    "layer": r["layer"],
                    "kl_divergence": r["kl_divergence"],
                    "logit_diff": r["logit_diff"],
                    "top_token_change": r["top_token_change"],
                    "baseline_top_token": r["baseline_top_token"],
                }
                for r in patching_results
            ],
        }

    if "generation" in mode_names:
        from .generation import GenerationTracker

        max_tokens = capture_config.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        tracker = GenerationTracker(model, adapter, tokenizer, max_new_tokens=max_tokens)
        result["generation"] = tracker.generate_with_tracking(
            input_ids, attention_mask, region_map,
        )

    # 12. Metadata
    peak_mem = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )
    result["metadata"] = {
        "model": adapter.model_name,
        "dtype": str(next(model.parameters()).dtype),
        "attn_implementation": "eager",
        "case_id": case_id,
        "total_tokens": len(token_ids),
        "peak_gpu_memory_gb": round(peak_mem, 2),
        "forward_pass_seconds": round(forward_time, 2),
        "total_seconds": round(time.time() - case_start, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "capture_modes": sorted(mode_names),
    }

    # Cleanup
    residual_cache.clear()
    attn_cache.clear()
    if mlp_cache:
        mlp_cache.clear()
    if grad_cache:
        grad_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def run_pipeline(
    model_path: str,
    input_path: str,
    output_dir: str,
    capture_config: CaptureConfig | None = None,
    top_k: int = 50,
    cases_filter: list[str] | None = None,
    multi_gpu: bool = False,
    tracked_tokens: list[str] | None = None,
) -> None:
    """Run the full analysis pipeline on all test cases."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load test cases
    with open(input_path) as f:
        test_data = json.load(f)

    system_prompt = test_data["system_prompt"]
    system_regions = test_data.get("system_regions", {})
    position_defs = test_data.get("query_positions", {})
    file_tracked = test_data.get("tracked_tokens", [])
    all_tracked = list(set((tracked_tokens or []) + file_tracked))
    cases = test_data["cases"]

    if capture_config is None:
        capture_config = CaptureConfig(
            attention=True, residual=True, logit_lens=True
        )

    # Merge file-level capture config
    file_capture = test_data.get("capture_config", {})
    for k, v in file_capture.items():
        if k not in capture_config:
            capture_config[k] = v

    # Load model
    print(f"Loading model from {model_path}...")
    device_map = "auto" if multi_gpu else {"": 0}
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device_map,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    if multi_gpu:
        _replace_accelerate_hooks(model)

    adapter = ModelAdapter.from_model(model, tokenizer)
    print(
        f"Model: {adapter.model_name}, {adapter.num_layers} layers, "
        f"{adapter.num_query_heads} heads, hidden={adapter.hidden_size}"
    )

    # Filter cases
    if cases_filter:
        cases = [c for c in cases if c.get("id") in cases_filter]

    # Process each case
    for i, case in enumerate(cases):
        cid = case.get("id", f"sample_{i:02d}")
        print(f"\nCase {i + 1}/{len(cases)}: {cid}")

        result = analyze_case(
            model=model,
            tokenizer=tokenizer,
            adapter=adapter,
            case=case,
            system_prompt=system_prompt,
            system_regions=system_regions,
            capture_config=capture_config,
            top_k=top_k,
            position_defs=position_defs,
            tracked_tokens=all_tracked,
        )

        out_file = output_path / f"{cid}.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(
            f"  Saved: {out_file} "
            f"({result['metadata']['total_seconds']:.1f}s, "
            f"{result['metadata']['peak_gpu_memory_gb']:.1f}GB)"
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Promptastic: mechanistic interpretability pipeline"
    )
    parser.add_argument("--input", required=True, help="Path to test_cases.json")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--model-path", required=True, help="Path to HuggingFace model")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k tokens for logit lens")
    parser.add_argument("--cases", nargs="*", help="Specific case IDs to process")
    parser.add_argument("--multi-gpu", action="store_true", help="Enable multi-GPU mode")
    parser.add_argument("--tracked-tokens", nargs="*", help="Tokens to track in logit lens")
    parser.add_argument("--no-per-token", action="store_true", help="Skip per-token capture")

    # Capture mode flags
    parser.add_argument(
        "--captures",
        default=",".join(DEFAULT_CAPTURES),
        help="Comma-separated capture modes (default: attention,residual,logit_lens)",
    )
    parser.add_argument("--per-head", action="store_true", help="Enable per-head attention")
    parser.add_argument("--mlp", action="store_true", help="Enable MLP capture")
    parser.add_argument("--patching", action="store_true", help="Enable activation patching")
    parser.add_argument(
        "--patching-method",
        default=DEFAULT_PATCHING_METHOD,
        choices=["zero", "mean"],
        help="Ablation method for patching",
    )
    parser.add_argument("--patching-layers", default="", help="Comma-separated layer indices to patch")
    parser.add_argument("--gradients", action="store_true", help="Enable gradient attribution")
    parser.add_argument(
        "--gradient-method",
        default=DEFAULT_GRADIENT_METHOD,
        choices=["vanilla", "integrated"],
    )
    parser.add_argument("--tuned-lens", default="", help="Path to tuned lens probes directory")
    parser.add_argument("--sae", default="", help="Path to SAE weights")
    parser.add_argument("--generation", action="store_true", help="Enable cross-step tracking")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)

    args = parser.parse_args()

    # Build capture config from args
    modes = set(args.captures.split(","))
    if args.per_head:
        modes.add("per_head")
    if args.mlp:
        modes.add("mlp")
    if args.patching:
        modes.add("patching")
    if args.gradients:
        modes.add("gradients")
    if args.tuned_lens:
        modes.add("tuned_lens")
    if args.sae:
        modes.add("sae")
    if args.generation:
        modes.add("generation")

    capture_config = CaptureConfig(
        attention="attention" in modes,
        per_head="per_head" in modes,
        residual="residual" in modes,
        logit_lens="logit_lens" in modes,
        mlp="mlp" in modes,
        tuned_lens="tuned_lens" in modes,
        sae="sae" in modes,
        patching="patching" in modes,
        gradients="gradients" in modes,
        generation="generation" in modes,
        patching_method=args.patching_method,
        patching_layers=args.patching_layers,
        gradient_method=args.gradient_method,
        max_new_tokens=args.max_new_tokens,
        sae_weights_path=args.sae,
        tuned_lens_path=args.tuned_lens,
    )

    run_pipeline(
        model_path=args.model_path,
        input_path=args.input,
        output_dir=args.output,
        capture_config=capture_config,
        top_k=args.top_k,
        cases_filter=args.cases,
        multi_gpu=args.multi_gpu,
        tracked_tokens=args.tracked_tokens,
    )


if __name__ == "__main__":
    main()
