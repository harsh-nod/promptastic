# Promptastic

Mechanistic interpretability pipeline for transformer models. Annotate regions in any prompt, run attention capture with causal attribution on any HuggingFace model, get back heatmaps, cooking curves, patching grids, head specialization charts, and comparative analysis showing what changed and where.

Existing MI tools (TransformerLens, NNsight, pyvene) are libraries that give you hooks and activation access to build your own analysis. Promptastic is a pipeline: structured prompt in, interpretability diagnostics out. It goes beyond attention observation to answer **why** a region matters, not just whether the model looks at it.

## What it does

- **Activation patching**: Zero or mean-ablate a region at any layer, measure KL divergence and logit diff. Establishes causal importance — not just correlation.
- **Per-head attention**: Full decomposition across all heads. Identifies specialist heads via Shannon entropy, reveals which heads focus on which prompt regions.
- **MLP capture**: Hooks MLP sublayers to capture what the MLP adds or overwrites. Projects MLP deltas through logit lens to see promoted concepts.
- **Tuned lens**: Per-layer affine probes (Belrose et al. 2023) for cleaner intermediate predictions than standard logit lens.
- **SAE decomposition**: Sparse autoencoders decompose the residual stream into interpretable features per region per layer.
- **Gradient attribution**: Vanilla and integrated gradient methods answering "how much does perturbing this region change the output?"
- **Cross-step tracking**: Tracks how prompt region influence evolves across autoregressive generation steps.
- **Region annotation**: Named spans via markers, regex, or char ranges. BPE-safe token mapping with cumulative decode.
- **11 renderers**: Heatmaps, cooking curves, layer GIFs, aggregates with confidence bands, patching grids, head grids, MLP curves, SAE dashboards, generation timelines.
- **N-variant comparison**: Delta tables, multi-seed stability, causal scoring, markdown reports.

## Quick start

```bash
# Install (local, rendering and analysis only)
pip install -e .

# Install with GPU dependencies (for running the engine)
pip install -e ".[gpu]"

# Install with all optional capabilities (SAE, tuned lens)
pip install -e ".[all]"
```

### 1. Define regions

Create a `regions.json` describing named spans in your prompt:

```json
{
  "system_prompt": {
    "regions": [
      {"name": "rules", "start_marker": "## Rules", "end_marker": "## Examples"},
      {"name": "examples", "start_marker": "## Examples", "end_marker": null}
    ]
  },
  "user_message": {
    "regions": [
      {"name": "context", "start_marker": "Previous:", "end_marker": "Current:"},
      {"name": "current", "start_marker": "Current:", "end_marker": null}
    ]
  }
}
```

### 2. Prepare inputs

```bash
python -m promptastic.prep.inputs \
    --prompt system_prompt.txt \
    --regions regions.json \
    --conversations conversations.json \
    --output test_cases.json
```

### 3. Run analysis

```bash
# Basic attention + logit lens (same as traditional MI)
promptastic --input test_cases.json --output results/ --model-path /path/to/model

# Full capture: attention + per-head + MLP + patching + gradients
promptastic --input test_cases.json --output results/ --model-path /path/to/model \
    --per-head --mlp --patching --gradients

# With tuned lens and SAE (requires pre-trained weights)
promptastic --input test_cases.json --output results/ --model-path /path/to/model \
    --tuned-lens /path/to/probes/ --sae /path/to/sae.pt

# Cross-step generation tracking
promptastic --input test_cases.json --output results/ --model-path /path/to/model \
    --generation --max-new-tokens 64
```

### 4. Render results

```bash
# Per-token attention heatmap
python -m promptastic.render.heatmap --result results/case_0.json --mask-chatml

# Per-region attention trajectories (cooking curves)
python -m promptastic.render.cooking_curves --result results/case_0.json --normalize per-region

# Activation patching grid (region x layer causal importance)
python -m promptastic.render.patching_heatmap --result results/case_0.json

# Per-head attention grid
python -m promptastic.render.head_grid --result results/case_0.json --layers final

# MLP contribution trajectories
python -m promptastic.render.mlp_curves --result results/case_0.json

# SAE feature dashboard
python -m promptastic.render.feature_dashboard --result results/case_0.json --top-k 10

# Cross-step attention timeline
python -m promptastic.render.generation_timeline --result results/case_0.json

# Animated layer sweep
python -m promptastic.render.layer_gif --result results/case_0.json --mask-chatml

# Multi-sample aggregate with confidence bands
python -m promptastic.render.aggregate --base-dir results/ --variants baseline:Baseline
```

### 5. Compare variants

```bash
python -m promptastic.analysis.compare \
    --base-dir results/ \
    --variants baseline:Baseline modified:Modified \
    --metrics all,patching,gradient

python -m promptastic.analysis.report \
    --base-dir results/ \
    --experiments baseline:Baseline:results_baseline modified:Modified:results_modified \
    --output-dir reports/
```

## Capture modes

Each capability is a composable capture mode enabled via CLI flags. Modes can be combined freely in a single run.

| Mode | Flag | What it captures | Extra passes |
|------|------|-----------------|-------------|
| Attention | `--captures attention` (default) | Per-region head-averaged attention at every layer | 0 |
| Per-head | `--per-head` | Full per-head attention without averaging | 0 |
| Residual | `--captures residual` (default) | Residual stream vectors at query positions | 0 |
| Logit lens | `--captures logit_lens` (default) | Residual projected through LM head at every layer | 0 |
| MLP | `--mlp` | Post-attention and post-MLP residuals, MLP delta | 0 |
| Tuned lens | `--tuned-lens PATH` | Learned affine probes for cleaner intermediate predictions | 0 |
| SAE | `--sae PATH` | Sparse autoencoder feature activations per layer | 0 |
| Patching | `--patching` | Zero/mean ablation per region per layer | regions x layers |
| Gradients | `--gradients` | Per-region gradient attribution (vanilla or integrated) | 0-1 |
| Generation | `--generation` | Per-step attention during autoregressive generation | max_new_tokens |

### Patching options

```bash
--patching-method zero|mean    # Ablation method (default: zero)
--patching-layers 0,8,16,24   # Specific layers to patch (default: every 4th)
```

### Gradient options

```bash
--gradient-method vanilla|integrated   # Attribution method (default: vanilla)
```

## Model support

The engine auto-discovers model architecture from any HuggingFace decoder-only transformer by inspecting `model.config` and walking the module tree.

| Family | Example | Layers | Notes |
|--------|---------|--------|-------|
| Llama 3 | Llama-3.1-8B-Instruct | 32 | Full support |
| Qwen | Qwen3-32B | 64 | Full support |
| Mistral | Mistral-7B-Instruct | 32 | Full support |
| Gemma | Gemma-2-9B-IT | 42 | System role auto-merged into user |
| GPT | gpt-oss-20b | 24 | Full support (MoE) |

Requirement: `attn_implementation="eager"` (flash attention doesn't materialize the attention matrix for capture, but computes the same mathematical function).

## GPU requirements

Rule of thumb: `model_params * 2 bytes + 5GB headroom` (fp16 weights + capture overhead).

| Model | VRAM needed | Recommended GPU |
|-------|-------------|-----------------|
| 8B params | ~21GB | A100 40GB, MI250X |
| 32B params | ~69GB | H100 80GB, MI300X |

Activation patching adds forward passes: `1 + (regions x patched_layers)`. Default stride of 4 means a 64-layer model with 10 regions runs 161 passes total.

## Project structure

```
src/promptastic/
  engine/
    model_adapter.py      # Auto-discovers any HF model architecture
    tokenization.py       # Chat template + BPE-safe region mapping
    hooks.py              # ResidualCache, AttentionCache, MLPCache, GradientCache
    captures.py           # 10 capture modes, compatibility validation
    patching.py           # Zero/mean ablation with KL divergence
    tuned_lens.py         # Per-layer affine probes (train + project)
    sae.py                # Sparse autoencoder decoder + analyzer
    gradients.py          # Vanilla + integrated gradient attribution
    generation.py         # Autoregressive generation with per-step hooks
    runner.py             # Main orchestrator composing all captures
  render/
    heatmap.py            # Per-token attention heatmap
    cooking_curves.py     # Per-region attention trajectories
    layer_gif.py          # Animated layer sweep
    aggregate.py          # Multi-sample with confidence bands
    patching_heatmap.py   # Region x layer causal importance grid
    head_grid.py          # Per-head attention + specialization
    mlp_curves.py         # MLP contribution trajectories
    feature_dashboard.py  # SAE feature visualization
    generation_timeline.py # Cross-step attention decay
  analysis/
    metrics.py            # Cooking stats, phase means, safe math
    compare.py            # N-variant comparison tables
    report.py             # Markdown reports with delta analysis
    causal.py             # Causal importance scoring
    head_analysis.py      # Head specialization + specialist detection
    feature_analysis.py   # SAE feature analysis + overlap
  prep/
    regions.py            # Region annotation (marker/regex/char-range)
    inputs.py             # Test case assembly
```

## Testing

```bash
pytest                    # 247 tests, ~0.3s
```

## Dependencies

**Core** (rendering + analysis, no GPU): `numpy`, `Pillow`

**GPU** (engine): `torch`, `transformers`, `accelerate`, `safetensors`, `huggingface_hub`

**Optional**: `sae-lens` (SAE feature decomposition), `tuned-lens` (reference probes)
