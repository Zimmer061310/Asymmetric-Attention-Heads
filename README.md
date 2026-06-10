# ENA-AAH-v3

Research code and compact result summaries for **Asymmetric Attention Heads
(AAH-v3)**, a Transformer attention-control mechanism that assigns different
context windows to different attention heads or head groups while preserving the
standard Transformer block interface.

<p align="center">
  <a href="#overview">Overview</a> |
  <a href="#method">Method</a> |
  <a href="#current-claim-boundary">Claim Boundary</a> |
  <a href="#results">Results</a> |
  <a href="#setup">Setup</a> |
  <a href="#citation">Citation</a>
</p>

This repository contains the AAH-v3 implementation, experiment configs, compact
paper-facing summaries, and diagnostic scripts. It intentionally excludes large
checkpoints, raw W&B run directories, virtual environments, server credentials,
and local scratch logs.

## Overview

Standard multi-head attention gives every head the same full causal attention
span. AAH-v3 keeps the usual Q/K/V projections, flat head concatenation, and
Transformer output interface, but adds a separate control path that chooses a
local causal window for each head or head group.

The current paper framing is **quality and structure**, not a hardware-FLOPs
reduction claim. AAH-v3 is best understood as a head-wise context-allocation
mechanism: it tests whether learned or structured head-window assignments can
improve or preserve language-model quality compared with full attention,
shuffled assignments, random windows, and simple fixed-window controls.

## Method

AAH-v3 builds smoothed features for heads and groups, constructs a hierarchy,
and applies constrained window decisions before mapping those decisions back to
individual heads. The final controller uses **wide joint sibling scoring**:
paired sibling groups are scored together so the scorer can compare them
directly before assigning window budgets.

<p align="center">
  <img src="figures/paper_fig2_joint_scorer.png" alt="Independent sibling scoring versus joint sibling scoring" width="900">
</p>

For execution, the implementation buckets heads by selected window and runs
causal local attention through the configured backend. The reference backend is
`dense_masked`; backend-realized experiments also cover FlashAttention and
PyTorch FlexAttention paths.

## Current Claim Boundary

The latest evidence changed the paper claim boundary:

- AAH-v3 is **not** currently presented as a measured GPU-FLOPs reduction
  method.
- `ACR`, `EAR`, and analytic FLOPs fields are routing diagnostics only. They
  describe selected or backend-accounted attention structure, not hardware
  FLOPs savings.
- Nsight Compute profiles for the FlashAttention/FlexAttention FLOPs lab did
  not show `gpu_flops_total_ratio_ncu < 1.0` against matched pure
  FlashAttention baselines.
- The current paper-facing direction is that AAH-v3 exposes and tests
  structured head-wise context allocation, with validation quality and routing
  structure as the main evidence.

See:

- `paper_results/aah_flops_reduction_lab/flopslab_pro6000_summary.md`
- `paper_results/aah_quality_structure_lab/phase1_quality_summary.md`
- `paper_results/aah_quality_structure_lab/README.md`

## Results

### 4096-token AAH-v3 quality/structure lab

The current quality lab screens seed-0, 3000-step controls for whether learned
or structured head-window assignment matters. It compares pure baseline,
current AAH references, post-selection shuffled windows, random windows, fixed
windows, fixed random grouping, and small stability-oriented AAH variants.

The Phase 1 summary ranks the rows by final validation loss:

| Rank | Row | Val loss | Val ppl | Notes |
|---:|---|---:|---:|---|
| 1 | `shallow-control-interval10` | 7.2775 | 1447.40 | Best short-budget AAH variant |
| 2 | `fixed-random-grouping` | 7.2831 | 1455.43 | Strong topology control |
| 3 | `fixed-1024` | 7.2834 | 1455.92 | Strong fixed-window control |
| 5 | `shallow-freeze` | 7.2920 | 1468.46 | Current AAH reference |
| 10 | `shallow-shuffle-post-select` | 7.3074 | 1491.33 | Preserves window histogram, disrupts head identity |
| 11 | `pure-baseline` | 7.3226 | 1514.16 | Full-attention reference |
| 12 | `full-adaptive-shuffle-post-select` | 7.3404 | 1541.36 | Shuffled full-adaptive control |

The full table is in
`paper_results/aah_quality_structure_lab/phase1_quality_summary.md`.

Interpretation: the short-budget controls support a conservative structure
story. AAH-style context allocation can improve validation loss over the pure
baseline in this setup, but simple fixed and randomized topology controls are
competitive, so the paper should avoid overstating adaptivity or FLOPs claims.

### Historical 4096-token AAH-v3 suite

The earlier 1B/4096 suite remains useful for mechanism diagnostics and
training dynamics. The figures show validation loss, selected-window behavior,
and hierarchy diagnostics for the main AAH regimes.

<p align="center">
  <img src="figures/paper_fig3_training_dynamics.png" alt="Training dynamics for the 1B 4096-token seed-0 suite" width="900">
</p>

<p align="center">
  <img src="figures/paper_fig4_window_bucket_heatmap.png" alt="Aggregate selected-window bucket heatmap" width="900">
</p>

These plots should be read as routing and quality diagnostics. They should not
be used as direct evidence of hardware FLOPs reduction.

### Qwen3-4B compatibility snapshot

The Qwen3-4B results are capped-subset compatibility checks for downstream
behavior. They are not official full benchmark scores.

## Repository Layout

- `src/models/transformer.py`: decoder-only Transformer baseline and AAH-v3
  attention/controller implementation.
- `scripts/train.py`: main local training entry point.
- `scripts/infer.py`: checkpoint evaluation and AAH diagnostics.
- `scripts/qwen3_aah_patch.py` and `scripts/qwen3_aah_paper.py`: Qwen3-4B
  compatibility utilities.
- `configs/`: experiment YAMLs, including paper-facing and diagnostic configs.
- `experiments/aah_quality_structure_lab/`: current quality/structure lab.
- `experiments/aah_flops_reduction_lab/`: retired FLOPs-reduction lab and
  diagnostic tooling.
- `paper_results/`: compact paper-facing CSV/Markdown summaries.

## What Is Not Included

- `.pt` checkpoints and adapter weights.
- Raw W&B run directories.
- Local logs, scratch outputs, and Python virtual environments.
- Datasets or downloaded Hugging Face model weights.
- Private tokens, server passwords, or machine-specific credentials.

For release-quality reproduction, publish large artifacts through an external
artifact store and record immutable hashes or model revisions. See
`REPRODUCIBILITY.md` and `PUBLIC_RELEASE.md` for the current release scope and
public-release checklist.

## Setup

Use Python 3.10+ with PyTorch:

```bash
python -m pip install -r requirements.txt
```

Optional backend-realized local-window experiments require FlashAttention or a
PyTorch build exposing `torch.nn.attention.flex_attention`. If those backends
are unavailable, configs may fall back to `dense_masked` and record the fallback
reason.

For Featurize-style runs, keep code and important model artifacts under
`/home/featurize/work`, and use `/home/featurize/data` only for fast scratch
storage.

## Typical Commands

Train from a YAML config:

```bash
python scripts/train.py --config configs/aah_v3_base.yaml
```

Run inference and collect diagnostics:

```bash
python scripts/infer.py --config configs/aah_v3_base.yaml --checkpoint path/to/checkpoint.pt
```

Generate the quality/structure lab configs:

```bash
python experiments/aah_quality_structure_lab/scripts/make_quality_configs.py
```

Run the Qwen3-4B AAH compatibility workflow:

```bash
python scripts/run_qwen3_aah_paper.py --benchmark-profile fast_paper
```

Exact flags may differ by config and checkpoint layout; inspect the target
script's `--help` output before launching expensive runs.

## Paper Result Files

Key compact result files include:

- `paper_results/aah_quality_structure_lab/phase1_quality_summary.md`
- `paper_results/aah_quality_structure_lab/phase1_quality_summary.csv`
- `paper_results/aah_flops_reduction_lab/flopslab_pro6000_summary.md`
- `paper_results/aah_v3_4096_table1_training.csv`
- `paper_results/aah_v3_4096_table2_inference.csv`
- `paper_results/qwen3_4b_aah/benchmarks/benchmark_paper_table.md`
- `paper_results/qwen3_4b_aah/benchmarks/benchmark_paper_table.tex`

The Qwen3 benchmark table is a capped-subset compatibility check, not an
official full benchmark report.

## License

This repository is released under the Apache License 2.0. See `LICENSE`.

Machine-readable citation metadata is provided in `CITATION.cff`. Update it
with the final arXiv identifier before public release.

## Repository Release Checklist

Before making the repository public:

1. Add the final arXiv citation and link.
2. Confirm unresolved provenance fields are either completed or clearly marked
   as limitations.
3. Keep checkpoints out of Git; publish large artifacts through an artifact
   store with SHA-256 hashes.
4. Verify no private tokens, server passwords, raw W&B credentials, or local
   machine paths are committed.

## Citation

The arXiv record is not public yet. Use this placeholder until the final record
exists:

```bibtex
@misc{zhao2026aahv3,
  title  = {Asymmetric Attention Heads: Hierarchical Group-Level Control for Head-Wise Context Allocation},
  author = {Zimu Zhao},
  year   = {2026},
  note   = {arXiv preprint, forthcoming}
}
```
