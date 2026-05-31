# Backend-realized local attention experiments

This directory is for the next AAH execution study. It is intentionally
separate from `src/` so backend experiments can be developed without changing
the main Transformer implementation.

## Goal

Test whether AAH's window policy can be paired with real local-attention
execution backends. The policy question and the kernel question are separated:

- AAH chooses per-head windows through grouping, hierarchy, and window selection.
- FlexAttention or FlashAttention executes the selected local causal windows.

The main comparison should use the same regime list as the existing paper
experiments:

1. pure backend full attention baseline, no AAH modules
2. grouping_off
3. full_adaptive
4. shallow_freeze
5. deep_practical_reuse

Do not add fixed-window-only rows such as fixed 512, fixed 1024, fixed 2048, or
fixed 4096 to the main comparison unless they are explicitly promoted as
diagnostics.

## Directory layout

```text
backend_realized_local_attention/
  FlexAttention/
    pure/          # no AAH code path
    aah_modified/  # AAH policy + FlexAttention execution
  FlashAttention/
    pure/          # no AAH code path
    aah_modified/  # AAH policy + FlashAttention execution
```

## Shared protocol

- Context length: 4096, matching the existing paper suite.
- Candidate AAH windows: `[512, 1024, 2048, 4096]`.
- Batch size: 1 unless the backend-specific runner proves a larger batch fits.
- Precision: bf16.
- Dropout: 0.0 for backend parity and deterministic local execution.
- Dataset: tokenized dataset file on the training server, not raw W&B logs.
- Main metrics: validation loss, validation perplexity, ACR, EAR, Nsight
  Compute GPU FLOPs ratio when hardware counters are available, token/s, peak
  memory, backend fallback rate, and backend timing.

## Configs

The checked-in configs mirror the existing `paper-main_4096_*_seed0` protocol:
1B shape, 4096 context, 10000 optimizer steps, eval every 200 steps, bf16, batch
size 1, and seed 0.

Pure backend baselines:

- `FlexAttention/pure/configs/backend_4096_pure_flex_seed0.yaml`
- `FlashAttention/pure/configs/backend_4096_pure_flash_seed0.yaml`

AAH-modified backend configs:

- `grouping_off`
- `full_adaptive`
- `shallow_freeze`
- `deep_practical_reuse`

Fixed-window-only runs are intentionally excluded from the main comparison.

## Run commands

```bash
experiments/backend_realized_local_attention/FlexAttention/pure/run.sh
experiments/backend_realized_local_attention/FlexAttention/aah_modified/run_all.sh
experiments/backend_realized_local_attention/FlashAttention/pure/run.sh
experiments/backend_realized_local_attention/FlashAttention/aah_modified/run_all.sh
```

The pure folders use a local pure backend Transformer with no AAH modules. The
AAH folders use a local backend-aware copy of the AAH Transformer. The main
`src/` tree remains the normal AAH implementation.

The legacy run scripts still write a compact Torch-profiler diagnostic JSON via
`_common/profile_flops_ratio.py`, but that file is not a paper FLOPs/FLOPs
source. It is kept only for debugging backend exposure. Paper FLOPs ratios must
come from `_common/profile_gpu_flops_ncu.py`.

## FLOPs rule

Do not reuse the old analytic `flops_ratio`, `analytic_flops_ratio`, or the
legacy Torch-profiler `measured_*_flops_ratio` as a paper FLOPs/FLOPs result.

Paper FLOPs ratios must be Nsight Compute GPU floating-point operation totals:

```text
gpu_flops_attention_ratio_ncu =
  Nsight GPU FP ops inside method attention ranges
  / Nsight GPU FP ops inside matched pure full-attention baseline attention ranges

gpu_flops_total_ratio_ncu =
  Nsight GPU FP ops in the method forward pass
  / Nsight GPU FP ops in the matched pure full-attention baseline forward pass
```

Profiler settings, hardware, dtype, batch size, sequence length, input batch,
checkpoint, and backend must match between numerator and denominator.

Run the hard preflight before training:

```bash
python -m experiments.backend_realized_local_attention._common.profile_gpu_flops_ncu \
  --preflight \
  --ncu /usr/local/cuda/bin/ncu \
  --output paper_results/backend_4096_realized_attention_ncu/ncu_preflight.json
```

If this returns `ERR_NVGPUCTRPERM`, the machine cannot produce true GPU FLOPs
for the paper. Stop before rerunning and use a server or container with NVIDIA
performance counters enabled (`RmProfilingAdminOnly=0` or equivalent platform
support). Do not replace this with ACR, EAR, analytic window ratios, token/s, or
Torch-profiler annotations.

## DeepSpeed FLOPs fallback

DeepSpeed FLOPs Profiler can be used as a software-profiler fallback when NCU
is blocked. It profiles the forward pass and reports model FLOPs, latency, and
parameter counts. For FlashAttention/FlexAttention custom kernels it may not
directly observe local-window kernel math, so the backend script records both:

- `deepspeed_raw_total_flops_ratio`: raw DeepSpeed profiler ratio.
- `deepspeed_adjusted_total_flops_ratio_est`: DeepSpeed total FLOPs adjusted by
  replacing or adding the backend-realized attention formula from EAR.

This is an estimated FLOPs ratio, not hardware-counter evidence. Install
DeepSpeed on the run machine, then profile existing checkpoints or run/profile:

```bash
pip install deepspeed

python -m experiments.backend_realized_local_attention._common.run_deepspeed_flops_suite \
  --run-root paper_results/backend_4096_realized_attention_deepspeed \
  --profile-only
```

If no checkpoints are present and a full rerun is intended, omit
`--profile-only`.

When counters are available, run the full suite:

```bash
python -m experiments.backend_realized_local_attention._common.run_ncu_suite \
  --ncu /usr/local/cuda/bin/ncu \
  --run-root paper_results/backend_4096_realized_attention_ncu \
  --delete-checkpoints
```

Then summarize:

```bash
python -m experiments.backend_realized_local_attention._common.summarize_ncu_results \
  --profile-dir paper_results/backend_4096_realized_attention_ncu/gpu_flops_profiles \
  --output-csv paper_results/backend_4096_realized_attention_ncu/backend_4096_ncu_summary.csv \
  --output-md paper_results/backend_4096_realized_attention_ncu/backend_4096_ncu_summary.md
```

By default the suite then runs one dense-masked memory sanity control:

```text
experiments/backend_realized_local_attention/DenseMasked/memory_sanity/configs/backend_4096_dense_memory_sanity_seed0.yaml
```

This run is not part of the Flex/Flash paper comparison. It is only a server
sanity check for memory accounting: compare its `gpu_alloc_max_mb` and
`gpu_reserved_max_mb` against the old dense-masked paper runs to see whether the
large dense-memory footprint reproduces on the new machine. Use
`--skip-dense-memory-sanity` to omit it.

## Current implementation rule

Keep the main `src/` implementation as the paper baseline code path. Backend
experiments in this directory should use local experiment modules or copies, so
the pure backend baseline remains pure and does not import disabled AAH modules.
