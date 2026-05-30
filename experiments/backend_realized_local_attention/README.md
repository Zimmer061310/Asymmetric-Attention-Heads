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
- Main metrics: validation loss, validation perplexity, ACR, measured attention
  FLOPs ratio, measured total FLOPs ratio, token/s, peak memory, backend fallback
  rate, and backend timing.

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

Each run script also writes a compact FLOPs profile JSON via
`_common/profile_flops_ratio.py`. The profiler loads the final checkpoint when
it exists; otherwise it profiles the same architecture with random weights,
which is still useful for static backend FLOP exposure checks.

## FLOPs rule

Do not reuse the old analytic `flops_ratio`.

Measured FLOPs ratios must be profiler-derived:

```text
measured_attention_flops_ratio =
  profiler GPU FP ops inside method attention ranges
  / profiler GPU FP ops inside full-attention baseline attention ranges

measured_total_flops_ratio =
  profiler GPU FP ops in the method forward or train step
  / profiler GPU FP ops in the matched full-attention baseline
```

Profiler settings, hardware, dtype, batch size, sequence length, input batch,
checkpoint, and backend must match between numerator and denominator.

If backend custom kernels do not expose profiler FLOP counters, the JSON keeps
the profiler FLOPs and backend-realized attention FLOP formula separate. Only
`measured_*_flops_ratio` should be used as the paper FLOPs/FLOPs ratio.

## Current implementation rule

Keep the main `src/` implementation as the paper baseline code path. Backend
experiments in this directory should use local experiment modules or copies, so
the pure backend baseline remains pure and does not import disabled AAH modules.
