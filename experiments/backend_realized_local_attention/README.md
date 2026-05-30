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

Do not add fixed-window-only rows such as fixed 1024, fixed 2048, fixed 4096, or
fixed 8192 to the main comparison unless they are explicitly promoted as
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

- Context length: 8192.
- Candidate AAH windows: `[1024, 2048, 4096, 8192]`.
- Batch size: 1 unless the backend-specific runner proves a larger batch fits.
- Precision: bf16.
- Dropout: 0.0 for backend parity and deterministic local execution.
- Dataset: tokenized dataset file on the training server, not raw W&B logs.
- Main metrics: validation loss, validation perplexity, ACR, measured attention
  FLOPs ratio, measured total FLOPs ratio, token/s, peak memory, backend fallback
  rate, and backend timing.

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

## Current implementation rule

Keep the main `src/` implementation as the paper baseline code path. Backend
experiments in this directory should use local experiment modules or copies, so
the pure backend baseline remains pure and does not import disabled AAH modules.
