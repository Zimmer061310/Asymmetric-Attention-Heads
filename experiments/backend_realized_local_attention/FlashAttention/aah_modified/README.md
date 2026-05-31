# AAH + FlashAttention

This folder is for the AAH policy path with FlashAttention execution.

AAH should choose per-head windows. FlashAttention should execute each selected
window bucket through causal sliding-window attention.

## Main runs

1. `flash_grouping_off`
2. `flash_full_adaptive`
3. `flash_shallow_freeze`
4. `flash_deep_practical_reuse`

Do not include fixed-window-only runs in the main table.

## Execution rule

For a selected window `W`, execute the bucket with the FlashAttention local
window:

```text
window_size = (W - 1, 0)
```

The full-window case `W = T` should reduce to full causal attention.

## Required checks

- AAH-selected ACR is logged.
- Backend fallback rate is zero for the main accepted runs.
- Nsight GPU FLOPs ratios are hardware-counter derived, not computed from ACR, EAR, or Torch profiler annotations.
- The pure FlashAttention baseline in `../pure` is used as the denominator.
