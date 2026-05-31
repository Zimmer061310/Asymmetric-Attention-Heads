# AAH + FlexAttention

This folder is for the AAH policy path with PyTorch FlexAttention execution.

AAH should choose per-head windows. FlexAttention should execute the selected
windows through block masks instead of dense masked attention.

## Main runs

1. `flex_grouping_off`
2. `flex_full_adaptive`
3. `flex_shallow_freeze`
4. `flex_deep_practical_reuse`

Do not include fixed-window-only runs in the main table.

## Execution rule

For a selected window `W`, use a trailing local causal mask:

```text
kv_idx <= q_idx and kv_idx >= q_idx - (W - 1)
```

Cache block masks by `(T, W, device, block_size)` and log backend fallback
reasons explicitly.

## Required checks

- AAH-selected ACR is logged.
- Backend fallback rate is zero for the main accepted runs.
- Nsight GPU FLOPs ratios are hardware-counter derived, not computed from ACR, EAR, or Torch profiler annotations.
- The pure FlexAttention baseline in `../pure` is used as the denominator.
