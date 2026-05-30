# Pure FlashAttention baseline

This folder is for the no-AAH FlashAttention baseline.

The baseline should not import or instantiate AAH controller, hierarchy,
grouping, window-selection, or diagnostic modules. It should be a standard
causal Transformer attention path whose attention execution is implemented with
FlashAttention.

## Main run

- `pure_flash_full_attention`

This is the denominator for FlashAttention measured FLOPs ratios.

## Required checks

- Output shape matches the standard model.
- Full-window causal attention is used for the baseline.
- No AAH fields are logged except absent/null placeholders in postprocessing.
- Profiler ranges include total forward or train-step scope and attention-only
  scope.
