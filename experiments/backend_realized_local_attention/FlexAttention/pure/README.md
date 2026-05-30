# Pure FlexAttention baseline

This folder is for the no-AAH FlexAttention baseline.

The baseline should not import or instantiate AAH controller, hierarchy,
grouping, window-selection, or diagnostic modules. It should be a standard
causal Transformer attention path whose attention execution is implemented with
PyTorch FlexAttention.

## Main run

- `pure_flex_full_attention`

This is the denominator for FlexAttention measured FLOPs ratios.

## Required checks

- Output shape matches the standard model.
- Causal rule is full causal attention, not a local window:
  `kv_idx <= q_idx`.
- No AAH fields are logged except absent/null placeholders in postprocessing.
- Profiler ranges include total forward or train-step scope and attention-only
  scope.
