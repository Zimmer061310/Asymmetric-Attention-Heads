# H4: Fixed Plan Granularity

Goal: test how much adaptivity can be frozen while preserving the AAH idea that
different layers and head groups may need different context spans.

Variants:

- `fixed-per-layer`
- `fixed-per-state`
- `fixed-per-head-group`
- `fixed-per-head`
- `slow-update-N200`
- `slow-update-N1000`

Expected positive signal: find the coarsest plan with acceptable validation
loss and lowest Nsight FLOPs ratio.

Failure interpretation: if quality requires highly dynamic per-forward routing,
Flash/Flex bucketing is probably insufficient; a custom fused kernel would be
needed.
