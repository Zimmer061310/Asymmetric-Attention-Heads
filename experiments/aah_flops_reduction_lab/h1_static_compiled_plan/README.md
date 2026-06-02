# H1: Static Compiled AAH Plan

Goal: test whether dynamic controller/hierarchy work causes the current Flash
AAH `~1.6x` Nsight FLOPs ratio.

Plan:

1. Export per-layer/per-head window plans from an AAH checkpoint.
2. Profile lab variants that use the exported plan.
3. Compare against pure FlashAttention.

Expected positive signal: ratio drops far below `1.6`; strong success is
`gpu_flops_total_ratio_ncu < 1.0`.

Failure interpretation: if the ratio stays near `1.5-1.6`, controller and
hierarchy overhead are not the main problem.
