# P3: Minimal-Runtime AAH Path

Goal: test whether the remaining total-forward Nsight FLOPs gap comes from
profile-time AAH diagnostics rather than the backend attention kernels.

This hypothesis uses the best H3 no-scatter execution plan but enables a
lab-only minimal-runtime flag. The flag skips per-forward diagnostic GPU
reductions such as output-head norms and attention entropy/usage statistics
during profiled inference. It also uses an uninitialized output buffer because
every head is filled exactly once.

Expected positive signal: total-forward `gpu_flops_total_ratio_ncu` falls from
the current best `1.015376x` toward the P2 attention-only ratio `1.000437x`, or
below `1.0`.

Failure interpretation: if the ratio remains around `1.015x`, the residual
overhead is not primarily diagnostic GPU reductions, and the next target should
be output assembly or a more fused execution path.
