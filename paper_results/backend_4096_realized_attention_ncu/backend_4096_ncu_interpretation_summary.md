# Pro 6000 Backend 4096 Nsight FLOPs Result Summary

This file summarizes the 10 backend-realized runs and the 3 post-suite diagnostic runs.
All rows use `seq_len=4096`, `batch_size=1`, `bf16`, and Nsight Compute GPU floating-point-operation counters.

`gpu_flops_total_ratio_ncu` is the paper-facing real GPU FLOPs/FLOPs ratio. It is not derived from ACR, EAR, Torch profiler FLOPs, or analytic attention-span formulas.

## 10 Backend Runs

| Backend | Method | Nsight GPU FLOPs | FLOPs Ratio | Status |
|---|---:|---:|---:|---|
| FlexAttention | pure | 240.008B | 1.0000 | OK |
| FlexAttention | grouping_off | 254.069B | 1.0586 | OK |
| FlexAttention | full_adaptive | - | - | profile failed/timeout |
| FlexAttention | shallow_freeze | 256.974B | 1.0707 | OK |
| FlexAttention | deep_practical_reuse | - | - | profile failed/timeout |
| FlashAttention | pure | 178.418B | 1.0000 | OK |
| FlashAttention | grouping_off | 284.336B | 1.5936 | OK |
| FlashAttention | full_adaptive | 284.931B | 1.5970 | OK |
| FlashAttention | shallow_freeze | 286.522B | 1.6059 | OK |
| FlashAttention | deep_practical_reuse | 286.869B | 1.6078 | OK |

## 3 Additional Diagnostic Runs

| Run | Nsight GPU FLOPs | FLOPs Ratio | What It Tests |
|---|---:|---:|---|
| Dense standard MHA baseline | 178.003B | 1.0000 | ordinary dense MHA denominator |
| Dense AAH full_adaptive window execution | 285.094B | 1.6016 | whether dense AAH reduces real FLOPs |
| Fixed-1024 AAH + FlashAttention | 284.735B | 1.5959 | whether simple fixed local Flash window saves FLOPs |

## Main Inference

The result is stronger than "AAH does not reduce compute." In the current implementation, AAH increases measured GPU FLOPs.

- ACR and EAR reductions do not predict real GPU FLOPs reduction.
- FlashAttention + AAH is consistently around `1.59x-1.61x` the FLOPs of pure FlashAttention.
- Dense AAH full_adaptive is around `1.60x` the FLOPs of dense standard MHA.
- Fixed-1024 AAH + FlashAttention is also `1.5959x`, so the problem is not only adaptive routing noise.
- FlexAttention AAH rows with valid profiles are less severe than FlashAttention, but still above the pure FlexAttention baseline.

## Claim Boundary

These results do not support a physical-compute-reduction claim for the current AAH implementation.

The defensible claim is narrower: AAH can route heads toward shorter selected windows and may preserve or improve validation quality under adaptive execution, but this implementation does not turn those selected windows into lower measured GPU FLOPs. For paper wording, ACR and EAR should be described as policy/accounting diagnostics, while Nsight-derived `gpu_flops_total_ratio_ncu` should be treated as the real hardware FLOPs metric.

