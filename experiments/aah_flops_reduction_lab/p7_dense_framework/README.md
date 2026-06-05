# P7 Dense Execution Framework

## Goal

Switch the active FLOPs-reduction work from pure FlashAttention comparisons to
a dense execution framework. The question is narrower:

```text
Can AAH produce lower Nsight GPU FLOPs than a matched standard dense MHA
baseline when both are measured in the dense/window-execution code family?
```

This is weaker than beating pure FlashAttention, but it is useful because the
current FlashAttention lab evidence suggests the remaining overhead comes from
bucketed layout / GEMM-shape fragmentation rather than dynamic controller work.

## Metric

Use a separate metric name so the denominator is never confused with the
FlashAttention reference framework:

```text
dense_gpu_flops_total_ratio_ncu =
  dense AAH gpu_flops_total
  / standard dense MHA gpu_flops_total
```

Do not report this as the paper-facing `gpu_flops_total_ratio_ncu` unless the
summary also states the denominator explicitly.

## Known Starting Point

The copied Pro 6000 dense diagnostic from the backend NCU suite is negative:

```text
standard dense MHA baseline GPU FLOPs = 178,003,140,306
dense AAH full_adaptive GPU FLOPs     = 285,093,795,549
dense_gpu_flops_total_ratio_ncu      = 1.601622
```

This means the previously measured dense full-adaptive/window-execution path
does not beat standard dense MHA. The dense framework should therefore start
with smaller controlled probes, not with the assumption that dense already
solves the problem.

## Next Dense Probes

1. Same-codepath dense full-window baseline.
   - Goal: verify whether standard dense MHA and the AAH dense wrapper are
     equal when AAH is forced to full windows.
   - Inference: if this is near `1.0`, the dense overhead is caused by the
     local/window AAH branch, not generic wrapper code.

2. Dense minimal-runtime AAH.
   - Goal: disable diagnostics, branch bookkeeping, entropy/norm reductions,
     and dynamic dictionaries in the dense AAH path.
   - Inference: if this drops sharply from `1.6016x`, the dense branch still
     has avoidable bookkeeping.

3. Dense static fixed-window probes.
   - Variants: fixed `1024`, fixed `2048`, two-bucket `1024/4096`.
   - Goal: identify whether dense local masking itself has irreducible overhead
     versus standard dense attention.

4. Dense attention-only NVTX scope.
   - Goal: separate dense attention math from non-attention model work.
   - Inference: if attention-only is still above `1.0`, dense masking/local
     execution is not reducing real FLOPs.

## Implemented Queue

The first dense queue applies the Flash-lab methods to dense AAH in total-forward
scope:

- dense denominator: `flopslab-4096-baseline-pure-dense-seed0`;
- P1 same-codepath full dense;
- H1 static compiled plan: per-layer, per-layer-head, majority;
- H2 quantized execution: single 1024, single 2048, two-bucket 1024/4096,
  two-bucket 2048/4096;
- H3 no-scatter prototype: contiguous 1024/4096, contiguous layer plan, matched
  scatter control;
- H4 fixed plan granularity: per-layer, per-state, per-head-group, per-head,
  slow-update N200, slow-update N1000;
- P3 minimal-runtime no-scatter;
- H5 head-reorder lower-bound.

Run handle:

```bash
experiments/aah_flops_reduction_lab/p7_dense_framework/scripts/profile_dense_framework_queue.sh
```

Outputs:

```text
paper_results/aah_flops_reduction_lab/profile_manifest_p7_dense_framework.jsonl
paper_results/aah_flops_reduction_lab/profile_status_p7_dense_framework.jsonl
paper_results/aah_flops_reduction_lab/logs/p7_dense_framework.log
paper_results/aah_flops_reduction_lab/gpu_flops_profiles/*dense*gpu_flops_profile.json
```

## Success / Failure Interpretation

Success is `dense_gpu_flops_total_ratio_ncu < 1.0` with finite logits and the
same `seq_len=4096`, `batch_size=1`, `bf16` input shape. That would justify a
dense-framework claim only: AAH beats a matched dense MHA implementation.

Failure means the AAH selected-window policy is still not translating into
hardware FLOPs reduction even under a dense denominator. In that case, the
remaining route is an execution rewrite or custom fused kernel, not more ACR/EAR
accounting.
