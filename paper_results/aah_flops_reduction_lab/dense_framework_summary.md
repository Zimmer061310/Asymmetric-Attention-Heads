# Dense Framework FLOPs Summary

This file separates the dense execution framework from the FlashAttention
reference framework used elsewhere in the FLOPs lab.

## Denominator

The P7 dense-framework rows divide by the matched standard dense MHA baseline
from the same full-forward profiler queue:

```text
flopslab-4096-baseline-pure-dense-seed0
gpu_flops_total = 6,171,093,127,247
dense_gpu_flops_total_ratio_ncu = 1.000000
```

The dense-framework metric is:

```text
dense_gpu_flops_total_ratio_ncu =
  dense AAH gpu_flops_total
  / standard dense MHA gpu_flops_total
```

All P7 rows below use `seq_len=4096`, `batch_size=1`, `bf16`, and the same
NVIDIA RTX PRO 6000 Blackwell Server Edition.

## Final P7 Dense Results

| Row | GPU FLOPs | Dense ratio | Interpretation |
|---|---:|---:|---|
| `baseline-pure-dense` | 6,171,093,127,247 | 1.000000 | Dense denominator |
| `same-codepath-full-dense` | 6,171,093,234,569 | 1.000000 | Codepath parity sanity check |
| `static-plan-per-layer-dense` | 6,267,650,323,778 | 1.015647 | Static plan still above dense baseline |
| `static-plan-per-layer-head-dense` | 6,267,531,926,068 | 1.015628 | Static per-head plan still above dense baseline |
| `static-plan-majority-dense` | 6,267,650,321,775 | 1.015647 | Majority plan still above dense baseline |
| `quantized-single-1024-dense` | 6,277,824,338,629 | 1.017295 | Quantized single bucket is worse than static |
| `quantized-single-2048-dense` | 6,276,817,194,910 | 1.017132 | Quantized single bucket is worse than static |
| `quantized-two-bucket-1024-4096-dense` | 6,276,975,957,492 | 1.017158 | Two buckets do not recover a FLOPs win |
| `quantized-two-bucket-2048-4096-dense` | 6,276,641,728,072 | 1.017104 | Two buckets do not recover a FLOPs win |
| `noscatter-contiguous-1024-4096-dense` | 6,265,980,620,124 | 1.015376 | No-scatter helps slightly but stays above 1.0 |
| `noscatter-contiguous-layer-plan-dense` | 6,266,345,585,426 | 1.015435 | No-scatter layer plan stays above 1.0 |
| `noscatter-scatter-control-matched-dense` | 6,267,166,957,995 | 1.015568 | Matched scatter control |
| `fixed-per-layer-dense` | 6,267,650,322,197 | 1.015647 | Fixed granularity does not change the result |
| `fixed-per-state-dense` | 6,267,531,918,541 | 1.015628 | Fixed granularity does not change the result |
| `fixed-per-head-group-dense` | 6,267,531,922,719 | 1.015628 | Fixed granularity does not change the result |
| `fixed-per-head-dense` | 6,267,531,928,420 | 1.015628 | Fixed granularity does not change the result |
| `slow-update-N200-dense` | 6,267,531,931,500 | 1.015628 | Slow update does not change the result |
| `slow-update-N1000-dense` | 6,267,531,928,269 | 1.015628 | Slow update does not change the result |
| `minruntime-noscatter-1024-4096-dense` | 6,173,714,483,308 | 1.000425 | Best valid dense-style lower-overhead probe |
| `headreorder-lowerbound-1024-4096-dense` | 6,171,928,926,856 | 1.000135 | Semantics-changing lower-bound probe |

## Interpretation

P7 confirms the same pattern as the FlashAttention lab, but in a dense
framework:

- the matched dense codepath can reproduce pure dense MHA at `1.000000x`;
- static plans, fixed plans, quantization, and ordinary no-scatter variants all
  remain around `1.015x` to `1.017x`;
- the best lower-overhead probes approach pure dense MHA, but still do not go
  below `1.0`;
- the semantics-changing head-reorder lower-bound is only `1.000135x`, which
  is too small and too artificial to support a compute-reduction claim.

The dense framework therefore does not rescue a measured FLOPs-reduction claim.
It does show that most excess FLOPs are removable overhead, but removing that
overhead brings AAH close to the dense baseline rather than below it.

## Claim Boundary

These dense profiles support the current reframing:

- AAH should not claim true GPU FLOPs reduction in the current implementation.
- ACR/EAR remain routing and accounting diagnostics, not paper-facing compute
  evidence.
- The defensible paper claim is quality plus head/window-structure analysis,
  with speed and memory reported only as measured backend-dependent diagnostics.

Compact artifacts:

```text
paper_results/aah_flops_reduction_lab/profile_manifest_p7_dense_framework.jsonl
paper_results/aah_flops_reduction_lab/profile_status_p7_dense_framework.jsonl
paper_results/aah_flops_reduction_lab/logs/p7_dense_framework.log
paper_results/aah_flops_reduction_lab/gpu_flops_profiles/*dense*gpu_flops_profile.json
```
