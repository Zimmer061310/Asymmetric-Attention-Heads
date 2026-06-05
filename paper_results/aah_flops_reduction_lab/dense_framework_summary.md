# Dense Framework FLOPs Summary

This file separates the dense execution framework from the FlashAttention
reference framework used elsewhere in the FLOPs lab.

## Denominator

Dense framework rows divide by the matched standard dense MHA baseline:

```text
standard dense MHA baseline GPU FLOPs = 178,003,140,306
```

The metric name for this framework is:

```text
dense_gpu_flops_total_ratio_ncu =
  dense AAH gpu_flops_total
  / standard dense MHA gpu_flops_total
```

## Current Dense Result

| Row | GPU FLOPs | Dense ratio | Interpretation |
|---|---:|---:|---|
| `dense_standard_mha_baseline_gpu_flops_profile` | 178,003,140,306 | 1.000000 | Dense denominator |
| `dense_aah_full_adaptive_window_exec_gpu_flops_profile` | 285,093,795,549 | 1.601622 | Negative result: dense AAH full_adaptive costs more than standard dense MHA |

The dense AAH / dense baseline ratio is therefore not `~1.01`; the copied
Nsight profile reports `1.601622x`.

## Framework Change

The active follow-up track is dense execution first. This does not change the
FlashAttention result interpretation: current AAH variants still do not beat
pure FlashAttention in measured Nsight GPU FLOPs. The dense framework is a
weaker but useful diagnostic target: determine whether AAH can beat a matched
standard dense MHA implementation after removing dense-path overhead.
