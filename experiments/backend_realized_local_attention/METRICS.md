# Metrics contract

Backend-realized local-attention experiments must report the following fields.

## Training/evaluation CSV fields

The backend-local trainer records:

- `val_loss`
- `val_ppl`
- `attn_ratio` / `effective_ACR`
- `tok_s`
- `gpu_alloc_max_mb`
- `gpu_reserved_max_mb`
- `backend_name`
- `backend_time_ms`
- `backend_fallback_reasons`

For paper tables, use `ACR` for the attention policy metric and memory in GiB
from the peak GPU fields.

## FLOPs ratio fields

Do not use the old analytic `flops_ratio`, `analytic_flops_ratio`, ACR, EAR, or
token/s as a measured FLOPs/FLOPs result.

Paper FLOPs ratios must come from a matched Nsight Compute pass with real GPU
floating-point operation counters:

```text
gpu_flops_attention_ratio_ncu =
  Nsight GPU FP ops inside method attention ranges
  / Nsight GPU FP ops inside pure full-attention backend baseline attention ranges

gpu_flops_total_ratio_ncu =
  Nsight GPU FP ops inside method forward-pass range
  / Nsight GPU FP ops inside pure full-attention backend baseline forward-pass range
```

The numerator and denominator must use the same:

- model shape
- checkpoint policy
- input batch
- context length
- precision
- hardware
- software stack
- backend family

If Nsight Compute counters are unavailable or return `ERR_NVGPUCTRPERM`, leave
the GPU FLOPs ratio blank, record the failure JSON, and report token/s, memory,
backend fallback rate, ACR, and EAR separately. Do not substitute Torch profiler
annotations or formula estimates.

The NCU path writes profiler outputs as `*_gpu_flops_profile.json`. Treat
`gpu_flops_total_ratio_ncu` and, when explicit attention ranges are profiled,
`gpu_flops_attention_ratio_ncu` as the only paper FLOPs/FLOPs fields. The
trainer's `analytic_flops_*` columns and the legacy Torch-profiler JSONs are
diagnostic only.
