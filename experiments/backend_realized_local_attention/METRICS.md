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

Do not use the old analytic `flops_ratio` as a measured FLOPs/FLOPs result.

Measured FLOPs ratios must come from a matched profiler pass:

```text
measured_attention_flops_ratio =
  profiler GPU FP ops inside method attention ranges
  / profiler GPU FP ops inside pure full-attention backend baseline attention ranges

measured_total_flops_ratio =
  profiler GPU FP ops inside method forward/train-step range
  / profiler GPU FP ops inside pure full-attention backend baseline forward/train-step range
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

If profiler FLOP counters are unavailable for a backend, leave the measured
FLOPs ratio blank and report token/s, memory, backend fallback rate, and ACR.

The run scripts write profiler outputs as `*_flops_profile.json` beside each
backend result. Treat these JSON files as the source of truth for measured FLOPs
ratios; the trainer's `analytic_flops_*` columns are diagnostic only.
