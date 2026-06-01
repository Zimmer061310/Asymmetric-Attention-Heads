# Backend 4096 Nsight FLOPs Summary

| Backend | Method | Val loss | ACR | EAR | Tok/s | GPU alloc max MB | Nsight FLOPs ratio | Computed FLOPs ratio | NCU status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FlexAttention | `pure` | 6.481104 |  |  |  |  | 1.0 | 1.0 | ok |
| FlexAttention | `grouping_off` |  |  |  |  |  | 1.0585880814153703 | 1.0585880814153703 | ok |
| FlexAttention | `full_adaptive` |  |  |  |  |  |  |  | missing |
| FlexAttention | `shallow_freeze` |  |  |  |  |  | 1.0706908503358656 | 1.0706908503358656 | ok |
| FlexAttention | `deep_practical_reuse` |  |  |  |  |  |  |  | missing |
| FlashAttention | `pure` | 6.464322 |  |  |  |  | 1.0 | 1.0 | ok |
| FlashAttention | `grouping_off` |  |  |  |  |  | 1.5936475367757315 | 1.5936475367757315 | ok |
| FlashAttention | `full_adaptive` |  |  |  |  |  | 1.5969842023601073 | 1.5969842023601073 | ok |
| FlashAttention | `shallow_freeze` |  |  |  |  |  | 1.6058978919710059 | 1.6058978919710059 | ok |
| FlashAttention | `deep_practical_reuse` |  |  |  |  |  | 1.607841646879165 | 1.607841646879165 | ok |

`gpu_flops_total_ratio_ncu` is the paper FLOPs/FLOPs field from Nsight Compute profiles. `computed_gpu_flops_total_ratio` recomputes the same ratio from raw `gpu_flops_total` and the matched pure backend baseline as a consistency check. Memory is `gpu_alloc_max_mb`, matching W&B `perf/gpu_alloc_max_mb`.
