# AAH FLOPs Lab Pro 6000 Nsight Summary

All rows use Nsight Compute GPU FLOP counters on RTX PRO 6000 Blackwell, seq_len=4096, batch_size=1, bf16. Total-forward rows divide by the matched pure FlashAttention total-forward baseline. Attention-scope rows divide by the matched pure FlashAttention attention NVTX-scope baseline. Lower is better; values below 1.0 would indicate lower measured GPU FLOPs than pure FlashAttention for that scope.

## Total Forward Scope

| Rank | Experiment | Type | Scope | Ratio | GPU FLOPs | Hypothesis | Mode |
|---:|---|---|---|---:|---:|---|---|
| 1 | `flopslab-4096-baseline-pure-flash-seed0` | denominator | total | 1.000000 | 6171093130434 | baseline | pure |
| 2 | `flopslab-4096-same-codepath-full-flash-seed0` | diagnostic_baseline | total | 1.000000 | 6171093238461 | p1_same_codepath_full_baseline | same_codepath_full |
| 3 | `flopslab-4096-noscatter-contiguous-1024-4096-flash-seed0` | aah | total | 1.015376 | 6265980625227 | h3_noscatter_prototype | noscatter_prototype |
| 4 | `flopslab-4096-minruntime-noscatter-1024-4096-flash-seed0` | aah | total | 1.015376 | 6265980625951 | p3_minimal_runtime | minimal_runtime |
| 5 | `flopslab-4096-noscatter-contiguous-layer-plan-flash-seed0` | aah | total | 1.015435 | 6266345599701 | h3_noscatter_prototype | noscatter_prototype |
| 6 | `flopslab-4096-noscatter-scatter-control-matched-flash-seed0` | aah | total | 1.015568 | 6267166948539 | h3_noscatter_prototype | noscatter_prototype |
| 7 | `flopslab-4096-slow-update-N200-flash-seed0` | aah | total | 1.015628 | 6267531920428 | h4_fixed_plan_granularity | fixed_plan |
| 8 | `flopslab-4096-static-plan-per-layer-head-flash-seed0` | aah | total | 1.015628 | 6267531924576 | h1_static_compiled_plan | static_compiled_plan |
| 9 | `flopslab-4096-fixed-per-head-flash-seed0` | aah | total | 1.015628 | 6267531926227 | h4_fixed_plan_granularity | fixed_plan |
| 10 | `flopslab-4096-fixed-per-state-flash-seed0` | aah | total | 1.015628 | 6267531926549 | h4_fixed_plan_granularity | fixed_plan |
| 11 | `flopslab-4096-fixed-per-head-group-flash-seed0` | aah | total | 1.015628 | 6267531926721 | h4_fixed_plan_granularity | fixed_plan |
| 12 | `flopslab-4096-slow-update-N1000-flash-seed0` | aah | total | 1.015628 | 6267531929695 | h4_fixed_plan_granularity | fixed_plan |
| 13 | `flopslab-4096-fixed-per-layer-flash-seed0` | aah | total | 1.015647 | 6267650317094 | h4_fixed_plan_granularity | fixed_plan |
| 14 | `flopslab-4096-static-plan-majority-flash-seed0` | aah | total | 1.015647 | 6267650321618 | h1_static_compiled_plan | static_compiled_plan |
| 15 | `flopslab-4096-static-plan-per-layer-flash-seed0` | aah | total | 1.015647 | 6267650321838 | h1_static_compiled_plan | static_compiled_plan |
| 16 | `flopslab-4096-quantized-two-bucket-1024-4096-flash-seed0` | aah | total | 1.017083 | 6276513219816 | h2_quantized_execution | quantized_execution |
| 17 | `flopslab-4096-quantized-two-bucket-2048-4096-flash-seed0` | aah | total | 1.017096 | 6276591445966 | h2_quantized_execution | quantized_execution |
| 18 | `flopslab-4096-quantized-single-2048-flash-seed0` | aah | total | 1.017132 | 6276817628095 | h2_quantized_execution | quantized_execution |
| 19 | `flopslab-4096-quantized-single-1024-flash-seed0` | aah | total | 1.017295 | 6277824394320 | h2_quantized_execution | quantized_execution |

## Attention NVTX Scope

| Rank | Experiment | Type | Scope | Ratio | GPU FLOPs | Hypothesis | Mode |
|---:|---|---|---|---:|---:|---|---|
| 1 | `flopslab-4096-baseline-pure-flash-attention-scope-seed0` | attention_denominator | attention | 1.000000 | 1808239165440 | baseline | pure_attention_scope |
| 2 | `flopslab-4096-same-codepath-full-flash-seed0` | attention_diagnostic_baseline | attention | 1.000000 | 1808239165440 | p1_same_codepath_full_baseline | same_codepath_full |
| 3 | `flopslab-4096-minruntime-noscatter-1024-4096-flash-seed0` | attention_aah | attention | 1.000437 | 1809030055936 | p3_minimal_runtime | minimal_runtime |
| 4 | `flopslab-4096-noscatter-contiguous-1024-4096-flash-seed0` | attention_aah | attention | 1.000437 | 1809030055936 | h3_noscatter_prototype | noscatter_prototype |

## Main Takeaway

The same-codepath full-window diagnostic baseline is `1.000000018x` in total-forward scope and `1.000000x` in attention scope, effectively identical to pure FlashAttention. This rules out generic AAH backend wrapper overhead as the source of the remaining total-forward gap.

Best total-forward AAH variant is `flopslab-4096-noscatter-contiguous-1024-4096-flash-seed0` at `1.015376x`; no tested AAH variant reached `<1.0` in total-forward GPU FLOPs.

P2 attention-only profiling shows the best no-scatter AAH row at `1.000437x` attention FLOPs versus pure FlashAttention. That means the `~1.54%` total-forward gap mostly collapses to a `~0.044%` attention-kernel gap when Nsight is filtered to backend attention ranges.

P3 minimal-runtime profiling did not improve the ratio: total-forward `1.015376x`, attention-scope `1.000437x`. Skipping profile-time diagnostic GPU reductions and using an uninitialized output buffer did not change measured GPU FLOPs, so the residual gap is not explained by those diagnostics.
