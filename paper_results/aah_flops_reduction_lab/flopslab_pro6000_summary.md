# AAH FLOPs Lab Pro 6000 Nsight Summary

All rows use Nsight Compute GPU FLOP counters on RTX PRO 6000 Blackwell, seq_len=4096, batch_size=1, bf16. Total-forward rows divide by the matched pure FlashAttention total-forward baseline. Attention-scope rows divide by the matched pure FlashAttention attention NVTX-scope baseline. Lower is better; values below 1.0 would indicate lower measured GPU FLOPs than pure FlashAttention for that scope.

## Total Forward Scope

| Rank | Experiment | Type | Scope | Ratio | GPU FLOPs | Hypothesis | Mode |
|---:|---|---|---|---:|---:|---|---|
| 1 | `flopslab-4096-baseline-pure-flash-seed0` | denominator | total | 1.000000 | 6171093130434 | baseline | pure |
| 2 | `flopslab-4096-same-codepath-full-flash-seed0` | diagnostic_baseline | total | 1.000000 | 6171093238461 | p1_same_codepath_full_baseline | same_codepath_full |
| 3 | `flopslab-4096-minruntime-noscatter-1024-4096-flash-seed0` | aah | total | 1.000425 | 6173714488149 | p3_minimal_runtime | minimal_runtime |
| 4 | `flopslab-4096-noscatter-contiguous-1024-4096-flash-seed0` | aah | total | 1.015376 | 6265980625227 | h3_noscatter_prototype | noscatter_prototype |
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

## Kernel Attribution Scope

| Rank | Experiment | Type | Scope | Ratio | GPU FLOPs | Hypothesis | Mode |
|---:|---|---|---|---:|---:|---|---|
| 1 | `backend-4096-pure-flash-seed0` | kernel_denominator | total | 1.000000 | 6171093131871 | p4_region_attribution | kernel_total_raw_csv |
| 2 | `flopslab-4096-minruntime-noscatter-1024-4096-flash-seed0` | kernel_aah | total | 1.000425 | 6173714490376 | p4_region_attribution | kernel_total_raw_csv |

## Main Takeaway

The same-codepath full-window diagnostic baseline is `1.000000018x` in total-forward scope and `1.000000x` in attention scope, effectively identical to pure FlashAttention. This rules out generic AAH backend wrapper overhead as the source of the remaining total-forward gap.

After fixing the config plumbing for `aah_flopslab_minimal_runtime`, the P3 minimal-runtime no-scatter path is the best total-forward AAH row: `1.000425x` total-forward and `1.000437x` attention-scope. It is still slightly above pure FlashAttention, but the gap is now about `0.0425%`, not `1.54%`.

P4 kernel attribution with raw Nsight CSVs shows the corrected AAH kernel-summary total is only `316225875` parsed FLOPs above pure in the summarized kernels. The largest positive deltas are extra/split CUTLASS GEMM shape variants and small index-select/copy kernels, while some full-shape GEMM variants decrease and mostly cancel the increase. This points toward bucketed head layout / GEMM-shape fragmentation as the remaining issue, not diagnostics or the FlashAttention kernel itself.

Top positive kernel deltas:
- `217796274742` FLOPs, count delta `7`: `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nn_align8>(T1::Params)`
- `43481255326` FLOPs, count delta `3`: `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x4_nn_align8>(T1::Params)`
- `4376786203` FLOPs, count delta `1`: `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(T1::Params)`
- `929687595` FLOPs, count delta `48`: `void at::<unnamed>::indexSelectSmallIndex<c10::BFloat16, long, unsigned int, (int)-1, (int)-1, (int)-1>(cuda::TensorInfo<T1, T3>, cuda::TensorInfo<const T1, T3>`
- `379275909` FLOPs, count delta `5`: `void at::<unnamed>::cunn_SoftMaxForwardReg<float, float, float, at::<unnamed>::SoftMaxForwardEpilogue, long, 4>(T3 *, const T1 *, T5)`
