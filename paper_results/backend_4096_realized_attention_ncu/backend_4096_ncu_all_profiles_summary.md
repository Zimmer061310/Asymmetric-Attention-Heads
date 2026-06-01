# Backend 4096 Nsight all-profile summary

This summary includes the original backend suite plus the post-suite dense and fixed-1024 checks. `gpu_flops_total_ratio_ncu` is the Nsight GPU FLOPs/FLOPs ratio; ACR/EAR are not used to compute it.

| label | gpu_flops_total | gpu_flops_total_ratio_ncu | ok |
|---|---:|---:|---|
| dense_aah_full_adaptive_window_exec | 285093795549.0 | 1.6016222807019223 | True |
| dense_standard_mha_baseline | 178003140306.0 | 1.0 | True |
| flashattention_deep_practical_reuse | 286868593890.0 | 1.607841646879165 | True |
| flashattention_fixed_1024_aah | 284734635245.0 | 1.5958812313606072 | True |
| flashattention_full_adaptive | 284931425607.0 | 1.5969842023601073 | True |
| flashattention_grouping_off | 284336102948.0 | 1.5936475367757315 | True |
| flashattention_pure | 178418437193.0 | 1.0 | True |
| flashattention_shallow_freeze | 286521792177.0 | 1.6058978919710059 | True |
| flexattention_grouping_off | 254069235543.0 | 1.0585880814153703 | True |
| flexattention_pure | 240007647926.0 | 1.0 | True |
| flexattention_shallow_freeze | 256973992645.0 | 1.0706908503358656 | True |
