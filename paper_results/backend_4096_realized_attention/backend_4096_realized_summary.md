# Backend-Realized 4096 Attention Summary

Final rows from the 4096 FlexAttention/FlashAttention backend-realized suite. ACR is the policy-selected attention ratio from the final training step; realized_attention_flops_formula_ratio is backend-window formula based; profiler_total_flops_ratio is the PyTorch profiler-reported total FLOPs ratio versus the pure backend baseline.

| backend | method | final_step | val_loss | val_ppl | attn_ratio_acr_last | tok_s_last | realized_attention_flops_formula_ratio | profiler_total_flops_ratio | peak_memory_mb_profile | backend_names_profile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FlexAttention | pure | 10000.0 | 6.4811 | 652.6909 | 1.0000 | 5854.6 | 1.0000 | 1.0000 | 6568.8 | flex_attention |
| FlexAttention | grouping_off | 10000.0 | 6.4598 | 638.9440 | 0.5898 | 4930.4 | 0.5801 | 1.0000 | 7067.7 | flex_attention |
| FlexAttention | full_adaptive | 10000.0 | 6.4494 | 632.3335 | 0.2331 | 4390.4 | 0.2493 | 1.0000 | 7377.2 | flex_attention |
| FlexAttention | shallow_freeze | 10000.0 | 6.4532 | 634.7131 | 0.3581 | 4558.4 | 0.4466 | 1.0000 | 7081.2 | flex_attention |
| FlexAttention | deep_practical_reuse | 10000.0 | 6.4420 | 627.6783 | 0.2773 | 4638.4 | 0.2917 | 1.0000 | 7411.5 | flex_attention |
| FlashAttention | pure | 10000.0 | 6.4660 | 642.8943 | 1.0000 | 16185.5 | 1.0000 | 1.0000 | 4543.6 | flash_attn |
| FlashAttention | grouping_off | 10000.0 | 6.4675 | 643.8887 | 0.5879 | 15448.4 | 0.5911 | 1.0000 | 4541.7 | flash_attn |
| FlashAttention | full_adaptive | 10000.0 | 6.4602 | 639.1940 | 0.3132 | 12323.5 | 0.3249 | 1.0000 | 4545.5 | flash_attn |
| FlashAttention | shallow_freeze | 10000.0 | 6.4472 | 630.9369 | 0.3945 | 13252.8 | 0.3359 | 1.0000 | 4542.5 | flash_attn |
| FlashAttention | deep_practical_reuse | 10000.0 | 6.4416 | 627.4379 | 0.3320 | 12631.9 | 0.3112 | 1.0000 | 4542.5 | flash_attn |
