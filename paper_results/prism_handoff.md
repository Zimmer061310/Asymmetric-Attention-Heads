# Prism Handoff: AAH-v3 4096 Paper Tables and Heatmap Data

Use these files in this folder:

- `aah_v3_4096_table1_training.csv`: fills paper Table 1 using final training-loop rows. Use `ACR_train_proxy` as the training ACR column; it is from `aah/attn_ratio` because W&B did not log `aah/ACR` on train rows. Memory is peak allocated GPU GiB.
- `aah_v3_4096_table2_inference.csv`: fills paper Table 2 using final `-infer` rows. This is the cleanest main table for final checkpoint quality and compute proxies.
- `prism_window_bucket_fractions.csv`: long-form window bucket fractions for grouped/bar plots.
- `prism_available_resolution_vector_proxy.csv`: available 12-value W&B vector converted to long form. Treat this as a layer/head-mean proxy strip only. It is not the true layer x head heatmap.

Recommended paper story:

1. Shallow freeze is the best final inference-quality row: val_loss 6.5583, val_ppl 705.09, ACR 0.3393, flops_ratio 0.7967.
2. Full adaptive gives the strongest compute-proxy reduction: ACR 0.2575 and flops_ratio 0.7715, but slower wall-clock inference.
3. grouping_off is not enough: it has worse final inference loss than baseline and higher ACR/flops_ratio than the learned-topology AAH rows.
4. Deep practical reuse is viable but not dominant in this seed-0 suite: val_loss 6.5833, ACR 0.3625, flops_ratio 0.8038.
5. Do not claim wall-clock speedup. The baseline has the highest measured inference tok/s; AAH results support compute-proxy reduction under quality constraints, with current implementation overhead.

True heatmap requirement:

For Figure 4, Prism needs the diagnostic heatmap CSVs, ideally with columns:

```text
regime,seed,checkpoint_step,layer_id,head_id,group_id,selected_window_idx,selected_window_size,pre_clamp_window_idx,post_clamp_window_idx,final_window_idx,final_window_size
```

The current `wandb_results.csv` does not contain those per-head rows. If the diagnostic CSVs can be recovered, use `final_window_size` as the heatmap color, y-axis `layer_id`, x-axis `head_id`, and create panels for `shallow_freeze` and `deep_practical_reuse` first, then optional `full_adaptive` and `grouping_off`.
