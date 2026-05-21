# AAH-v3 4096 Result Analysis

Source: W&B CSV export for the paper-facing 4096-token runs. Rows ending in `-infer` are final checkpoint inference rows; non-infer rows are final training-loop rows.

## Table 1 Training Values

| method | grouping | hierarchy | joint_scorer | context | seed | val_loss | ACR_train_proxy | flops_ratio | tokens_per_second_train | peak_gpu_alloc_GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pure baseline | off | none | off | 4096 | 0 | 6.5672 | 1.0000 | 1.0000 | 4418.4558 | 35.1780 |
| grouping_off | off | none | off | 4096 | 0 | 6.5655 | 0.5814 | 0.8712 | 3633.7810 | 35.9961 |
| Full adaptive | adaptive learned | [2,2,2,2] adaptive | wide joint | 4096 | 0 | 6.5590 | 0.3066 | 0.7867 | 3399.0929 | 36.7819 |
| Shallow freeze | learned frozen | [2] | wide joint | 4096 | 0 | 6.5367 | 0.3724 | 0.8069 | 3534.4544 | 36.7691 |
| Deep practical reuse | cached level-0 | [2,2,2,2] reuse | wide joint | 4096 | 0 | 6.5549 | 0.2891 | 0.7812 | 3394.9929 | 36.6157 |

## Table 2 Inference Values

| method | grouping | hierarchy | joint_scorer | context | checkpoint_step | val_loss | val_ppl | ACR | flops_ratio | tokens_per_second_infer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pure baseline | off | none | off | 4096 | 10000 | 6.5878 | 726.2109 | 1.0000 | 1.0000 | 14419.8467 |
| grouping_off | off | none | off | 4096 | 10000 | 6.5893 | 727.2689 | 0.5794 | 0.8706 | 8193.3611 |
| Full adaptive | adaptive learned | [2,2,2,2] adaptive | wide joint | 4096 | 10000 | 6.5824 | 722.2594 | 0.2575 | 0.7715 | 4454.8306 |
| Shallow freeze | learned frozen | [2] | wide joint | 4096 | 10000 | 6.5583 | 705.0939 | 0.3393 | 0.7967 | 7053.0189 |
| Deep practical reuse | cached level-0 | [2,2,2,2] reuse | wide joint | 4096 | 10000 | 6.5833 | 722.9317 | 0.3625 | 0.8038 | 4844.5606 |

## Key Interpretation

- Pure baseline: inference val_loss 6.5878 (+0.0000 vs baseline), val_ppl 726.21 (+0.00), ACR 1.0000, flops_ratio 1.0000, infer tok/s 14419.8.
- grouping_off: inference val_loss 6.5893 (+0.0015 vs baseline), val_ppl 727.27 (+1.06), ACR 0.5794, flops_ratio 0.8706, infer tok/s 8193.4.
- Full adaptive: inference val_loss 6.5824 (-0.0055 vs baseline), val_ppl 722.26 (-3.95), ACR 0.2575, flops_ratio 0.7715, infer tok/s 4454.8.
- Shallow freeze: inference val_loss 6.5583 (-0.0295 vs baseline), val_ppl 705.09 (-21.12), ACR 0.3393, flops_ratio 0.7967, infer tok/s 7053.0.
- Deep practical reuse: inference val_loss 6.5833 (-0.0045 vs baseline), val_ppl 722.93 (-3.28), ACR 0.3625, flops_ratio 0.8038, infer tok/s 4844.6.

Best final inference quality is Shallow freeze. Full adaptive has the lowest ACR/flops_ratio but is also the slowest in wall-clock inference, so the paper should phrase efficiency claims as compute-proxy savings rather than measured speedups. Learned grouping is supported by comparing Shallow freeze and Full adaptive against grouping_off: both improve quality and reduce ACR/flops_ratio, though with wall-clock overhead.

## Heatmap Caveat

The W&B CSV does not include true per-layer/per-head heatmap rows (`layer_id`, `head_id`, `final_window_size`). It only includes aggregate bucket fractions and a 12-value `aah/resolution_per_head_mean` vector. Use `prism_available_resolution_vector_proxy.csv` only as a proxy/summary strip, not as the true Figure 4 heatmap. The true heatmap needs diagnostic CSVs such as `*_step10000_heatmap.csv` from the run diagnostics directory.
