# Prompt for Prism: Update AAH-v3 Experiment Table

Please update the AAH-v3 paper experiment section and experiment table to match the revised runnable protocol below.

The old table described a 512-token main comparison with 3 seeds plus a separate 1024-token stress suite. That is no longer the protocol. Replace it with a single 4096-token comparison suite using one seed only and 10k optimizer steps.

Required main runs:

| Run ID | Regime | Context | AAH? | Grouping | Hierarchy | Joint Scorer | Seeds |
|---|---:|---:|---:|---|---|---|---:|
| `main_4096_pure_baseline` | Pure baseline | `4096` | no | none | none | none | `1` |
| `main_4096_grouping_off` | `grouping_off` | `4096` | yes | off | none | off | `1` |
| `main_4096_full_adaptive` | Full adaptive | `4096` | yes | adaptive learned | `[2,2,2,2]` adaptive | wide joint | `1` |
| `main_4096_shallow_freeze` | Shallow freeze | `4096` | yes | learned frozen | `[2]` | wide joint | `1` |
| `main_4096_deep_practical_reuse` | Deep practical reuse | `4096` | yes | cached level-0 | `[2,2,2,2]` reuse | wide joint | `1` |

Main-run total: 5 regimes x 1 seed = 5 training runs.

Use these shared settings:

- `model_scale`: `1B`
- `context_length`: `4096`
- `optimizer_steps`: `10000`
- `seed`: `0`
- `aah_v3_candidate_windows`: `[512,1024,2048,4096]`
- `batch_size`: `1`
- `checkpoint_steps`: `[1000,5000,10000]`
- `eval_interval`: `1000`
- same tokenizer, data pipeline, optimizer schedule, model width/depth, and flat Transformer output interface across all rows

Remove the separate long-context stress table, because 4096 is now the main context length. If the text still wants a stress/capacity section, describe it as future or optional follow-up rather than required evidence.

Appendix diagnostics should also be revised to 4096 context, 10k steps, and one seed:

| Appendix Run | Variable Changed |
|---|---|
| `appendix_4096_control_off` | disable controller decisions / use no adaptive control |
| `appendix_4096_fixed_random_grouping` | replace learned grouping with fixed random grouping |
| `appendix_4096_freeze_after_warmup_passthrough` | freeze/pass through topology after warmup |
| `appendix_4096_independent_scoring` | use independent scoring instead of joint sibling scoring |
| `appendix_4096_no_parent_constraint` | disable parent index constraint |
| `appendix_4096_no_feature_ema` | disable feature EMA smoothing |

Appendix total: 6 runs. Full package total: 11 runs.

Please update any prose that says "3 seeds", "15 main runs", "24 mandatory runs", "42 full runs", "512-token main comparison", "1024-token main comparison", "2048-token main comparison", "1000 optimizer steps", or "separate 1024-token stress test". Replace those statements with the new 4096-token, 10k-step, single-seed protocol above.

For figures and diagnostics, keep the same requested logs and plots, but update labels from 512/1024/2048 to 4096 where appropriate. Heatmaps should be reported for `main_4096_shallow_freeze`, `main_4096_deep_practical_reuse`, and optionally `main_4096_full_adaptive` / `main_4096_grouping_off`.
