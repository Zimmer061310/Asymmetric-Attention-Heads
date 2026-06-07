# AAH Quality / Structure Phase 1 Summary

ACR/EAR and analytic FLOPs fields are routing diagnostics only in this lab.

| Rank | Run | Val loss | Val ppl | Train loss | Tok/s | GPU alloc max MB | Window ablation |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `quality-4096-phase1-shallow-control-interval10-seed0_shallow-control-interval10` | 7.277523 | 1447.3977 | 8.568014 | 1052.77 | 37641.52 | {'adaptive': 1.0} |
| 2 | `quality-4096-phase1-fixed-random-grouping-seed0_fixed-random-grouping` | 7.283056 | 1455.4297 | 8.574179 | 1239.22 | 37386.53 | {'adaptive': 1.0} |
| 3 | `quality-4096-phase1-fixed-1024-seed0_fixed-1024` | 7.283394 | 1455.9215 | 8.560406 | 1247.92 | 37193.72 | {'fixed_window': 1.0} |
| 4 | `quality-4096-phase1-shallow-resolution-ema030-seed0_shallow-resolution-ema030` | 7.283815 | 1456.5338 | 8.631633 | 1244.52 | 37737.65 | {'adaptive': 1.0} |
| 5 | `quality-4096-phase1-shallow-freeze-seed0_shallow-freeze` | 7.291971 | 1468.4625 | 8.529487 | 1338.53 | 37644.53 | {'adaptive': 1.0} |
| 6 | `quality-4096-phase1-shallow-no512-seed0_shallow-no512` | 7.293451 | 1470.6375 | 8.910921 | 1207.89 | 37117.70 | {'adaptive': 1.0} |
| 7 | `quality-4096-phase1-shallow-random-uniform-seed0_shallow-random-uniform` | 7.295536 | 1473.7066 | 8.550664 | 1238.74 | 38111.03 | {'random_uniform': 1.0} |
| 8 | `quality-4096-phase1-full-adaptive-seed0_full-adaptive` | 7.299678 | 1479.8233 | 8.582784 | 1284.85 | 37726.39 | {'adaptive': 1.0} |
| 9 | `quality-4096-phase1-fixed-2048-seed0_fixed-2048` | 7.303891 | 1486.0704 | 8.554174 | 1245.59 | 37195.77 | {'fixed_window': 1.0} |
| 10 | `quality-4096-phase1-shallow-shuffle-post-select-seed0_shallow-shuffle-post-select` | 7.307424 | 1491.3306 | 8.585091 | 1339.99 | 37644.52 | {'shuffle_post_select': 1.0} |
| 11 | `quality-4096-phase1-pure-baseline-seed0_pure-baseline` | 7.322616 | 1514.1601 | 8.622456 | 1395.18 | 36022.24 |  |
| 12 | `quality-4096-phase1-full-adaptive-shuffle-post-select-seed0_full-adaptive-shuffle-post-select` | 7.340420 | 1541.3586 | 8.723300 | 1109.70 | 37691.34 | {'shuffle_post_select': 1.0} |

## Promotion Rule

Promote only 3-4 rows to 5000 steps: best AAH reference, any decisive random/shuffle control, best fixed control, and best optimization variant.
