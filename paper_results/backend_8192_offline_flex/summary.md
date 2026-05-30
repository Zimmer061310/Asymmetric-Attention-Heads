# 8192 Offline Wikitext FlexAttention Smoke Results

- Date: 2026-05-30
- Server: AutoDL RTX PRO 6000 Blackwell Server Edition, 96 GB VRAM
- Branch/commit: `AAH-v3` at `8e1417e` on the server run checkout
- Dataset: offline tokenized Wikitext-2 (`wikitext2_gpt2.pt`)
- Sequence length: 8192
- Candidate windows: `[1024, 2048, 4096, 8192]`
- Backend: PyTorch FlexAttention
- Training budget: 500 steps per config, eval every 100 steps, checkpoints disabled
- Result type: backend/compute smoke, not a final quality experiment

## Result Table

| Config | Step | Train loss | Val loss | ACR | Realized ACR est. | Tok/s | GPU reserved MB | Backend | Kernel calls | Backend time ms | Avg window | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| `backend_8192_full_flex` | 500 | 18.660 | 17.678 | 1.000 | 1.000 | 4157.9 | 39634 | `flex_attention` | 16 | 15.16 | 8192 | none |
| `backend_8192_fixed_1024_flex` | 500 | 18.727 | 17.643 | 0.125 | 0.125 | 4102.5 | 39616 | `flex_attention` | 16 | 53.02 | 1024 | none |
| `backend_8192_fixed_2048_flex` | 500 | 18.808 | 17.729 | 0.250 | 0.250 | 4098.7 | 39616 | `flex_attention` | 16 | 53.12 | 2048 | none |
| `backend_8192_fixed_4096_flex` | 500 | 18.820 | 17.787 | 0.500 | 0.500 | 4098.9 | 39616 | `flex_attention` | 16 | 53.24 | 4096 | none |
| `backend_8192_full_adaptive_flex` | 500 | 18.796 | 17.766 | 0.380 | 0.380 | 3840.4 | 39698 | `flex_attention` | 27 | 56.92 | 3109 | none |
| `backend_8192_deep_practical_reuse_flex` | 500 | 18.792 | 17.759 | 0.383 | 0.383 | 3852.2 | 39738 | `flex_attention` | 27 | 56.53 | 3136 | none |

## Notes

- All six selected configs completed without OOM.
- All final train rows report `backend_name=flex_attention` and no fallback reasons.
- Fixed-window ACR matched the expected ratios: 1024 -> 0.125, 2048 -> 0.250, 4096 -> 0.500, full 8192 -> 1.000.
- AAH adaptive policies used real local FlexAttention execution and selected an effective ACR around 0.38 at the final step.
- Full 8192 FlexAttention used fewer backend calls/time than local bucketed modes in this implementation; quality/speed interpretation should separate policy selection from kernel-launch overhead.

