# P6: Context Scaling Profiles

Goal: test whether AAH only becomes FLOPs-competitive when context length is
larger than the 4096-token lab runs.

Rows:

- `flopslab-8192-baseline-pure-flash-seed0`
- `flopslab-8192-minruntime-noscatter-1024-8192-flash-seed0`
- `flopslab-8192-headreorder-lowerbound-1024-8192-flash-seed0`

All rows are profile-only:

- `seq_len=8192`
- `batch_size=1`
- `bf16`
- FlashAttention backend
- Nsight Compute hardware/derived FLOP counters

The AAH rows reuse the learned/static 4096 head-window pattern from:

```text
paper_results/aah_flops_reduction_lab/plans/flopslab-4096-noscatter-contiguous-1024-4096-flash-seed0.json
```

For 8192, the execution quantizer maps selected windows below 4096 to `1024`
and selected 4096/full heads to `8192`. This preserves the same short/full head
pattern while making the full bucket equal to the new context length.

Expected interpretation:

- If P3 or H5 falls below `1.0`, AAH may only be useful as a longer-context
  systems result.
- If both remain `>=1.0`, the current FlashAttention + AAH path still does not
  prove true GPU FLOPs reduction.
- H5 remains a semantics-changing lower-bound profile, not a quality result.
