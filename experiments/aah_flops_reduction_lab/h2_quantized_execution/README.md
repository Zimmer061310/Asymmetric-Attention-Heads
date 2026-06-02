# H2: Quantized Execution

Goal: test whether multi-bucket fragmentation causes current Flash AAH to use
more GPU FLOPs than pure FlashAttention.

Variants:

- `quantized-single-1024`
- `quantized-single-2048`
- `quantized-two-bucket-1024-4096`
- `quantized-two-bucket-2048-4096`

Expected positive signal: fewer execution buckets substantially reduce
`gpu_flops_total_ratio_ncu`.

Failure interpretation: if fixed/quantized buckets remain near `1.6`, bucket
count and kernel launch count are not the only problem.
