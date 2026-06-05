# H5 Scripts

`profile_headreorder_lowerbound.sh` runs the lab-only lower-bound profile:

```text
flopslab-4096-headreorder-lowerbound-1024-4096-flash-seed0
```

It records total-forward and attention-scope Nsight ratios against the matched
pure FlashAttention denominators under `paper_results/aah_flops_reduction_lab`.

This script does not validate model quality. The config intentionally assumes
heads are already physically reordered, which is only a proxy for a future clean
implementation.
