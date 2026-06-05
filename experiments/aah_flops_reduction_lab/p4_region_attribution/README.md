# P4: Region Attribution

Goal: identify which non-attention region accounts for the remaining
total-forward Nsight FLOPs gap after P2/P3.

This step does not change execution semantics. It adds NVTX ranges around the
best current AAH path and profiles them individually:

- `aah_ncu_qkv`
- `aah_ncu_bucket_select`
- `aah_ncu_attention`
- `aah_ncu_output_assembly`
- `aah_ncu_output_projection`
- `aah_ncu_mlp`

Expected result: one of the non-attention regions explains most of the
difference between total-forward `1.015376x` and attention-scope `1.000437x`.

Failure interpretation: if the region totals do not explain the gap, Nsight
attribution is being split across untagged runtime/library kernels and the next
step should use native stack or kernel-name reports rather than more model code
changes.
