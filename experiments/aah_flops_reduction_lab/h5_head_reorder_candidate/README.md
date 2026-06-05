# H5: Head Reorder Candidate

Goal: determine whether full head reordering is worth implementing after P3/P4
reduced the best AAH row to roughly `1.000425x` total-forward FLOPs.

## Lower-Bound Probe

The first H5 row is:

```text
flopslab-4096-headreorder-lowerbound-1024-4096-flash-seed0
```

It enables:

- `aah_flopslab_minimal_runtime=true`
- `aah_flopslab_assume_preordered_heads=true`

This is a semantics-changing lower-bound profile. It assumes the checkpoint's
Q/K/V heads are already physically ordered by the 1024/full bucket plan, so the
profiled forward can skip runtime Q/K/V `index_select` and concatenate bucket
outputs directly. That is not a valid model-quality result until a real
per-layer head permutation and output-projection adjustment is implemented.

Question answered:

```text
If head weights were already stored in bucket order, would removing runtime
head permutation and output scatter push Nsight gpu_flops_total_ratio_ncu below
1.0?
```

Success:

- ratio drops below the corrected P3 `1.000425x`;
- strong success is `<1.0`;
- if strong success holds, implement real head-reorder metadata, checkpoint
  loading rules, and output-projection permutation tests.

Failure:

- if the ratio stays around `1.0004x`, full head reordering is unlikely to
  matter enough by itself;
- if the ratio rises, the remaining residual is probably GEMM-shape/kernel
  selection noise rather than runtime scatter.

## Full Implementation Gate

Only after the lower-bound probe is positive:

- derive a per-layer head permutation from the best fixed/quantized plan;
- reorder Q/K/V head layout by window bucket;
- adjust output projection so model semantics match the original head order;
- save permutation metadata in checkpoints/configs;
- add round-trip permutation and logits-equivalence tests;
- keep `src/models/transformer.py` untouched until the lab result is clearly
  useful.
