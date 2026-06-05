# AAH FLOPs Reduction Lab Plan And Log

This is the living plan for the isolated `aah-flops-reduction-lab` branch. The
primary target is still:

```text
gpu_flops_total_ratio_ncu < 1.0
```

The denominator is the matched pure FlashAttention row at `seq_len=4096`,
`batch_size=1`, `bf16`, measured with Nsight Compute GPU FLOP counters on the
same hardware.

## Current Evidence

The Pro 6000 Blackwell profile-only sweep completed cleanly. The compact
artifacts are committed under:

```text
paper_results/aah_flops_reduction_lab/
```

The pure FlashAttention baseline was:

```text
gpu_flops_total = 6,171,093,130,434
gpu_flops_total_ratio_ncu = 1.000000
```

Best AAH row:

```text
flopslab-4096-noscatter-contiguous-1024-4096-flash-seed0
gpu_flops_total = 6,265,980,625,227
gpu_flops_total_ratio_ncu = 1.015376
```

So the best tested AAH path is still about `1.54%` above pure FlashAttention in
measured total GPU FLOPs.

## Completed Hypotheses

### H2: Quantized Execution

Rows:

- `quantized-single-1024`: `1.017295`
- `quantized-single-2048`: `1.017132`
- `quantized-two-bucket-1024-4096`: `1.017083`
- `quantized-two-bucket-2048-4096`: `1.017096`

Interpretation: bucket count and coarse bucket shape are not the dominant
remaining cause. All four rows cluster around `1.017x`.

### H1: Static Compiled Plan

Rows:

- `static-plan-per-layer`: `1.015647`
- `static-plan-per-layer-head`: `1.015628`
- `static-plan-majority`: `1.015647`

Interpretation: skipping dynamic hierarchy/controller work helps, but only
slightly. Dynamic control is not the dominant remaining cause.

### H4: Fixed Plan Granularity

Rows:

- `fixed-per-layer`: `1.015647`
- `fixed-per-state`: `1.015628`
- `fixed-per-head-group`: `1.015628`
- `fixed-per-head`: `1.015628`
- `slow-update-N200`: `1.015628`
- `slow-update-N1000`: `1.015628`

Interpretation: fixed-plan granularity barely changes measured total GPU FLOPs.
The remaining overhead is not explained by adaptivity granularity.

### H3: No-Scatter Prototype

Rows:

- `noscatter-contiguous-1024-4096`: `1.015376`
- `noscatter-contiguous-layer-plan`: `1.015435`
- `noscatter-scatter-control-matched`: `1.015568`

Interpretation: no-scatter gives the best result, but the gain is only about
`0.019%` versus the matched scatter control and about `0.025%` versus the best
static/fixed rows. This is not enough to justify full head-reordering surgery
yet.

## Main Conclusion

The broad cheap hypotheses are exhausted:

- dynamic control overhead is not the main issue;
- bucket count is not the main issue;
- fixed plan granularity is not the main issue;
- scatter removal helps only marginally.

The remaining gap is small, about `1.5%` over pure FlashAttention. The next
work should target measurement/control overhead and codepath parity before
larger transformer surgery.

## Next Phase Plan

### P1: Same-Codepath Full Baseline

Status: completed on Pro 6000.

Result:

```text
flopslab-4096-same-codepath-full-flash-seed0
gpu_flops_total = 6,171,093,238,461
gpu_flops_total_ratio_ncu = 1.0000000175
```

Goal: determine whether the remaining `~1.5%` comes from the AAH lab/backend
transformer wrapper rather than AAH local execution itself.

Run a row that uses the AAH backend transformer codepath but forces full-window
execution with no dynamic controller or local-window reduction. Compare it
against pure FlashAttention.

Implementation handle:

```text
experiments/aah_flops_reduction_lab/baselines/configs/flopslab-4096-same-codepath-full-flash-seed0.yaml
experiments/aah_flops_reduction_lab/baselines/scripts/profile_same_codepath_full.sh
```

Expected outcomes:

- If the same-codepath full baseline is also around `1.015x`, then the measured
  overhead is mostly wrapper/codepath overhead, not AAH routing.
- If the same-codepath full baseline is near `1.0`, then AAH execution still
  adds real extra FLOPs that must be removed.

Interpretation: the same-codepath full-window diagnostic is effectively equal
to pure FlashAttention. This rules out generic AAH backend wrapper overhead as
the source of the `~1.5%` gap. The remaining overhead comes from the AAH
local/window execution path or its non-attention bookkeeping during that path.

### P2: Attention-Only Nsight Range

Status: implemented locally; ready for Pro 6000 profile-only launch.

Goal: separate attention-kernel FLOPs from total forward FLOPs.

Keep `gpu_flops_total_ratio_ncu` as the primary metric, but add an
attention-region profile row or range-tagged profile if the existing profiler
can support it cleanly.

Implementation handle:

```text
experiments/backend_realized_local_attention/_common/profile_gpu_flops_ncu.py
experiments/backend_realized_local_attention/_common/pure_backend_transformer.py
experiments/backend_realized_local_attention/_common/aah_backend_transformer.py
experiments/aah_flops_reduction_lab/baselines/scripts/profile_attention_scope.sh
```

The profiler now has an opt-in `--profile-scope attention` mode. It uses an
explicit `aah_ncu_attention` NVTX range around backend attention execution and
passes `--nvtx --nvtx-include aah_ncu_attention` to Nsight Compute. The default
scope remains total forward, so existing paper-facing total-FLOPs profiles are
unchanged.

Rows to launch:

- pure FlashAttention attention denominator;
- best AAH no-scatter row attention profile;
- same-codepath full-window AAH attention sanity check.

Expected outcomes:

- If attention-only is below `1.0` but total is `1.015x`, the attention saving
  exists but is washed out by non-attention overhead.
- If attention-only is also above `1.0`, the local Flash execution path itself
  still does not beat the pure FlashAttention kernel.

### P3: Minimal-Runtime AAH Path

Goal: shave the remaining measured overhead by disabling everything not needed
for inference execution.

Candidate removals for the profiled forward:

- diagnostic packing;
- branch-frequency bookkeeping;
- entropy/norm/head-usage statistics;
- per-forward debug dictionaries;
- avoidable tensor/list conversions;
- any timing/stat collection not required for the profile JSON.

Acceptance:

- preserve finite logits and output shape;
- profile against the same pure FlashAttention denominator;
- report total Nsight GPU FLOPs;
- promote only if `gpu_flops_total_ratio_ncu` drops meaningfully, ideally below
  `1.0`.

### P4: H5 Head Reordering Gate

Do not start full head-reordering implementation yet. H3 did not show a large
enough improvement. Revisit H5 only if P1-P3 show that codepath overhead is
under control and scatter/head layout remains the major residual cost.

## Open Decisions

- Whether to run P1-P3 immediately on the idle Pro 6000 instance.
- Whether P1-P3 should remain profile-only, or include a short `3000`-step
  quality probe after a promising profile result.
- Whether to keep the existing pure FlashAttention denominator or rerun it once
  alongside P1 for a fresh same-session denominator.
