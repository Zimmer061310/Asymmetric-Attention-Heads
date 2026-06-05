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

Best AAH row after P3/P5 follow-up probes:

```text
flopslab-4096-headreorder-lowerbound-1024-4096-flash-seed0
gpu_flops_total = 6,171,928,931,891
gpu_flops_total_ratio_ncu = 1.000135
```

This is a semantics-changing lower-bound probe, not a valid quality result. It
is still about `0.0135%` above pure FlashAttention in measured total GPU FLOPs.
The best non-H5 lab path remains corrected P3 at `1.000425x`.

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

Status: completed on Pro 6000.

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
explicit `aah_ncu_attention` NVTX push/pop range around backend attention
execution and passes `--nvtx --nvtx-include aah_ncu_attention]` to Nsight
Compute. The closing bracket is required by this Nsight version for pushed
NVTX ranges. The default scope remains total forward, so existing paper-facing
total-FLOPs profiles are unchanged.

Rows to launch:

- pure FlashAttention attention denominator;
- best AAH no-scatter row attention profile;
- same-codepath full-window AAH attention sanity check.

Launch note: the first pure-row P2 attempt used `aah_ncu_attention` without the
push/pop closing bracket and failed with `ncu_metric_parse_empty`. A direct
bracket-pattern test succeeded for the pure Flash attention denominator:

```text
flashattention_pure_attention_gpu_flops_profile_bracket_test
gpu_flops_total = 1,808,239,165,440
gpu_flops_total_ratio_ncu = 1.000000
```

Final P2 rows:

```text
flashattention_pure_attention_gpu_flops_profile
gpu_flops_total = 1,808,239,165,440
gpu_flops_total_ratio_ncu = 1.000000

flopslab-4096-noscatter-contiguous-1024-4096-flash-seed0_attention
gpu_flops_total = 1,809,030,055,936
gpu_flops_total_ratio_ncu = 1.0004373816

flopslab-4096-same-codepath-full-flash-seed0_attention
gpu_flops_total = 1,808,239,165,440
gpu_flops_total_ratio_ncu = 1.000000
```

Interpretation: the attention-only Nsight range collapses the best AAH gap from
`1.015376x` total-forward FLOPs to `1.000437x` attention-scope FLOPs. AAH still
does not beat pure FlashAttention in the profiled attention kernels, but the
remaining attention-only gap is only about `0.044%`. The larger total-forward
gap is therefore mostly outside the backend attention kernels or in non-attention
work around local/window execution.

Expected outcomes:

- If attention-only is below `1.0` but total is `1.015x`, the attention saving
  exists but is washed out by non-attention overhead.
- If attention-only is also above `1.0`, the local Flash execution path itself
  still does not beat the pure FlashAttention kernel.

### P3: Minimal-Runtime AAH Path

Status: completed on Pro 6000.

Goal: shave the remaining measured overhead by disabling everything not needed
for inference execution.

Candidate removals for the profiled forward:

- diagnostic packing;
- branch-frequency bookkeeping;
- entropy/norm/head-usage statistics;
- per-forward debug dictionaries;
- avoidable tensor/list conversions;
- any timing/stat collection not required for the profile JSON.

Implementation handle:

```text
experiments/backend_realized_local_attention/_common/aah_backend_transformer.py
experiments/aah_flops_reduction_lab/p3_minimal_runtime/
```

The P3 config reuses the best no-scatter 1024/full plan and enables the
lab-only `aah_flopslab_minimal_runtime` flag. This skips profile-time GPU
diagnostic reductions such as `y_h.float().norm(...)`, attention entropy/usage
statistics, and full diagnostic dictionary packing during the profiled forward.
It also uses an uninitialized output buffer because every head is written once.

Rows to launch:

- total-forward P3 minimal-runtime no-scatter profile, divided by the pure
  FlashAttention total-forward denominator;
- attention-scope P3 minimal-runtime no-scatter profile, divided by the pure
  FlashAttention attention denominator.

Initial P3 rows before config-plumbing fix:

```text
flopslab-4096-minruntime-noscatter-1024-4096-flash-seed0
gpu_flops_total = 6,265,980,625,951
gpu_flops_total_ratio_ncu = 1.0153761244

flopslab-4096-minruntime-noscatter-1024-4096-flash-seed0_attention
gpu_flops_total = 1,809,030,055,936
gpu_flops_total_ratio_ncu = 1.0004373816
```

These initial rows were invalid for the intended P3 test because
`load_model()` did not pass `aah_flopslab_minimal_runtime` into `GPTConfig`.
After adding that plumbing, the server verified
`model.blocks[0].attn.flopslab_minimal_runtime == True` and P3 was rerun.

Corrected P3 rows:

```text
flopslab-4096-minruntime-noscatter-1024-4096-flash-seed0
gpu_flops_total = 6,173,714,488,149
gpu_flops_total_ratio_ncu = 1.0004247801

flopslab-4096-minruntime-noscatter-1024-4096-flash-seed0_attention
gpu_flops_total = 1,809,030,055,936
gpu_flops_total_ratio_ncu = 1.0004373816
```

Interpretation: P3 almost closes the original gap. The total-forward AAH ratio
falls from `1.015376x` to `1.000425x` once diagnostic GPU reductions and
profile-time stats are actually disabled. It is still slightly above pure
FlashAttention, but the gap is now about `0.0425%`.

Acceptance:

- preserve finite logits and output shape;
- profile against the same pure FlashAttention denominator;
- report total Nsight GPU FLOPs;
- promote only if `gpu_flops_total_ratio_ncu` drops meaningfully, ideally below
  `1.0`.

### P4: Region Attribution

Status: completed on Pro 6000.

Goal: identify which non-attention region accounted for the original gap between
attention-scope `1.000437x` and the pre-fix total-forward `1.015376x`, then
rerun attribution after corrected P3.

Implementation handle:

```text
experiments/backend_realized_local_attention/_common/aah_backend_transformer.py
experiments/backend_realized_local_attention/_common/profile_gpu_flops_ncu.py
experiments/aah_flops_reduction_lab/p4_region_attribution/
```

The AAH backend now tags the following NVTX push/pop ranges:

- `aah_ncu_qkv`
- `aah_ncu_bucket_select`
- `aah_ncu_attention`
- `aah_ncu_output_assembly`
- `aah_ncu_output_projection`
- `aah_ncu_mlp`

The profiler now supports `--profile-scope cuda_region --profile-label <range>`,
which records raw `gpu_flops_region` for a selected region by opening
CUDA-profiler start/stop inside that range. A plain NVTX-only first attempt
failed for `aah_ncu_qkv` with `ncu_metric_parse_empty`, likely because
PyTorch/cuBLAS kernels are not reliably attributed to the Python NVTX push/pop
range. P4 profiles the P3 minimal-runtime no-scatter config so diagnostic
reductions stay out of the attribution.

Launch note: the CUDA-gated first attempt also returned
`ncu_metric_parse_empty` for both `aah_ncu_qkv` and `aah_ncu_attention`, so P4
fell back to full-total kernel-name attribution. The profiler now supports
`--raw-csv-output`; P4 reran pure Flash and corrected P3-minimal AAH total
profiles, preserved raw Nsight CSVs on the server, and copied compact kernel
summaries back under `paper_results/aah_flops_reduction_lab/kernel_summaries/`.

Corrected P4 rows:

```text
flopslab-4096-kernel-pure-flash-seed0
gpu_flops_total = 6,171,093,131,871
gpu_flops_total_ratio_ncu = 1.000000

flopslab-4096-kernel-minruntime-noscatter-1024-4096-flash-seed0
gpu_flops_total = 6,173,714,490,376
gpu_flops_total_ratio_ncu = 1.0004247803
```

Kernel-summary interpretation: the parsed kernel-summary total is only
`316,225,875` FLOPs above pure. Positive deltas come mainly from extra/split
CUTLASS GEMM variants and small index-select/copy kernels, while removed or
smaller full-shape GEMM variants mostly cancel them. This points to bucketed
head layout / GEMM-shape fragmentation as the remaining issue, not diagnostics
or FlashAttention itself.

Expected outcomes:

- If bucket selection or output assembly is large, target tensor layout and
  reassembly before full head-reordering.
- If output projection or MLP dominates equally to pure Flash, the residual
  gap is likely untagged runtime/library kernels or shape/layout differences.
- If region totals do not explain the gap, switch to Nsight kernel-name/native
  stack attribution instead of making more model changes.

### P5: H5 Head-Reorder Lower-Bound Probe

Status: completed on Pro 6000.

Do not start full head-reordering surgery yet. The corrected P3/P4 results show
that codepath overhead is mostly under control and the remaining gap is only
about `0.0425%`, with small positive deltas from runtime head selection/copy and
split GEMM shape variants. The next step is therefore a lower-bound profile,
not a checkpoint-compatible implementation.

Row to launch:

```text
flopslab-4096-headreorder-lowerbound-1024-4096-flash-seed0
```

Implementation handle:

```text
experiments/aah_flops_reduction_lab/h5_head_reorder_candidate/
experiments/backend_realized_local_attention/_common/aah_backend_transformer.py
```

The row enables `aah_flopslab_assume_preordered_heads=true` together with the
corrected P3 minimal-runtime path. During profiled forward, the lab path assumes
the Q/K/V heads are already physically stored in bucket order, so it skips
runtime Q/K/V `index_select` and concatenates bucket outputs directly. This is a
semantics-changing lower-bound probe, not a valid model-quality result.

Question answered:

```text
If future head-reordered weights made bucket order physical, would removing the
remaining runtime permutation/output-scatter path push the Nsight ratio below
1.0?
```

Success:

- ratio falls below corrected P3 `1.000425x`;
- strong success is `<1.0`;
- only then implement real per-layer head permutation, output-projection
  adjustment, checkpoint metadata, and equivalence tests.

Failure:

- if the row stays around `1.0004x`, full head reordering is probably too small
  to matter alone;
- if it rises, the remaining residual is likely GEMM-shape/kernel-selection
  noise rather than explicit scatter.

Observed rows:

```text
flopslab-4096-headreorder-lowerbound-1024-4096-flash-seed0
gpu_flops_total = 6,171,928,931,891
gpu_flops_total_ratio_ncu = 1.0001354382

flopslab-4096-headreorder-lowerbound-1024-4096-flash-seed0_attention
gpu_flops_total = 1,809,030,055,936
gpu_flops_total_ratio_ncu = 1.0004373816
```

Interpretation: the lower-bound head-reorder proxy improves total-forward from
corrected P3 `1.000425x` to `1.000135x`, so runtime Q/K/V permutation and output
scatter were part of the residual. However, even the semantics-changing
lower-bound probe remains above pure FlashAttention, and attention-scope FLOPs
are unchanged. Full head-reordering alone is therefore not a strong enough path
to a paper-facing `<1.0` FLOPs ratio. A real implementation may still be useful
for cleanup or runtime, but the next compute-reduction attempt should target the
attention kernel schedule or reduce the number/shape fragmentation of attention
and projection/GEMM launches, not just head order.

## Open Decisions

- Whether to run P1-P3 immediately on the idle Pro 6000 instance.
- Whether P1-P3 should remain profile-only, or include a short `3000`-step
  quality probe after a promising profile result.
- Whether to keep the existing pure FlashAttention denominator or rerun it once
  alongside P1 for a fresh same-session denominator.
