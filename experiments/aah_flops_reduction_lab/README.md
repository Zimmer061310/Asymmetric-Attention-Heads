# AAH FLOPs Reduction Lab

This lab isolates experiments whose only systems goal is to make AAH produce a
true Nsight Compute GPU FLOPs ratio below `1.0`.

The paper branch evidence shows that current AAH execution increases measured
GPU FLOPs despite lower ACR/EAR. This lab therefore treats ACR and EAR as
diagnostics only. The primary metric is:

```text
gpu_flops_total_ratio_ncu =
  Nsight GPU FP ops for the AAH variant
  / Nsight GPU FP ops for the matched pure FlashAttention baseline
```

## Isolation rule

Do not edit `src/models/transformer.py` for lab ideas until a local prototype
shows a real Nsight improvement. Lab code lives here and may import the existing
backend-realized experiment modules.

## Naming rule

All experiment names use:

```text
flopslab-4096-{axis}-{variant}-{backend}-seed0
```

Examples:

- `flopslab-4096-static-plan-per-layer-flash-seed0`
- `flopslab-4096-quantized-two-bucket-flash-seed0`
- `flopslab-4096-noscatter-contiguous-flash-seed0`
- `flopslab-4096-fixed-per-head-group-flash-seed0`

## Directory map

```text
aah_flops_reduction_lab/
  _common/                 # shared lab utilities and schemas
  baselines/               # pure Flash denominators and dense diagnostics
  h1_static_compiled_plan/ # compile dynamic AAH decisions into static plans
  h2_quantized_execution/  # collapse decisions into one/two GPU-friendly buckets
  h3_noscatter_prototype/  # contiguous head blocks without arbitrary scatter
  h4_fixed_plan_granularity/
  h5_head_reorder_candidate/
```

Each hypothesis folder contains its own `README.md`, `configs/`, `scripts/`,
and `results/`. Checkpoints and raw W&B folders do not belong in Git.

## First-run order

1. Export static AAH plans with `h1_static_compiled_plan/scripts/export_static_plan.sh`.
2. Generate/update variant configs with `_common/make_lab_configs.py`.
3. Profile only the pure Flash denominator and one cheap H1/H2 variant.
4. Continue only if the ratio drops meaningfully below the current `~1.60`.

## Training budget

Lab configs are short probes by default, not paper reruns. Generated configs use
`max_steps=3000` and checkpoints at `1000/2000/3000`. To test a longer probe,
regenerate configs with:

```bash
python -m experiments.aah_flops_reduction_lab._common.make_lab_configs \
  --max-steps 5000 \
  --checkpoint-steps 1000,3000,5000
```
