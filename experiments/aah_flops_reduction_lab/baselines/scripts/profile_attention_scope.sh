#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-paper_results/aah_flops_reduction_lab}"
NCU="${NCU:-/usr/local/cuda/bin/ncu}"
TIMEOUT="${PROFILE_TIMEOUT:-7200}"
PURE_ATTENTION_JSON="$ROOT/gpu_flops_profiles/flashattention_pure_attention_gpu_flops_profile.json"

python -m experiments.backend_realized_local_attention._common.profile_gpu_flops_ncu \
  --module pure \
  --config experiments/backend_realized_local_attention/FlashAttention/pure/configs/backend_4096_pure_flash_seed0.yaml \
  --ncu "$NCU" \
  --warmup 1 \
  --repeats 1 \
  --timeout "$TIMEOUT" \
  --profile-scope attention \
  --output "$PURE_ATTENTION_JSON"

python -m experiments.backend_realized_local_attention._common.profile_gpu_flops_ncu \
  --module aah \
  --config experiments/aah_flops_reduction_lab/h3_noscatter_prototype/configs/flopslab-4096-noscatter-contiguous-1024-4096-flash-seed0.yaml \
  --ncu "$NCU" \
  --warmup 1 \
  --repeats 1 \
  --timeout "$TIMEOUT" \
  --profile-scope attention \
  --baseline-json "$PURE_ATTENTION_JSON" \
  --output "$ROOT/gpu_flops_profiles/flopslab-4096-noscatter-contiguous-1024-4096-flash-seed0_attention_gpu_flops_profile.json"

python -m experiments.backend_realized_local_attention._common.profile_gpu_flops_ncu \
  --module aah \
  --config experiments/aah_flops_reduction_lab/baselines/configs/flopslab-4096-same-codepath-full-flash-seed0.yaml \
  --ncu "$NCU" \
  --warmup 1 \
  --repeats 1 \
  --timeout "$TIMEOUT" \
  --profile-scope attention \
  --baseline-json "$PURE_ATTENTION_JSON" \
  --output "$ROOT/gpu_flops_profiles/flopslab-4096-same-codepath-full-flash-seed0_attention_gpu_flops_profile.json"
