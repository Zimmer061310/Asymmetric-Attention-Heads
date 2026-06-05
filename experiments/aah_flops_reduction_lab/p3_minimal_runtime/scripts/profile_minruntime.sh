#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-paper_results/aah_flops_reduction_lab}"
NCU="${NCU:-/usr/local/cuda/bin/ncu}"
TIMEOUT="${PROFILE_TIMEOUT:-7200}"
TOTAL_BASELINE_JSON="${TOTAL_BASELINE_JSON:-$ROOT/gpu_flops_profiles/flashattention_pure_gpu_flops_profile.json}"
ATTENTION_BASELINE_JSON="${ATTENTION_BASELINE_JSON:-$ROOT/gpu_flops_profiles/flashattention_pure_attention_gpu_flops_profile.json}"
CONFIG="experiments/aah_flops_reduction_lab/p3_minimal_runtime/configs/flopslab-4096-minruntime-noscatter-1024-4096-flash-seed0.yaml"

python -m experiments.backend_realized_local_attention._common.profile_gpu_flops_ncu \
  --module aah \
  --config "$CONFIG" \
  --ncu "$NCU" \
  --warmup 1 \
  --repeats 1 \
  --timeout "$TIMEOUT" \
  --baseline-json "$TOTAL_BASELINE_JSON" \
  --output "$ROOT/gpu_flops_profiles/flopslab-4096-minruntime-noscatter-1024-4096-flash-seed0_gpu_flops_profile.json"

python -m experiments.backend_realized_local_attention._common.profile_gpu_flops_ncu \
  --module aah \
  --config "$CONFIG" \
  --ncu "$NCU" \
  --warmup 1 \
  --repeats 1 \
  --timeout "$TIMEOUT" \
  --profile-scope attention \
  --baseline-json "$ATTENTION_BASELINE_JSON" \
  --output "$ROOT/gpu_flops_profiles/flopslab-4096-minruntime-noscatter-1024-4096-flash-seed0_attention_gpu_flops_profile.json"
