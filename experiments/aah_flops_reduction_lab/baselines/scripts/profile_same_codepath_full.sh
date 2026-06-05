#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-paper_results/aah_flops_reduction_lab}"
NCU="${NCU:-/usr/local/cuda/bin/ncu}"
BASELINE_JSON="${BASELINE_JSON:-$ROOT/gpu_flops_profiles/flashattention_pure_gpu_flops_profile.json}"

python -m experiments.backend_realized_local_attention._common.profile_gpu_flops_ncu \
  --module aah \
  --config experiments/aah_flops_reduction_lab/baselines/configs/flopslab-4096-same-codepath-full-flash-seed0.yaml \
  --ncu "$NCU" \
  --warmup 1 \
  --repeats 1 \
  --timeout "${PROFILE_TIMEOUT:-7200}" \
  --baseline-json "$BASELINE_JSON" \
  --output "$ROOT/gpu_flops_profiles/flopslab-4096-same-codepath-full-flash-seed0_gpu_flops_profile.json"
