#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-paper_results/aah_flops_reduction_lab}"
NCU="${NCU:-/usr/local/cuda/bin/ncu}"

python -m experiments.backend_realized_local_attention._common.profile_gpu_flops_ncu \
  --module pure \
  --config experiments/backend_realized_local_attention/FlashAttention/pure/configs/backend_4096_pure_flash_seed0.yaml \
  --ncu "$NCU" \
  --warmup 1 \
  --repeats 1 \
  --timeout "${PROFILE_TIMEOUT:-7200}" \
  --output "$ROOT/gpu_flops_profiles/flashattention_pure_gpu_flops_profile.json"
