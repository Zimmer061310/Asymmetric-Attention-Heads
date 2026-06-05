#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-paper_results/aah_flops_reduction_lab}"
NCU="${NCU:-/usr/local/cuda/bin/ncu}"
TIMEOUT="${PROFILE_TIMEOUT:-7200}"
BASELINE_JSON="${BASELINE_JSON:-$ROOT/gpu_flops_profiles/flashattention_pure_gpu_flops_profile.json}"
CONFIG="experiments/aah_flops_reduction_lab/p3_minimal_runtime/configs/flopslab-4096-minruntime-noscatter-1024-4096-flash-seed0.yaml"

profile_region() {
  local region="$1"
  local out_name="$2"
  python -m experiments.backend_realized_local_attention._common.profile_gpu_flops_ncu \
    --module aah \
    --config "$CONFIG" \
    --ncu "$NCU" \
    --warmup 1 \
    --repeats 1 \
    --timeout "$TIMEOUT" \
    --profile-scope nvtx \
    --profile-label "$region" \
    --baseline-json "$BASELINE_JSON" \
    --output "$ROOT/gpu_flops_profiles/${out_name}_gpu_flops_profile.json"
}

profile_region "aah_ncu_qkv" "flopslab-4096-region-qkv-minruntime-noscatter-1024-4096-flash-seed0"
profile_region "aah_ncu_bucket_select" "flopslab-4096-region-bucket-select-minruntime-noscatter-1024-4096-flash-seed0"
profile_region "aah_ncu_attention" "flopslab-4096-region-attention-minruntime-noscatter-1024-4096-flash-seed0"
profile_region "aah_ncu_output_assembly" "flopslab-4096-region-output-assembly-minruntime-noscatter-1024-4096-flash-seed0"
profile_region "aah_ncu_output_projection" "flopslab-4096-region-output-projection-minruntime-noscatter-1024-4096-flash-seed0"
profile_region "aah_ncu_mlp" "flopslab-4096-region-mlp-minruntime-noscatter-1024-4096-flash-seed0"
