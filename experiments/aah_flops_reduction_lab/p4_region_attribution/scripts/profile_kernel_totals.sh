#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-paper_results/aah_flops_reduction_lab}"
NCU="${NCU:-/usr/local/cuda/bin/ncu}"
TIMEOUT="${PROFILE_TIMEOUT:-7200}"
RAW_DIR="$ROOT/ncu_raw"
SUMMARY_DIR="$ROOT/kernel_summaries"
PURE_CONFIG="experiments/backend_realized_local_attention/FlashAttention/pure/configs/backend_4096_pure_flash_seed0.yaml"
AAH_CONFIG="experiments/aah_flops_reduction_lab/p3_minimal_runtime/configs/flopslab-4096-minruntime-noscatter-1024-4096-flash-seed0.yaml"
PURE_JSON="$ROOT/gpu_flops_profiles/flopslab-4096-kernel-pure-flash-seed0_gpu_flops_profile.json"
AAH_JSON="$ROOT/gpu_flops_profiles/flopslab-4096-kernel-minruntime-noscatter-1024-4096-flash-seed0_gpu_flops_profile.json"
PURE_RAW="$RAW_DIR/flopslab-4096-kernel-pure-flash-seed0_ncu_raw.csv"
AAH_RAW="$RAW_DIR/flopslab-4096-kernel-minruntime-noscatter-1024-4096-flash-seed0_ncu_raw.csv"

python -m experiments.backend_realized_local_attention._common.profile_gpu_flops_ncu \
  --module pure \
  --config "$PURE_CONFIG" \
  --ncu "$NCU" \
  --warmup 1 \
  --repeats 1 \
  --timeout "$TIMEOUT" \
  --raw-csv-output "$PURE_RAW" \
  --output "$PURE_JSON"

python -m experiments.backend_realized_local_attention._common.profile_gpu_flops_ncu \
  --module aah \
  --config "$AAH_CONFIG" \
  --ncu "$NCU" \
  --warmup 1 \
  --repeats 1 \
  --timeout "$TIMEOUT" \
  --baseline-json "$PURE_JSON" \
  --raw-csv-output "$AAH_RAW" \
  --output "$AAH_JSON"

python experiments/aah_flops_reduction_lab/p4_region_attribution/scripts/summarize_ncu_kernels.py \
  --raw-csv "$PURE_RAW" \
  --output-json "$SUMMARY_DIR/flopslab-4096-kernel-pure-flash-seed0_kernel_summary.json" \
  --output-csv "$SUMMARY_DIR/flopslab-4096-kernel-pure-flash-seed0_kernel_summary.csv"

python experiments/aah_flops_reduction_lab/p4_region_attribution/scripts/summarize_ncu_kernels.py \
  --raw-csv "$AAH_RAW" \
  --output-json "$SUMMARY_DIR/flopslab-4096-kernel-minruntime-noscatter-1024-4096-flash-seed0_kernel_summary.json" \
  --output-csv "$SUMMARY_DIR/flopslab-4096-kernel-minruntime-noscatter-1024-4096-flash-seed0_kernel_summary.csv"
