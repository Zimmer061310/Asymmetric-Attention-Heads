#!/usr/bin/env bash
set -euo pipefail
cd /home/featurize/work/AAH-v3-pro6000
export PATH=/home/featurize/work/bin:$PATH
export PIP_CACHE_DIR=/home/featurize/data/pip-cache
ROOT=paper_results/backend_4096_realized_attention_ncu
OUT=$ROOT/gpu_flops_profiles
LOG=$ROOT/post_suite_dense_ncu/post_suite_dense_ncu.log
mkdir -p "$OUT" "$ROOT/post_suite_dense_ncu"
{
  echo "$(date -u +%FT%TZ) post-suite dense Nsight job armed"
  while screen -ls | grep -q 'aah_backend_pro6000'; do
    echo "$(date -u +%FT%TZ) waiting for aah_backend_pro6000 to finish"
    sleep 300
  done
  echo "$(date -u +%FT%TZ) main suite ended; starting dense baseline profile"
  python -m experiments.backend_realized_local_attention._common.profile_gpu_flops_ncu \
    --module pure \
    --config configs/paper_required/main_4096_pure_baseline_seed0.yaml \
    --ncu /home/featurize/work/bin/ncu-sudo \
    --warmup 1 \
    --repeats 1 \
    --timeout 7200 \
    --output "$OUT/dense_standard_mha_baseline_gpu_flops_profile.json"
  echo "$(date -u +%FT%TZ) dense baseline profile done; starting dense AAH full_adaptive profile"
  python -m experiments.backend_realized_local_attention._common.profile_gpu_flops_ncu \
    --module aah \
    --config configs/paper_required/main_4096_full_adaptive_seed0.yaml \
    --checkpoint experiments/backend_realized_local_attention/FlexAttention/aah_modified/results/backend-4096-full_adaptive-flex-seed0.pt \
    --baseline-json "$OUT/dense_standard_mha_baseline_gpu_flops_profile.json" \
    --ncu /home/featurize/work/bin/ncu-sudo \
    --warmup 1 \
    --repeats 1 \
    --timeout 7200 \
    --output "$OUT/dense_aah_full_adaptive_window_exec_gpu_flops_profile.json"
  echo "$(date -u +%FT%TZ) dense AAH full_adaptive profile done; starting fixed 1024 AAH + FlashAttention profile"
  python -m experiments.backend_realized_local_attention._common.profile_gpu_flops_ncu \
    --module aah \
    --config "$ROOT/post_suite_dense_ncu/backend_4096_fixed_1024_flash_seed0.yaml" \
    --baseline-json "$OUT/flashattention_pure_gpu_flops_profile.json" \
    --ncu /home/featurize/work/bin/ncu-sudo \
    --warmup 1 \
    --repeats 1 \
    --timeout 7200 \
    --output "$OUT/flashattention_fixed_1024_aah_gpu_flops_profile.json"
  echo "$(date -u +%FT%TZ) fixed 1024 AAH + FlashAttention profile done"
} 2>&1 | tee -a "$LOG"
