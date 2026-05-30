#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"

BASE_CFG="experiments/backend_realized_local_attention/FlashAttention/pure/configs/backend_4096_pure_flash_seed0.yaml"
BASE_JSON="experiments/backend_realized_local_attention/FlashAttention/pure/results/backend_4096_pure_flash_flops_profile.json"
if [[ ! -f "$BASE_JSON" ]]; then
  python -m experiments.backend_realized_local_attention._common.profile_flops_ratio \
    --module pure \
    --config "$BASE_CFG" \
    --output "$BASE_JSON"
fi

for cfg in \
  experiments/backend_realized_local_attention/FlashAttention/aah_modified/configs/backend_4096_grouping_off_flash_seed0.yaml \
  experiments/backend_realized_local_attention/FlashAttention/aah_modified/configs/backend_4096_full_adaptive_flash_seed0.yaml \
  experiments/backend_realized_local_attention/FlashAttention/aah_modified/configs/backend_4096_shallow_freeze_flash_seed0.yaml \
  experiments/backend_realized_local_attention/FlashAttention/aah_modified/configs/backend_4096_deep_practical_reuse_flash_seed0.yaml
do
  python -m experiments.backend_realized_local_attention._common.run_train \
    --module aah \
    --config "$cfg"
  base="$(basename "$cfg" .yaml)"
  python -m experiments.backend_realized_local_attention._common.profile_flops_ratio \
    --module aah \
    --config "$cfg" \
    --baseline-json "$BASE_JSON" \
    --output "experiments/backend_realized_local_attention/FlashAttention/aah_modified/results/${base}_flops_profile.json"
done
