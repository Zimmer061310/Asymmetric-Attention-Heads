#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-experiments/backend_realized_local_attention/FlashAttention/aah_modified/configs/backend_4096_full_adaptive_flash_seed0.yaml}"
CHECKPOINT="${CHECKPOINT:-}"
ROOT="${ROOT:-paper_results/aah_flops_reduction_lab}"
OUT="${OUT:-$ROOT/plans/flopslab-4096-base-aah-plan-flash-seed0.json}"

cmd=(
  python -m experiments.aah_flops_reduction_lab._common.export_static_plan
  --config "$CONFIG"
  --output "$OUT"
  --calibration-batches "${CALIBRATION_BATCHES:-16}"
  --start-step "${START_STEP:-10000}"
  --step-stride "${STEP_STRIDE:-5}"
)

if [[ -n "$CHECKPOINT" ]]; then
  cmd+=(--checkpoint "$CHECKPOINT")
fi

"${cmd[@]}"

python -m experiments.aah_flops_reduction_lab._common.copy_plan_to_required_variants \
  --source "$OUT" \
  --root "$ROOT"
