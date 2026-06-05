#!/usr/bin/env bash
set -euo pipefail

ROOT="paper_results/aah_flops_reduction_lab"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR" "$ROOT/gpu_flops_profiles"

NCU_BIN="${NCU_BIN:-/home/featurize/work/bin/ncu-sudo}"
MANIFEST="$ROOT/profile_manifest_p7_dense_framework.jsonl"
STATUS="$ROOT/profile_status_p7_dense_framework.jsonl"
LOG="$LOG_DIR/p7_dense_framework.log"

{
  echo "$(date -u +%FT%TZ) p7 dense framework queue starting"
  python -m experiments.aah_flops_reduction_lab.p7_dense_framework.scripts.make_dense_framework_configs
  python -m experiments.aah_flops_reduction_lab.p7_dense_framework.scripts.make_dense_profile_manifest \
    --ncu "$NCU_BIN" \
    --output "$MANIFEST"
  python -m experiments.aah_flops_reduction_lab._common.run_profile_manifest \
    --manifest "$MANIFEST" \
    --status "$STATUS" \
    --skip-existing \
    --cwd "."
  echo "$(date -u +%FT%TZ) p7 dense framework queue finished"
} 2>&1 | tee -a "$LOG"
