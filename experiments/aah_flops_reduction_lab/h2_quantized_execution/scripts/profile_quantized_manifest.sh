#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-paper_results/aah_flops_reduction_lab}"
MANIFEST="${MANIFEST:-$ROOT/profile_manifest.jsonl}"

python -m experiments.aah_flops_reduction_lab._common.make_profile_manifest \
  --output "$MANIFEST" \
  --ncu "${NCU:-/usr/local/cuda/bin/ncu}" \
  --profile-timeout "${PROFILE_TIMEOUT:-7200}"

echo "Review $MANIFEST before launching remote profiles."
