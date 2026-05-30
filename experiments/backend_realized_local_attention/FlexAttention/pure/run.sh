#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"

python -m experiments.backend_realized_local_attention._common.run_train \
  --module pure \
  --config experiments/backend_realized_local_attention/FlexAttention/pure/configs/backend_4096_pure_flex_seed0.yaml

python -m experiments.backend_realized_local_attention._common.profile_flops_ratio \
  --module pure \
  --config experiments/backend_realized_local_attention/FlexAttention/pure/configs/backend_4096_pure_flex_seed0.yaml \
  --output experiments/backend_realized_local_attention/FlexAttention/pure/results/backend_4096_pure_flex_flops_profile.json
