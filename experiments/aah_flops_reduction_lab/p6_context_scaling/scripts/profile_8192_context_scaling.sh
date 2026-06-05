#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-paper_results/aah_flops_reduction_lab}"
NCU="${NCU:-ncu}"
TIMEOUT="${PROFILE_TIMEOUT:-21600}"
OUT="$ROOT/gpu_flops_profiles"
STATUS="$ROOT/profile_status_p6_context_scaling_8192.jsonl"

PURE_CONFIG="experiments/aah_flops_reduction_lab/p6_context_scaling/configs/flopslab-8192-baseline-pure-flash-seed0.yaml"
P3_CONFIG="experiments/aah_flops_reduction_lab/p6_context_scaling/configs/flopslab-8192-minruntime-noscatter-1024-8192-flash-seed0.yaml"
H5_CONFIG="experiments/aah_flops_reduction_lab/p6_context_scaling/configs/flopslab-8192-headreorder-lowerbound-1024-8192-flash-seed0.yaml"

PURE_TOTAL="$OUT/flopslab-8192-baseline-pure-flash-seed0_gpu_flops_profile.json"
PURE_ATTN="$OUT/flopslab-8192-baseline-pure-flash-seed0_attention_gpu_flops_profile.json"
P3_TOTAL="$OUT/flopslab-8192-minruntime-noscatter-1024-8192-flash-seed0_gpu_flops_profile.json"
P3_ATTN="$OUT/flopslab-8192-minruntime-noscatter-1024-8192-flash-seed0_attention_gpu_flops_profile.json"
H5_TOTAL="$OUT/flopslab-8192-headreorder-lowerbound-1024-8192-flash-seed0_gpu_flops_profile.json"
H5_ATTN="$OUT/flopslab-8192-headreorder-lowerbound-1024-8192-flash-seed0_attention_gpu_flops_profile.json"

mkdir -p "$OUT" "$(dirname "$STATUS")"

log_status() {
  python - "$STATUS" "$1" "$2" <<'PY'
import json, sys, time
path, event, name = sys.argv[1:4]
with open(path, "a") as f:
    f.write(json.dumps({"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event, "name": name}) + "\n")
PY
}

run_profile() {
  local name="$1"
  local module="$2"
  local config="$3"
  local output="$4"
  shift 4
  log_status start "$name"
  python -m experiments.backend_realized_local_attention._common.profile_gpu_flops_ncu \
    --module "$module" \
    --config "$config" \
    --ncu "$NCU" \
    --warmup 1 \
    --repeats 1 \
    --timeout "$TIMEOUT" \
    --output "$output" \
    "$@"
  log_status finish "$name"
}

run_profile "pure-total-8192" pure "$PURE_CONFIG" "$PURE_TOTAL"
run_profile "pure-attention-8192" pure "$PURE_CONFIG" "$PURE_ATTN" --profile-scope attention
run_profile "p3-total-8192" aah "$P3_CONFIG" "$P3_TOTAL" --baseline-json "$PURE_TOTAL"
run_profile "p3-attention-8192" aah "$P3_CONFIG" "$P3_ATTN" --profile-scope attention --baseline-json "$PURE_ATTN"
run_profile "h5-total-8192" aah "$H5_CONFIG" "$H5_TOTAL" --baseline-json "$PURE_TOTAL"
run_profile "h5-attention-8192" aah "$H5_CONFIG" "$H5_ATTN" --profile-scope attention --baseline-json "$PURE_ATTN"

log_status queue_finish "p6-context-scaling-8192"
