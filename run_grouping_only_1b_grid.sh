#!/usr/bin/env bash
ROOT=/home/featurize/work/ENA-AAH-v3
CONFIGS=(
  configs/aah_v3_grouping_only_1b_mgs1_d1.yaml
  configs/aah_v3_grouping_only_1b_mgs1_d2.yaml
  configs/aah_v3_grouping_only_1b_mgs1_d4.yaml
  configs/aah_v3_grouping_only_1b_mgs2_d1.yaml
  configs/aah_v3_grouping_only_1b_mgs2_d2.yaml
  configs/aah_v3_grouping_only_1b_mgs2_d4.yaml
  configs/aah_v3_grouping_only_1b_mgs4_d1.yaml
  configs/aah_v3_grouping_only_1b_mgs4_d2.yaml
  configs/aah_v3_grouping_only_1b_mgs4_d4.yaml
)
mkdir -p "$ROOT/logs"
for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  ts=$(date +%Y%m%d_%H%M%S)
  log="$ROOT/logs/${name}_${ts}.log"
  echo "[$(date -Is)] START $cfg log=$log"
  python "$ROOT/scripts/train.py" --config "$ROOT/$cfg" > "$log" 2>&1
  code=$?
  echo "[$(date -Is)] END $cfg code=$code log=$log"
  if [ "$code" -ne 0 ]; then
    echo "[$(date -Is)] STOP queue after failure in $cfg"
    exit "$code"
  fi
  sleep 5
done
