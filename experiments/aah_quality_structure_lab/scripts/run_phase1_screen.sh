#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

CONFIG_DIR="experiments/aah_quality_structure_lab/configs/phase1"
RESULT_DIR="experiments/aah_quality_structure_lab/results/phase1"
LOG_DIR="paper_results/aah_quality_structure_lab/logs"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

python experiments/aah_quality_structure_lab/scripts/make_quality_configs.py

while IFS= read -r cfg; do
  name="$(basename "$cfg" .yaml)"
  if python - "$name" "$RESULT_DIR" "$cfg" <<'PY'
import csv
import glob
import os
import sys
import yaml

name, result_dir, cfg_path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
target_step = int(cfg.get("train", {}).get("max_steps", 3000))
max_step = 0
for path in glob.glob(os.path.join(result_dir, f"{name}*.csv")):
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    max_step = max(max_step, int(float(row.get("step") or 0)))
                except ValueError:
                    pass
    except FileNotFoundError:
        pass
sys.exit(0 if max_step >= target_step else 1)
PY
  then
    echo "run_skip_complete ${name} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_DIR/phase1_screen.log"
    continue
  fi
  log="$LOG_DIR/${name}.log"
  echo "run_start ${name} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_DIR/phase1_screen.log"
  python scripts/train.py --config "$cfg" 2>&1 | tee "$log"
  echo "run_finish ${name} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_DIR/phase1_screen.log"
  rm -f "$RESULT_DIR/${name}"_step*.pt "$RESULT_DIR/${name}"_step*.pt.meta.json
done < <(python -c 'import yaml; m=yaml.safe_load(open("experiments/aah_quality_structure_lab/configs/phase1/phase1_manifest.yaml")); [print(r["config"]) for r in m["runs"]]')

python experiments/aah_quality_structure_lab/scripts/summarize_quality_runs.py
