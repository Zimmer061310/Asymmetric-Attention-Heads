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
  log="$LOG_DIR/${name}.log"
  echo "run_start ${name} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_DIR/phase1_screen.log"
  python scripts/train.py --config "$cfg" 2>&1 | tee "$log"
  echo "run_finish ${name} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_DIR/phase1_screen.log"
  rm -f "$RESULT_DIR/${name}"_step*.pt "$RESULT_DIR/${name}"_step*.pt.meta.json
done < <(python -c 'import yaml; m=yaml.safe_load(open("experiments/aah_quality_structure_lab/configs/phase1/phase1_manifest.yaml")); [print(r["config"]) for r in m["runs"]]')

python experiments/aah_quality_structure_lab/scripts/summarize_quality_runs.py
