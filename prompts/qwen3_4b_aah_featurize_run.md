# Qwen3-4B AAH Featurize Run Prompt

Use this after renting a 96GB Featurize instance with the repo mirror attached.

## Goal

Run the pretrained Qwen3-4B-Base AAH experiment package for the paper:

- `qwen3_4b_full_attention_baseline`
- `qwen3_4b_grouping_off`
- `qwen3_4b_full_adaptive`
- `qwen3_4b_shallow_freeze`
- `qwen3_4b_deep_practical_reuse`

The current custom 1B experiments remain the controlled AAH mechanism table. This run is for the pretrained downstream benchmark table.

## Storage Rules

- Repo and important artifacts: `/home/featurize/work`
- Scratch only: `/home/featurize/data`
- Persistent output root:
  `/home/featurize/work/ENA-AAH-v3-persistent/AAH-qwen3-4b-paper`
- Active scratch root:
  `/home/featurize/data/AAH-qwen3-4b-paper`

## Command

```bash
cd /home/featurize/work/ENA-AAH-v3-benchmark
git pull
mkdir -p /home/featurize/data/AAH-qwen3-4b-paper
tmux new-session -d -s qwen3_4b_aah_paper \
  "python3 scripts/run_qwen3_aah_paper.py \
    --resume \
    --continue-on-benchmark-error \
    --adapt-steps 1000 \
    --batch-size 1 \
    --seq-len 4096 \
    --precision bf16 \
    2>&1 | tee /home/featurize/data/AAH-qwen3-4b-paper/driver.log"
```

If the 96GB GPU has room, increase `--batch-size` first, then `--adapt-steps`.

## Monitoring

Check:

```bash
tmux ls
nvidia-smi
tail -n 120 /home/featurize/data/AAH-qwen3-4b-paper/master.log
tail -n 120 /home/featurize/data/AAH-qwen3-4b-paper/driver.log
find /home/featurize/work/ENA-AAH-v3-persistent/AAH-qwen3-4b-paper -maxdepth 2 -type f | sort | tail -n 80
```

## Completion Criteria

- Adapter files exist for all four AAH regimes under `adapters/`.
- Smoke summaries exist for all five regimes under `summaries/`.
- Heatmap CSVs exist under `diagnostics/`.
- Benchmark files exist under `benchmarks/`, including:
  - `benchmark_results_by_task.csv`
  - `benchmark_results_by_model.csv`
  - `benchmark_paper_table.md`
  - `benchmark_paper_table.tex`

After completion, copy benchmark summary files into the local repo, commit and push, then release the instance with:

```bash
featurize instance release
```
