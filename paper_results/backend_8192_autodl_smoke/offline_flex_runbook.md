# Offline 8192 FlexAttention Runbook

This runbook is for AutoDL or any server that cannot reliably reach Hugging Face
datasets during training.

## Prepare Token File

On a machine with Hugging Face access:

```bash
python scripts/prepare_tokenized_dataset.py \
  --output data/tokenized/wikitext2_gpt2.pt \
  --dataset wikitext \
  --dataset-name wikitext-2-raw-v1 \
  --tokenizer gpt2
```

The generated `.pt` file is ignored by Git. Copy it to the server:

```bash
scp -P <port> data/tokenized/wikitext2_gpt2.pt \
  data/tokenized/wikitext2_gpt2.pt.meta.json \
  root@<host>:/root/autodl-tmp/datasets/
```

## Short Real-Backend Suite

Use this first to verify real validation/training with the offline token file:

```bash
python scripts/run_paper_experiments.py \
  --suite backend_8192 \
  --config-dir /root/autodl-tmp/aah-runs/offline_8192_flex/configs \
  --log-dir /root/autodl-tmp/aah-runs/offline_8192_flex/logs \
  --summary-dir /root/autodl-tmp/aah-runs/offline_8192_flex/summaries \
  --diagnostics-dir /root/autodl-tmp/aah-runs/offline_8192_flex/diagnostics \
  --offline-token-file /root/autodl-tmp/datasets/wikitext2_gpt2.pt \
  --out-dir /root/autodl-tmp/aah-runs/offline_8192_flex/experiments \
  --train-max-steps 500 \
  --train-eval-interval 100 \
  --train-eval-batches 5 \
  --train-log-interval 20 \
  --disable-checkpoints \
  --only 'full_flex|fixed_1024_flex|fixed_2048_flex|fixed_4096_flex|full_adaptive_flex|deep_practical_reuse_flex' \
  --write-configs \
  --run train \
  --continue-on-error
```

Mandatory checks in logs/W&B:

- `aah/backend_name` should be `flex_attention`.
- `aah/backend_fallback_reasons` should be empty.
- `aah/backend_realized_ACR_est` should track the selected windows.
- `aah/attn_ratio` should be less than `1.0` for local/adaptive policies after warmup.

## Full Run

After the 500-step suite is clean, remove `--disable-checkpoints` and increase
`--train-max-steps`. Keep `dropout=0.0` for real FlexAttention backend runs
unless dropout support is implemented in the backend path.
