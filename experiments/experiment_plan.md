# AAH-v1 Small-Model Experiment Plan (v0)

## Goal
Compare **baseline MHA** vs **AAH-v1** on a small model to measure efficiency and quality deltas.

## Model + Data (Locked for v0)
* Model: decoder-only Transformer (8L, 8H, d_model 512, d_ff 2048)
* Seq length: 512
* Tokenizer: GPT-2
* Dataset: Wikitext-2 (train/validation)

## Variants
1) **Baseline MHA**
2) **AAH-v1** (Local + Global heads, fixed partition)

## Metrics to Record (per run)
* Validation perplexity
* Train tokens/sec
* Peak GPU memory (MB)
* Runtime per 1k steps (sec)

## Runs (Minimal)
* Baseline: 2,000 steps
* AAH-v1: 2,000 steps

## Success Criteria (Pilot)
* AAH-v1 shows **lower cost** (tokens/sec ↑ or memory ↓)
* Perplexity degradation is **small** (<= +3–5% vs baseline)

## Notes
* Keep **all hyperparameters identical** between variants.
* Only modify the attention module for AAH-v1.
