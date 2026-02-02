# AAH-v1 Baseline Small-Model Report

## Setup
* Date: 2026-02-02
* Dataset: Wikitext-2 (wikitext-2-raw-v1)
* Tokenizer: GPT-2
* Model: 8 layers, 8 heads, d_model 512, d_ff 2048, seq_len 512
* Steps: 2000

## Results (Baseline)
* Final train loss: **8.8502**
* Final validation loss: **7.7234**
* Final validation perplexity: **2260.69**

## Artifacts
* Model: `experiments/baseline-small-wt2.pt`
* CSV log: `experiments/baseline-small-wt2_baseline.csv`
* W&B offline logs: `wandb/offline-run-20260202_160357-yvvenq5e`
