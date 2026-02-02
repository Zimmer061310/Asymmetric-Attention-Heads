# AAH-v1 Small-Model Report

## Setup
* Date: 2026-02-02
* Dataset: Wikitext-2 (wikitext-2-raw-v1)
* Tokenizer: GPT-2
* Model: 8 layers, 8 heads, d_model 512, d_ff 2048, seq_len 512
* Steps: 2000

## AAH-v1 Configuration
* Head groups: **local + global**
* Local heads: 4
* Global heads: 4
* Local window (W): 128
* Global downsample stride (s): 4

## Results (AAH)
* Final train loss: **8.8780**
* Final validation loss: **7.7631**
* Final validation perplexity: **2352.14**
* Mean tokens/sec (from log points): **4927.50**

## Artifacts
* Model: `experiments/aah-small-wt2.pt`
* CSV log: `experiments/aah-small-wt2_aah.csv`
* W&B offline logs: `wandb/offline-run-20260202_170330-8k3y584l`

## Notes
* `mem_mb` is **0.0** because we only track CUDA memory; for MPS/CPU it will not report GPU memory.
