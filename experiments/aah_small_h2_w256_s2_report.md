# AAH-v1 Small-Model Report (H_local=H/4, W=256, s=2)

## Setup
* Date: 2026-02-02
* Dataset: Wikitext-2 (wikitext-2-raw-v1)
* Tokenizer: GPT-2
* Model: 8 layers, 8 heads, d_model 512, d_ff 2048, seq_len 512
* Steps: 2000

## AAH Configuration
* H = 8
* H_local = 2 (H/4)
* H_global = 6 (3H/4)
* Local window W = 256
* Global downsample stride s = 2

## Results
* Final train loss: **8.8521**
* Final validation loss: **7.7735**
* Final validation perplexity: **2376.72**
* Mean tokens/sec (from log points): **5234.83**

## Artifacts
* Model: `experiments/aah-small-wt2-h2w256s2.pt`
* CSV log: `experiments/aah-small-wt2-h2w256s2_aah_h2w256s2.csv`
* W&B offline logs: `wandb/offline-run-20260202_175008-js0yh31e`

## Notes
* Memory tracking is CUDA-only in the script; on MPS it will report 0.0 MB.
