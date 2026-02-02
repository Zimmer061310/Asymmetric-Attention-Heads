# Baseline vs AAH-v1 (Small Model) — Quick Comparison

## Runs
* Baseline report: `experiments/baseline_small_report.md`
* AAH report (v1): `experiments/aah_small_report.md`
* AAH report (H_local=H/4, W=256, s=2): `experiments/aah_small_h2_w256_s2_report.md`

## Final Metrics
### Baseline (MHA)
* Train loss: 8.8502
* Val loss: 7.7234
* Val ppl: 2260.69
* Mean tokens/sec: 5211.46

### AAH-v1 (local_heads=4, W=128, s=4)
* Train loss: 8.8780
* Val loss: 7.7631
* Val ppl: 2352.14
* Mean tokens/sec: 4927.50

### AAH-v1 (H_local=2, W=256, s=2)
* Train loss: 8.8521
* Val loss: 7.7735
* Val ppl: 2376.72
* Mean tokens/sec: 5234.83

## Deltas vs Baseline
### AAH-v1 (local_heads=4, W=128, s=4)
* Val ppl: **+4.05%** (worse)
* Throughput (tokens/sec): **-5.45%** (slower)

### AAH-v1 (H_local=2, W=256, s=2)
* Val ppl: **+5.13%** (worse)
* Throughput (tokens/sec): **+0.45%** (faster)

## Interpretation (Pilot)
* Variant (H_local=2, W=256, s=2) recovered **throughput** (slightly faster than baseline) but still shows **worse perplexity** than baseline.
* Next iteration options (keep fixed model/training):
  * Increase global heads further (H_local=1) OR increase local window (W=384)
  * Keep s=2 but reduce global head count? (trade-off check)
  * Add a small sweep (W ∈ {128,256,384}, s ∈ {2,4}, H_local ∈ {1,2,4})

## Notes
* Memory numbers are not comparable here because CUDA memory tracking reports 0.0 on MPS/CPU.
* H_Global has not developed yet.