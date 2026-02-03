# Experiments

Place experiment logs, notes, and results summaries here.

3‑experiment comparison (small model, WT‑2, 2000 steps, 2026‑02‑02)

•  Baseline (MHA): val loss 7.7234, val ppl 2260.69, train loss 8.8502, throughput 5211.46 tok/s.  
•  AAH‑v1 (local_heads=4, W=128, s=4): val loss 7.7631, val ppl 2352.14, train loss 8.8780, throughput 4927.50 tok/s.  
•  AAH‑v1 (H_local=2, W=256, s=2): val loss 7.7735, val ppl 2376.72, train loss 8.8521, throughput 5234.83 tok/s.

Deltas vs baseline (from comparison file):
•  AAH‑v1 (W=128, s=4): +4.05% val ppl (worse), –5.45% throughput (slower).
•  AAH‑v1 (W=256, s=2): +5.13% val ppl (worse), +0.45% throughput (slightly faster).