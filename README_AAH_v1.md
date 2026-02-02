# AAH-v1 Research & Implementation Plan

This document defines the **locked plan for AAH-v1**, the first concrete and reproducible implementation of **Asymmetric Attention Heads**. It is scoped intentionally narrow to serve as a clean baseline for future variants (AAH-v2, AAH-v3, …).

---

## 0. Purpose of AAH-v1

AAH-v1 answers one question only:

> **Can explicit head-level computational asymmetry reduce attention cost while preserving model quality?**

AAH-v1 is **not** intended to be optimal or adaptive. It is intended to be:

* minimal
* analyzable
* reproducible
* extensible

---

## 1. Scope Lock (Very Important)

### Included

* Decoder-only Transformer
* Modified **Attention module only**
* Two head groups:

  * Local (sliding window)
  * Global (downsampled K/V)
* Fixed, deterministic head partition

### Explicitly Excluded

* Dynamic routing or learned grouping
* Sparse / approximate attention
* KV cache modifications
* Kernel-level or CUDA optimizations
* Any changes outside the attention module

If a change violates this list, it is **not AAH-v1**.

---

## 2. Architecture Definition (Locked)

### Head Groups

| Group | Role   | Behavior                                               |
| ----- | ------ | ------------------------------------------------------ |
| A     | Local  | Sliding window attention (window = W)                  |
| B     | Global | Full-context attention on downsampled K/V (stride = s) |

Constraint:

```
H_local + H_global = H
```

---

## 3. Attention Computation Summary

### Local Heads

* Context: last `W` tokens
* Resolution: full
* Cost: `O(L × W)` per head

### Global Heads

* Context: full sequence
* Resolution: `L / s`
* Cost: `O(L × (L / s))` per head

Outputs from both groups are concatenated and projected identically to standard MHA.

---

## 4. Hyperparameters (To Be Swept)

| Symbol  | Meaning               | Typical Values |
| ------- | --------------------- | -------------- |
| H_local | number of local heads | {H/2, H/4}     |
| W       | local window size     | {64, 128, 256} |
| s       | downsampling stride   | {2, 4}         |

All other model hyperparameters are **fixed to baseline**.

---

## 5. Experimental Protocol

### Baseline

* Vanilla Multi-Head Attention (MHA)
* Same model size, same training setup

### Comparisons

* MHA vs AAH-v1 under equal parameter counts
* Sweep one AAH hyperparameter at a time

### Metrics

* Validation perplexity / loss
* Training throughput
* Inference latency
* Attention entropy per head group

---

## 6. Expected Outcomes

### Hypotheses

1. Many heads can be constrained locally without accuracy loss
2. A small number of global heads preserves long-range dependencies
3. Moderate downsampling (s = 2–4) is tolerable

### Failure Modes (To Observe)

* Long-context degradation when `H_global` too small
* Precision loss when `s` too large
* Over-reliance on local heads

---

## 7. Deliverables for AAH-v1

AAH-v1 is considered **complete** when:

* [ ] Attention module implemented and verified
* [ ] Baseline parity test passed (degenerates to MHA)
* [ ] Full hyperparameter sweep completed
* [ ] Metrics logged and analyzed
* [ ] Clear conclusions written (even if negative)

---

## 8. Relationship to Future Versions

* **AAH-v1**: fixed, static, two-group asymmetry (this document)
* **AAH-v2**: may introduce learned or adaptive grouping
* **AAH-v3**: may combine AAH with KV-cache strategies

All future versions must reference AAH-v1 as the baseline.
