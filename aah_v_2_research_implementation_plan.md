# AAH-v2 Research & Implementation Plan

This document defines the **locked research and engineering plan for AAH-v2**, the second-stage evolution of Asymmetric Attention Heads. AAH-v2 is explicitly motivated by the negative results of AAH-v1 and targets **true compute reduction through dynamic, in-computation control**, while preserving architectural clarity.

AAH-v2 is the **core scientific contribution** of the AAH project.

---

## 0. Motivation (Why AAH-v2 Exists)

AAH-v1 demonstrated a key result:

> **Static head asymmetry does not reliably reduce compute nor preserve quality.**

Empirical evidence shows:
- Heads that contribute little *still consume full compute*
- Static role assignment removes capacity before optimization can adapt
- FLOP savings must occur **before matmul**, not after

Therefore, AAH-v2 addresses the missing capability:

> **Dynamic, execution-time control over attention head resolution and participation.**

---

## 1. Core Research Question

> **Can attention heads dynamically reduce their own effective computation during training and inference, while preserving or improving model quality?**

This question is answered **only if**:
- Tensor sizes change *before* expensive operators
- Control signals are learnable or data-dependent
- Overhead is lower than saved compute

---

## 2. Scope Lock (Strict)

### Included
- Decoder-only Transformer
- Changes limited to **Attention module internals**
- Dynamic head-level or group-level control
- Tensor-shape or resolution changes *before matmul*
- Deterministic and reproducible execution paths

### Explicitly Excluded
- KV-cache tricks or post-hoc pruning
- Token dropping outside attention
- CUDA / kernel rewrites
- Model-wide routing or MoE
- External controller models (reserved for v3)

If a mechanism violates these, it is **not AAH-v2**.

### Compliance Checklist (Non-Negotiable)
- Control changes tensor sizes **before** QK^T
- No masking or zeroing **after** computation
- No KV-cache sharing or paging tricks
- No token dropping outside attention
- No kernel/CUDA rewrites
- No MoE or external controllers

---

## 3. High-Level Architecture

AAH-v2 extends AAH-v1 by inserting **control ports** inside the attention black box.

```
X
│
├─ QKV projection (unchanged)
│
├─ Head Control Module (NEW)
│   ├─ observes head-local statistics
│   ├─ outputs control signals per head / group
│
├─ Controlled Attention Computation
│   ├─ adaptive window size W_h
│   ├─ adaptive downsampling stride s_h
│   └─ optional head participation scaling
│
├─ Concatenate head outputs
│
└─ Output projection (unchanged)
```

---

## 4. Control Port Design (Key Innovation)

### Control Location (Non-Negotiable)

Control must occur:
- **Before** QK^T matmul
- By changing **tensor shapes or resolutions**

Not allowed:
- Zeroing outputs after computation
- Masking after attention weights are formed

---

## 5. Control Signal Definition (v2.1)

Initial AAH-v2 uses **low-bandwidth control signals**.

### Control Granularity
- Per-head or per-head-group
- Same control for all tokens in a forward pass

### Control Outputs (examples)
- `W_h ∈ {64, 128, 256, L}`
- `s_h ∈ {1, 2, 4}`

These are **discrete but differentiable** via:
- straight-through estimators, or
- soft-to-hard annealing

---

## 6. Control Module (Internal, Lightweight)

### Inputs (Readable Diagnostics)
- Attention entropy per head
- Mean / variance of attention scores
- Norm of head output
- (Optional) token position statistics

### Architecture Constraints
- ≤ 1% of base model parameters
- No recurrence
- No cross-layer state (v2)

This module is **inside** the Transformer and trained end-to-end.

---

## 7. Compute Model (What Actually Saves FLOPs)

For head `h` at layer `l`:

```
Cost_h ≈ L × (W_h + L / s_h)
```

AAH-v2 enables:
- Heads to shrink `W_h` when confident
- Heads to increase `s_h` when global detail is unnecessary

Savings are **input- and training-stage dependent**.

### 7.1 Execution-Time Compute Accounting (Required)
- Log per-head matmul shapes `(L_q, L_k)` for every layer
- Compute effective attention elements: `Σ_h (L_q × L_k)_h`
- Treat post-hoc masking as **no savings**

---

## 8. Training Protocol

### Phase 1 — Warm-up
- Disable control (AAH-v1 behavior)
- Allow representations to form

### Phase 2 — Control Activation
- Gradually enable control signals
- Apply regularization toward lower compute

### Phase 3 — Stabilization
- Freeze control policy or anneal to discrete

---

## 9. Metrics & Diagnostics (Expanded)

In addition to v1 metrics:
- Average effective context per head
- Compute saved per layer
- Control signal entropy
- Head collapse / dead-head detection

All metrics logged per layer.

---

## 10. Success Criteria

AAH-v2 is successful if **any** of the following hold:
- Same perplexity with ≥10–20% attention FLOP reduction
- Better perplexity at equal compute
- Emergent head specialization with measurable savings

---

## 11. Failure Modes (Explicit)

- Controller collapses to static policy → v1
- Control oscillation destabilizes training
- Overhead outweighs savings
- Heads game the metric (low compute, low usefulness)

These outcomes are valid research results.

---

## 12. Relationship to AAH-v3

- AAH-v2: *internal, lightweight, end-to-end control*
- AAH-v3: *external, side-loaded controller with global view*

AAH-v2 defines the **upper bound of what internal control can achieve**.

---

## 13. Deliverables

- [ ] Control port implementation
- [ ] Head-level readable diagnostics
- [ ] FLOP accounting per layer
- [ ] Ablation vs AAH-v1 and MHA
- [ ] Written analysis (positive or negative)

### 13.1 Phase Mapping (README Alignment)
- Phase 1–2: Control port + diagnostics
- Phase 3: Ablations + FLOP accounting
- Phase 4–5: Analysis + write-up

---

## 14. Branching Strategy

- `main`: stable baselines only
- `AAH-v1`: frozen, reference baseline
- `AAH-v2`: active research branch

`main` is updated **only when a version is complete and analyzed**.

---

## 15. Guiding Principle

> **If computation is not removed before matmul, it does not count.**

AAH-v2 lives or dies by this rule.

