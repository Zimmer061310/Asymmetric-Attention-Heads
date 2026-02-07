# AAH‑v3 Plan — Hierarchical Resolution Control

## 0. Purpose

AAH‑v3 is motivated by a clear empirical conclusion from AAH‑v2:

> **Flat, per‑head, per‑step dynamic control introduces excessive runtime overhead and instability, preventing real speedups.**

AAH‑v3 replaces flat dynamic control with **hierarchical, aggregated resolution control** that is:
- slower‑moving
- more stable
- cheaper to execute
- resistant to oscillation and kernel‑shape thrashing

---

## 1. Core Hypothesis

> **Attention resolution can be controlled hierarchically and infrequently using aggregated statistics, reducing effective compute while preserving model quality and runtime stability.**

---

## 2. Scope Lock (Strict)

### Included
- Decoder‑only Transformer
- Modifications limited to the **Attention module**
- **Window size (W)** as the primary control variable
- Hierarchical aggregation of control signals
- Deterministic inference behavior

### Explicitly Excluded
- Per‑head dynamic grouping
- Per‑step discrete routing
- Stride control (disabled by default)
- KV‑cache tricks or post‑hoc pruning
- Kernel rewrites / CUDA hacks
- External or side‑loaded controllers (reserved for v4)

If a mechanism violates these constraints, it is **not AAH‑v3**.

---

## 3. High-Level Architecture (Adaptive Multi-Level)

```
Heads (fixed identity)
│
├─ Head statistics
│   ├─ attention entropy
│   ├─ output norm
│   └─ usage metrics
│
├─ Adaptive grouping (Level 1)
│   └─ clusters heads by similarity (variable size)
│
├─ Super‑grouping (Levels 2..N)
│   └─ clusters groups into supergroups (variable depth)
│
└─ Global group (top)
    └─ a single root group
```

**Design rule:**
- Heads **report** statistics
- Grouping **emerges** from similarity (no fixed group size)
- Controllers **constrain** resolution
- Heads **do not decide** their own resolution

---

## 4. Resolution Control Mechanism (Hierarchical, Adaptive)

### Controller Outputs
- Allowed resolution bounds per group / supergroup:

```
W ∈ {64, 128, 256}
max_W = 128
```

### Head Behavior
- Each head computes attention using:

```
W_h ≤ max_W
```

- No per‑head discrete switching
- No per‑token routing
- Group membership updates only on control intervals

## 4.5 Control Policy Contract (Locked)

### Policy
- **Primary policy:** argmax over controller logits.
- **Fallback policy:** **deterministic, non‑learning safeguard** (specified but **not implemented yet**).
- No probabilistic sampling in v3.

### Fallback Rule (Contract Only)
- If **entropy < e_min** AND **norm < n_min**, shrink to the next smaller window.
- Fallback is **diagnostic‑triggered only** and **must not** introduce stochasticity.

### Diagnostics‑Only Metrics (No Gradient / No Control Effect)
- **Group lifespan:** EMA of how long groups stay unchanged.
- **Group overlap:** Jaccard overlap with previous grouping.
- **Head reassignment count:** number / rate of heads switching groups.

These metrics are **observational only** and **do not** influence control decisions.

---

## 5. Control Update Schedule (Non‑Negotiable)

Controllers update:
- every **K steps** (e.g. 50–200), or
- at explicit **training phase boundaries**

Warmup:
- For the first **W warmup steps**, all heads use **full window** (attn_ratio = 1.0).
- After warmup, hierarchical control begins at the next control interval.

Controllers must **never** update:
- every forward pass
- every token

This prevents:
- kernel shape thrashing
- throughput oscillation
- controller‑induced noise

---

## 6. Controller + Grouping Constraints

- Lightweight MLPs or linear layers
- Aggregation only (mean / EMA / pooled statistics)
- Similarity‑based clustering with **variable** group sizes
- No recurrence
- No cross‑layer feedback loops
- Parameter budget ≤ **0.5%** of base model

---

## 7. Guardrails (Reinterpreted)

In AAH‑v3, guardrails are **diagnostic only**.

Used for:
- logging entropy collapse
- logging norm collapse
- validating controller decisions
- logging group stability (how often heads change groups)

Not used for:
- runtime enforcement
- forcing resolution changes
- hard constraints during attention computation

---

## 8. Compute Expectations

AAH‑v3 does **not** promise immediate wall‑clock speedups.

Primary targets:
- Reduced average attention window
- Reduced attention element count
- Stable kernel shapes across steps
- Lower variance in step time

---

## 9. Evaluation Metrics

### Primary
- Eval PPL vs baseline
- Average effective window per layer
- Attention element ratio vs baseline

### Secondary
- Throughput stability (variance, not peak)
- Memory usage
- Controller decision entropy

---

## 10. Success Criteria

AAH‑v3 is considered successful if **any** of the following hold:
- ≤1% PPL degradation with ≥20–30% fewer attention elements
- More stable throughput than AAH‑v2
- Emergent depth‑wise resolution hierarchy (early layers small W, later layers larger W)

---

## 11. Known Risks

- Controller collapses to always‑max window
- Control adapts too slowly
- No measurable compute reduction

All outcomes are valid research results.

---

## 12. Version Positioning

- AAH‑v1: static asymmetry (negative result)
- AAH‑v2: flat dynamic control (quality parity, speed failure)
- **AAH‑v3: hierarchical resolution control (current target)**
- AAH‑v4: external / side‑loaded controller (future work)

---

## 13. Status

**Design locked. Ready for implementation.**

---

## 14. Immediate Next Steps (Overhead + Baselines)

We are **not scaling up yet**. The next steps focus on **overhead removal** and **clean baselines**, not new mechanisms.

1) **Make AAH‑v3 control‑off path baseline‑like**
   - No grouping, no controller, no extra tensors
   - Target ≤5% slowdown vs baseline

2) **Increase `control_interval` aggressively (200–500) + cache decisions**
   - No per‑step recomputation

3) **Add explicit profiling**
   - time spent in control / grouping
   - time spent outside attention matmul

4) **After control‑off is clean, prepare one medium‑scale config (100M–300M)**
   - baseline vs v3 full
   - short run is enough

**Goal:** find the break‑even scale where AAH wins.

