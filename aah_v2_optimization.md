# AAH-v2 — Optimization Guidance & Guardrail Policy (for Warp)

This note summarizes **what optimization space is still legitimate for AAH-v2**, and how we should **handle Entropy / Norm guardrails going forward**, based on completed v2 experiments and profiling.

This is **not a new design plan** (no v3/v4 ideas here). Everything below stays strictly within **AAH-v2 scope**.

---

## 1. Current v2 Status (Baseline)

Empirical results show:

- AAH-v2 achieves **true pre-matmul shape control**
- Window control is **correct and memory-safe**
- Full v2 (window + stride + grouping + guardrails) is:
  - ~25% slower than baseline
  - Slightly worse PPL
  - Same memory footprint (CPU RSS)

Conclusion:
> v2 is **functionally correct**, but **overhead dominates savings** in its current form.

Therefore, remaining work is **overhead minimization and stabilization**, not new features.

---

## 2. Remaining Optimization Space (Legitimate for v2)

The following optimizations are **allowed, valid, and aligned with v2 goals**:

### 2.1 Reduce Control-Path Overhead (High Priority)

Still legitimate:
- Fewer controller invocations
- Lower-frequency control updates (rate limiting)
- Cache control decisions for multiple steps

Examples:
- Update (W_h, s_h) every *N* steps instead of every step
- Freeze control after stabilization phase

This directly targets the measured **step-time inflation**.

---

### 2.2 Prefer Window-Only Control (Empirically Justified)

Based on results:
- Window-only control:
  - Trains stably
  - Has predictable behavior
- Stride control:
  - Information-destructive
  - Adds overhead without clear gains

Action:
- Keep stride **optional or ablated**
- Treat window control as the **core v2 mechanism**

This is consistent with v2 scope.

---

### 2.3 Simplify Grouping (Static or Per-Head)

Dynamic grouping:
- Works functionally
- Adds control noise and overhead
- No quality or speed win observed

Legitimate v2 optimization:
- Static grouping **or**
- Pure per-head control

Do **not** add hierarchy here (that is v3).

---

### 2.4 Eliminate Kernel-Shape Thrashing

Ensure:
- Control choices come from a **small discrete set**
- No per-step shape churn beyond W_h choices

This helps MPS / backend caching behavior.

---

## 3. Guardrail Policy (Important)

### 3.1 Keep Guardrails — But Only During Control Learning

Entropy / Norm guardrails are **not useless**, but they are **not runtime features**.

They should be treated as:
> **training stabilizers**, not permanent mechanisms.

**Policy:**
- Warm-up phase: ❌ guardrails OFF
- Control activation phase: ✅ guardrails ON
- Stabilization / late training: ❌ guardrails OFF
- Inference: ❌ guardrails OFF

---

### 3.2 Convert Hard Guardrails → Soft Penalties

Do **not** enforce guardrails via:
- clamps
- forced overrides
- branching logic inside attention

Instead, use **loss-level penalties only**.

Example:
```python
loss += λ_entropy * relu(H_min − H_h)
loss += λ_norm * relu(N_min − ||O_h||)
```
