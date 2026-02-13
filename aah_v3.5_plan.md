# AAH-v3.5 — Hierarchical Control with Adjustable Authority (Power Distance)

## 0. Version Identity

AAH-v3.5 extends AAH-v3.

- Transformer architecture remains unchanged.
- Attention formula remains unchanged.
- Output remains flat (standard head concatenation).
- Hierarchy is used only for execution control.

New addition:
> A continuous authority parameter controlling the strength of hierarchical coordination.

---

## 1. Core Research Question

> How does authority distribution within hierarchical head control affect compute reduction, stability, and model quality?

AAH-v3 introduced hierarchical grouping.
AAH-v3.5 introduces **authority strength modulation**.

---

## 2. Motivation

Observations from V3:

- Group-level coordination improves compute targeting.
- Fully centralized control may over-prune.
- Fully distributed control may under-prune.
- Stability varies across control patterns.

Key hypothesis:

> There exists an optimal balance between local autonomy and global authority.

---

## 3. Conceptual Model

Each head group has:

- Local control signal: \( c_i \)
- Global controller signal: \( C \)

Introduce authority coefficient:

\[
\alpha \in [0, 1]
\]

Define effective control:

\[
\tilde{c}_i = (1 - \alpha)c_i + \alpha C
\]

Interpretation:

- \( \alpha = 0 \): Fully decentralized (head autonomy)
- \( \alpha = 1 \): Fully centralized (top-down control)
- Intermediate α values: Mixed authority

---

## 4. What Changes from V3

V3:
- Hierarchical control structure
- Implicit authority distribution

V3.5:
- Explicit, tunable authority parameter
- Continuous spectrum instead of discrete modes
- Allows empirical mapping of control regimes

No change to:
- Attention math
- QKV computation
- Output topology
- Transformer block layout

---

## 5. Control Integration

Authority blending occurs:

- Before attention matmul
- During resolution/window decision
- Not after attention computation

This ensures:
- Real tensor shape change
- Real FLOPs reduction
- Not post-hoc masking

---

## 6. Experimental Plan

### 6.1 Authority Sweep

Run experiments across:

\[
\alpha \in \{0.0, 0.25, 0.5, 0.75, 1.0\}
\]

Measure:

- Validation perplexity
- Attention FLOPs
- Wall-clock time
- Stability (loss oscillation, gradient norms)

---

### 6.2 Compute–Quality Tradeoff Curve

Plot:

- Effective compute vs validation quality
- Authority coefficient vs stability

Goal:
Identify the sweet spot between over-centralization and over-autonomy.

---

### 6.3 Stability Diagnostics

Measure:

- Resolution variance per head
- Group disagreement statistics
- Pruning frequency
- Collapse modes

Hypothesis:
Mid-range authority produces smoother execution dynamics.

---

## 7. Expected Outcomes

Possible regimes:

1. Low α:
   - High autonomy
   - Weak compute reduction
   - High redundancy

2. High α:
   - Strong pruning
   - Risk of over-pruning
   - Potential instability

3. Medium α:
- Balanced compute reduction
- Stable training
- Best efficiency-quality trade-off

---

## 8. Success Criteria

AAH-v3.5 is successful if:

- A non-trivial α produces better compute–quality tradeoff than V3.
- Authority parameter explains observed execution behavior.
- Results are reproducible across model sizes.

---

## 9. Positioning

AAH-v3.5 reframes attention execution as:

> A controllable hierarchical coordination system.

It does not modify representation structure.

It studies:

- Execution dynamics
- Authority distribution
- Structured compute allocation

This keeps the research within execution-aware Transformer design.