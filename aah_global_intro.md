# Asymmetric Attention Heads (AAH) — Global Introduction

## 1. Motivation

Multi-Head Attention (MHA) assumes:

- All heads are equally important
- All heads always compute full-resolution attention
- Outputs are concatenated in flat parallel form

However:

- Head redundancy exists
- Compute is expensive
- Specialization emerges
- Attention execution is not adaptive

AAH explores structured attention execution under a fixed Transformer architecture.

---

## 2. AAH-v1 — Static Asymmetric Attention

### Question
Can static head asymmetry reduce compute while maintaining quality?

### Design
- Local heads (windowed attention)
- Global heads (downsampled attention)
- Fixed assignment

### Result
- Quality degradation
- Runtime gains unreliable

Conclusion:
> Static asymmetry alone is insufficient.

---

## 3. AAH-v2 — Dynamic Resolution Control

### Question
Can attention resolution be dynamically adjusted per head?

### Design
- Control port per head/group
- Resolution changed before matmul
- Attention formula unchanged

### Goal
Reduce real FLOPs by changing tensor shapes.

---

## 4. AAH-v3 — Hierarchical Execution Control

### Question
Can head groups coordinate execution hierarchically?

### Design
- Tree grouping
- Group-level control
- Centralized or distributed resolution decisions

Hierarchy exists only for execution management.

Output remains flat.

---

## 5. AAH-v3.5 — Power Distance Control

### Question
How does authority distribution affect attention execution?

### Concept
Introduce centralization coefficient:

\[
\tilde{c}_i = (1 - \alpha)c_i + \alpha C
\]

- α = 0 → fully distributed
- α = 1 → fully centralized

This models different hierarchical authority strengths.

Goal:
Understand the trade-off between autonomy and global coordination.

---

## 6. AAH-v4 — Hierarchical Output Architecture

### Question
Should attention outputs remain flat?

### Design Shift
Replace flat concatenation with hierarchical composition:

\[
H \rightarrow G \rightarrow Root \rightarrow Output
\]

This changes representation topology.

Focus:
- Structural inductive bias
- Semantic routing
- Optional selective branch execution

---

## 7. Evolution Summary

| Version | Focus | Transformer Changed? | Compute Reduction? |
|----------|--------|----------------------|--------------------|
| V1 | Static asymmetry | No | Weak |
| V2 | Dynamic resolution | No | Yes |
| V3 | Hierarchical control | No | Yes |
| V3.5 | Authority modeling | No | Yes |
| V4 | Hierarchical output | Yes | Optional |

---

## 8. Positioning

AAH is a research program studying:

> Execution-aware and structurally-aware attention design under Transformer constraints.

The trajectory moves from:

Efficiency → Control Theory → Organizational Structure → Architectural Redesign.