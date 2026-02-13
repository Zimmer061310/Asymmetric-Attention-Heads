# AAH-v4 — Hierarchical Output Attention Architecture

## 0. Version Identity

AAH-v4 is the first version that **modifies the attention output topology itself**.

Unlike previous versions:
- V1–V3.5 preserved flat head concatenation.
- V4 introduces **hierarchical output composition**.

This version moves AAH from execution-aware design to structural redesign.

---

## 1. Core Research Question

> Can hierarchical head composition improve representational efficiency and selective computation beyond flat Multi-Head Attention?

---

## 2. Motivation

Observations from prior versions:

- Heads specialize (syntax, locality, induction, entity linking, etc.).
- Many heads are partially redundant.
- Some downstream representations may not require all semantic components.
- Flat concatenation assumes equal and parallel importance.

V4 hypothesis:

> Tree-structured head composition better reflects semantic hierarchy and may allow selective routing.

---

## 3. Architectural Change

### Standard MHA Output

\[
O = W_o [H_1 \| H_2 \| \dots \| H_n]
\]

Flat, parallel, linear mixing.

---

### AAH-v4 Output

Heads are grouped hierarchically:

        Root
      /      \
   Group A   Group B
   /    \     /    \
 H1     H2   H3    H4

Composition rule:

1. Leaf level: compute head outputs \(H_i\)
2. Group level:
   \[
   G_j = W_{g_j}[H_{j1} \| H_{j2}]
   \]
3. Root level:
   \[
   O = W_{root}[G_1 \| G_2]
   \]

This replaces flat mixing with hierarchical mixing.

---

## 4. Design Variants

### Variant A — Static Hierarchy
- Fixed grouping
- No routing
- All heads computed
- Pure structural inductive bias

### Variant B — Selective Branch Activation
- Some branches skipped
- Conditional execution
- Enables compute reduction

### Variant C — Learned Routing
- Controller selects active branches
- Dynamic tree execution

---

## 5. Compute Implications

Without routing:
- FLOPs unchanged
- Architectural bias only

With routing:
- Entire branches skipped
- Real FLOPs reduction possible

---

## 6. Research Goals

Evaluate:

1. Does hierarchical composition improve quality?
2. Does it improve training stability?
3. Does selective routing enable efficient inference?
4. Does specialization become more explicit?

---

## 7. Risks

- Added depth increases optimization difficulty
- May degrade performance
- May not outperform flat mixing
- Routing instability

---

## 8. Experimental Plan

### Phase 1 — Static Hierarchy
- Replace flat projection with tree projection
- Keep compute unchanged
- Compare perplexity and training curves

### Phase 2 — Controlled Routing
- Add binary branch gating
- Measure compute vs quality tradeoff

### Phase 3 — Hybrid with V3 Control
- Combine hierarchical output + dynamic resolution control

---

## 9. Success Criteria

V4 is successful if:

- Quality ≥ baseline
- Hierarchical specialization measurable
- Selective execution reduces compute without major degradation