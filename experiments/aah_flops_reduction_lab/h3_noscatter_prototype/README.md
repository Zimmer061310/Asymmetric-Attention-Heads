# H3: No-Scatter Contiguous Prototype

Goal: test whether arbitrary head gather/scatter and head reassembly dominate
the current AAH FLOPs increase.

Prototype rule:

- Heads in the same execution bucket must be contiguous.
- The first prototype may ignore checkpoint compatibility.
- Always run a matched scatter-control row.
- The lab-only implementation may reorder heads before attention and skip the
  inverse scatter. Treat this as a FLOPs-overhead probe only; it is not a
  quality-valid model path until a real output-projection permutation is added.

Expected positive signal: contiguous no-scatter execution is materially cheaper
than the matched scatter control.

Failure interpretation: if no-scatter barely improves Nsight ratio, permanent
head reordering is unlikely to solve the issue alone.
