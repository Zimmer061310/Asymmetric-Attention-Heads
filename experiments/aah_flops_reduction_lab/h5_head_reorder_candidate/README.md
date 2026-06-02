# H5: Head Reorder Candidate

Do not start this implementation until H3 shows a material no-scatter FLOPs
improvement.

Goal: convert a useful no-scatter prototype into a clean implementation with
layer-specific head permutations, adjusted output projection semantics, and
checkpoint/config metadata.

Acceptance gate:

- round-trip permutation test passes;
- logits match the no-scatter prototype within numerical tolerance;
- Nsight ratio improves over scatter-based AAH;
- no changes are promoted into `src/models/transformer.py` until the lab result
  is positive.
