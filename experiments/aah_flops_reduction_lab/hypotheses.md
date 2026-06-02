# AAH FLOPs Reduction Lab Hypotheses

## H1: Static compiled AAH plan

Question: is dynamic controller/hierarchy work the reason Flash AAH measured
around `1.6x` pure FlashAttention?

Method: export per-layer/per-head window plans from an AAH checkpoint, then
profile a compiled execution mode that skips hierarchy construction, controller
scoring, EMA updates, and dynamic grouping during the profiled forward pass.

Success: Nsight ratio drops sharply; strong success is `<1.0`.

Failure: if the ratio stays near `1.5-1.6`, controller overhead is not the main
cause.

## H2: Single-window / two-bucket quantized execution

Question: is multi-bucket fragmentation the reason for the FLOPs increase?

Method: collapse AAH-selected windows into one or two FlashAttention buckets,
for example `1024` or `[1024, 4096]`.

Success: ratio falls sharply versus current Flash AAH.

Failure: if fixed or quantized buckets remain near `1.6`, launch count and
bucket count are not the only problem.

## H3: No-scatter contiguous prototype

Question: is gather/scatter/head reassembly the major cost?

Method: use a controlled head order where heads in the same execution bucket
are contiguous, then compare against a matched scatter-based control.

Success: no-scatter ratio is materially lower than the scatter control.

Failure: if no-scatter barely changes the result, full permanent head reordering
is unlikely to solve the problem alone.

## H4: Fixed plan granularity

Question: how much adaptivity can be frozen while preserving the AAH idea?

Method: compare per-layer, per-state, per-head-group, per-head, and slow-update
plans.

Success: identify the coarsest plan with acceptable validation loss and the
lowest true FLOPs ratio.

Failure: if quality requires highly dynamic per-forward routing, Flash/Flex
bucketing is probably not enough; a custom fused kernel would be required.

## H5: Head reordering candidate

Question: if no-scatter helps, can a clean implementation preserve model
semantics while removing arbitrary head scatter?

Method: derive layer-specific head permutations, reorder Q/K/V head layout by
bucket, and adjust the output projection/permutation metadata.

Gate: do not start H5 until H3 shows a meaningful Nsight improvement.
