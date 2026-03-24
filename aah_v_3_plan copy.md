Latest Draft — AAH Paper (Working Draft for Prism / Warp Review)

Title

Asymmetric Attention Heads v3: Hierarchical Group-Level Control for Quality-Constrained Attention Compute Efficiency

⸻

Abstract

Standard multi-head attention (MHA) allocates attention computation uniformly across heads, even though heads often differ in specialization and likely execution utility. This work studies whether structured, unequal allocation can reduce attention computation while preserving model quality. We present AAH-v3, an execution-aware extension of MHA that applies hierarchical, group-level control to select discrete attention windows from execution statistics, while preserving the Transformer block interface and flat output merge. AAH-v3 changes the internal execution policy of attention rather than the output topology of the model. In the current implementation, AAH-v3 reduces effective attention computation as measured by Attention Compute Ratio (ACR) and aligned FLOPs-style proxies derived from executed attention widths, evaluated under explicit quality constraints. We therefore frame the main contribution as a quality-constrained efficiency frontier in compute-proxy space. Wall-clock throughput is treated as a secondary systems outcome, since runtime gains depend on masking behavior, kernel implementation, and hardware characteristics.

⸻

1. Introduction

Transformers rely on multi-head attention (MHA), where each head is typically executed with a uniform attention pattern and full attended span. This design is simple and robust, but it may over-allocate computation: not all heads require the same attended region to make useful contributions at every stage of execution. This motivates a central question: can attention computation be reduced through structured, unequal head allocation while maintaining model quality?

We investigate this question through the Asymmetric Attention Heads (AAH) line of methods, with AAH-v3 as the primary method in this paper. AAH-v3 introduces hierarchical group-level control over discrete attention windows while preserving the standard Transformer block interface and flat output merge. The method therefore targets the execution policy inside attention, not architectural changes to the block output structure.

Our evaluation scope is intentionally constrained. We treat effective attention computation as the primary optimization target, measured by ACR and aligned FLOPs-style proxies, and treat validation quality as the primary guardrail. Runtime metrics such as inference tokens per second are reported as secondary outcomes, because reductions in compute proxy do not guarantee proportional wall-clock gains across different masking implementations, kernels, and hardware platforms.

To keep the paper focused, AAH-v1 and AAH-v2 are included only as brief design-evolution context in the main text, with detailed formulations and additional ablations deferred to the appendix. The core method, claims, and experimental analysis center on AAH-v3.

⸻

2. Background: Standard Multi-Head Attention

Let X \in \mathbb{R}^{L \times d} denote the input sequence representation, where L is sequence length and d is model dimension. For each head h \in \{1,\dots,H\},

Q_h = XW_h^Q,\qquad K_h = XW_h^K,\qquad V_h = XW_h^V.

The head output is

O_h = \operatorname{softmax}\!\left(\frac{Q_hK_h^\top}{\sqrt{d_h}} + M\right)V_h,

where d_h is head dimension and M is the causal mask. The final multi-head attention output is

O = \operatorname{Concat}(O_1,\dots,O_H)W^O.

This formulation has two important properties. First, all heads operate in parallel on the same layer input. Second, all heads use the same full-width attention pattern over the sequence. In other words, standard MHA allocates uniform attention computation across heads.

If each head attends over the full causal context, the dominant attention-score computation scales proportionally to

\sum_{h=1}^{H} L^2 d_h,

ignoring constant factors and non-attention terms. This quadratic dependence on sequence length makes attention an attractive target for execution-aware compute reduction.

⸻

3. AAH Design Evolution

AAH was developed through a sequence of versions, each designed to isolate a different hypothesis about head-level asymmetry.

3.1 AAH-v1: Static asymmetric heads

AAH-v1 asked whether fixed asymmetric head roles were sufficient. Heads were partitioned statically into groups such as local-window heads and reduced-resolution heads. This version established a minimal asymmetry baseline, but results indicated that static asymmetry alone was not enough: quality degraded and runtime gains were unreliable.

3.2 AAH-v2: Dynamic resolution control

AAH-v2 introduced dynamic execution control before or during attention computation. Instead of assigning fixed head roles, attention width and effective key span could vary according to control signals. This shifted the project from static structural asymmetry to dynamic resolution control, but also introduced new stability and implementation issues.

3.3 AAH-v3: Hierarchical execution control

AAH-v3 is the main method of this paper. It adds hierarchical grouping and group-level control while preserving the standard Transformer output interface. Heads are grouped, control features are aggregated at the group level, and a controller selects discrete execution widths. Smoothing, ramping, and strict evaluation procedures are used to stabilize training and maintain reproducibility.

Because AAH-v3 represents the first stable and reproducible execution-aware design in this family, it is the primary method studied in the main text. Additional details of earlier versions are included only to motivate the design and are deferred to the appendix when necessary.

⸻

4. Method: AAH-v3

4.1 Overview

AAH-v3 augments standard multi-head attention with hierarchical group-level execution control. It does not alter the Transformer block layout, the existence of Q/K/V projections, or the flat concatenation-plus-projection output interface. Instead, it changes how attention is executed inside the attention module.

The core idea is simple: heads are partitioned into groups, group-level features are computed from execution statistics, and a controller selects a discrete attention width for each group. Heads in the same group inherit the same effective execution width.

4.2 Grouping map

Let the head set be partitioned into groups:

\mathcal H = \bigcup_{g=1}^{G}\mathcal H_g,
\qquad
\mathcal H_g \cap \mathcal H_{g'} = \varnothing \;\; \text{for } g\neq g'.

We define a grouping map

\pi:\{1,\dots,H\}\to\{1,\dots,G\},\qquad \pi(h)=g,

which assigns head h to group g.

In the current implementation, grouping is built hierarchically from head statistics and may be cached across control intervals. Thus, hierarchy is used for control aggregation, while execution remains organized by per-head window assignment.

4.3 Group features

For each head h, AAH-v3 computes a feature vector

\psi_h=
\left[
q_h^{\mathrm{mean}},\,
q_h^{\mathrm{std}},\,
k_h^{\mathrm{mean}},\,
k_h^{\mathrm{std}},\,
v_h^{\mathrm{mean}},\,
v_h^{\mathrm{std}},\,
e_h,\,
n_h,\,
u_h
\right],

where:
    •    the mean/std terms are computed from the current batch’s Q, K, and V tensors,
    •    e_h, n_h, and u_h denote the previous entropy, norm, and usage statistics.

Group features are formed by mean aggregation over heads in the group:

\phi_g = \operatorname{Mean}_{h\in\mathcal H_g}\psi_h.

This yields a 9-dimensional group-level control feature vector.

4.4 Group controller

For each group g, AAH-v3 computes a control signal from the group features:

c_g = f(\phi_g).

In the current implementation, f is a two-layer MLP with GELU nonlinearity that outputs logits over a discrete set of candidate attention widths:

z_g = W_2\,\sigma(W_1\phi_g+b_1)+b_2,

where \sigma is GELU and z_g \in \mathbb R^{|\mathcal W|} corresponds to the window set

\mathcal W = \{64,128,256,512\}.

The raw selected index is

\hat c_g = \arg\max z_g.

A hierarchical parent constraint is then enforced:

c_g = \min(\hat c_g,\; c_{\mathrm{parent}(g)}),

so that a child group cannot select a wider window than its parent.

4.5 Resolution map

The selected group index is propagated to heads through \pi(h), yielding per-head discrete window assignments. Optional temporal smoothing and post-warmup ramping are then applied.

Let i_h^{(t)} be the discrete window index assigned to head h at control step t. With optional EMA smoothing,

\tilde i_h^{(t)} = \alpha \tilde i_h^{(t-1)} + (1-\alpha) i_h^{(t)},

where \alpha is the resolution EMA coefficient.

After smoothing and post-warmup ramping, the selected head width is mapped to a discrete value

L_{k,h} = r(c_{\pi(h)}) \in \mathcal W.

For execution and logging, this width is clamped by a minimum GPU-efficiency width:

\tilde L_{k,h} = \operatorname{clamp}(L_{k,h}, L_{\min}, T),

where L_{\min}=W_{\min,\text{gpu}} and T is the sequence length.

4.6 Controlled attention execution

Given Q_h, K_h, V_h, AAH-v3 executes head h under its group-controlled effective width \tilde L_{k,h}. At the abstract level, this is written as

O_h = \operatorname{Attn}(Q_h, K_h, V_h;\tilde L_{k,h}),

where \tilde L_{k,h} determines the effective attended region through causal window masking.

In the current implementation, grouped execution is performed by window branch, and local attention is realized through window-aware causal masking. This detail matters for claim scope: AAH-v3 reduces the effective attended region and the corresponding compute proxy, but does not necessarily shrink kernel-level score-matrix formation in the current implementation.

4.7 Output interface

A crucial property of AAH-v3 is that the output interface remains unchanged. Head outputs are still merged by flat concatenation and output projection:

O = \operatorname{Concat}(O_1,\dots,O_H)W^O.

Thus, AAH-v3 changes execution policy inside attention, not the output topology of the Transformer block.

⸻

5. Measuring Effective Attention Computation

Because runtime improvements are implementation- and hardware-dependent, we separate effective attention computation from wall-clock speed.

We define the Attention Compute Ratio (ACR) as

\operatorname{ACR}
=
\frac{\sum_{l=1}^{N}\sum_{h=1}^{H} L_q^{(l,h)}L_k^{(l,h)}}{\sum_{l=1}^{N}\sum_{h=1}^{H}L^2},

where L_q^{(l,h)} and L_k^{(l,h)} are the effective query and key lengths used by head h in layer l.

In the decoder-only setting used here, L_q^{(l,h)}=L, so

\operatorname{ACR}
=
\frac{\sum_{l=1}^{N}\sum_{h=1}^{H}L\cdot L_k^{(l,h)}}{NH\,L^2}.

This metric corresponds to the logged attn_ratio in the implementation. It should be interpreted as an algorithmic compute proxy proportional to effective attention score size, not as a direct measurement of kernel-level FLOPs or runtime.

We also define a FLOPs-style proxy by combining effective attention elements with the non-attention cost terms kept fixed by model architecture. Let B denote batch size, L sequence length, d model dimension, d_h head dimension, n_{\mathrm{ff}} feed-forward dimension, and N_{\mathrm{layer}} the number of layers. Then

\mathrm{attn}_{\mathrm{full}} = N_{\mathrm{layer}} \cdot 4BL^2d,

\mathrm{attn}_{\mathrm{est}} = 4Bd_h \cdot \mathrm{attn\_elements}_{\mathrm{total}},

\mathrm{non\_attn}
=
N_{\mathrm{layer}}
\cdot
\left(8BLd^2 + 4BLd\,n_{\mathrm{ff}}\right),

\mathrm{flops}_{\mathrm{total,est}}
=
\mathrm{attn}_{\mathrm{est}} + \mathrm{non\_attn},

\mathrm{flops}_{\mathrm{ratio}}
=
\frac{\mathrm{flops}_{\mathrm{total,est}}}{\mathrm{flops}_{\mathrm{total,full}}}.

⸻

6. Implementation Differences Between Baseline and AAH-v3

This section isolates what changes in AAH-v3 and what remains identical to baseline MHA, in order to keep claims implementation-faithful.

6.1 Components that remain unchanged

AAH-v3 preserves the standard Transformer block contract. Specifically, it keeps:
    •    the Q/K/V linear projection structure for each attention layer,
    •    the scaled dot-product attention formulation with causal masking,
    •    the residual block layout (attention + MLP with layer normalization),
    •    the flat output merge O=\mathrm{Concat}(O_1,\dots,O_H)W^O.

Accordingly, AAH-v3 is not an architectural replacement of the Transformer block; it is an execution-policy augmentation inside the attention module.

6.2 Components introduced by AAH-v3

AAH-v3 introduces four execution-control components:
    1.    Hierarchical group-level control aggregation.
Heads are grouped hierarchically, and group-level features drive window selection.
    2.    Discrete window selection from execution statistics.
A lightweight controller predicts logits over a fixed window set \mathcal W, followed by discrete selection and hierarchical parent-constrained refinement.
    3.    Temporal stabilization of control decisions.
Optional smoothing and post-warmup ramping are applied to reduce unstable control transitions during training.
    4.    Execution diagnostics and compute-proxy instrumentation.
The implementation logs effective attention-width statistics and aligned compute proxies (ACR/attn_ratio, FLOPs-style proxies, and stability diagnostics).

6.3 Practical behavior and conservative interpretation

AAH-v3 implements hierarchical control aggregation, but practical head-to-group assignment is conservative in operation because grouping can be cached across control intervals. Thus, the strongest accurate claim is hierarchical control over execution resolution, rather than fully dynamic re-grouping at every control interval.

6.4 Control-off ablation interpretation

When control is disabled, AAH-v3 reverts to full-span causal attention behavior under the same block interface, with the remaining difference primarily being wrapper and instrumentation overhead. This ablation is therefore used to isolate the contribution of adaptive control, rather than to claim full architectural equivalence beyond attention execution behavior.

6.5 Systems truthfulness boundary

In the current implementation, local attention uses window-aware masking to control the effective attended region, while score tensors are still formed before masking. Therefore, AAH-v3 supports strong claims about effective attention compute proxy reduction, but not unconditional claims of kernel-level FLOPs reduction or universal runtime acceleration.

⸻

7. Experimental Protocol

7.1 Reporting objective

This section reports the quality-constrained attention compute efficiency of AAH-v3. Primary evidence is based on ACR / flops_ratio, val_ppl, and val_loss. Inference throughput (infer tok/s) is reported as a secondary systems reference. All reported inference results are obtained with strict checkpoint matching.

7.2 Fixed setup

All compared runs use fixed model scale, data pipeline, and training budget, with explicit reporting of:
    •    dataset and tokenizer,
    •    sequence length,
    •    training steps,
    •    precision and hardware context,
    •    config and checkpoint lineage.

Inference evaluation is performed with strict checkpoint matching to prevent checkpoint mismatch and result contamination.

7.3 Metrics and priority

Primary metrics:
    1.    ACR / flops_ratio
    2.    val_ppl
    3.    val_loss

Secondary metric:
4. infer tok/s

Efficiency is defined as compute-proxy reduction under quality constraints, not raw throughput improvement.

7.4 Quality gate

Relative to the canonical AAH-v3 reference, a candidate run is rejected if either condition holds:
    •    val_ppl degrades by more than 1.0%
    •    val_loss degrades by more than 0.5%

Only quality-passing runs proceed to frontier analysis.

7.5 Ranking and Pareto selection

Among quality-passing runs, a Pareto filter is applied in (\text{flops_ratio}, \text{val_ppl}, \text{val_loss}) space. Among non-dominated runs, preference is given to lower flops_ratio, then lower val_ppl, then lower val_loss, with infer tok/s used only as a tie-break.

A run is dominated if another run is no worse on all three primary metrics and strictly better on at least one.

7.6 Canonical replacement rule

A run replaces the canonical configuration only if:
    •    it passes the quality gate,
    •    it improves flops_ratio by at least 0.001 absolute,
    •    and it shows no meaningful quality degradation.

Otherwise, it may be retained as a frontier point but not promoted to canonical.

7.7 Noise-aware comparison thresholds

Changes below the following are treated as practically equivalent:
    •    val_ppl < 0.5
    •    val_loss < 0.005
    •    flops_ratio < 0.001
    •    infer tok/s < 3%

7.8 Required reporting set

Each experiment table should include:
    •    baseline MHA,
    •    control-off ablation,
    •    canonical AAH-v3,
    •    best quality-passing frontier point.

Each reported result should include:
    •    config identifier,
    •    val_ppl, val_loss, flops_ratio, infer tok/s,
    •    quality-gate pass/fail,
    •    Pareto membership,
    •    canonical candidacy decision.

⸻

8. Results Presentation Skeleton

8.1 Reporting objective

This section reports the quality-constrained attention compute efficiency of AAH-v3. Primary evidence is based on ACR / flops_ratio, val_ppl, and val_loss. Inference throughput (infer tok/s) is reported as a secondary systems reference. All reported inference results are obtained with strict checkpoint matching.

8.2 Main comparison table (Table 1)

Purpose: headline comparison across baseline, ablation, canonical, and best frontier candidate.

Rows (fixed):
    1.    Baseline MHA
    2.    AAH-v3 control-off ablation
    3.    AAH-v3 canonical
    4.    Best quality-passing frontier point

Columns (recommended):
    •    Method
    •    val_ppl
    •    val_loss
    •    ACR / flops_ratio
    •    infer tok/s
    •    Role
    •    Notes

8.3 Frontier / sweep table (Table 2)

Purpose: full decision trace over candidate runs.

Rows: all strict-checkpoint sweep runs included in analysis.

Columns:
    •    Config ID
    •    Key knobs
    •    val_ppl
    •    val_loss
    •    ACR / flops_ratio
    •    infer tok/s
    •    Quality-gate pass/fail
    •    Pareto-optimal
    •    Canonical-eligible
    •    Canonical decision

8.4 Figure plan

Figure 1 (primary): val_ppl vs ACR / flops_ratio
    •    shows baseline, control-off, canonical, and Pareto-optimal points
    •    defines the quality-constrained efficiency frontier

Figure 2 (secondary): infer tok/s vs val_ppl
    •    reported as a systems reference
    •    explicitly marked as secondary to compute-proxy objectives

Optional appendix figure: val_loss vs ACR / flops_ratio for robustness.

8.5 Caption policy

Each table and figure caption should:
    1.    state that ACR / flops_ratio are compute proxies,
    2.    state that runtime is implementation- and hardware-dependent,
    3.    state that results are strict-checkpoint only,
    4.    avoid implying proxy reduction equals kernel-level FLOPs reduction,
    5.    avoid implying that AAH-v3 changes Transformer output topology.

8.6 Results narrative template
    1.    setup reminder
    2.    quality gate outcome
    3.    Pareto frontier outcome
    4.    canonical decision
    5.    secondary runtime observation

8.7 Reproducibility block

Report once, consistently:
    •    commit hash
    •    experiment/run name
    •    canonical config path
    •    canonical checkpoint path
    •    strict-checkpoint infer command
    •    dataset / tokenizer / sequence length
    •    precision and hardware context

⸻

9. Current Canonical Result Pack (to be inserted / verified)

9.1 Final clean experiment table currently available

Row    val_ppl    val_loss    flops_ratio    infer tok/s    Notes
Baseline (10k)    46646.6801    10.750357    1.000000*    28508.67    strict-like infer row exists but appears inconsistent with baseline train metrics; likely historical mismatch artifact
Control-off (10k)    574.6579    6.353775    N/A    N/A    no strict-checkpoint infer row in exports; train-only metrics available
Canonical V3 (wmin64)    555.2797    6.319472    0.966826    8498.32    canonical config/checkpoint lineage
Best frontier point    555.2797    6.319472    0.966826    8498.32    same as canonical; sole non-dominated point under locked policy

* Baseline flops_ratio set to 1.0 by definition of full attention.

9.2 Canonical config block
    •    aah_v3_windows: [64, 128, 256, 512]
    •    aah_v3_control_interval: 400
    •    aah_v3_sim_threshold: 0.72
    •    aah_v3_super_threshold: 0.76
    •    aah_v3_max_depth: 4
    •    aah_v3_resolution_ema_alpha: 0.15
    •    aah_v3_post_warmup_ramp_steps: 2000
    •    aah_v3_W_min_gpu: 64
    •    aah_v3_warmup_steps: 2000
    •    aah_v3_resolution_collapse_min_frac: 0.98
    •    aah_v3_resolution_collapse_max_frac: 0.98
    •    precision: bf16
    •    dataset: wikitext-2-raw-v1
    •    seq_len: 512

Run name:
    •    aah-v3-full-1b-10k-v3-final-wmin64-wt2

9.3 Reproducibility items
    •    Commit hash: 905cc938c89b4af815e1b75895f3ba7580df8742
    •    Canonical config path: configs/aah_v3_full_1b_10000_v3_final.yaml
    •    Canonical checkpoint path: experiments/aah-v3-full-1b-10k-v3-final-wmin64-wt2.pt
    •    Frozen candidate path (if used): experiments/final/aah-v3-final-candidate.pt
    •    Strict infer command:

python3 scripts/infer.py \
  --config configs/aah_v3_full_1b_10000_v3_final.yaml \
  --checkpoint experiments/aah-v3-full-1b-10k-v3-final-wmin64-wt2.pt \
  --strict-checkpoint \
  --eval-batches 50


⸻

10. Discussion / Limitations (skeleton)

AAH-v3 should be interpreted as a compute-efficiency method whose strongest evidence lies in effective attention compute proxies under quality constraints. In the current implementation, grouped window control reduces the effective attended region but does not guarantee proportional kernel-level FLOPs reduction or runtime acceleration. Moreover, practical grouping behavior is conservative because cached grouping limits aggressive reassignment across control intervals. These limitations define clear next steps: more kernel-faithful execution paths, cleaner strict-checkpoint baselines, and stronger multi-scale evaluation.

⸻

11. Conclusion (skeleton)

We presented AAH-v3, a hierarchical group-level execution-control method for multi-head attention that preserves the Transformer block interface while reducing effective attention compute proxy under quality constraints. By framing efficiency in terms of ACR and FLOPs-style proxies, rather than assuming direct runtime gains, AAH-v3 provides a careful and reproducible design study of head-level compute allocation in Transformers. The current results support AAH-v3 as a quality-constrained compute-efficiency method, with runtime improvements remaining implementation- and hardware-dependent.

⸻

When you say start, the next messages can be written as relay messages to Warp, each beginning with:

ChatGPT >>> Warp

