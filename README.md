# AI Research Plan

## Goals

* Do AI research during vacation using Python in this repository.
* Track both repo/code progress and research progress.
* Produce a thesis with Prism based on our experiments and research.

## Roles

* ChatGPT (teacher): research guidance, advice, teaching.
* Warp (terminal runner): run code and manage repo files.
* Prism (thesis writer): write thesis from experiment and research summaries.
* This passage will sync to all three AI's prompt, AIs must update their Idea, Plan, Summary in this passage. current idea on the "Idea" paragraph, and the current works summarize on the "Summary" paragraph, and the current plan update on the "Plan" paragraph.
* The repo is [https://github.com/Zimmer061310/Ena](https://github.com/Zimmer061310/Ena)

## Basic Plan

1. Find the topic
2. Do the research
3. Write the thesis

## Hardware

* MacBook Pro (M1 Pro)
* PC with Radeon 7900XT GPU

## Environment Snapshot (2026-01-31)

### Conda

* base: /opt/homebrew/Caskroom/miniconda/base
* torch: /opt/homebrew/Caskroom/miniconda/base/envs/torch

### Python venvs

* ~/venvs/base (Python 3.12.10)
* ~/venvs/torch (Python 3.12.10)

### Key installed packages (venv base)

* torch 2.10.0, transformers 5.0.0, datasets 4.5.0, accelerate 1.12.0, tokenizers 0.22.2
* numpy 2.4.1, scipy 1.17.0, pandas 3.0.0, matplotlib 3.10.8
* wandb 0.24.1, rich 14.3.1, loguru 0.7.3

### Key installed packages (venv torch)

* torch 2.10.0, torchvision 0.25.0, numpy 2.4.1, sympy 1.14.0, pillow 12.1.0

## Plan

### Phase 0 — Baseline and Scope Lock

* Implement or reuse a standard decoder-only Transformer with vanilla multi-head attention (MHA).
* Fix model scale (e.g. 100M–300M parameters) and datasets for all experiments.
* Establish baseline metrics: perplexity, decode latency, attention head statistics.

### Phase 1 — Asymmetric Attention Heads (AAH) Design

* Define head partitions (e.g. short-range vs long-range heads).
* Design at least two asymmetry mechanisms:

  * **Range asymmetry:** different attention masks / context lengths per head group.
  * **Resolution asymmetry:** downsampled keys/values for selected head groups.
* Ensure output interface matches standard attention.

### Phase 2 — Implementation

* Modify the attention module to support heterogeneous head configurations.
* Keep all non-attention components unchanged.
* Support toggling AAH on/off for controlled ablation.

### Phase 3 — Experiments

* Sweep the proportion of asymmetric heads.
* Measure:

  * Accuracy / perplexity
  * Training and inference cost
  * Contribution and entropy per head group

### Phase 4 — Analysis and Interpretation

* Analyze which head types dominate information flow.
* Identify failure modes (e.g. long-context degradation).
* Compare against vanilla MHA under equal compute budgets.

### Phase 5 — Write-up

* Formalize AAH as an attention generalization.
* Present empirical trade-offs and mechanistic insights.

## Idea

### Idea 1 — Head-Grouped KV Cache (HG-KV)

**Core question:** How much attention head independence is actually necessary during autoregressive inference?

**Description:**
Standard multi-head attention maintains a full, independent KV cache per head, leading to high memory usage and bandwidth pressure during decoding. This idea proposes **grouping attention heads** so that multiple heads share the same K/V representations, forming a *continuous design space* between:

* **MHA** (H groups, fully independent KV)
* **GQA** (intermediate number of groups)
* **MQA** (1 group, fully shared KV)

By sweeping the number of KV groups, we explicitly study the trade-off between expressiveness and efficiency.

**What changes in the Transformer chain:**

* Replace per-head K/V projection with **per-group K/V projection**
* Heads map deterministically to KV groups during inference
* KV cache is stored per group instead of per head

**Why this is interesting:**

* Reduces KV cache size and memory bandwidth
* Provides a unifying framework for MHA / GQA / MQA
* Enables controlled ablation on head redundancy

**Planned analysis:**

* Decode latency and memory usage vs group count
* Accuracy / perplexity degradation curves
* Attention entropy and inter-head similarity analysis

**Scope:**

* Decoder-only Transformer
* Autoregressive inference
* Small–medium scale models for reproducibility

---

### Idea 2 — Asymmetric Attention Heads (AAH)

**Core question:** Do all attention heads need the same attention resolution and computation pattern?

**Description:**
Standard Transformers enforce *homogeneous attention heads*: each head uses the same sequence length, attention computation, and update frequency. This idea proposes **asymmetric attention heads**, where different heads are explicitly assigned different computational roles.

Typical head roles include:

* Short-range, high-resolution heads
* Long-range, low-resolution heads
* Optional global or summary heads

The external attention interface remains unchanged, while internal heads operate under heterogeneous constraints.

**What changes in the Transformer chain:**

* Attention heads are partitioned into functional classes
* Each class applies a different attention mask, context length, or resolution
* Outputs are concatenated and projected identically to standard MHA

**Why this is interesting:**

* Makes head redundancy explicit and controllable
* Redistributes compute inside the attention module
* Pure structural modification (no KV cache, no sparse indexing)

---

### Originality and Related-Work Boundary (Important)

To ensure the research topic is **clearly original**, we explicitly separate this work from existing literature:

**What existing work studies:**

* *Asymmetric attention formulations* (e.g. query–key asymmetry, theoretical expressivity)
* *Sparse / approximate attention* (e.g. asymmetric indexing, LSH, retrieval-style attention)
* *KV cache optimizations* (e.g. MQA, GQA, paging, quantization)

**What this work does NOT do:**

* Does not modify the attention formula itself
* Does not approximate attention via sparsity or external indexing
* Does not reuse existing asymmetric attention theory

**What this work uniquely studies:**

* Head-level **computational asymmetry** inside multi-head attention
* Attention heads with different context ranges and resolutions
* Controlled ablation of head heterogeneity under fixed model size

**Open gap addressed by this work:**

> How attention computation should be *unevenly allocated across heads* to maximize efficiency without sacrificing model quality.

This head-centric, compute-allocation perspective is not directly addressed in prior work.

## Summary

### Current Status

* Research direction **locked** on **Asymmetric Attention Heads (AAH)** as the primary topic.
* AAH is defined as a **head-level structural modification** of multi-head attention, not a change to the attention formula, KV cache, or sparse indexing.
* The core hypothesis is that **attention heads do not need uniform computation**, and that uneven allocation of context range and resolution can preserve model quality while reducing cost.

### Structural Progress

* Established a clear baseline: **standard homogeneous Multi-Head Attention (MHA)**.
* Defined the conceptual transition from homogeneous heads to **explicit head specialization**.
* Formalized AAH as partitioning heads into functional groups (short-range / long-range / global), while keeping the external Transformer interface unchanged.

### Originality Check

* Compared AAH against:

  * Asymmetric attention *theory* (query–key asymmetry)
  * Sparse / approximate attention (indexing, LSH, retrieval-based)
  * KV-cache optimizations (MQA, GQA)
* Confirmed AAH occupies a **distinct, underexplored design axis**: head-level computational asymmetry.

### Next Steps

* Implement **AAH-v1** with range asymmetry as the minimal, safe starting point.
* Run controlled ablations against vanilla MHA under equal parameter budgets.
* Measure accuracy, compute cost, and head-level contribution statistics.

This establishes a clean, original foundation for experimentation and thesis development.
