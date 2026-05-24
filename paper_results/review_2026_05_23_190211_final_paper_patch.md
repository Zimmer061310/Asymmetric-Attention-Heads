# Final Paper Patch For Review 2026-05-23 19:02:11

The editable LaTeX/Prism source for draft 38 is not present in this repository.
Apply the following changes in Prism or the paper source. These fixes are
designed to avoid another oscillating review cycle: where Prism previously
asked for a visual/data change and then objected to the side effect, keep the
data as-is and make the caption/prose explicit.

## 1. Five-Row Two-Commit Provenance

Do not claim independently verified same-code provenance. Replace the
strongest shared-run wording with an intended-control statement.

Suggested replacement for the main protocol table row:

```text
Intended controlled fields: The local run configuration is intended to hold
tokenizer, data pipeline, optimizer schedule, batch construction, model
width/depth, seed, and Transformer output interface fixed across the five
paper-facing regimes. The public artifact bundle does not yet contain enough
per-run immutable metadata to independently verify every field; unresolved
fields are listed in the release-provenance appendix.
```

Suggested appendix addition:

```text
The current W&B-derived export records two git commits across the five main
rows: Full attention, Grouping off, and Full adaptive use commit
4b660cc7f3cc629deadce30b9d93382b2e5a0f7f, while Shallow freeze and Deep
practical reuse use commit aa92c473024dabd679a49bc51c2c0b3433abb441. The paper
treats these rows as the intended controlled comparison exported from
wandb_results_new/, but independent release requires a per-row manifest with
source run, checkpoint basename, checkpoint step, config basename/hash, git
commit, and artifact hash, or a rerun under one commit. Until that manifest is
complete, the comparison should be read as controlled by local experiment
intent rather than independently verified public provenance.
```

This is better than pretending the issue is solved.

## 2. Training-Dynamics Final X Coordinate

If Figure training_dynamics keeps the lower-panel final diagnostic row visually
aligned at x=250, do not say all plotted coordinates are literal logged values.

Suggested caption sentence:

```text
Validation points use their logged W&B coordinates. For the ACR and hierarchy
panels, the final diagnostic rows exported at W&B step 249 are visually aligned
to the 10000-step endpoint at x=250 so the final diagnostic values correspond
to the same selected final checkpoint as the validation curve.
```

Alternative if you want literal coordinates only:

```text
Validation logging ends at W&B step 250, while ACR and hierarchy diagnostics
end at W&B step 249 because they are emitted by a different logging path; the
table values are selected from the final 10000-step checkpoint row.
```

Use one convention. Do not combine "literal logged coordinates" with relabeled
lower-panel x=250.

## 3. Q/K/V Magnitude Feature

The implementation uses mean absolute activation. The paper should use
`mu(|q_h|)`, not `|bar(q_h)|`.

Replace:

```text
x_h = [|bar(q_h)|, sigma(q_h), |bar(k_h)|, sigma(k_h), |bar(v_h)|, sigma(v_h), e_h, n_h, rho_h]
```

with:

```text
x_h = [mu(|q_h|), sigma(q_h), mu(|k_h|), sigma(k_h), mu(|v_h|), sigma(v_h), e_h, n_h, rho_h]
```

Add:

```text
Here mu(|q_h|), mu(|k_h|), and mu(|v_h|) are mean absolute activations reduced
over batch, sequence, and head-feature dimensions while preserving the head
index. The implementation does not compute |mu(q_h)|, |mu(k_h)|, or
|mu(v_h)|. Group-level Q/K/V magnitude entries use the same mean-absolute
convention, with the reduction additionally averaging over the heads inside
the group.
```

Also change the formula-lock note to use `rho_h` for usage. Do not use `u_h`
there because `u_i^{(r)}` already denotes enriched controller input.

## 4. Shallow Freeze Parent Constraint

Add this sentence near the regime definitions or parent-constraint subsection:

```text
For the one-level Shallow freeze [2] topology, the parent-index clamp is
vacuous because the level-0 groups are already top-level items; nontrivial
parent-child clamping is exercised only in multi-level hierarchies such as
[2,2,2,2].
```

Where the Shallow freeze prose says "constrained choices", use:

```text
raw or vacuously constrained choices
```

or:

```text
the same decision pipeline, with the parent-index clamp vacuous in this
one-level topology
```

## 5. Qwen3 Downstream Table Caveat

Strengthen the Table 6 caption itself, because captions are what readers quote.

Suggested caption replacement:

```text
Internal pretrained compatibility smoke test on capped deterministic
Qwen3-4B-Base subsets at 4096-token context. These are not official full
benchmark scores, are not independently reproducible from the current public
artifact bundle until the manifest blockers are resolved, and should not be
cited as benchmark results or used for external model comparisons.
```

Suggested downstream paragraph addition:

```text
The exact percentages in this table are included to check whether the AAH
patch preserves the pretrained Qwen3 execution interface under a fixed local
protocol. They should not be cited as standalone benchmark results. Independent
reproduction requires the unresolved model-revision, subset-ID, prompt,
scoring, execution-harness, transfer-load, and adapter-hash fields listed in
the provenance manifest.
```

Suggested appendix replacement:

```text
For the Qwen3 compatibility check, the base model is Qwen/Qwen3-4B-Base at
Hugging Face revision 906bfd4b4dc7f14ee4320094d8b41684abff8539, with model-card
license apache-2.0 as checked on 2026-05-23. AAH variants patch the pretrained
Qwen3 attention modules, load only .aah_state.* adapter/controller/topology
keys, and leave the full-attention baseline on the original Qwen3 attention
path. The public manifest records the displayed subset counts and scoring
protocol. Exact adapter files or artifact URLs, adapter/controller/topology
hashes, dataset revisions, subset item IDs, server package versions, and
code-task execution-environment details remain required before the table can be
treated as independently reproducible.
```

## 6. Wide Joint Sibling Scoring Claim Strength

Keep wide joint sibling scoring as a design description, but add the caveat in
the abstract or contribution list so readers cannot read it as isolated causal
evidence.

Suggested abstract sentence after the wide-joint sentence:

```text
The present experiments evaluate the complete AAH-v3 execution package;
isolating the contribution of wide joint sibling scoring requires the planned
independent-scoring and simple-policy diagnostics.
```

Suggested contribution-list wording:

```text
Controller design. We introduce wide joint sibling scoring as a design intended
to contrast sibling groups directly; the reported experiments evaluate it as
part of the complete AAH-v3 execution package rather than isolating its causal
contribution.
```

## Final Recommendation

These six edits should be enough for an arXiv-ready claim boundary. Do not add
new experiments unless you want to resolve issue 1 fully. For arXiv, it is
acceptable to keep issue 1 as a release-provenance limitation if the language
above is used.
