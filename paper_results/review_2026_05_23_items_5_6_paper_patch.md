# Paper Patch For Review Items 5 And 6

The editable paper source is not present in this repository. Apply the following
text changes to the Prism/LaTeX source for draft 37.

## Item 5: Q/K/V Magnitude Feature

Replace any formula using absolute scalar means, such as:

```text
x_h = [|bar(q_h)|, sigma(q_h), |bar(k_h)|, sigma(k_h), |bar(v_h)|, sigma(v_h), e_h, n_h, rho_h]
```

with:

```text
x_h = [mu(|q_h|), sigma(q_h), mu(|k_h|), sigma(k_h), mu(|v_h|), sigma(v_h), e_h, n_h, rho_h]
```

Add the following sentence immediately after the formula:

```text
Here mu(|q_h|), mu(|k_h|), and mu(|v_h|) are mean absolute activations reduced over batch, sequence, and head-feature dimensions while preserving the head index; the implementation does not compute |mu(q_h)|, |mu(k_h)|, or |mu(v_h)|.
```

For group features, use:

```text
Group-level Q/K/V magnitude entries use the same mean-absolute convention, with
the reduction additionally averaging over the heads inside the group.
```

## Item 6: Qwen3 Downstream Reproducibility

In the downstream benchmark subsection, keep the table but make the label
unambiguously non-official and non-reproducible from the public repo alone:

```text
The downstream table is an internal capped-subset compatibility smoke test, not
an official full benchmark report. The public repository records the transfer
and loading procedure, displayed sample counts, scoring protocol, and recorded
adapter/run hashes in downstream_provenance_manifest.md, but exact adapter
artifacts, dataset revisions, subset item IDs, server package versions, and
code-task execution-environment details remain outside the public artifact
bundle. Therefore these scores should be cited only as a compatibility check,
not as independently reproducible benchmark evidence.
```

In the appendix provenance paragraph, replace the generic unresolved sentence
with:

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
