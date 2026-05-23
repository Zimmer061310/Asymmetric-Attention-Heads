# Reproducibility Scope

This repository is a compact research-code release, not a full artifact dump.
It contains implementation code, experiment configs, paper-facing result
summaries, and selected diagnostics. It intentionally excludes large model
weights, raw W&B directories, local logs, and downloaded datasets.

## Included Evidence

- The AAH-v3 implementation and baseline Transformer code.
- Training and inference scripts used by the custom 1B AAH-v3 experiments.
- Qwen3-4B compatibility scripts used for capped downstream checks.
- Compact CSV, JSON, Markdown, and LaTeX result summaries under
  `paper_results/`.
- Heatmap-style diagnostics for the Qwen3-4B compatibility runs.

## Not Included

- Full checkpoints and adapter weights.
- Downloaded Qwen or dataset caches.
- Raw server logs and W&B run folders.
- Private review material or temporary paper drafts.

## Interpreting The Results

The custom 1B / 4096-token suite is the main mechanism and efficiency evidence
for AAH-v3. The Qwen3-4B table is a capped-subset compatibility check designed
to show that the Hugging Face patch preserves a pretrained model interface and
can run standard downstream evaluations. It should not be described as an
official full benchmark report.

The AAH-v3 Q/K/V magnitude feature is mean absolute activation, `mu(abs(q_h))`,
not absolute value after scalar reduction, `abs(mu(q_h))`. See
`paper_results/aah_v3_feature_definition.md` for the formula lock. The Qwen3
downstream transfer/load and benchmark provenance status is recorded in
`paper_results/qwen3_4b_aah/downstream_provenance_manifest.md`.

When publishing an artifact bundle, include:

- exact Git commit;
- exact config file;
- checkpoint or adapter SHA-256 hash;
- base model name and revision;
- dataset/task revision where available;
- hardware type;
- precision, batch size, context length, and benchmark sample cap.
