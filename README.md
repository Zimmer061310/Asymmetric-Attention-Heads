# ENA-AAH-v3

Research code and paper-result summaries for **Asymmetric Attention Heads
(AAH-v3)**, an execution-aware extension of multi-head attention. AAH-v3 keeps
the standard Transformer block interface while using a separate control path to
assign different local causal attention windows to different heads or head
groups.

The repository is currently intended as a private research artifact until the
paper is posted publicly. It does not include large model checkpoints, raw W&B
run directories, or local virtual environments.

## What Is Included

- `src/models/transformer.py` contains the decoder-only Transformer baseline
  and the AAH-v3 attention/controller implementation.
- `scripts/train.py` and `scripts/infer.py` are the main local training and
  inference/diagnostic entry points for the custom Transformer runs.
- `scripts/qwen3_aah_patch.py` and `scripts/qwen3_aah_paper.py` contain the
  Qwen3-4B compatibility utilities used for the capped downstream checks.
- `configs/` contains experiment configuration files, including paper-facing
  AAH-v3 regimes and earlier diagnostic variants.
- `paper_results/` contains compact paper-facing result summaries, tables, and
  diagnostic CSVs that are small enough to version.

## What Is Not Included

- `.pt` checkpoints and adapter weights.
- Raw W&B run directories.
- Local logs, scratch outputs, and Python virtual environments.
- Datasets or downloaded Hugging Face model weights.

For release-quality reproduction, store large artifacts in an external artifact
store and record immutable hashes or model revisions. The paper appendix lists
remaining provenance fields that should be filled before claiming independent
reproducibility.

See `REPRODUCIBILITY.md` for the precise scope of the released artifacts and
`PUBLIC_RELEASE.md` for the safe public-release checklist.

## Setup

Use Python 3.10+ with PyTorch. A minimal local setup is:

```bash
python -m pip install -r requirements.txt
```

For Featurize-style remote runs, keep code and important model artifacts under
`/home/featurize/work`, and use `/home/featurize/data` only for fast scratch
storage.

## Typical Commands

Train from a YAML config:

```bash
python scripts/train.py --config configs/aah_v3_base.yaml
```

Run inference and collect diagnostics:

```bash
python scripts/infer.py --config configs/aah_v3_base.yaml --checkpoint path/to/checkpoint.pt
```

Run the Qwen3-4B AAH paper workflow:

```bash
python scripts/run_qwen3_aah_paper.py --benchmark-profile fast_paper
```

Exact flags may differ by config and checkpoint layout; inspect the target
script's `--help` output before launching expensive runs.

## Paper Result Files

Key compact result files are:

- `paper_results/aah_v3_4096_table1_training.csv`
- `paper_results/aah_v3_4096_table2_inference.csv`
- `paper_results/wandb_results_new/`
- `paper_results/qwen3_4b_aah/benchmarks/benchmark_paper_table.md`
- `paper_results/qwen3_4b_aah/benchmarks/benchmark_paper_table.tex`

The Qwen3 benchmark table is a capped-subset compatibility check, not an
official full benchmark report. The custom 1B/4096 suite is the main
mechanism/efficiency evidence for ACR and hierarchy diagnostics.

## License

This repository is released under the Apache License 2.0. See `LICENSE`.

Machine-readable citation metadata is provided in `CITATION.cff`. Update it
with the final arXiv identifier before the public release.

## Repository Release Checklist

Before making the repository public:

1. Add the final arXiv citation and link.
2. Confirm the paper's unresolved provenance fields are either completed or
   clearly marked as limitations.
3. Keep checkpoints out of Git; publish large artifacts through an artifact
   store with SHA-256 hashes.
4. Verify no private tokens, server passwords, raw W&B credentials, or local
   machine paths are committed.

## Citation

The arXiv record is not public yet. Use this placeholder until the final record
exists:

```bibtex
@misc{zhao2026aahv3,
  title  = {Asymmetric Attention Heads: Hierarchical Group-Level Control for Quality-Constrained Attention Compute Efficiency},
  author = {Zimu Zhao},
  year   = {2026},
  note   = {arXiv preprint, forthcoming}
}
```
