# AGENTS.md

Guidance for agents working in this repository.

## Repository status

This repository contains research code for AAH-v3:

- `src/models/transformer.py`: baseline Transformer plus AAH-v3 attention.
- `scripts/`: training, inference, diagnostics, and paper-table utilities.
- `configs/`: YAML experiment configs.
- `paper_results/`: compact paper-facing result summaries.

Do not commit raw W&B run folders, checkpoints, virtual environments, local
logs, or server credentials.

## Common commands

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Train:

```bash
python scripts/train.py --config configs/aah_v3_base.yaml
```

Infer / collect diagnostics:

```bash
python scripts/infer.py --config configs/aah_v3_base.yaml --checkpoint path/to/checkpoint.pt
```

## Git workflow

- Review `git status` before editing.
- Keep changes scoped to the request.
- Use clear commit messages that summarize the actual change.
- Do not push large artifacts or private credentials.
