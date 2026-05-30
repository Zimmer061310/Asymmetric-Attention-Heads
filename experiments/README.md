# Experiments

This directory is intentionally kept light in the public repo.

Raw experiment logs, local CSV dumps, checkpoints, and W&B run folders are not
versioned here. Compact paper-facing summaries live in `paper_results/`.

Use this directory only for small, intentionally curated experiment notes. Keep
large artifacts in an external artifact store and record their hashes in the
paper or release notes.

`backend_realized_local_attention/` contains the curated protocol scaffold for
AAH with real local-attention backends. It is split by backend and by pure
backend baseline versus AAH-modified execution so the main `src/` code path can
remain separate from backend-specific experiments.
