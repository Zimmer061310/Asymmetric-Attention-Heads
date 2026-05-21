# Scripts

Main entry points:

- `train.py`: train the custom decoder-only Transformer from a YAML config.
- `infer.py`: run checkpoint evaluation and AAH diagnostics.
- `run_paper_experiments.py`: launch the custom AAH-v3 paper experiment suite.
- `run_qwen3_aah_paper.py`: run the Qwen3-4B compatibility workflow.
- `qwen3_aah_patch.py`: patch Hugging Face Qwen attention modules with AAH-style execution logic.
- `benchmark_paper_tasks.py`: aggregate capped benchmark-task outputs into paper tables.

Older sweep scripts are retained for traceability. Check each script's `--help`
or top-level constants before launching a costly run.
