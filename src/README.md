# Source

- `data.py` contains dataset/tokenization helpers used by local training.
- `models/transformer.py` contains the baseline Transformer and AAH-v3
  implementation, including hierarchy construction, controller decisions,
  per-head window propagation, grouped local causal execution, and diagnostics.

The public paper-facing method is AAH-v3. Earlier internal variants are kept in
configs and scripts only where they help explain diagnostic history.
