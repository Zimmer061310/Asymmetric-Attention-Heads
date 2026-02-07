# Baseline Model Snapshot

This folder contains a **vanilla GPT baseline** (no AAH-v2 or AAH-v3 logic).

Contents:
- `transformer.py`: baseline-only model (CausalSelfAttention).
- `baseline.yaml`: training config with `aah_v2_enabled: false` and `aah_v3_enabled: false` (AAH off).
