# Common Lab Utilities

This folder contains code shared by the FLOPs reduction hypotheses.

The utilities are allowed to import existing backend-realized local-attention
modules, but they must not modify the paper baseline implementation in `src/`.

Current utilities:

- `naming.py`: naming constants and variant records.
- `make_lab_configs.py`: writes the lab config matrix from the Flash backend
  templates.
- `export_static_plan.py`: exports per-layer window/group statistics from an
  AAH checkpoint.
- `copy_plan_to_required_variants.py`: copies one calibration export to every
  plan path referenced by plan-requiring lab configs.
- `make_profile_manifest.py`: writes a JSONL command manifest for remote Nsight
  profiling.
