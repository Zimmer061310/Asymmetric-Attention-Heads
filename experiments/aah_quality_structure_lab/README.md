# AAH Quality / Structure Lab

This lab replaces the FLOPs-reduction track with quality and structure
controls. ACR/EAR and analytic FLOPs fields are routing diagnostics here, not
paper-facing compute claims.

## Phase 1

Run the 12 seed-0, 3000-step screening rows:

```bash
python experiments/aah_quality_structure_lab/scripts/make_quality_configs.py
bash experiments/aah_quality_structure_lab/scripts/run_phase1_screen.sh
python experiments/aah_quality_structure_lab/scripts/summarize_quality_runs.py
```

Main question: does adaptive/head-specific AAH improve or preserve validation
quality more than shuffled, random, or fixed window assignment?

## Outputs

Compact summaries are written under:

```text
paper_results/aah_quality_structure_lab/
```

Do not commit checkpoints, raw W&B directories, or server credentials.
