#!/usr/bin/env python3
import argparse
import os
import sys

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.run_paper_experiments import apply_regime_overrides, base_config


PHASE1_ROWS = [
    {
        "slug": "pure-baseline",
        "source": "main_4096_pure_baseline",
        "group": "reference",
        "purpose": "budget-matched quality denominator",
    },
    {
        "slug": "shallow-freeze",
        "source": "main_4096_shallow_freeze",
        "group": "reference",
        "purpose": "current best-quality AAH reference",
    },
    {
        "slug": "full-adaptive",
        "source": "main_4096_full_adaptive",
        "group": "reference",
        "purpose": "adaptive structure reference",
    },
    {
        "slug": "shallow-shuffle-post-select",
        "source": "main_4096_shallow_freeze",
        "group": "random_control",
        "purpose": "preserve AAH window histogram but destroy head-window identity",
        "model": {"aah_v3_window_ablation_mode": "shuffle_post_select"},
    },
    {
        "slug": "full-adaptive-shuffle-post-select",
        "source": "main_4096_full_adaptive",
        "group": "random_control",
        "purpose": "same histogram shuffle control for full adaptive hierarchy",
        "model": {"aah_v3_window_ablation_mode": "shuffle_post_select"},
    },
    {
        "slug": "shallow-random-uniform",
        "source": "main_4096_shallow_freeze",
        "group": "random_control",
        "purpose": "arbitrary local-window noise control",
        "model": {"aah_v3_window_ablation_mode": "random_uniform"},
    },
    {
        "slug": "fixed-1024",
        "source": "main_4096_shallow_freeze",
        "group": "fixed_control",
        "purpose": "simple fixed 1024 local-window control",
        "model": {"aah_v3_window_ablation_mode": "fixed_window", "aah_v3_fixed_window": 1024},
    },
    {
        "slug": "fixed-2048",
        "source": "main_4096_shallow_freeze",
        "group": "fixed_control",
        "purpose": "simple fixed 2048 local-window control",
        "model": {"aah_v3_window_ablation_mode": "fixed_window", "aah_v3_fixed_window": 2048},
    },
    {
        "slug": "fixed-random-grouping",
        "source": "appendix_4096_fixed_random_grouping",
        "group": "topology_control",
        "purpose": "existing random topology/grouping control",
    },
    {
        "slug": "shallow-no512",
        "source": "main_4096_shallow_freeze",
        "group": "optimization",
        "purpose": "test whether removing the 512 window improves quality",
        "model": {"aah_v3_windows": [1024, 2048, 4096]},
    },
    {
        "slug": "shallow-control-interval10",
        "source": "main_4096_shallow_freeze",
        "group": "optimization",
        "purpose": "test whether slower routing improves stability",
        "model": {"aah_v3_control_interval": 10},
    },
    {
        "slug": "shallow-resolution-ema030",
        "source": "main_4096_shallow_freeze",
        "group": "optimization",
        "purpose": "test whether smoother execution improves quality",
        "model": {"aah_v3_resolution_ema_alpha": 0.30},
    },
]


def write_yaml(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def build_config(row, seed, max_steps, out_dir):
    cfg = base_config(row["source"], seed, 4096, aah_enabled=not row["source"].endswith("pure_baseline"))
    cfg = apply_regime_overrides(cfg, row["source"])
    name = f"quality-4096-phase1-{row['slug']}-seed{seed}"
    cfg["experiment"]["name"] = name
    cfg["experiment"]["variant"] = row["slug"]
    cfg["experiment"]["out_dir"] = out_dir
    cfg["model"]["aah_v3_window_ablation_mode"] = "adaptive"
    cfg["model"]["aah_v3_fixed_window"] = 0
    cfg["model"]["aah_v3_window_ablation_seed"] = int(seed)
    cfg["model"].update(row.get("model", {}))
    cfg["train"]["max_steps"] = int(max_steps)
    cfg["train"]["checkpoint_steps"] = [int(max_steps)]
    cfg["train"]["eval_interval"] = 200
    cfg["train"]["eval_batches"] = 20
    cfg["train"]["log_interval"] = 50
    cfg["train"]["use_wandb"] = True
    cfg["train"]["log_csv"] = True
    cfg["train"]["save_checkpoints"] = True
    cfg["quality_lab"] = {
        "phase": "phase1_screen",
        "group": row["group"],
        "purpose": row["purpose"],
        "source_run_id": row["source"],
        "promotion_budget_steps": 5000,
    }
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="experiments/aah_quality_structure_lab/configs/phase1")
    parser.add_argument("--out-dir", default="experiments/aah_quality_structure_lab/results/phase1")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest = []
    for row in PHASE1_ROWS:
        cfg = build_config(row, args.seed, args.max_steps, args.out_dir)
        path = os.path.join(args.config_dir, f"{cfg['experiment']['name']}.yaml")
        write_yaml(path, cfg)
        manifest.append({
            "name": cfg["experiment"]["name"],
            "config": path,
            "group": row["group"],
            "purpose": row["purpose"],
            "source": row["source"],
            "max_steps": int(args.max_steps),
        })

    manifest_path = os.path.join(args.config_dir, "phase1_manifest.yaml")
    write_yaml(manifest_path, {"runs": manifest})
    print(f"wrote {len(manifest)} configs")
    print(f"manifest {manifest_path}")


if __name__ == "__main__":
    main()
