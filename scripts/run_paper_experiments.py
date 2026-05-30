#!/usr/bin/env python3
import argparse
import copy
import os
import re
import subprocess
import sys
from datetime import datetime

import yaml


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_SEEDS = [0]


MAIN_4096 = [
    "main_4096_pure_baseline",
    "main_4096_grouping_off",
    "main_4096_full_adaptive",
    "main_4096_shallow_freeze",
    "main_4096_deep_practical_reuse",
]

APPENDIX_4096 = [
    "appendix_4096_control_off",
    "appendix_4096_fixed_random_grouping",
    "appendix_4096_freeze_after_warmup_passthrough",
    "appendix_4096_independent_scoring",
    "appendix_4096_no_parent_constraint",
    "appendix_4096_no_feature_ema",
]

def deep_update(dst, src):
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def base_config(run_id, seed, context_length, aah_enabled=True):
    if int(context_length) >= 8192:
        windows = [1024, 2048, 4096, context_length]
    else:
        windows = [512, 1024, 2048, context_length]
    batch_size = 1
    return {
        "experiment": {
            "name": f"paper-{run_id}-seed{seed}",
            "seed": int(seed),
            "variant": run_id,
            "out_dir": "experiments/paper",
        },
        "data": {
            "dataset": "wikitext-2-raw-v1",
            "tokenizer": "gpt2",
            "seq_len": int(context_length),
            "num_workers": 2,
        },
        "model": {
            "n_layer": 16,
            "n_head": 12,
            "n_embd": 1536,
            "n_ff": 6144,
            "dropout": 0.1,
            "aah_v3_enabled": bool(aah_enabled),
            "aah_v3_windows": windows,
            "aah_v3_grouping_enabled": True,
            "aah_v3_control_enabled": True,
            "aah_v3_control_dim": 16,
            "aah_v3_control_interval": 5,
            "aah_v3_sim_threshold": 0.72,
            "aah_v3_super_threshold": 0.76,
            "aah_v3_max_depth": 4,
            "aah_v3_ema_alpha": 0.9,
            "aah_v3_churn_penalty": 0.02,
            "aah_v3_min_group_size": 3,
            "aah_v3_warmup_steps": 100,
            "aah_v3_W_min_gpu": 64,
            "aah_v3_resolution_ema_alpha": 0.15,
            "aah_v3_post_warmup_ramp_steps": 0,
            "aah_v3_group_feature_mode": "mean",
            "aah_v3_upper_cluster_metric": "cosine_normdiff",
            "aah_v3_upper_l2_threshold": 0.0,
            "aah_v3_cosine_normdiff_scale": 16.0,
            "aah_v3_controller_input_mode": "enriched",
            "aah_v3_controller_arch": "mlp",
            "aah_v3_controller_logit_scale": 1.0,
            "aah_v3_controller_rng_reference_dim": 16,
            "aah_v3_controller_choice_mode": "learned",
            "aah_v3_controller_pairwise_mode": "joint_sibling",
            "aah_v3_pairwise_bias_scale": 1.0,
            "aah_v3_joint_output_scale": 1.0,
            "aah_v3_joint_hidden_dim": 64,
            "aah_v3_diagnostic_detail": "light",
            "aah_v3_resolution_collapse_min_frac": 0.98,
            "aah_v3_resolution_collapse_max_frac": 0.98,
            "aah_v3_build_hierarchy": True,
            "aah_v3_apply_window_control": True,
            "aah_v3_reuse_group_hierarchy": False,
            "aah_v3_hierarchy_ablation_mode": "adaptive",
            "aah_v3_fixed_hierarchy_seed": int(seed),
            "aah_v3_parent_constraint": True,
        },
        "train": {
            "batch_size": batch_size,
            "grad_accum": 1,
            "max_steps": 10000,
            "lr": 0.0003,
            "weight_decay": 0.1,
            "warmup_steps": 100,
            "eval_interval": 200,
            "eval_batches": 20,
            "eval_log_progress": False,
            "log_interval": 50,
            "device": "cuda",
            "precision": "bf16",
            "use_wandb": True,
            "log_csv": True,
            "save_checkpoints": True,
            "checkpoint_steps": [1000, 5000, 10000],
        },
    }


def apply_regime_overrides(cfg, run_id):
    model = cfg["model"]
    if run_id.endswith("pure_baseline"):
        model.update(
            {
                "aah_v3_enabled": False,
                "aah_v3_grouping_enabled": False,
                "aah_v3_control_enabled": False,
                "aah_v3_build_hierarchy": False,
                "aah_v3_apply_window_control": False,
                "aah_v3_controller_pairwise_mode": "none",
            }
        )
    elif run_id.endswith("grouping_off"):
        model.update(
            {
                "aah_v3_enabled": True,
                "aah_v3_grouping_enabled": False,
                "aah_v3_control_enabled": True,
                "aah_v3_build_hierarchy": False,
                "aah_v3_apply_window_control": True,
                "aah_v3_controller_pairwise_mode": "none",
                "aah_v3_reuse_group_hierarchy": False,
                "aah_v3_hierarchy_ablation_mode": "adaptive",
            }
        )
    elif run_id.endswith("full_adaptive"):
        model.update(
            {
                "aah_v3_max_depth": 4,
                "aah_v3_reuse_group_hierarchy": False,
                "aah_v3_hierarchy_ablation_mode": "adaptive",
            }
        )
    elif run_id.endswith("shallow_freeze"):
        model.update(
            {
                "aah_v3_max_depth": 1,
                "aah_v3_reuse_group_hierarchy": False,
                "aah_v3_hierarchy_ablation_mode": "freeze_learned_topology",
            }
        )
    elif run_id.endswith("deep_practical_reuse"):
        model.update(
            {
                "aah_v3_max_depth": 4,
                "aah_v3_reuse_group_hierarchy": True,
                "aah_v3_hierarchy_ablation_mode": "adaptive",
            }
        )
    elif run_id.endswith("control_off"):
        model.update(
            {
                "aah_v3_enabled": True,
                "aah_v3_grouping_enabled": True,
                "aah_v3_control_enabled": False,
                "aah_v3_build_hierarchy": True,
                "aah_v3_apply_window_control": False,
                "aah_v3_controller_pairwise_mode": "joint_sibling",
            }
        )
    elif run_id.endswith("fixed_random_grouping"):
        model.update(
            {
                "aah_v3_hierarchy_ablation_mode": "fixed_random",
                "aah_v3_reuse_group_hierarchy": False,
            }
        )
    elif run_id.endswith("freeze_after_warmup_passthrough"):
        model.update(
            {
                "aah_v3_hierarchy_ablation_mode": "freeze_after_warmup",
                "aah_v3_reuse_group_hierarchy": False,
            }
        )
    elif run_id.endswith("independent_scoring"):
        model.update(
            {
                "aah_v3_controller_pairwise_mode": "none",
                "aah_v3_reuse_group_hierarchy": False,
                "aah_v3_hierarchy_ablation_mode": "adaptive",
            }
        )
    elif run_id.endswith("no_parent_constraint"):
        # The current model always applies the parent clamp inside _select_windows.
        # Keep a marker in the config so this run is visible, but it requires model
        # support before it becomes a true no-parent-constraint ablation.
        model["aah_v3_parent_constraint"] = False
    elif run_id.endswith("no_feature_ema"):
        model["aah_v3_ema_alpha"] = 0.0
    else:
        raise ValueError(f"Unknown run_id: {run_id}")
    return cfg


def suite_run_ids(suite):
    if suite == "mandatory":
        return MAIN_4096
    if suite == "appendix":
        return APPENDIX_4096
    if suite == "all":
        return MAIN_4096 + APPENDIX_4096
    raise ValueError(f"Unknown suite: {suite}")


def context_for_run(run_id):
    return 4096


def config_path_for(config_dir, run_id, seed):
    return os.path.join(config_dir, f"{run_id}_seed{seed}.yaml")


def generate_configs(
    config_dir,
    suite,
    seeds,
    offline_token_file="",
    out_dir="",
    train_max_steps=None,
    train_eval_interval=None,
    train_eval_batches=None,
    train_log_interval=None,
    checkpoint_steps=None,
    disable_checkpoints=False,
):
    os.makedirs(config_dir, exist_ok=True)
    paths = []
    for run_id in suite_run_ids(suite):
        for seed in seeds:
            context_length = context_for_run(run_id)
            cfg = base_config(run_id, seed, context_length, aah_enabled=not run_id.endswith("pure_baseline"))
            cfg = apply_regime_overrides(cfg, run_id)
            if offline_token_file:
                cfg["data"]["dataset"] = f"tokenized:{offline_token_file}"
                cfg["data"]["tokenizer"] = os.path.basename(offline_token_file)
                cfg["data"]["num_workers"] = 0
            if out_dir:
                cfg["experiment"]["out_dir"] = out_dir
            if train_max_steps is not None:
                cfg["train"]["max_steps"] = int(train_max_steps)
                cfg["train"]["checkpoint_steps"] = [int(train_max_steps)]
            if train_eval_interval is not None:
                cfg["train"]["eval_interval"] = int(train_eval_interval)
            if train_eval_batches is not None:
                cfg["train"]["eval_batches"] = int(train_eval_batches)
            if train_log_interval is not None:
                cfg["train"]["log_interval"] = int(train_log_interval)
            if checkpoint_steps is not None:
                cfg["train"]["checkpoint_steps"] = [int(s) for s in checkpoint_steps]
            if disable_checkpoints:
                cfg["train"]["save_checkpoints"] = False
                cfg["train"]["checkpoint_steps"] = []
            path = config_path_for(config_dir, run_id, seed)
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            paths.append(path)
    return paths


def load_run_name(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["experiment"]["name"], cfg


def final_checkpoint_for(cfg):
    return os.path.join(cfg["experiment"].get("out_dir", "experiments"), f"{cfg['experiment']['name']}.pt")


def should_include(path, only_regex):
    if not only_regex:
        return True
    return re.search(only_regex, path) is not None


def run_command(cmd, log_path, dry_run=False):
    print(" ".join(cmd))
    print(f"log: {log_path}")
    if dry_run:
        return 0
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=f, stderr=subprocess.STDOUT, text=True)
        return proc.wait()


def main():
    parser = argparse.ArgumentParser(description="Generate and run the AAH-v3 paper experiment suite.")
    parser.add_argument("--suite", choices=["mandatory", "appendix", "all"], default="mandatory")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--config-dir", default="configs/paper_required")
    parser.add_argument("--log-dir", default="logs/paper_required")
    parser.add_argument("--summary-dir", default="experiments/paper_summaries")
    parser.add_argument("--diagnostics-dir", default="experiments/paper_diagnostics")
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument(
        "--offline-token-file",
        default="",
        help="Optional tokenized dataset .pt path. Generated configs use data.dataset=tokenized:<path>.",
    )
    parser.add_argument("--out-dir", default="", help="Override experiment.out_dir in generated configs.")
    parser.add_argument("--train-max-steps", type=int, default=None)
    parser.add_argument("--train-eval-interval", type=int, default=None)
    parser.add_argument("--train-eval-batches", type=int, default=None)
    parser.add_argument("--train-log-interval", type=int, default=None)
    parser.add_argument("--checkpoint-steps", nargs="*", type=int, default=None)
    parser.add_argument("--disable-checkpoints", action="store_true")
    parser.add_argument("--only", default=None, help="Regex filter over generated config paths.")
    parser.add_argument("--write-configs", action="store_true", help="Generate configs and exit unless --run is also set.")
    parser.add_argument("--run", choices=["none", "train", "infer", "all"], default="none")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip finished checkpoints/summaries.")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    config_dir = os.path.abspath(os.path.join(PROJECT_ROOT, args.config_dir))
    paths = generate_configs(
        config_dir,
        args.suite,
        args.seeds,
        offline_token_file=args.offline_token_file,
        out_dir=args.out_dir,
        train_max_steps=args.train_max_steps,
        train_eval_interval=args.train_eval_interval,
        train_eval_batches=args.train_eval_batches,
        train_log_interval=args.train_log_interval,
        checkpoint_steps=args.checkpoint_steps,
        disable_checkpoints=args.disable_checkpoints,
    )
    paths = [p for p in paths if should_include(p, args.only)]
    print(f"generated_configs={len(paths)} dir={config_dir}")

    if args.write_configs and args.run == "none":
        for path in paths:
            print(path)
        return
    if args.run == "none":
        print("No runs requested. Use --run train, --run infer, or --run all.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in paths:
        run_name, cfg = load_run_name(path)
        ckpt = final_checkpoint_for(cfg)
        stem = os.path.splitext(os.path.basename(path))[0]

        if args.run in ("train", "all"):
            if args.resume and os.path.exists(ckpt):
                print(f"skip_train existing_checkpoint={ckpt}")
            else:
                log_path = os.path.join(PROJECT_ROOT, args.log_dir, f"train_{stem}_{timestamp}.log")
                cmd = [sys.executable, "scripts/train.py", "--config", path]
                rc = run_command(cmd, log_path, dry_run=args.dry_run)
                if rc != 0:
                    print(f"train_failed config={path} exit={rc}")
                    if not args.continue_on_error:
                        sys.exit(rc)

        if args.run in ("infer", "all"):
            summary_path = os.path.join(PROJECT_ROOT, args.summary_dir, f"{stem}_infer.json")
            if args.resume and os.path.exists(summary_path):
                print(f"skip_infer existing_summary={summary_path}")
                continue
            log_path = os.path.join(PROJECT_ROOT, args.log_dir, f"infer_{stem}_{timestamp}.log")
            cmd = [
                sys.executable,
                "scripts/infer.py",
                "--config",
                path,
                "--eval-batches",
                str(args.eval_batches),
                "--summary-json",
                summary_path,
                "--diagnostics-dir",
                os.path.join(PROJECT_ROOT, args.diagnostics_dir),
                "--strict-checkpoint",
            ]
            rc = run_command(cmd, log_path, dry_run=args.dry_run)
            if rc != 0:
                print(f"infer_failed config={path} exit={rc}")
                if not args.continue_on_error:
                    sys.exit(rc)


if __name__ == "__main__":
    main()
