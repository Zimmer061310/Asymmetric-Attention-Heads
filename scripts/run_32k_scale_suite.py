#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime

import yaml


def run_and_log(cmd, log_path):
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        return proc.wait()


def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def experiment_name(cfg_path):
    cfg = load_config(cfg_path)
    return cfg["experiment"]["name"]


def parse_csv_ints(s):
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    return [int(p) for p in parts]


def infer_method_tag(text):
    lower = text.lower()
    if "baseline" in lower:
        return "baseline"
    if "control_off" in lower or "control-off" in lower:
        return "control-off"
    return "aah"


def infer_model_size_tag(text):
    lower = text.lower()
    for token in ["100m", "200m", "500m", "1b", "qwen2b", "2b"]:
        if token in lower:
            return token
    return "unknown"


def materialize_config_for_seed(cfg_path, seed, generated_dir):
    cfg = load_config(cfg_path)
    exp = cfg["experiment"]
    original_seed = int(exp["seed"])
    original_name = exp["name"]
    if int(seed) == original_seed:
        return cfg_path, original_name, original_seed
    os.makedirs(generated_dir, exist_ok=True)
    exp["seed"] = int(seed)
    exp["name"] = f"{original_name}-s{seed}"
    out_name = f"{os.path.splitext(os.path.basename(cfg_path))[0]}_seed{seed}.yaml"
    out_path = os.path.join(generated_dir, out_name)
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path, exp["name"], int(seed)


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs", help="Directory for logs")
    parser.add_argument("--eval-batches", type=int, default=50, help="Inference eval batches")
    parser.add_argument("--checkpoint-steps", default="9000,9500,10000", help="Comma-separated checkpoint steps to evaluate")
    parser.add_argument("--seeds", default="1337", help="Comma-separated seeds (e.g. 1337 or 1337,2024,777)")
    parser.add_argument("--summary-dir", default="experiments/summaries", help="Directory for detailed+summary CSV exports")
    parser.add_argument("--generated-config-dir", default="experiments/generated_configs", help="Directory for generated seed-overridden configs")
    parser.add_argument("--include-qwen2b", action="store_true", help="Include qwen2b triplet in suite")
    args = parser.parse_args()

    checkpoint_steps = parse_csv_ints(args.checkpoint_steps)
    seeds = parse_csv_ints(args.seeds)
    if not checkpoint_steps:
        raise ValueError("checkpoint steps cannot be empty")
    if not seeds:
        raise ValueError("seed list cannot be empty")

    configs = [
        "configs/baseline_100m_10000.yaml",
        "configs/aah_v3_control_off_final_100m_10000.yaml",
        "configs/aah_v3_final_100m_10000.yaml",
        "configs/baseline_200m_10000.yaml",
        "configs/aah_v3_control_off_final_200m_10000.yaml",
        "configs/aah_v3_final_200m_10000.yaml",
        "configs/baseline_500m_10000.yaml",
        "configs/aah_v3_control_off_final_500m_10000.yaml",
        "configs/aah_v3_final_500m_10000.yaml",
        "configs/baseline_1b_10000.yaml",
        "configs/aah_v3_control_off_final_1b_10000.yaml",
        "configs/aah_v3_final_1b_10000.yaml",
    ]
    if args.include_qwen2b:
        configs.extend(
            [
                "configs/baseline_qwen2b_10000.yaml",
                "configs/aah_v3_control_off_final_qwen2b_10000.yaml",
                "configs/aah_v3_final_qwen2b_10000.yaml",
            ]
        )

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.summary_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    detailed_rows = []
    summary_rows = []

    for cfg in configs:
        for seed in seeds:
            run_cfg, run_name, run_seed = materialize_config_for_seed(
                cfg,
                seed,
                args.generated_config_dir,
            )
            tag = f"{os.path.splitext(os.path.basename(cfg))[0]}_s{run_seed}_{ts}"

            train_log = os.path.join(args.log_dir, f"train_{tag}.log")
            print(f"=== Training: {run_cfg} -> {train_log} ===", flush=True)
            rc = run_and_log([sys.executable, "scripts/train.py", "--config", run_cfg], train_log)
            if rc != 0:
                print(f"Training failed for {run_cfg} (exit {rc}). Stopping.", flush=True)
                sys.exit(rc)

            ckpts = [os.path.join("experiments", f"{run_name}_step{step}.pt") for step in checkpoint_steps]
            infer_log = os.path.join(args.log_dir, f"infer_{tag}.log")
            infer_summary_json = os.path.join(args.log_dir, f"infer_summary_{tag}.json")
            print(
                f"=== Inference: {run_cfg} | checkpoints={ckpts} -> {infer_log} ===",
                flush=True,
            )
            infer_cmd = [
                sys.executable,
                "scripts/infer.py",
                "--config",
                run_cfg,
                "--checkpoints",
                *ckpts,
                "--strict-checkpoint",
                "--deterministic-eval",
                "--eval-batches",
                str(args.eval_batches),
                "--summary-json",
                infer_summary_json,
            ]
            rc = run_and_log(infer_cmd, infer_log)
            if rc != 0:
                print(f"Inference failed for {run_cfg} (exit {rc}). Stopping.", flush=True)
                sys.exit(rc)

            with open(infer_summary_json, "r") as f:
                payload = json.load(f)

            method_tag = infer_method_tag(run_name)
            model_size_tag = infer_model_size_tag(run_name)
            for row in payload["results"]:
                detailed_rows.append(
                    {
                        "suite_timestamp": ts,
                        "source_config": cfg,
                        "resolved_config": run_cfg,
                        "run_name": row.get("run_name", run_name),
                        "method_tag": method_tag,
                        "model_size_tag": model_size_tag,
                        "seed": row.get("seed", run_seed),
                        "checkpoint_path": row.get("checkpoint_path"),
                        "checkpoint_step": row.get("checkpoint_step"),
                        "val_loss": row.get("val_loss"),
                        "val_ppl": row.get("val_ppl"),
                        "flops_ratio": row.get("flops_ratio"),
                        "infer_tok_s": row.get("tok_s"),
                        "git_commit": row.get("git_commit"),
                        "config_hash": row.get("config_hash"),
                    }
                )

            s = payload["summary"]
            summary_rows.append(
                {
                    "suite_timestamp": ts,
                    "source_config": cfg,
                    "resolved_config": run_cfg,
                    "run_name": run_name,
                    "method_tag": method_tag,
                    "model_size_tag": model_size_tag,
                    "seed": run_seed,
                    "n_checkpoints": s.get("n_checkpoints"),
                    "val_loss_mean": s.get("val_loss_mean"),
                    "val_loss_std": s.get("val_loss_std"),
                    "val_ppl_mean": s.get("val_ppl_mean"),
                    "val_ppl_std": s.get("val_ppl_std"),
                    "flops_ratio_mean": s.get("flops_ratio_mean"),
                    "flops_ratio_std": s.get("flops_ratio_std"),
                    "infer_tok_s_mean": s.get("tok_s_mean"),
                    "infer_tok_s_std": s.get("tok_s_std"),
                    "best_checkpoint": s.get("best_checkpoint"),
                    "best_checkpoint_step": s.get("best_checkpoint_step"),
                    "best_val_ppl": s.get("best_val_ppl"),
                    "best_val_loss": s.get("best_val_loss"),
                    "last_checkpoint": s.get("last_checkpoint"),
                    "last_checkpoint_step": s.get("last_checkpoint_step"),
                    "last_val_ppl": s.get("last_val_ppl"),
                    "last_val_loss": s.get("last_val_loss"),
                    "checkpoint_sensitivity_val_ppl": s.get("checkpoint_sensitivity_val_ppl"),
                    "checkpoint_sensitivity_val_loss": s.get("checkpoint_sensitivity_val_loss"),
                    "git_commit": payload["meta"].get("git_commit"),
                    "config_hash": payload["meta"].get("config_hash"),
                }
            )

    detailed_path = os.path.join(args.summary_dir, f"scale_suite_detailed_{ts}.csv")
    summary_path = os.path.join(args.summary_dir, f"scale_suite_summary_{ts}.csv")
    write_csv(
        detailed_path,
        detailed_rows,
        fieldnames=[
            "suite_timestamp",
            "source_config",
            "resolved_config",
            "run_name",
            "method_tag",
            "model_size_tag",
            "seed",
            "checkpoint_path",
            "checkpoint_step",
            "val_loss",
            "val_ppl",
            "flops_ratio",
            "infer_tok_s",
            "git_commit",
            "config_hash",
        ],
    )
    write_csv(
        summary_path,
        summary_rows,
        fieldnames=[
            "suite_timestamp",
            "source_config",
            "resolved_config",
            "run_name",
            "method_tag",
            "model_size_tag",
            "seed",
            "n_checkpoints",
            "val_loss_mean",
            "val_loss_std",
            "val_ppl_mean",
            "val_ppl_std",
            "flops_ratio_mean",
            "flops_ratio_std",
            "infer_tok_s_mean",
            "infer_tok_s_std",
            "best_checkpoint",
            "best_checkpoint_step",
            "best_val_ppl",
            "best_val_loss",
            "last_checkpoint",
            "last_checkpoint_step",
            "last_val_ppl",
            "last_val_loss",
            "checkpoint_sensitivity_val_ppl",
            "checkpoint_sensitivity_val_loss",
            "git_commit",
            "config_hash",
        ],
    )

    print(f"Detailed export: {detailed_path}", flush=True)
    print(f"Summary export: {summary_path}", flush=True)
    print("Scale suite completed.", flush=True)


if __name__ == "__main__":
    main()
