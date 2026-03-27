#!/usr/bin/env python3
import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime

import yaml


DEFAULT_CONFIGS = [
    "configs/baseline_500m_10000.yaml",
    "configs/aah_v3_control_off_final_500m_10000.yaml",
    "configs/aah_v3_final_500m_10000.yaml",
    "configs/baseline_1b_10000.yaml",
    "configs/aah_v3_control_off_final_1b_10000.yaml",
    "configs/aah_v3_final_1b_10000.yaml",
]


def run_and_log(cmd, log_path):
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        return proc.wait()


def parse_csv_ints(s):
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    return [int(p) for p in parts]


def experiment_name(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["experiment"]["name"]


def infer_method_tag(text):
    lower = text.lower()
    if "baseline" in lower:
        return "baseline"
    if "control_off" in lower or "control-off" in lower:
        return "control-off"
    return "aah"


def infer_model_size_tag(text):
    lower = text.lower()
    for token in ["500m", "1b"]:
        if token in lower:
            return token
    return "unknown"


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean_std(values):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return sum(vals) / len(vals), statistics.pstdev(vals)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run 500M+1B suite in order: baseline/control-off/final, "
            "with 1 train + 3 strict single-checkpoint inferences per config"
        )
    )
    parser.add_argument("--log-dir", default="logs", help="Directory for logs")
    parser.add_argument("--eval-batches", type=int, default=50, help="Inference eval batches")
    parser.add_argument(
        "--checkpoint-steps",
        default="9000,9500,10000",
        help="Comma-separated checkpoint steps to evaluate",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="Config paths to run in order (default: 500M then 1B triplets)",
    )
    parser.add_argument(
        "--summary-dir",
        default="experiments/summaries",
        help="Directory for detailed+summary CSV exports",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable for subprocesses")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining configs even if one training/inference run fails",
    )
    args = parser.parse_args()

    checkpoint_steps = parse_csv_ints(args.checkpoint_steps)
    if not checkpoint_steps:
        raise ValueError("checkpoint steps cannot be empty")

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.summary_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    detailed_rows = []
    summary_rows = []
    failures = []

    for cfg in args.configs:
        if not os.path.exists(cfg):
            msg = f"Config not found: {cfg}"
            print(msg, flush=True)
            failures.append((cfg, "missing"))
            if not args.continue_on_error:
                sys.exit(1)
            continue

        run_name = experiment_name(cfg)
        tag = f"{os.path.splitext(os.path.basename(cfg))[0]}_{ts}"

        train_log = os.path.join(args.log_dir, f"train_{tag}.log")
        print(f"=== Training: {cfg} -> {train_log} ===", flush=True)
        rc = run_and_log([args.python, "scripts/train.py", "--config", cfg], train_log)
        if rc != 0:
            print(f"Training failed for {cfg} (exit {rc}).", flush=True)
            failures.append((cfg, f"train:{rc}"))
            if not args.continue_on_error:
                sys.exit(rc)
            continue

        method_tag = infer_method_tag(run_name)
        model_size_tag = infer_model_size_tag(run_name)
        checkpoint_paths = [os.path.join("experiments", f"{run_name}_step{step}.pt") for step in checkpoint_steps]

        cfg_results = []
        payload_meta = None
        for step, ckpt_path in zip(checkpoint_steps, checkpoint_paths):
            infer_log = os.path.join(args.log_dir, f"infer_{tag}_step{step}.log")
            infer_summary_json = os.path.join(args.log_dir, f"infer_summary_{tag}_step{step}.json")
            print(
                f"=== Inference: {cfg} | checkpoint={ckpt_path} -> {infer_log} ===",
                flush=True,
            )
            infer_cmd = [
                args.python,
                "scripts/infer.py",
                "--config",
                cfg,
                "--checkpoint",
                ckpt_path,
                "--strict-checkpoint",
                "--deterministic-eval",
                "--eval-batches",
                str(args.eval_batches),
                "--summary-json",
                infer_summary_json,
            ]
            rc = run_and_log(infer_cmd, infer_log)
            if rc != 0:
                print(f"Inference failed for {cfg} step={step} (exit {rc}).", flush=True)
                failures.append((cfg, f"infer:{step}:{rc}"))
                if not args.continue_on_error:
                    sys.exit(rc)
                continue

            with open(infer_summary_json, "r") as f:
                payload = json.load(f)
            if payload_meta is None:
                payload_meta = payload.get("meta", {})
            if not payload.get("results"):
                print(f"Inference summary missing results for {cfg} step={step}.", flush=True)
                failures.append((cfg, f"infer:{step}:no-results"))
                if not args.continue_on_error:
                    sys.exit(1)
                continue

            row = payload["results"][0]
            cfg_results.append(row)
            detailed_rows.append(
                {
                    "suite_timestamp": ts,
                    "source_config": cfg,
                    "run_name": row.get("run_name", run_name),
                    "method_tag": method_tag,
                    "model_size_tag": model_size_tag,
                    "seed": row.get("seed"),
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

        if not cfg_results:
            continue

        val_loss_mean, val_loss_std = mean_std([r.get("val_loss") for r in cfg_results])
        val_ppl_mean, val_ppl_std = mean_std([r.get("val_ppl") for r in cfg_results])
        flops_ratio_mean, flops_ratio_std = mean_std([r.get("flops_ratio") for r in cfg_results])
        infer_tok_s_mean, infer_tok_s_std = mean_std([r.get("tok_s") for r in cfg_results])
        best_row = min(cfg_results, key=lambda r: float(r.get("val_ppl", 1e30)))
        last_row = cfg_results[-1]
        ppl_values = [float(r.get("val_ppl")) for r in cfg_results if r.get("val_ppl") is not None]
        loss_values = [float(r.get("val_loss")) for r in cfg_results if r.get("val_loss") is not None]
        checkpoint_sensitivity_val_ppl = max(ppl_values) - min(ppl_values) if ppl_values else None
        checkpoint_sensitivity_val_loss = max(loss_values) - min(loss_values) if loss_values else None
        summary_rows.append(
            {
                "suite_timestamp": ts,
                "source_config": cfg,
                "run_name": run_name,
                "method_tag": method_tag,
                "model_size_tag": model_size_tag,
                "n_checkpoints": len(cfg_results),
                "val_loss_mean": val_loss_mean,
                "val_loss_std": val_loss_std,
                "val_ppl_mean": val_ppl_mean,
                "val_ppl_std": val_ppl_std,
                "flops_ratio_mean": flops_ratio_mean,
                "flops_ratio_std": flops_ratio_std,
                "infer_tok_s_mean": infer_tok_s_mean,
                "infer_tok_s_std": infer_tok_s_std,
                "best_checkpoint": best_row.get("checkpoint_path"),
                "best_checkpoint_step": best_row.get("checkpoint_step"),
                "best_val_ppl": best_row.get("val_ppl"),
                "best_val_loss": best_row.get("val_loss"),
                "last_checkpoint": last_row.get("checkpoint_path"),
                "last_checkpoint_step": last_row.get("checkpoint_step"),
                "last_val_ppl": last_row.get("val_ppl"),
                "last_val_loss": last_row.get("val_loss"),
                "checkpoint_sensitivity_val_ppl": checkpoint_sensitivity_val_ppl,
                "checkpoint_sensitivity_val_loss": checkpoint_sensitivity_val_loss,
                "git_commit": (payload_meta or {}).get("git_commit"),
                "config_hash": (payload_meta or {}).get("config_hash"),
            }
        )

    detailed_path = os.path.join(args.summary_dir, f"scale_500m_1b_detailed_{ts}.csv")
    summary_path = os.path.join(args.summary_dir, f"scale_500m_1b_summary_{ts}.csv")
    write_csv(
        detailed_path,
        detailed_rows,
        fieldnames=[
            "suite_timestamp",
            "source_config",
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
            "run_name",
            "method_tag",
            "model_size_tag",
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

    if failures:
        print("Completed with failures:", flush=True)
        for cfg, reason in failures:
            print(f"  - {cfg}: {reason}", flush=True)
        print(f"Detailed export: {detailed_path}", flush=True)
        print(f"Summary export: {summary_path}", flush=True)
        sys.exit(1)

    print(f"Detailed export: {detailed_path}", flush=True)
    print(f"Summary export: {summary_path}", flush=True)
    print("500M/1B suite completed.", flush=True)


if __name__ == "__main__":
    main()
