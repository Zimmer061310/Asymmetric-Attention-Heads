#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from datetime import datetime

import yaml

DEFAULT_CONFIGS = [
    "configs/baseline_500m_10000.yaml",
    "configs/baseline_1b_10000.yaml",
    "configs/baseline_qwen2b_10000.yaml",
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


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline 500M->1B->2B with 1 train + 3 strict checkpoint inferences per config"
    )
    parser.add_argument("--log-dir", default="logs", help="Directory for train/infer logs")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="Config paths to run in order (default: 500M, 1B, 2B baselines)",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable for subprocesses")
    parser.add_argument("--eval-batches", type=int, default=50, help="Inference eval batches")
    parser.add_argument(
        "--checkpoint-steps",
        default="9000,9500,10000",
        help="Comma-separated checkpoint steps for strict inference",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining configs when one train/infer task fails",
    )
    args = parser.parse_args()

    ckpt_steps = parse_csv_ints(args.checkpoint_steps)
    if not ckpt_steps:
        raise ValueError("checkpoint steps cannot be empty")

    os.makedirs(args.log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

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
        train_cmd = [args.python, "scripts/train.py", "--config", cfg]
        rc = run_and_log(train_cmd, train_log)
        if rc != 0:
            print(f"Training failed for {cfg} (exit {rc})", flush=True)
            failures.append((cfg, f"train:{rc}"))
            if not args.continue_on_error:
                sys.exit(rc)
            continue

        checkpoints = [os.path.join("experiments", f"{run_name}_step{step}.pt") for step in ckpt_steps]
        infer_log = os.path.join(args.log_dir, f"infer_{tag}.log")
        infer_summary_json = os.path.join(args.log_dir, f"infer_summary_{tag}.json")
        print(
            f"=== Inference: {cfg} | checkpoints={checkpoints} -> {infer_log} ===",
            flush=True,
        )
        infer_cmd = [
            args.python,
            "scripts/infer.py",
            "--config",
            cfg,
            "--checkpoints",
            *checkpoints,
            "--strict-checkpoint",
            "--deterministic-eval",
            "--eval-batches",
            str(args.eval_batches),
            "--summary-json",
            infer_summary_json,
        ]
        rc = run_and_log(infer_cmd, infer_log)
        if rc != 0:
            print(f"Inference failed for {cfg} (exit {rc})", flush=True)
            failures.append((cfg, f"infer:{rc}"))
            if not args.continue_on_error:
                sys.exit(rc)

    if failures:
        print("Completed with failures:", flush=True)
        for cfg, reason in failures:
            print(f"  - {cfg}: {reason}", flush=True)
        sys.exit(1)

    print("500M/1B/2B train+3x strict infer suite completed.", flush=True)


if __name__ == "__main__":
    main()
