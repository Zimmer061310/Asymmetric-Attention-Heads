#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from datetime import datetime

import yaml


def run_and_log(cmd, log_path):
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        return proc.wait()


def experiment_name(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["experiment"]["name"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs", help="Directory for logs")
    parser.add_argument("--eval-batches", type=int, default=50, help="Inference eval batches")
    args = parser.parse_args()

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
        "configs/baseline_qwen2b_10000.yaml",
        "configs/aah_v3_control_off_final_qwen2b_10000.yaml",
        "configs/aah_v3_final_qwen2b_10000.yaml",
    ]

    os.makedirs(args.log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for cfg in configs:
        exp_name = experiment_name(cfg)
        tag = f"{os.path.splitext(os.path.basename(cfg))[0]}_{ts}"

        train_log = os.path.join(args.log_dir, f"train_{tag}.log")
        print(f"=== Training: {cfg} -> {train_log} ===", flush=True)
        rc = run_and_log([sys.executable, "scripts/train.py", "--config", cfg], train_log)
        if rc != 0:
            print(f"Training failed for {cfg} (exit {rc}). Stopping.", flush=True)
            sys.exit(rc)

        ckpt = os.path.join("experiments", f"{exp_name}.pt")
        infer_log = os.path.join(args.log_dir, f"infer_{tag}.log")
        print(f"=== Inference: {cfg} | checkpoint={ckpt} -> {infer_log} ===", flush=True)
        rc = run_and_log(
            [
                sys.executable,
                "scripts/infer.py",
                "--config",
                cfg,
                "--checkpoint",
                ckpt,
                "--strict-checkpoint",
                "--eval-batches",
                str(args.eval_batches),
            ],
            infer_log,
        )
        if rc != 0:
            print(f"Inference failed for {cfg} (exit {rc}). Stopping.", flush=True)
            sys.exit(rc)

    print("32k scale suite completed.", flush=True)


if __name__ == "__main__":
    main()
