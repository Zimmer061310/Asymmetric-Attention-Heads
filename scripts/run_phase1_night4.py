#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from datetime import datetime


def run_and_log(cmd, log_path):
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        return proc.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs", help="Directory for logs")
    parser.add_argument("--run-inference", action="store_true", help="Run inference after each training run")
    parser.add_argument("--eval-batches", type=int, default=50, help="Inference eval batches")
    args = parser.parse_args()

    configs = [
        "configs/aah_v3_full_1b_5000_night1.yaml",
        "configs/aah_v3_full_1b_5000_night2.yaml",
        "configs/aah_v3_full_1b_5000_night3.yaml",
        "configs/aah_v3_full_1b_5000_night4.yaml",
    ]

    os.makedirs(args.log_dir, exist_ok=True)

    for cfg in configs:
        name = os.path.splitext(os.path.basename(cfg))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_log = os.path.join(args.log_dir, f"train_{name}_{ts}.log")
        print(f"=== Training: {cfg} -> {train_log} ===", flush=True)
        rc = run_and_log([sys.executable, "scripts/train.py", "--config", cfg], train_log)
        if rc != 0:
            print(f"Training failed for {cfg} (exit {rc}). Stopping.", flush=True)
            sys.exit(rc)

        if args.run_inference:
            infer_log = os.path.join(args.log_dir, f"infer_{name}_{ts}.log")
            print(f"=== Inference: {cfg} -> {infer_log} ===", flush=True)
            rc = run_and_log(
                [
                    sys.executable,
                    "scripts/infer.py",
                    "--config",
                    cfg,
                    "--eval-batches",
                    str(args.eval_batches),
                ],
                infer_log,
            )
            if rc != 0:
                print(f"Inference failed for {cfg} (exit {rc}). Stopping.", flush=True)
                sys.exit(rc)

    print("All night runs completed.", flush=True)


if __name__ == "__main__":
    main()
