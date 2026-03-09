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
    parser.add_argument("--eval-batches", type=int, default=50, help="Inference eval batches")
    parser.add_argument(
        "--config",
        default="configs/aah_v3_full_1b_10000_night3.yaml",
        help="Training config path",
    )
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = os.path.splitext(os.path.basename(args.config))[0]

    train_log = os.path.join(args.log_dir, f"train_{name}_{ts}.log")
    print(f"=== Training: {args.config} -> {train_log} ===", flush=True)
    rc = run_and_log([sys.executable, "scripts/train.py", "--config", args.config], train_log)
    if rc != 0:
        print(f"Training failed (exit {rc}). Stopping.", flush=True)
        sys.exit(rc)

    infer_log = os.path.join(args.log_dir, f"infer_{name}_{ts}.log")
    print(f"=== Inference: {args.config} -> {infer_log} ===", flush=True)
    rc = run_and_log(
        [
            sys.executable,
            "scripts/infer.py",
            "--config",
            args.config,
            "--eval-batches",
            str(args.eval_batches),
        ],
        infer_log,
    )
    if rc != 0:
        print(f"Inference failed (exit {rc}).", flush=True)
        sys.exit(rc)

    print("Night3 10k train+infer completed.", flush=True)


if __name__ == "__main__":
    main()
