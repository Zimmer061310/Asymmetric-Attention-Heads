#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from datetime import datetime


def run(cmd, log_path):
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        return proc.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs", help="Directory for inference logs")
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/baseline_1b_5000.yaml",
            "configs/aah_v3_control_off_1b_5000.yaml",
            "configs/aah_v3_full_1b_5000_stable2.yaml",
        ],
    )
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    for cfg in args.configs:
        name = os.path.splitext(os.path.basename(cfg))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(args.log_dir, f"infer_{name}_{ts}.log")
        print(f"=== Inference {cfg} -> {log_path} ===", flush=True)
        cmd = [sys.executable, "scripts/infer.py", "--config", cfg, "--eval-batches", str(args.eval_batches)]
        rc = run(cmd, log_path)
        if rc != 0:
            print(f"Inference failed for {cfg} (exit {rc}). Stopping.", flush=True)
            sys.exit(rc)


if __name__ == "__main__":
    main()
