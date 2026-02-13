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
    parser.add_argument("--log-dir", default="logs", help="Directory to store logs")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/baseline_1b_5000.yaml",
            "configs/aah_v3_control_off_1b_5000.yaml",
            "configs/aah_v3_full_1b_5000.yaml",
            "configs/aah_v3_full_1b_5000_stable.yaml",
        ],
        help="List of config paths to run in order",
    )
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    for cfg in args.configs:
        name = os.path.splitext(os.path.basename(cfg))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(args.log_dir, f"{name}_{ts}.log")
        print(f"=== Running {cfg} -> {log_path} ===", flush=True)
        rc = run([sys.executable, "scripts/train.py", "--config", cfg], log_path)
        if rc != 0:
            print(f"Run failed for {cfg} (exit {rc}). Stopping.", flush=True)
            sys.exit(rc)


if __name__ == "__main__":
    main()
