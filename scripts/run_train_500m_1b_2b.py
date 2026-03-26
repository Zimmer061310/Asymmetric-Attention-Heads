#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from datetime import datetime

DEFAULT_CONFIGS = [
    "configs/baseline_500m_10000.yaml",
    "configs/baseline_1b_10000.yaml",
    "configs/baseline_qwen2b_10000.yaml",
]


def run_and_log(cmd, log_path):
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        return proc.wait()


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline training sequentially for 500M, 1B, and 2B model configs"
    )
    parser.add_argument("--log-dir", default="logs", help="Directory for training logs")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="Config paths to run in order (default: 500M, 1B, 2B baselines)",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable used for training")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining configs even if one training run fails",
    )
    args = parser.parse_args()

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

        tag = f"{os.path.splitext(os.path.basename(cfg))[0]}_{ts}"
        train_log = os.path.join(args.log_dir, f"train_{tag}.log")
        cmd = [args.python, "scripts/train.py", "--config", cfg]

        print(f"=== Training: {cfg} -> {train_log} ===", flush=True)
        rc = run_and_log(cmd, train_log)
        if rc != 0:
            print(f"Training failed for {cfg} (exit {rc})", flush=True)
            failures.append((cfg, str(rc)))
            if not args.continue_on_error:
                sys.exit(rc)

    if failures:
        print("Completed with failures:", flush=True)
        for cfg, reason in failures:
            print(f"  - {cfg}: {reason}", flush=True)
        sys.exit(1)

    print("500M/1B/2B baseline training suite completed.", flush=True)


if __name__ == "__main__":
    main()
