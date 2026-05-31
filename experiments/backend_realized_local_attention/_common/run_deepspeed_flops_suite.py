"""Run DeepSpeed FLOPs profiling for the 4096 Flex/Flash backend suite."""

import argparse
import subprocess
import sys
from pathlib import Path

from experiments.backend_realized_local_attention._common.run_ncu_suite import ROWS, checkpoint_path


def run(cmd, dry_run=False, continue_on_error=False):
    print("+ " + " ".join(str(x) for x in cmd), flush=True)
    if dry_run:
        return 0
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0 and not continue_on_error:
        raise SystemExit(proc.returncode)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="paper_results/backend_4096_realized_attention_deepspeed")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    profile_dir = run_root / "deepspeed_flops_profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    baselines = {}
    for backend, method, module, cfg in ROWS:
        if not args.profile_only:
            run(
                [
                    sys.executable,
                    "-m",
                    "experiments.backend_realized_local_attention._common.run_train",
                    "--module",
                    module,
                    "--config",
                    cfg,
                ],
                dry_run=args.dry_run,
                continue_on_error=args.continue_on_error,
            )

        out = profile_dir / f"{backend.lower()}_{method}_deepspeed_flops_profile.json"
        cmd = [
            sys.executable,
            "-m",
            "experiments.backend_realized_local_attention._common.profile_deepspeed_flops",
            "--module",
            module,
            "--config",
            cfg,
            "--device",
            args.device,
            "--warmup",
            str(args.warmup),
            "--output",
            str(out),
        ]
        ckpt = checkpoint_path(cfg)
        if ckpt.exists():
            cmd.extend(["--checkpoint", str(ckpt)])
        if method != "pure" and backend in baselines:
            cmd.extend(["--baseline-json", str(baselines[backend])])
        run(cmd, dry_run=args.dry_run, continue_on_error=args.continue_on_error)
        if method == "pure":
            baselines[backend] = out

    print(f"DeepSpeed FLOPs suite finished. Profiles: {profile_dir}", flush=True)


if __name__ == "__main__":
    main()
