"""Run/profile the 4096 backend suite with Nsight Compute FLOP counters."""

import argparse
import subprocess
import sys
from pathlib import Path


ROWS = [
    ("FlexAttention", "pure", "pure", "experiments/backend_realized_local_attention/FlexAttention/pure/configs/backend_4096_pure_flex_seed0.yaml"),
    ("FlexAttention", "grouping_off", "aah", "experiments/backend_realized_local_attention/FlexAttention/aah_modified/configs/backend_4096_grouping_off_flex_seed0.yaml"),
    ("FlexAttention", "full_adaptive", "aah", "experiments/backend_realized_local_attention/FlexAttention/aah_modified/configs/backend_4096_full_adaptive_flex_seed0.yaml"),
    ("FlexAttention", "shallow_freeze", "aah", "experiments/backend_realized_local_attention/FlexAttention/aah_modified/configs/backend_4096_shallow_freeze_flex_seed0.yaml"),
    ("FlexAttention", "deep_practical_reuse", "aah", "experiments/backend_realized_local_attention/FlexAttention/aah_modified/configs/backend_4096_deep_practical_reuse_flex_seed0.yaml"),
    ("FlashAttention", "pure", "pure", "experiments/backend_realized_local_attention/FlashAttention/pure/configs/backend_4096_pure_flash_seed0.yaml"),
    ("FlashAttention", "grouping_off", "aah", "experiments/backend_realized_local_attention/FlashAttention/aah_modified/configs/backend_4096_grouping_off_flash_seed0.yaml"),
    ("FlashAttention", "full_adaptive", "aah", "experiments/backend_realized_local_attention/FlashAttention/aah_modified/configs/backend_4096_full_adaptive_flash_seed0.yaml"),
    ("FlashAttention", "shallow_freeze", "aah", "experiments/backend_realized_local_attention/FlashAttention/aah_modified/configs/backend_4096_shallow_freeze_flash_seed0.yaml"),
    ("FlashAttention", "deep_practical_reuse", "aah", "experiments/backend_realized_local_attention/FlashAttention/aah_modified/configs/backend_4096_deep_practical_reuse_flash_seed0.yaml"),
]

DENSE_MEMORY_SANITY_CONFIG = (
    "experiments/backend_realized_local_attention/"
    "DenseMasked/memory_sanity/configs/backend_4096_dense_memory_sanity_seed0.yaml"
)


def run(cmd, dry_run=False, continue_on_error=False):
    print("+ " + " ".join(str(x) for x in cmd), flush=True)
    if dry_run:
        return 0
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0 and not continue_on_error:
        raise SystemExit(proc.returncode)
    return proc.returncode


def checkpoint_path(config_path):
    import yaml

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    out_dir = Path(cfg["experiment"].get("out_dir", "experiments"))
    return out_dir / f"{cfg['experiment']['name']}.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="paper_results/backend_4096_realized_attention_ncu")
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--delete-checkpoints", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--skip-dense-memory-sanity",
        action="store_true",
        help="Skip the final dense-masked memory sanity run.",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root)
    profile_dir = run_root / "gpu_flops_profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    preflight_json = run_root / "ncu_preflight.json"

    preflight = [
        sys.executable,
        "-m",
        "experiments.backend_realized_local_attention._common.profile_gpu_flops_ncu",
        "--preflight",
        "--ncu",
        args.ncu,
        "--output",
        str(preflight_json),
    ]
    rc = run(preflight, dry_run=args.dry_run, continue_on_error=True)
    if rc != 0:
        print("NCU preflight failed; stopping before training/reruns.", flush=True)
        raise SystemExit(rc)

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

        out = profile_dir / f"{backend.lower()}_{method}_gpu_flops_profile.json"
        cmd = [
            sys.executable,
            "-m",
            "experiments.backend_realized_local_attention._common.profile_gpu_flops_ncu",
            "--module",
            module,
            "--config",
            cfg,
            "--ncu",
            args.ncu,
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
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
        if args.delete_checkpoints and ckpt.exists():
            ckpt.unlink()

    if not args.profile_only and not args.skip_dense_memory_sanity:
        print("Running final dense-masked memory sanity run.", flush=True)
        run(
            [
                sys.executable,
                "-m",
                "experiments.backend_realized_local_attention._common.run_train",
                "--module",
                "pure",
                "--config",
                DENSE_MEMORY_SANITY_CONFIG,
            ],
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
        )
        ckpt = checkpoint_path(DENSE_MEMORY_SANITY_CONFIG)
        if args.delete_checkpoints and ckpt.exists():
            ckpt.unlink()

    print(f"NCU suite finished. Profiles: {profile_dir}", flush=True)


if __name__ == "__main__":
    main()
