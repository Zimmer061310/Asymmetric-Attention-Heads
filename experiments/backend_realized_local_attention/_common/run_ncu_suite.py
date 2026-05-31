"""Run/profile the 4096 backend suite with Nsight Compute FLOP counters."""

import argparse
import csv
import json
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


def metrics_csv_path(config_path):
    import yaml

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    out_dir = Path(cfg["experiment"].get("out_dir", "experiments"))
    safe_name = cfg["experiment"]["name"].replace("_", "-")
    variant = cfg["experiment"].get("variant", "")
    return out_dir / f"{safe_name}_{variant}.csv"


def latest_metrics_row(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    for row in reversed(rows):
        if row.get("val_loss"):
            return row
    return rows[-1]


def pick(row, *names):
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return ""


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def append_status(run_root, payload):
    path = Path(run_root) / "status.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def log_profile_summary_to_wandb(args, backend, method, cfg, profile_json, baseline_json):
    if args.no_wandb_profile_summary or args.dry_run:
        return
    try:
        import wandb
    except Exception as exc:
        if args.require_wandb:
            raise RuntimeError(f"wandb import failed: {exc}") from exc
        print(f"wandb profile summary skipped: {exc}", flush=True)
        return

    profile = load_json(profile_json)
    metrics = latest_metrics_row(metrics_csv_path(cfg))
    baseline_total = None
    if baseline_json:
        baseline_total = as_float(load_json(baseline_json).get("gpu_flops_total"))
    gpu_total = as_float(profile.get("gpu_flops_total"))
    computed_ratio = (gpu_total / baseline_total) if gpu_total is not None and baseline_total else None

    payload = {
        "backend_ncu/val_loss": as_float(pick(metrics, "val_loss")),
        "backend_ncu/ACR": as_float(pick(metrics, "effective_ACR", "attn_ratio")),
        "backend_ncu/EAR": as_float(pick(metrics, "backend_realized_ACR_est")),
        "backend_ncu/tok_s": as_float(pick(metrics, "tok_s")),
        "backend_ncu/memory_gpu_alloc_max_mb": as_float(pick(metrics, "gpu_alloc_max_mb")),
        "backend_ncu/gpu_flops_total": gpu_total,
        "backend_ncu/gpu_flops_total_ratio_ncu": as_float(profile.get("gpu_flops_total_ratio_ncu")),
        "backend_ncu/computed_gpu_flops_total_ratio": computed_ratio,
        "backend_ncu/ncu_permission_ok": 1 if profile.get("ncu_permission_ok") else 0,
        "backend_ncu/backend": backend,
        "backend_ncu/method": method,
        "backend_ncu/profile_json": str(profile_json),
    }
    run = wandb.init(
        project=args.wandb_project,
        name=f"backend-4096-ncu-{backend.lower()}-{method}-seed0",
        job_type="backend-ncu-summary",
        config={"backend": backend, "method": method, "config": cfg},
        reinit=True,
    )
    try:
        run.log(payload)
    finally:
        run.finish()


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
    parser.add_argument("--wandb-project", default="ENA-AAH")
    parser.add_argument("--require-wandb", action="store_true")
    parser.add_argument("--no-wandb-profile-summary", action="store_true")
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
        row_status = {"backend": backend, "method": method, "config": cfg, "train_rc": None, "profile_rc": None}
        if not args.profile_only:
            train_rc = run(
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
            row_status["train_rc"] = train_rc
            if train_rc != 0:
                row_status["status"] = "train_failed"
                append_status(run_root, row_status)
                if args.continue_on_error:
                    continue

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
        baseline_json = baselines.get(backend)
        if method != "pure" and baseline_json:
            cmd.extend(["--baseline-json", str(baseline_json)])
        profile_rc = run(cmd, dry_run=args.dry_run, continue_on_error=args.continue_on_error)
        row_status["profile_rc"] = profile_rc
        row_status["profile_json"] = str(out)
        if profile_rc != 0:
            row_status["status"] = "profile_failed"
            append_status(run_root, row_status)
            if args.continue_on_error:
                continue
        if out.exists():
            log_profile_summary_to_wandb(args, backend, method, cfg, out, baseline_json)
        if method == "pure":
            baselines[backend] = out
        if args.delete_checkpoints and ckpt.exists():
            ckpt.unlink()
        row_status["status"] = "ok"
        append_status(run_root, row_status)

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

    summary_csv = run_root / "backend_4096_ncu_summary.csv"
    summary_md = run_root / "backend_4096_ncu_summary.md"
    run(
        [
            sys.executable,
            "-m",
            "experiments.backend_realized_local_attention._common.summarize_ncu_results",
            "--profile-dir",
            str(profile_dir),
            "--output-csv",
            str(summary_csv),
            "--output-md",
            str(summary_md),
        ],
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
    )
    print(f"NCU suite finished. Profiles: {profile_dir}", flush=True)


if __name__ == "__main__":
    main()
