"""Summarize backend 4096 training CSVs and Nsight FLOP profiles."""

import argparse
import csv
import json
from pathlib import Path


ROWS = [
    ("FlexAttention", "pure", "experiments/backend_realized_local_attention/FlexAttention/pure/results/backend-4096-pure-flex-seed0_metrics.csv"),
    ("FlexAttention", "grouping_off", "experiments/backend_realized_local_attention/FlexAttention/aah_modified/results/backend-4096-grouping-off-flex-seed0_metrics.csv"),
    ("FlexAttention", "full_adaptive", "experiments/backend_realized_local_attention/FlexAttention/aah_modified/results/backend-4096-full-adaptive-flex-seed0_metrics.csv"),
    ("FlexAttention", "shallow_freeze", "experiments/backend_realized_local_attention/FlexAttention/aah_modified/results/backend-4096-shallow-freeze-flex-seed0_metrics.csv"),
    ("FlexAttention", "deep_practical_reuse", "experiments/backend_realized_local_attention/FlexAttention/aah_modified/results/backend-4096-deep-practical-reuse-flex-seed0_metrics.csv"),
    ("FlashAttention", "pure", "experiments/backend_realized_local_attention/FlashAttention/pure/results/backend-4096-pure-flash-seed0_metrics.csv"),
    ("FlashAttention", "grouping_off", "experiments/backend_realized_local_attention/FlashAttention/aah_modified/results/backend-4096-grouping-off-flash-seed0_metrics.csv"),
    ("FlashAttention", "full_adaptive", "experiments/backend_realized_local_attention/FlashAttention/aah_modified/results/backend-4096-full-adaptive-flash-seed0_metrics.csv"),
    ("FlashAttention", "shallow_freeze", "experiments/backend_realized_local_attention/FlashAttention/aah_modified/results/backend-4096-shallow-freeze-flash-seed0_metrics.csv"),
    ("FlashAttention", "deep_practical_reuse", "experiments/backend_realized_local_attention/FlashAttention/aah_modified/results/backend-4096-deep-practical-reuse-flash-seed0_metrics.csv"),
]


def latest_csv_row(path):
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


def load_profile(profile_dir, backend, method):
    path = Path(profile_dir) / f"{backend.lower()}_{method}_gpu_flops_profile.json"
    if not path.exists():
        return {}, path
    with open(path, "r") as f:
        return json.load(f), path


def pick(row, *names):
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    out_rows = []
    for backend, method, metrics_csv in ROWS:
        row = latest_csv_row(metrics_csv)
        profile, profile_path = load_profile(args.profile_dir, backend, method)
        out_rows.append(
            {
                "backend": backend,
                "method": method,
                "val_loss": pick(row, "val_loss"),
                "ACR": pick(row, "effective_ACR", "attn_ratio"),
                "EAR": pick(row, "backend_realized_ACR_est"),
                "tok_s": pick(row, "tok_s"),
                "peak_memory_mb": pick(row, "gpu_reserved_max_mb", "gpu_alloc_max_mb"),
                "gpu_flops_total": profile.get("gpu_flops_total", ""),
                "gpu_flops_total_ratio_ncu": profile.get("gpu_flops_total_ratio_ncu", ""),
                "gpu_flops_attention_ratio_ncu": profile.get("gpu_flops_attention_ratio_ncu", ""),
                "ncu_permission_ok": profile.get("ncu_permission_ok", ""),
                "ncu_error_kind": profile.get("ncu_error_kind", ""),
                "ncu_metrics_used": ";".join(profile.get("ncu_metrics_used", []) or []),
                "profile_json": str(profile_path),
            }
        )

    fields = list(out_rows[0].keys())
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    with open(args.output_md, "w") as f:
        f.write("# Backend 4096 Nsight FLOPs Summary\n\n")
        f.write("| Backend | Method | Val loss | ACR | EAR | Tok/s | Peak mem MB | GPU FLOPs ratio NCU | NCU status |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in out_rows:
            status = "ok" if row["ncu_permission_ok"] is True else (row["ncu_error_kind"] or "missing")
            f.write(
                f"| {row['backend']} | `{row['method']}` | {row['val_loss']} | {row['ACR']} | "
                f"{row['EAR']} | {row['tok_s']} | {row['peak_memory_mb']} | "
                f"{row['gpu_flops_total_ratio_ncu']} | {status} |\n"
            )
        f.write("\n`gpu_flops_total_ratio_ncu` is the only paper FLOPs/FLOPs field here. It is populated only from Nsight Compute hardware/derived FLOP counters.\n")

    print(f"wrote {args.output_csv} and {args.output_md}")


if __name__ == "__main__":
    main()
