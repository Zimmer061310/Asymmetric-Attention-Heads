#!/usr/bin/env python3
import ast
import csv
import glob
import os
from pathlib import Path


ROOT = Path("experiments/aah_quality_structure_lab/results/phase1")
OUT = Path("paper_results/aah_quality_structure_lab")


def last_float(row, key):
    val = row.get(key, "")
    if val in ("", None):
        return None
    try:
        return float(val)
    except ValueError:
        return None


def final_rows(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    train_rows = [r for r in rows if r.get("train_loss")]
    val_rows = [r for r in rows if r.get("val_loss")]
    return (train_rows[-1] if train_rows else {}), (val_rows[-1] if val_rows else {})


def parse_branch_usage(text):
    if not text:
        return {}
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return {}
    return {str(k): float(v) for k, v in parsed.items()} if isinstance(parsed, dict) else {}


def summarize():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(glob.glob(str(ROOT / "quality-4096-phase1-*.csv"))):
        train, val = final_rows(path)
        name = os.path.basename(path).replace(".csv", "")
        if name.endswith("_run"):
            name = name[:-4]
        branch_usage = parse_branch_usage(train.get("branch_usage_freq", ""))
        rows.append({
            "run": name,
            "final_step": train.get("step") or val.get("step"),
            "train_loss": train.get("train_loss", ""),
            "val_loss": val.get("val_loss", ""),
            "val_ppl": val.get("val_ppl", ""),
            "tok_s": train.get("tok_s", ""),
            "gpu_alloc_max_mb": train.get("gpu_alloc_max_mb", ""),
            "avg_window": train.get("avg_window", ""),
            "resolution_std": train.get("resolution_std", ""),
            "branch_usage_freq": branch_usage,
            "window_ablation_mode_freq": train.get("window_ablation_mode_freq", ""),
            "window_ablation_changed_frac": train.get("window_ablation_changed_frac", ""),
            "csv": path,
        })

    csv_path = OUT / "phase1_quality_summary.csv"
    fields = [
        "run",
        "final_step",
        "train_loss",
        "val_loss",
        "val_ppl",
        "tok_s",
        "gpu_alloc_max_mb",
        "avg_window",
        "resolution_std",
        "branch_usage_freq",
        "window_ablation_mode_freq",
        "window_ablation_changed_frac",
        "csv",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    ranked = sorted(
        rows,
        key=lambda r: last_float(r, "val_loss") if last_float(r, "val_loss") is not None else float("inf"),
    )
    md_path = OUT / "phase1_quality_summary.md"
    with open(md_path, "w") as f:
        f.write("# AAH Quality / Structure Phase 1 Summary\n\n")
        f.write("ACR/EAR and analytic FLOPs fields are routing diagnostics only in this lab.\n\n")
        f.write("| Rank | Run | Val loss | Val ppl | Train loss | Tok/s | GPU alloc max MB | Window ablation |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|---|\n")
        for i, row in enumerate(ranked, start=1):
            f.write(
                f"| {i} | `{row['run']}` | {row['val_loss']} | {row['val_ppl']} | "
                f"{row['train_loss']} | {row['tok_s']} | {row['gpu_alloc_max_mb']} | "
                f"{row['window_ablation_mode_freq']} |\n"
            )
        f.write("\n## Promotion Rule\n\n")
        f.write("Promote only 3-4 rows to 5000 steps: best AAH reference, any decisive random/shuffle control, best fixed control, and best optimization variant.\n")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    summarize()
