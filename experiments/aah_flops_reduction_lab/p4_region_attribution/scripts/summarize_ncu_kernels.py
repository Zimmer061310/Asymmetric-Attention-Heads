"""Summarize Nsight Compute raw CSV FLOP metrics by kernel name."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_float(value: str) -> float:
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"n/a", "nan"}:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def summarize(raw_csv: Path) -> dict:
    rows = list(csv.reader(raw_csv.open(newline="")))
    metric_cols = []
    kernel_idx = None
    header = None
    data_rows = []
    for i, row in enumerate(rows):
        lowered = [c.strip().lower() for c in row]
        metric_cols = [idx for idx, name in enumerate(row) if "op_" in name or "ops_path" in name]
        if metric_cols:
            header = row
            for candidate in ("kernel name", "name", "kernel"):
                if candidate in lowered:
                    kernel_idx = lowered.index(candidate)
                    break
            data_rows = rows[i + 1 :]
            break
    if header is None:
        return {"raw_csv": str(raw_csv), "total_flops": 0.0, "kernels": []}

    grouped = {}
    for row in data_rows:
        if not row or row[0] in {"ID", ""}:
            continue
        kernel = row[kernel_idx].strip() if kernel_idx is not None and len(row) > kernel_idx else "unknown"
        flops = 0.0
        metrics = {}
        for idx in metric_cols:
            if len(row) <= idx:
                continue
            val = parse_float(row[idx])
            if val:
                metrics[header[idx]] = metrics.get(header[idx], 0.0) + val
                flops += val
        if flops <= 0.0:
            continue
        entry = grouped.setdefault(kernel, {"kernel": kernel, "flops": 0.0, "count": 0, "metrics": {}})
        entry["flops"] += flops
        entry["count"] += 1
        for name, val in metrics.items():
            entry["metrics"][name] = entry["metrics"].get(name, 0.0) + val

    kernels = sorted(grouped.values(), key=lambda x: x["flops"], reverse=True)
    total = float(sum(k["flops"] for k in kernels))
    for k in kernels:
        k["share"] = (float(k["flops"]) / total) if total > 0 else 0.0
    return {"raw_csv": str(raw_csv), "total_flops": total, "kernels": kernels}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    summary = summarize(Path(args.raw_csv))
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True))

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["kernel", "count", "flops", "share"], lineterminator="\n")
        writer.writeheader()
        for row in summary["kernels"]:
            writer.writerow({k: row[k] for k in ("kernel", "count", "flops", "share")})


if __name__ == "__main__":
    main()
