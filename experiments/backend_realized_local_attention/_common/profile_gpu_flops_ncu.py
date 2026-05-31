"""Profile backend-local attention with Nsight Compute GPU FLOP counters.

This script is intentionally separate from ``profile_flops_ratio.py``. The old
script uses PyTorch profiler FLOP annotations and formula fallbacks; this script
only reports paper FLOPs ratios when Nsight Compute hardware/derived FLOP
counters are available.
"""

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from experiments.backend_realized_local_attention._common.profile_flops_ratio import (
    autocast_context,
    collect_backend_stats,
    get_device,
    load_config,
    load_model,
    sync,
)


DEFAULT_NCU = os.environ.get("NCU_BIN", "ncu")
TOTAL_RANGE = "aah_ncu_total_forward"

FLOP_METRIC_PATTERNS = (
    re.compile(r"flop.*(count|sum|total)", re.IGNORECASE),
    re.compile(r"derived__.*flop", re.IGNORECASE),
    re.compile(r"derived__.*sass.*op_[fdh](add|mul|fma).*", re.IGNORECASE),
    re.compile(r"derived__.*(hmma|mma|tensor).*", re.IGNORECASE),
)

COUNTER_PERMISSION_MARKERS = (
    "ERR_NVGPUCTRPERM",
    "profiling permission",
    "permission issue",
    "ERR_NVGPUCTR",
)


def run_cmd(cmd, timeout=None):
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def ncu_preflight(ncu):
    proc = run_cmd([ncu, "--query-metrics"], timeout=60)
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    permission_error = any(marker in text for marker in COUNTER_PERMISSION_MARKERS)
    return {
        "ncu_permission_ok": proc.returncode == 0 and not permission_error,
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-4000:],
        "error_kind": "ERR_NVGPUCTRPERM" if permission_error else (None if proc.returncode == 0 else "ncu_query_failed"),
    }


def discover_flop_metrics(ncu):
    proc = run_cmd([ncu, "--query-metrics"], timeout=60)
    full_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    permission_error = any(marker in full_text for marker in COUNTER_PERMISSION_MARKERS)
    preflight = {
        "ncu_permission_ok": proc.returncode == 0 and not permission_error,
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-4000:],
        "error_kind": "ERR_NVGPUCTRPERM" if permission_error else (None if proc.returncode == 0 else "ncu_query_failed"),
    }
    if not preflight["ncu_permission_ok"]:
        return [], preflight
    lines = full_text.splitlines()
    metrics = []
    for raw in lines:
        token = raw.strip().split()[0] if raw.strip() else ""
        token = token.strip(",")
        if not token or token.startswith(("#", "==")):
            continue
        if any(pattern.search(token) for pattern in FLOP_METRIC_PATTERNS):
            metrics.append(token)
    metrics = sorted(set(metrics))
    return metrics, preflight


def parse_numeric(value):
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"n/a", "nan", "inf", "-inf"}:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def parse_ncu_csv(path, metric_names):
    metric_names = set(metric_names)
    values = {name: [] for name in metric_names}
    with open(path, newline="") as f:
        rows = list(csv.reader(f))

    header = None
    for i, row in enumerate(rows):
        lowered = [c.strip().lower() for c in row]
        if "metric name" in lowered and "metric value" in lowered:
            header = {name: lowered.index(name) for name in ("metric name", "metric value")}
            data_rows = rows[i + 1 :]
            break
    else:
        data_rows = rows

    if header is not None:
        name_idx = header["metric name"]
        value_idx = header["metric value"]
        for row in data_rows:
            if len(row) <= max(name_idx, value_idx):
                continue
            name = row[name_idx].strip()
            if name in metric_names:
                val = parse_numeric(row[value_idx])
                if val is not None:
                    values[name].append(val)
    else:
        for row in data_rows:
            for name in metric_names:
                if name not in row:
                    continue
                idx = row.index(name)
                for candidate in row[idx + 1 :]:
                    val = parse_numeric(candidate)
                    if val is not None:
                        values[name].append(val)
                        break

    totals = {name: float(sum(vals)) for name, vals in values.items() if vals}
    return totals


def write_json(path, payload):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def collect_model_metadata(config_path, module_key, device_name, checkpoint=None, warmup=1):
    cfg = load_config(config_path)
    train = cfg.get("train", {})
    device = get_device(device_name or train.get("device", "cuda"))
    precision = train.get("precision", "bf16")
    model, gpt_cfg, checkpoint_loaded, checkpoint_path = load_model(cfg, module_key, device, checkpoint)
    batch_size = int(train.get("batch_size", 1))
    seq_len = int(cfg["data"]["seq_len"])
    idx = torch.randint(0, int(gpt_cfg.vocab_size), (batch_size, seq_len), device=device)
    with torch.no_grad():
        for _ in range(max(0, int(warmup))):
            with autocast_context(device, precision):
                model(idx)
            sync(device)
    stats = collect_backend_stats(model, batch_size=batch_size, head_dim=int(gpt_cfg.n_embd) // int(gpt_cfg.n_head))
    device_name_out = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    return {
        "config_path": os.path.abspath(config_path),
        "module": module_key,
        "checkpoint_loaded": checkpoint_loaded,
        "checkpoint_path": checkpoint_path,
        "device": str(device),
        "device_name": device_name_out,
        "precision": precision,
        "batch_size": batch_size,
        "seq_len": seq_len,
        **stats,
    }


def child_forward(args):
    cfg = load_config(args.config)
    train = cfg.get("train", {})
    device = get_device(args.device or train.get("device", "cuda"))
    if device.type != "cuda":
        raise SystemExit("Nsight Compute profiling requires a CUDA device")
    precision = train.get("precision", "bf16")
    model, gpt_cfg, _, _ = load_model(cfg, args.module, device, args.checkpoint)
    batch_size = int(train.get("batch_size", 1))
    seq_len = int(cfg["data"]["seq_len"])
    torch.manual_seed(int(args.seed))
    idx = torch.randint(0, int(gpt_cfg.vocab_size), (batch_size, seq_len), device=device)

    with torch.no_grad():
        for _ in range(max(0, int(args.warmup))):
            with autocast_context(device, precision):
                model(idx)
            sync(device)

        torch.cuda.nvtx.range_push(TOTAL_RANGE)
        try:
            for _ in range(max(1, int(args.repeats))):
                with autocast_context(device, precision):
                    model(idx)
                sync(device)
        finally:
            torch.cuda.nvtx.range_pop()
    print("ncu_child_forward_done")


def run_ncu_profile(args, metrics):
    with tempfile.TemporaryDirectory(prefix="aah-ncu-") as tmp:
        raw_csv = Path(tmp) / "ncu_raw.csv"
        cmd = [
            args.ncu,
            "--target-processes",
            "all",
            "--csv",
            "--page",
            "raw",
            "--nvtx",
            "--nvtx-include",
            TOTAL_RANGE,
            "--metrics",
            ",".join(metrics),
            "--log-file",
            str(raw_csv),
            sys.executable,
            "-m",
            "experiments.backend_realized_local_attention._common.profile_gpu_flops_ncu",
            "--child-forward",
            "--config",
            args.config,
            "--module",
            args.module,
            "--device",
            args.device or "cuda",
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
            "--seed",
            str(args.seed),
        ]
        if args.checkpoint:
            cmd.extend(["--checkpoint", args.checkpoint])
        proc = run_cmd(cmd, timeout=args.timeout)
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0 or any(marker in text for marker in COUNTER_PERMISSION_MARKERS):
            return {
                "ok": False,
                "ncu_command": cmd,
                "ncu_stdout_tail": (proc.stdout or "")[-4000:],
                "ncu_stderr_tail": (proc.stderr or "")[-4000:],
                "ncu_error_kind": "ERR_NVGPUCTRPERM" if "ERR_NVGPUCTRPERM" in text else "ncu_profile_failed",
            }
        if not raw_csv.exists():
            return {
                "ok": False,
                "ncu_command": cmd,
                "ncu_stdout_tail": (proc.stdout or "")[-4000:],
                "ncu_stderr_tail": (proc.stderr or "")[-4000:],
                "ncu_error_kind": "ncu_csv_missing",
            }
        values = parse_ncu_csv(raw_csv, metrics)
        return {
            "ok": bool(values),
            "ncu_command": cmd,
            "ncu_metric_values": values,
            "ncu_stdout_tail": (proc.stdout or "")[-2000:],
            "ncu_stderr_tail": (proc.stderr or "")[-2000:],
            "ncu_error_kind": None if values else "ncu_metric_parse_empty",
        }


def add_baseline_ratios(result, baseline_json):
    if not baseline_json:
        result["gpu_flops_total_ratio_ncu"] = 1.0 if result.get("gpu_flops_total") else None
        result["gpu_flops_attention_ratio_ncu"] = None
        return result
    with open(baseline_json, "r") as f:
        baseline = json.load(f)
    denom = float(baseline.get("gpu_flops_total") or 0.0)
    result["gpu_flops_total_ratio_ncu"] = (float(result.get("gpu_flops_total") or 0.0) / denom) if denom > 0 else None
    result["gpu_flops_attention_ratio_ncu"] = None
    result["baseline_json"] = os.path.abspath(baseline_json)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--module", choices=("pure", "aah"))
    parser.add_argument("--output", required=False)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ncu", default=DEFAULT_NCU)
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--baseline-json")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--child-forward", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    if args.child_forward:
        child_forward(args)
        return

    if args.preflight:
        payload = ncu_preflight(args.ncu)
        payload["ncu"] = args.ncu
        if args.output:
            write_json(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit(0 if payload["ncu_permission_ok"] else 2)

    if not args.config or not args.module or not args.output:
        parser.error("--config, --module, and --output are required unless --preflight or --child-forward is used")

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metadata = collect_model_metadata(args.config, args.module, args.device, args.checkpoint, warmup=1)
    metrics = list(args.metric)
    preflight = None
    if not metrics:
        metrics, preflight = discover_flop_metrics(args.ncu)
    else:
        preflight = ncu_preflight(args.ncu)

    result = {
        **metadata,
        "started_at_utc": started_at,
        "ncu": args.ncu,
        "ncu_permission_ok": bool(preflight and preflight["ncu_permission_ok"]),
        "ncu_preflight": preflight,
        "ncu_metrics_used": metrics,
        "gpu_flops_total": None,
        "gpu_flops_attention_or_forward": "forward_total",
        "gpu_flops_attention": None,
        "gpu_flops_total_ratio_ncu": None,
        "gpu_flops_attention_ratio_ncu": None,
        "torch_profiler_total_flops_ratio": None,
        "paper_metric_source": "Nsight Compute hardware/derived FLOP counters only",
    }

    if not result["ncu_permission_ok"]:
        result["ncu_error_kind"] = preflight.get("error_kind") if preflight else "ncu_query_failed"
        write_json(args.output, result)
        print(f"ncu_profile_failed {result['ncu_error_kind']} wrote {args.output}")
        raise SystemExit(2)

    if not metrics:
        result["ncu_error_kind"] = "no_flop_counter_metrics_discovered"
        write_json(args.output, result)
        print(f"ncu_profile_failed no FLOP counter metrics discovered wrote {args.output}")
        raise SystemExit(3)

    profile = run_ncu_profile(args, metrics)
    result.update(profile)
    if profile.get("ok"):
        metric_values = profile.get("ncu_metric_values", {})
        result["gpu_flops_total"] = float(sum(float(v) for v in metric_values.values()))
        result = add_baseline_ratios(result, args.baseline_json)
        result["ncu_error_kind"] = None
        write_json(args.output, result)
        print(
            "wrote_gpu_flops_profile "
            f"{args.output} gpu_flops_total={result['gpu_flops_total']} "
            f"gpu_flops_total_ratio_ncu={result['gpu_flops_total_ratio_ncu']}"
        )
        return

    result["ncu_error_kind"] = profile.get("ncu_error_kind", "ncu_profile_failed")
    write_json(args.output, result)
    print(f"ncu_profile_failed {result['ncu_error_kind']} wrote {args.output}")
    raise SystemExit(4)


if __name__ == "__main__":
    main()
