#!/usr/bin/env python3
"""Upload Qwen3 AAH paper logs and benchmark summaries to Weights & Biases.

This is intentionally separate from the experiment orchestrator so it can be
started after a long run is already in progress.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import time
from typing import Dict, Iterable, List


DEFAULT_PERSIST_ROOT = "/home/featurize/work/ENA-AAH-v3-persistent/AAH-qwen3-4b-paper"
DEFAULT_PROJECT = "ENA-AAH"

ADAPT_RE = re.compile(r"step\s+(\d+)\s+\|\s+loss\s+([0-9.eE+-]+)\s+\|\s+mean_ACR\s+([0-9.eE+-]+)")


def load_json(path: str, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def file_sig(path: str) -> str:
    st = os.stat(path)
    return f"{int(st.st_mtime_ns)}:{st.st_size}"


def method_from_adapt_log(path: str) -> str:
    name = os.path.basename(path)
    prefix = "adapt_"
    suffix = ".log"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix) : -len(suffix)]
    return os.path.splitext(name)[0]


def regime_from_method(method: str) -> str:
    prefix = "qwen3_4b_"
    return method[len(prefix) :] if method.startswith(prefix) else method


def parse_adapt_log(path: str) -> List[Dict[str, float]]:
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            m = ADAPT_RE.search(line)
            if not m:
                continue
            rows.append({"step": int(m.group(1)), "loss": float(m.group(2)), "mean_ACR": float(m.group(3))})
    return rows


def csv_rows(path: str) -> Iterable[Dict[str, str]]:
    with open(path, newline="") as f:
        yield from csv.DictReader(f)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def init_wandb(project: str, name: str, run_id: str, config: Dict[str, object]):
    import wandb

    return wandb.init(
        project=project,
        name=name,
        id=run_id,
        resume="allow",
        config=config,
        reinit=True,
    )


def upload_adapt_logs(args, state: Dict[str, str]) -> int:
    import wandb

    uploaded = 0
    for path in sorted(glob.glob(os.path.join(args.persist_root, "logs", "adapt_*.log"))):
        sig = file_sig(path)
        key = f"adapt:{path}"
        if state.get(key) == sig:
            continue
        rows = parse_adapt_log(path)
        if not rows:
            continue
        method = method_from_adapt_log(path)
        regime = regime_from_method(method)
        run = init_wandb(
            args.project,
            f"qwen3-4b-aah-adapt-{regime}",
            f"qwen3_4b_aah_adapt_{regime}",
            {"model": "Qwen/Qwen3-4B-Base", "method": method, "regime": regime, "stage": "adapt"},
        )
        for row in rows:
            wandb.log(
                {
                    "adapt/loss": row["loss"],
                    "adapt/mean_ACR": row["mean_ACR"],
                    "step": row["step"],
                },
                step=row["step"],
            )
        adapter = os.path.join(args.persist_root, "adapters", f"{regime}_aah_adapter.pt")
        meta = f"{adapter}.json"
        if args.upload_artifacts and os.path.exists(adapter):
            art = wandb.Artifact(f"qwen3-4b-aah-adapter-{regime}", type="model")
            art.add_file(adapter)
            if os.path.exists(meta):
                art.add_file(meta)
            run.log_artifact(art)
        run.finish()
        state[key] = sig
        uploaded += 1
    return uploaded


def upload_smoke_summaries(args, state: Dict[str, str]) -> int:
    import wandb

    uploaded = 0
    for path in sorted(glob.glob(os.path.join(args.persist_root, "summaries", "*_smoke_summary.json"))):
        sig = file_sig(path)
        key = f"smoke:{path}"
        if state.get(key) == sig:
            continue
        data = load_json(path, {})
        method = os.path.basename(path).replace("_smoke_summary.json", "")
        regime = data.get("regime") or regime_from_method(method)
        run = init_wandb(
            args.project,
            f"qwen3-4b-aah-smoke-{regime}",
            f"qwen3_4b_aah_smoke_{regime}",
            {"model": "Qwen/Qwen3-4B-Base", "method": method, "regime": regime, "stage": "smoke"},
        )
        metrics = {}
        for k, v in data.items():
            if isinstance(v, (int, float)):
                metrics[f"smoke/{k}"] = v
        if metrics:
            wandb.log(metrics)
        if args.upload_artifacts:
            art = wandb.Artifact(f"qwen3-4b-aah-smoke-{regime}", type="diagnostic")
            art.add_file(path)
            for diag in glob.glob(os.path.join(args.persist_root, "diagnostics", f"{method}_*.csv")):
                art.add_file(diag)
            run.log_artifact(art)
        run.finish()
        state[key] = sig
        uploaded += 1
    return uploaded


def upload_benchmarks(args, state: Dict[str, str]) -> int:
    import wandb

    uploaded = 0
    pattern = os.path.join(args.persist_root, "benchmarks", "**", "*_benchmark_results_by_task.csv")
    for path in sorted(glob.glob(pattern, recursive=True)):
        sig = file_sig(path)
        key = f"benchmark:{path}"
        if state.get(key) == sig:
            continue
        rows = list(csv_rows(path))
        if not rows:
            continue
        method = rows[0].get("method") or os.path.basename(path).replace("_benchmark_results_by_task.csv", "")
        stage = "full" if f"{os.sep}full{os.sep}" in path else "sanity"
        run = init_wandb(
            args.project,
            f"qwen3-4b-aah-{stage}-{method}",
            f"qwen3_4b_aah_{stage}_{method}",
            {"model": "Qwen/Qwen3-4B-Base", "method": method, "stage": f"benchmark_{stage}"},
        )
        scores_by_group: Dict[str, List[float]] = {}
        metrics = {}
        for row in rows:
            task = row.get("task", "")
            metric = row.get("metric", "score")
            score = safe_float(row.get("score"))
            if score is None:
                continue
            metrics[f"benchmark/{task}/{metric}"] = score
            group = row.get("task_group", "other")
            scores_by_group.setdefault(group, []).append(score)
            scores_by_group.setdefault("overall", []).append(score)
            n_examples = safe_float(row.get("n_examples"))
            if n_examples is not None:
                metrics[f"benchmark/{task}/n_examples"] = n_examples
        for group, vals in scores_by_group.items():
            if vals:
                metrics[f"benchmark/{group}/average"] = sum(vals) / len(vals)
        metrics["benchmark/n_tasks"] = len(rows)
        wandb.log(metrics)
        if args.upload_artifacts:
            art = wandb.Artifact(f"qwen3-4b-aah-{stage}-{method}-results", type="evaluation")
            art.add_file(path)
            stem = path.replace("_benchmark_results_by_task.csv", "")
            for suffix in ["_benchmark_results_raw.jsonl", "_benchmark_failures.json", "_benchmark_missing_tasks.json"]:
                other = stem + suffix
                if os.path.exists(other):
                    art.add_file(other)
            run.log_artifact(art)
        run.finish()
        state[key] = sig
        uploaded += 1
    return uploaded


def upload_aggregate(args, state: Dict[str, str]) -> int:
    import wandb

    files = [
        "benchmark_results_raw.jsonl",
        "benchmark_results_by_task.csv",
        "benchmark_results_by_model.csv",
        "benchmark_paper_table.md",
        "benchmark_paper_table.tex",
        "benchmark_failures.json",
        "benchmark_missing_tasks.json",
    ]
    paths = [os.path.join(args.persist_root, "benchmarks", name) for name in files]
    existing = [p for p in paths if os.path.exists(p)]
    if not existing:
        return 0
    sig = "|".join(f"{os.path.basename(p)}:{file_sig(p)}" for p in existing)
    key = "aggregate:benchmark_tables"
    if state.get(key) == sig:
        return 0
    run = init_wandb(
        args.project,
        "qwen3-4b-aah-paper-aggregate",
        "qwen3_4b_aah_paper_aggregate",
        {"model": "Qwen/Qwen3-4B-Base", "stage": "benchmark_aggregate"},
    )
    model_csv = os.path.join(args.persist_root, "benchmarks", "benchmark_results_by_model.csv")
    if os.path.exists(model_csv):
        for row in csv_rows(model_csv):
            method = row.get("method", "")
            for key_name in ["core_average", "math_code_average", "chinese_average", "overall_average"]:
                val = safe_float(row.get(key_name))
                if val is not None:
                    import wandb

                    wandb.log({f"aggregate/{method}/{key_name}": val})
    if args.upload_artifacts:
        art = wandb.Artifact("qwen3-4b-aah-paper-benchmark-tables", type="evaluation")
        for path in existing:
            art.add_file(path)
        run.log_artifact(art)
    run.finish()
    state[key] = sig
    return 1


def sync_once(args) -> int:
    state_path = os.path.join(args.persist_root, "wandb_sync_state.json")
    state = load_json(state_path, {})
    uploaded = 0
    uploaded += upload_adapt_logs(args, state)
    uploaded += upload_smoke_summaries(args, state)
    uploaded += upload_benchmarks(args, state)
    uploaded += upload_aggregate(args, state)
    save_json(state_path, state)
    print(json.dumps({"uploaded_items": uploaded, "state_path": state_path}, sort_keys=True), flush=True)
    return uploaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist-root", default=DEFAULT_PERSIST_ROOT)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT))
    parser.add_argument("--interval-s", type=int, default=600)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--upload-artifacts", action="store_true")
    args = parser.parse_args()
    if args.watch:
        while True:
            try:
                sync_once(args)
            except Exception as exc:
                print(json.dumps({"error": repr(exc)}, sort_keys=True), flush=True)
            time.sleep(max(30, args.interval_s))
    else:
        sync_once(args)


if __name__ == "__main__":
    main()
