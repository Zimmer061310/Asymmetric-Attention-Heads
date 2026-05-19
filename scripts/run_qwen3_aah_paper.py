#!/usr/bin/env python3
"""Featurize orchestration for Qwen3-4B-Base AAH paper experiments."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUN_ROOT = "/home/featurize/data/AAH-qwen3-4b-paper"
PERSIST_ROOT = "/home/featurize/work/ENA-AAH-v3-persistent/AAH-qwen3-4b-paper"
RESULT_REPO_DIR = os.path.join(PROJECT_ROOT, "paper_results", "qwen3_4b_aah")
MODEL_NAME = "Qwen/Qwen3-4B-Base"

REGIMES = [
    ("qwen3_4b_full_attention_baseline", "full_attention_baseline"),
    ("qwen3_4b_grouping_off", "grouping_off"),
    ("qwen3_4b_full_adaptive", "full_adaptive"),
    ("qwen3_4b_shallow_freeze", "shallow_freeze"),
    ("qwen3_4b_deep_practical_reuse", "deep_practical_reuse"),
]

TASKS = [
    "mmlu",
    "mmlu_pro",
    "gpqa_diamond",
    "bbh",
    "arc_challenge",
    "hellaswag",
    "triviaqa",
    "gsm8k",
    "mgsm",
    "math",
    "cmath",
    "humaneval",
    "mbpp",
    "cmmlu",
    "ceval",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Runner:
    def __init__(self, args):
        self.args = args
        self.run_root = os.path.abspath(args.run_root)
        self.persist_root = os.path.abspath(args.persist_root)
        self.log_dir = os.path.join(self.run_root, "logs")
        self.scratch_dir = os.path.join(self.run_root, "scratch")
        self.adapters_dir = os.path.join(self.persist_root, "adapters")
        self.summaries_dir = os.path.join(self.persist_root, "summaries")
        self.diagnostics_dir = os.path.join(self.persist_root, "diagnostics")
        self.benchmarks_dir = os.path.join(self.persist_root, "benchmarks")
        self.persist_logs_dir = os.path.join(self.persist_root, "logs")
        self.master_log = os.path.join(self.run_root, "master.log")

    def setup(self):
        for path in [
            self.run_root,
            self.log_dir,
            self.scratch_dir,
            self.adapters_dir,
            self.summaries_dir,
            self.diagnostics_dir,
            self.benchmarks_dir,
            self.persist_logs_dir,
            RESULT_REPO_DIR,
        ]:
            os.makedirs(path, exist_ok=True)

    def log(self, message: str) -> None:
        line = f"[{now()}] {message}"
        print(line, flush=True)
        os.makedirs(os.path.dirname(self.master_log), exist_ok=True)
        with open(self.master_log, "a") as f:
            f.write(line + "\n")

    def run_cmd(self, cmd, log_name, check=True):
        log_path = os.path.join(self.log_dir, log_name)
        self.log(f"run command={' '.join(cmd)} log={log_path}")
        start = time.time()
        env = os.environ.copy()
        env.setdefault("HF_HOME", "/home/featurize/work/hf-cache")
        env.setdefault("TRANSFORMERS_CACHE", "/home/featurize/work/hf-cache")
        env.setdefault("HF_DATASETS_CACHE", "/home/featurize/work/hf-cache/datasets")
        with open(log_path, "w") as f:
            proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=f, stderr=subprocess.STDOUT, text=True, env=env)
            rc = proc.wait()
        elapsed = time.time() - start
        self.log(f"done rc={rc} runtime_s={elapsed:.1f} log={log_path}")
        shutil.copy2(log_path, os.path.join(self.persist_logs_dir, os.path.basename(log_path)))
        if check and rc != 0:
            raise RuntimeError(f"Command failed rc={rc}: {' '.join(cmd)}")
        return rc

    def install_deps(self):
        if self.args.skip_pip_install:
            return
        packages = [
            "transformers>=4.53.0",
            "accelerate",
            "datasets",
            "tokenizers",
            "sentencepiece",
            "protobuf",
            "numpy",
            "scipy",
            "pandas",
            "pyyaml",
            "tqdm",
            "wandb",
            "lm-eval",
        ]
        self.run_cmd([sys.executable, "-m", "pip", "install", "--user", *packages], "pip_install_qwen3.log", check=False)

    def adapter_path(self, regime: str) -> str:
        return os.path.join(self.adapters_dir, f"{regime}_aah_adapter.pt")

    def smoke_summary_path(self, method: str) -> str:
        return os.path.join(self.summaries_dir, f"{method}_smoke_summary.json")

    def run_smoke(self, method: str, regime: str):
        if self.args.resume and os.path.exists(self.smoke_summary_path(method)):
            self.log(f"skip_smoke existing method={method}")
            return
        out_dir = os.path.join(self.scratch_dir, method, "smoke")
        cmd = [
            sys.executable,
            "scripts/qwen3_aah_paper.py",
            "smoke",
            "--model",
            self.args.model,
            "--regime",
            regime,
            "--out-dir",
            out_dir,
            "--seq-len",
            str(self.args.seq_len),
            "--precision",
            self.args.precision,
            "--device",
            self.args.device,
        ]
        adapter = self.adapter_path(regime)
        if regime != "full_attention_baseline" and os.path.exists(adapter):
            cmd.extend(["--adapter", adapter])
        self.run_cmd(cmd, f"smoke_{method}.log")
        for path in glob.glob(os.path.join(out_dir, "*.json")):
            shutil.copy2(path, self.smoke_summary_path(method))
        for path in glob.glob(os.path.join(out_dir, "*.csv")):
            shutil.copy2(path, os.path.join(self.diagnostics_dir, f"{method}_{os.path.basename(path)}"))

    def run_adapt(self, method: str, regime: str):
        if regime == "full_attention_baseline":
            return
        adapter = self.adapter_path(regime)
        if self.args.resume and os.path.exists(adapter):
            self.log(f"skip_adapt existing method={method} adapter={adapter}")
            return
        out_dir = os.path.join(self.scratch_dir, method, "adapt")
        cmd = [
            sys.executable,
            "scripts/qwen3_aah_paper.py",
            "adapt",
            "--model",
            self.args.model,
            "--regime",
            regime,
            "--out-dir",
            out_dir,
            "--seq-len",
            str(self.args.adapt_seq_len),
            "--precision",
            self.args.precision,
            "--device",
            self.args.device,
            "--steps",
            str(self.args.adapt_steps),
            "--batch-size",
            str(self.args.batch_size),
            "--dataset",
            self.args.dataset,
            "--dataset-config",
            self.args.dataset_config,
            "--split",
            self.args.split,
        ]
        if self.args.unfreeze_outputs:
            cmd.append("--unfreeze-outputs")
        self.run_cmd(cmd, f"adapt_{method}.log")
        produced = os.path.join(out_dir, f"{regime}_aah_adapter.pt")
        if os.path.exists(produced):
            shutil.copy2(produced, adapter)
            meta = f"{produced}.json"
            if os.path.exists(meta):
                shutil.copy2(meta, f"{adapter}.json")
        for path in glob.glob(os.path.join(out_dir, "*.csv")):
            shutil.copy2(path, os.path.join(self.diagnostics_dir, f"{method}_{os.path.basename(path)}"))

    def run_benchmark(self, method: str, regime: str, tasks, max_samples, log_suffix):
        out_dir = os.path.join(self.benchmarks_dir, log_suffix)
        marker = os.path.join(out_dir, f"{method}_benchmark_results_by_task.csv")
        if self.args.resume and os.path.exists(marker):
            self.log(f"skip_benchmark existing method={method} suffix={log_suffix}")
            return
        cmd = [
            sys.executable,
            "scripts/qwen3_aah_paper.py",
            "benchmark",
            "--model",
            self.args.model,
            "--regime",
            regime,
            "--method",
            method,
            "--out-dir",
            out_dir,
            "--seq-len",
            str(self.args.seq_len),
            "--precision",
            self.args.precision,
            "--device",
            self.args.device,
            "--max-samples-per-task",
            str(max_samples),
            "--tasks",
            ",".join(tasks),
        ]
        adapter = self.adapter_path(regime)
        if regime != "full_attention_baseline":
            cmd.extend(["--adapter", adapter])
        self.run_cmd(cmd, f"benchmark_{log_suffix}_{method}.log", check=not self.args.continue_on_benchmark_error)

    def aggregate(self):
        rows = []
        failures = []
        missing = []
        for path in glob.glob(os.path.join(self.benchmarks_dir, "full", "*_benchmark_results_raw.jsonl")):
            with open(path) as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
        for path in glob.glob(os.path.join(self.benchmarks_dir, "full", "*_benchmark_failures.json")):
            with open(path) as f:
                failures.extend(json.load(f))
        for path in glob.glob(os.path.join(self.benchmarks_dir, "full", "*_benchmark_missing_tasks.json")):
            with open(path) as f:
                missing.extend(json.load(f))

        raw_path = os.path.join(self.benchmarks_dir, "benchmark_results_raw.jsonl")
        task_path = os.path.join(self.benchmarks_dir, "benchmark_results_by_task.csv")
        model_path = os.path.join(self.benchmarks_dir, "benchmark_results_by_model.csv")
        md_path = os.path.join(self.benchmarks_dir, "benchmark_paper_table.md")
        tex_path = os.path.join(self.benchmarks_dir, "benchmark_paper_table.tex")
        with open(raw_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        fields = ["method", "checkpoint_step", "task", "metric", "score", "stderr_or_std", "n_examples", "checkpoint_sha256", "eval_runtime_s", "task_group"]
        with open(task_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fields})
        aggregates = self.aggregate_rows(rows)
        with open(model_path, "w", newline="") as f:
            fields_model = ["method", "core_average", "math_code_average", "chinese_average", "overall_average", "n_tasks"]
            writer = csv.DictWriter(f, fieldnames=fields_model)
            writer.writeheader()
            writer.writerows(aggregates)
        self.write_tables(md_path, tex_path, aggregates)
        for path in [raw_path, task_path, model_path, md_path, tex_path]:
            shutil.copy2(path, RESULT_REPO_DIR)
        with open(os.path.join(self.benchmarks_dir, "benchmark_failures.json"), "w") as f:
            json.dump(failures, f, indent=2, sort_keys=True)
        with open(os.path.join(self.benchmarks_dir, "benchmark_missing_tasks.json"), "w") as f:
            json.dump(missing, f, indent=2, sort_keys=True)
        self.log(f"aggregated rows={len(rows)} failures={len(failures)} missing={len(missing)}")

    def aggregate_rows(self, rows):
        grouped = {}
        for row in rows:
            try:
                score = float(row.get("score"))
            except Exception:
                continue
            if score != score:
                continue
            method = row.get("method", "")
            group = row.get("task_group", "other")
            grouped.setdefault(method, {}).setdefault(group, []).append(score)
            grouped.setdefault(method, {}).setdefault("overall", []).append(score)
        out = []
        for method in sorted(grouped):
            def avg(name):
                vals = grouped[method].get(name, [])
                return f"{sum(vals) / len(vals):.6f}" if vals else ""
            out.append({
                "method": method,
                "core_average": avg("core"),
                "math_code_average": avg("math_code"),
                "chinese_average": avg("chinese"),
                "overall_average": avg("overall"),
                "n_tasks": len(grouped[method].get("overall", [])),
            })
        return out

    def write_tables(self, md_path, tex_path, rows):
        headers = ["method", "core_average", "math_code_average", "chinese_average", "overall_average", "n_tasks"]
        with open(md_path, "w") as f:
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
            for row in rows:
                f.write("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n")
        with open(tex_path, "w") as f:
            f.write("\\begin{tabular}{lrrrrr}\n\\toprule\n")
            f.write("Method & Core Avg. & Math/Code Avg. & Chinese Avg. & Overall Avg. & Tasks \\\\\n\\midrule\n")
            for row in rows:
                vals = [str(row.get(k, "")) for k in headers]
                vals[0] = vals[0].replace("_", "\\_")
                f.write(" & ".join(vals) + " \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")

    def run(self):
        self.setup()
        self.install_deps()
        self.log("qwen3_orchestrator_start")
        for method, regime in REGIMES:
            self.run_smoke(method, regime)
            self.run_adapt(method, regime)
        self.run_benchmark(REGIMES[0][0], REGIMES[0][1], ["arc_challenge", "hellaswag", "gsm8k"], self.args.sanity_samples, "sanity")
        for method, regime in REGIMES:
            self.run_benchmark(method, regime, TASKS, self.args.max_samples_per_task, "full")
        self.aggregate()
        self.log("qwen3_orchestrator_complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=RUN_ROOT)
    parser.add_argument("--persist-root", default=PERSIST_ROOT)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--adapt-seq-len", type=int, default=1024)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--adapt-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples-per-task", type=int, default=0)
    parser.add_argument("--sanity-samples", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-pip-install", action="store_true")
    parser.add_argument("--continue-on-benchmark-error", action="store_true")
    parser.add_argument("--unfreeze-outputs", action="store_true")
    args = parser.parse_args()
    Runner(args).run()


if __name__ == "__main__":
    main()
