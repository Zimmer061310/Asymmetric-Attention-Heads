#!/usr/bin/env python3
"""Remote orchestration for the AAH 4096 checkpoint rerun and benchmarks."""

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

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.run_paper_experiments import MAIN_4096, generate_configs  # noqa: E402


RUN_ROOT = "/home/featurize/data/AAH-paper-4096-benchmark-rerun"
PERSIST_ROOT = "/home/featurize/work/ENA-AAH-v3-persistent/AAH-paper-4096-benchmark-rerun"
RESULT_REPO_DIR = os.path.join(PROJECT_ROOT, "paper_results", "benchmark_rerun")
TMUX_SESSION = "aah_4096_benchmark_rerun"

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


def now():
    return datetime.now(timezone.utc).isoformat()


class Orchestrator:
    def __init__(self, args):
        self.args = args
        self.run_root = os.path.abspath(args.run_root)
        self.persist_root = os.path.abspath(args.persist_root)
        self.log_dir = os.path.join(self.run_root, "logs")
        self.config_dir = os.path.join(self.run_root, "configs")
        self.checkpoint_dir = os.path.join(self.run_root, "checkpoints")
        self.summary_dir = os.path.join(self.run_root, "summaries")
        self.diagnostics_dir = os.path.join(self.run_root, "diagnostics")
        self.benchmark_dir = os.path.join(self.persist_root, "benchmarks")
        self.master_log = os.path.join(self.run_root, "master.log")

        self.persist_dirs = {
            "checkpoints": os.path.join(self.persist_root, "checkpoints"),
            "logs": os.path.join(self.persist_root, "logs"),
            "summaries": os.path.join(self.persist_root, "summaries"),
            "diagnostics": os.path.join(self.persist_root, "diagnostics"),
            "benchmarks": self.benchmark_dir,
            "configs": os.path.join(self.persist_root, "configs"),
        }

    def setup_dirs(self):
        for path in [
            self.run_root,
            self.log_dir,
            self.config_dir,
            self.checkpoint_dir,
            self.summary_dir,
            self.diagnostics_dir,
            RESULT_REPO_DIR,
            *self.persist_dirs.values(),
        ]:
            os.makedirs(path, exist_ok=True)

    def log(self, message):
        line = f"[{now()}] {message}"
        print(line, flush=True)
        os.makedirs(os.path.dirname(self.master_log), exist_ok=True)
        with open(self.master_log, "a") as f:
            f.write(line + "\n")

    def run_cmd(self, cmd, log_name, env=None, check=True):
        log_path = os.path.join(self.log_dir, log_name)
        self.log(f"run command={' '.join(cmd)} log={log_path}")
        start = time.time()
        with open(log_path, "w") as f:
            proc = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            rc = proc.wait()
        elapsed = time.time() - start
        self.log(f"done rc={rc} runtime_s={elapsed:.1f} log={log_path}")
        self.copy_file(log_path, self.persist_dirs["logs"])
        if check and rc != 0:
            raise RuntimeError(f"Command failed rc={rc}: {' '.join(cmd)}")
        return rc

    def copy_file(self, path, dest_dir):
        if os.path.exists(path):
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(path, os.path.join(dest_dir, os.path.basename(path)))

    def copy_glob(self, pattern, dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        for path in glob.glob(pattern):
            if os.path.isfile(path):
                shutil.copy2(path, os.path.join(dest_dir, os.path.basename(path)))

    def install_optional_deps(self):
        if self.args.skip_pip_install:
            self.log("skip optional pip install")
            return
        base_packages = [
            "transformers==4.41.2",
            "datasets",
            "accelerate",
            "tokenizers",
            "numpy",
            "scipy",
            "pandas",
            "matplotlib",
            "tqdm",
            "pyyaml",
            "wandb",
            "rich",
            "loguru",
        ]
        self.run_cmd(
            [sys.executable, "-m", "pip", "install", "--user", *base_packages],
            "pip_install_requirements.log",
            check=False,
        )
        self.run_cmd(
            [sys.executable, "-m", "pip", "install", "--user", "lm-eval"],
            "pip_install_lm_eval.log",
            check=False,
        )

    def write_configs(self):
        generated = generate_configs(self.config_dir, "mandatory", [0])
        paths = []
        for path in generated:
            with open(path, "r") as f:
                cfg = yaml.safe_load(f)
            name = cfg["experiment"]["name"]
            cfg["experiment"]["out_dir"] = os.path.join(self.checkpoint_dir, name)
            cfg["train"]["use_wandb"] = bool(self.args.use_wandb)
            cfg["train"]["save_checkpoints"] = True
            cfg["train"]["checkpoint_steps"] = [1000, 5000, 10000]
            cfg["train"]["eval_interval"] = 200
            cfg["train"]["max_steps"] = 10000
            cfg["train"]["precision"] = "bf16"
            cfg["train"]["batch_size"] = 1
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            self.copy_file(path, self.persist_dirs["configs"])
            paths.append(path)
        self.log(f"prepared_configs={len(paths)}")
        return paths

    def final_checkpoint_path(self, cfg):
        return os.path.join(cfg["experiment"]["out_dir"], f"{cfg['experiment']['name']}.pt")

    def persistent_checkpoint_path(self, cfg):
        return os.path.join(self.persist_dirs["checkpoints"], f"{cfg['experiment']['name']}.pt")

    def load_config(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def persist_training_outputs(self, cfg):
        out_dir = cfg["experiment"]["out_dir"]
        run_name = cfg["experiment"]["name"]
        for suffix in [".pt", ".pt.meta.json"]:
            self.copy_file(os.path.join(out_dir, f"{run_name}{suffix}"), self.persist_dirs["checkpoints"])
        self.copy_glob(os.path.join(out_dir, "*.csv"), self.persist_dirs["logs"])
        self.copy_glob(os.path.join(out_dir, "*crash*.log"), self.persist_dirs["logs"])
        self.log(f"persisted_training_outputs run={cfg['experiment']['name']}")

    def upload_wandb_artifact(self, cfg):
        status_path = os.path.join(self.persist_root, "wandb_artifact_status.jsonl")
        ckpt = self.persistent_checkpoint_path(cfg)
        meta = f"{ckpt}.meta.json"
        payload = {
            "time": now(),
            "run_name": cfg["experiment"]["name"],
            "checkpoint": ckpt,
            "status": "skipped",
            "reason": "not attempted",
        }
        if not self.args.use_wandb:
            payload["reason"] = "wandb disabled"
        elif not os.path.exists(ckpt):
            payload["reason"] = "checkpoint missing"
        else:
            code = (
                "import os, wandb; "
                "run=wandb.init(project='ENA-AAH', job_type='artifact-upload', reinit=True); "
                f"art=wandb.Artifact('{cfg['experiment']['name']}', type='model'); "
                f"art.add_file(r'{ckpt}'); "
                f"art.add_file(r'{meta}') if os.path.exists(r'{meta}') else None; "
                "run.log_artifact(art); run.finish()"
            )
            rc = self.run_cmd([sys.executable, "-c", code], f"wandb_artifact_{cfg['experiment']['name']}.log", check=False)
            payload["status"] = "uploaded" if rc == 0 else "failed"
            payload["reason"] = "" if rc == 0 else f"wandb artifact command rc={rc}"
        with open(status_path, "a") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
        self.copy_file(status_path, self.persist_dirs["logs"])
        self.log(f"wandb_artifact run={cfg['experiment']['name']} status={payload['status']} reason={payload['reason']}")

    def train_and_infer(self, config_paths):
        for path in config_paths:
            cfg = self.load_config(path)
            run_name = cfg["experiment"]["name"]
            final_persist = self.persistent_checkpoint_path(cfg)
            summary_persist = os.path.join(self.persist_dirs["summaries"], f"{run_name}_infer.json")

            train_done = os.path.exists(final_persist)
            infer_done = os.path.exists(summary_persist)
            if self.args.resume and train_done and infer_done:
                self.log(f"skip_train_infer existing run={run_name}")
                continue

            if self.args.resume and train_done:
                self.log(f"skip_train existing_checkpoint={final_persist}")
                self.upload_wandb_artifact(cfg)
            else:
                self.log(f"stage=train run={run_name}")
                self.run_cmd([sys.executable, "scripts/train.py", "--config", path], f"train_{run_name}.log")
                self.persist_training_outputs(cfg)
                self.upload_wandb_artifact(cfg)

            ckpt = self.final_checkpoint_path(cfg)
            if not os.path.exists(ckpt):
                ckpt = final_persist
            summary_path = os.path.join(self.summary_dir, f"{run_name}_infer.json")
            self.log(f"stage=infer run={run_name}")
            self.run_cmd(
                [
                    sys.executable,
                    "scripts/infer.py",
                    "--config",
                    path,
                    "--checkpoint",
                    ckpt,
                    "--eval-batches",
                    str(self.args.infer_eval_batches),
                    "--summary-json",
                    summary_path,
                    "--diagnostics-dir",
                    self.diagnostics_dir,
                    "--strict-checkpoint",
                ],
                f"infer_{run_name}.log",
            )
            self.copy_file(summary_path, self.persist_dirs["summaries"])
            self.copy_glob(os.path.join(self.diagnostics_dir, f"{run_name}_*.csv"), self.persist_dirs["diagnostics"])

    def run_benchmarks(self, config_paths):
        methods = []
        for path in config_paths:
            cfg = self.load_config(path)
            methods.append((cfg["experiment"]["variant"], cfg["experiment"]["name"], path, self.persistent_checkpoint_path(cfg)))
        if self.args.skip_benchmarks:
            self.log("skip benchmark stage")
            return

        if methods:
            method, run_name, cfg_path, ckpt = methods[0]
            self.log(f"stage=benchmark_sanity method={method}")
            self.run_cmd(
                [
                    sys.executable,
                    "scripts/benchmark_paper_tasks.py",
                    "--config",
                    cfg_path,
                    "--checkpoint",
                    ckpt,
                    "--method",
                    f"{method}_sanity",
                    "--out-dir",
                    self.benchmark_dir,
                    "--tasks",
                    "arc_challenge,gsm8k",
                    "--max-samples-per-task",
                    "2",
                ],
                "benchmark_sanity.log",
            )

        for method, run_name, cfg_path, ckpt in methods:
            if not os.path.exists(ckpt):
                raise FileNotFoundError(f"Persistent checkpoint missing before benchmark: {ckpt}")
            self.log(f"stage=benchmark method={method} checkpoint={ckpt}")
            self.run_cmd(
                [
                    sys.executable,
                    "scripts/benchmark_paper_tasks.py",
                    "--config",
                    cfg_path,
                    "--checkpoint",
                    ckpt,
                    "--method",
                    method,
                    "--out-dir",
                    self.benchmark_dir,
                    "--tasks",
                    ",".join(TASKS),
                    "--max-samples-per-task",
                    str(self.args.max_samples_per_task),
                ],
                f"benchmark_{method}.log",
                check=not self.args.continue_on_benchmark_error,
            )
        self.aggregate_benchmarks()

    def aggregate_benchmarks(self):
        raw_rows = []
        failures = []
        missing = []
        for path in glob.glob(os.path.join(self.benchmark_dir, "*_benchmark_results_raw.jsonl")):
            if "_sanity_" in os.path.basename(path):
                continue
            with open(path, "r") as f:
                for line in f:
                    if line.strip():
                        raw_rows.append(json.loads(line))
        for path in glob.glob(os.path.join(self.benchmark_dir, "*_benchmark_failures.json")):
            if "_sanity_" in os.path.basename(path):
                continue
            with open(path, "r") as f:
                for row in json.load(f):
                    row["source_file"] = os.path.basename(path)
                    failures.append(row)
        for path in glob.glob(os.path.join(self.benchmark_dir, "*_benchmark_missing_tasks.json")):
            if "_sanity_" in os.path.basename(path):
                continue
            with open(path, "r") as f:
                for row in json.load(f):
                    row["source_file"] = os.path.basename(path)
                    missing.append(row)

        raw_path = os.path.join(self.benchmark_dir, "benchmark_results_raw.jsonl")
        by_task_path = os.path.join(self.benchmark_dir, "benchmark_results_by_task.csv")
        by_model_path = os.path.join(self.benchmark_dir, "benchmark_results_by_model.csv")
        failures_path = os.path.join(self.benchmark_dir, "benchmark_failures.json")
        missing_path = os.path.join(self.benchmark_dir, "benchmark_missing_tasks.json")
        md_path = os.path.join(self.benchmark_dir, "benchmark_paper_table.md")
        tex_path = os.path.join(self.benchmark_dir, "benchmark_paper_table.tex")

        with open(raw_path, "w") as f:
            for row in raw_rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")

        fields = [
            "method",
            "checkpoint_step",
            "task",
            "metric",
            "score",
            "stderr_or_std",
            "n_examples",
            "checkpoint_sha256",
            "eval_runtime_s",
            "task_group",
        ]
        with open(by_task_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in raw_rows:
                writer.writerow({k: row.get(k, "") for k in fields})

        aggregates = self.make_aggregates(raw_rows)
        with open(by_model_path, "w", newline="") as f:
            fields_model = ["method", "core_average", "math_code_average", "chinese_average", "overall_average", "n_tasks"]
            writer = csv.DictWriter(f, fieldnames=fields_model)
            writer.writeheader()
            for row in aggregates:
                writer.writerow(row)

        with open(failures_path, "w") as f:
            json.dump(failures, f, indent=2, sort_keys=True)
        with open(missing_path, "w") as f:
            json.dump(missing, f, indent=2, sort_keys=True)
        self.write_markdown_table(md_path, aggregates)
        self.write_tex_table(tex_path, aggregates)

        for path in [raw_path, by_task_path, by_model_path, failures_path, missing_path, md_path, tex_path]:
            self.copy_file(path, RESULT_REPO_DIR)
        self.log(f"aggregated_benchmarks rows={len(raw_rows)} failures={len(failures)} missing={len(missing)}")

    def make_aggregates(self, rows):
        by_method = {}
        for row in rows:
            method = row.get("method", "")
            group = row.get("task_group", "other")
            try:
                score = float(row.get("score"))
            except Exception:
                continue
            if score != score:
                continue
            by_method.setdefault(method, {}).setdefault(group, []).append(score)
            by_method.setdefault(method, {}).setdefault("overall", []).append(score)
        out = []
        for method in sorted(by_method):
            groups = by_method[method]
            def avg(name):
                vals = groups.get(name, [])
                return f"{(sum(vals) / len(vals)):.6f}" if vals else ""
            out.append(
                {
                    "method": method,
                    "core_average": avg("core"),
                    "math_code_average": avg("math_code"),
                    "chinese_average": avg("chinese"),
                    "overall_average": avg("overall"),
                    "n_tasks": len(groups.get("overall", [])),
                }
            )
        return out

    def write_markdown_table(self, path, rows):
        headers = ["method", "core_average", "math_code_average", "chinese_average", "overall_average", "n_tasks"]
        with open(path, "w") as f:
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
            for row in rows:
                f.write("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n")

    def write_tex_table(self, path, rows):
        headers = ["Method", "Core Avg.", "Math/Code Avg.", "Chinese Avg.", "Overall Avg.", "Tasks"]
        keys = ["method", "core_average", "math_code_average", "chinese_average", "overall_average", "n_tasks"]
        with open(path, "w") as f:
            f.write("\\begin{tabular}{lrrrrr}\n\\toprule\n")
            f.write(" & ".join(headers) + " \\\\\n\\midrule\n")
            for row in rows:
                vals = [str(row.get(k, "")) for k in keys]
                vals[0] = vals[0].replace("_", "\\_")
                f.write(" & ".join(vals) + " \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")

    def verify_completion(self):
        ckpts = glob.glob(os.path.join(self.persist_dirs["checkpoints"], "paper-main_4096_*-seed0.pt"))
        summaries = glob.glob(os.path.join(self.persist_dirs["summaries"], "paper-main_4096_*-seed0_infer.json"))
        required_tables = [
            os.path.join(self.benchmark_dir, "benchmark_results_by_task.csv"),
            os.path.join(self.benchmark_dir, "benchmark_results_by_model.csv"),
            os.path.join(self.benchmark_dir, "benchmark_paper_table.md"),
            os.path.join(self.benchmark_dir, "benchmark_paper_table.tex"),
        ]
        missing_tables = [p for p in required_tables if not os.path.exists(p)]
        self.log(f"verify_completion ckpts={len(ckpts)} summaries={len(summaries)} missing_tables={missing_tables}")
        if len(ckpts) < len(MAIN_4096):
            raise RuntimeError("Not all final persistent checkpoints exist")
        if len(summaries) < len(MAIN_4096):
            raise RuntimeError("Not all inference summaries exist")
        if missing_tables:
            raise RuntimeError(f"Benchmark tables missing: {missing_tables}")

    def git_finalize(self):
        if self.args.skip_git_finalize:
            self.log("skip git finalize")
            return
        self.run_cmd(["git", "status", "--short"], "git_status_before_finalize.log", check=False)
        add_paths = [
            "scripts/benchmark_paper_tasks.py",
            "scripts/run_paper_benchmark_rerun.py",
            os.path.relpath(RESULT_REPO_DIR, PROJECT_ROOT),
        ]
        self.run_cmd(["git", "add", *add_paths], "git_add_finalize.log", check=False)
        diff_rc = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=PROJECT_ROOT).returncode
        if diff_rc == 0:
            self.log("git_finalize no staged changes")
            return
        self.run_cmd(["git", "commit", "-m", "Add AAH benchmark rerun results"], "git_commit_finalize.log", check=False)
        self.run_cmd(["git", "push"], "git_push_finalize.log", check=False)

    def release_instance(self):
        if not self.args.release_on_complete:
            self.log("skip instance release")
            return
        self.run_cmd(["featurize", "instance", "release"], "featurize_release.log", check=False)

    def run(self):
        self.setup_dirs()
        self.log("orchestrator_start")
        self.install_optional_deps()
        config_paths = self.write_configs()
        if not self.args.skip_training:
            self.train_and_infer(config_paths)
        else:
            self.log("skip training stage")
        self.run_benchmarks(config_paths)
        if not (self.args.skip_training or self.args.skip_benchmarks):
            self.verify_completion()
        self.git_finalize()
        self.release_instance()
        self.log("orchestrator_complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=RUN_ROOT)
    parser.add_argument("--persist-root", default=PERSIST_ROOT)
    parser.add_argument("--infer-eval-batches", type=int, default=50)
    parser.add_argument("--max-samples-per-task", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--skip-pip-install", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-benchmarks", action="store_true")
    parser.add_argument("--continue-on-benchmark-error", action="store_true")
    parser.add_argument("--skip-git-finalize", action="store_true")
    parser.add_argument("--release-on-complete", action="store_true")
    args = parser.parse_args()
    Orchestrator(args).run()


if __name__ == "__main__":
    main()
