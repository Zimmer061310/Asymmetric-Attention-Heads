#!/usr/bin/env python3
"""Best-effort paper benchmark runner for local AAH GPT checkpoints.

The runner intentionally avoids hiding task failures. Each requested task either
emits a result row, a missing-task entry, or a failure entry so the paper table
can distinguish measured numbers from unavailable harness coverage.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import yaml
from datasets import get_dataset_config_names, load_dataset
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from infer import build_model  # noqa: E402


TASK_ALIASES = {
    "mmlu": "mmlu",
    "mmlu-pro": "mmlu_pro",
    "mmlu_pro": "mmlu_pro",
    "gpqa-diamond": "gpqa_diamond",
    "gpqa_diamond": "gpqa_diamond",
    "bbh": "bbh",
    "arc-challenge": "arc_challenge",
    "arc_challenge": "arc_challenge",
    "hellaswag": "hellaswag",
    "triviaqa": "triviaqa",
    "gsm8k": "gsm8k",
    "mgsm": "mgsm",
    "math": "math",
    "cmath": "cmath",
    "humaneval": "humaneval",
    "human_eval": "humaneval",
    "mbpp": "mbpp",
    "cmmlu": "cmmlu",
    "c-eval": "ceval",
    "ceval": "ceval",
}

DEFAULT_TASKS = [
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

TASK_GROUPS = {
    "mmlu": "core",
    "mmlu_pro": "core",
    "gpqa_diamond": "core",
    "bbh": "core",
    "arc_challenge": "core",
    "hellaswag": "core",
    "triviaqa": "core",
    "gsm8k": "math_code",
    "mgsm": "math_code",
    "math": "math_code",
    "cmath": "math_code",
    "humaneval": "math_code",
    "mbpp": "math_code",
    "cmmlu": "chinese",
    "ceval": "chinese",
}


@dataclass
class BenchmarkResult:
    method: str
    checkpoint_step: int
    task: str
    metric: str
    score: float
    stderr_or_std: float
    n_examples: int
    checkpoint_sha256: str
    eval_runtime_s: float
    task_group: str


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint_step(path):
    meta_path = f"{path}.meta.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        try:
            return int(meta.get("step", -1))
        except Exception:
            return -1
    m = re.search(r"_step(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def normalize_tasks(task_arg):
    if not task_arg or task_arg == ["all"]:
        return list(DEFAULT_TASKS)
    tasks = []
    for item in task_arg:
        for part in item.split(","):
            key = part.strip().lower()
            if not key:
                continue
            tasks.append(TASK_ALIASES.get(key, key))
    return tasks


def choose_split(ds, preferred=("test", "validation", "eval", "dev", "train")):
    for split in preferred:
        if split in ds:
            return split
    return list(ds.keys())[0]


def safe_configs(dataset_name, fallback):
    try:
        names = get_dataset_config_names(dataset_name)
        return names or fallback
    except Exception:
        return fallback


def limited_iter(ds, max_samples):
    count = 0
    for row in ds:
        yield row
        count += 1
        if max_samples and count >= max_samples:
            break


def cap_reached(max_samples, count):
    return bool(max_samples and count >= max_samples)


def answer_to_index(answer, labels=None):
    if isinstance(answer, int):
        return answer
    text = str(answer).strip()
    if text.isdigit():
        idx = int(text)
        if labels and str(idx) in labels:
            return labels.index(str(idx))
        return idx
    if len(text) == 1 and text.isalpha():
        return ord(text.upper()) - ord("A")
    if labels and text in labels:
        return labels.index(text)
    return -1


def build_mc_prompt(question, choices):
    labels = [chr(ord("A") + i) for i in range(len(choices))]
    lines = [str(question).strip(), ""]
    for label, choice in zip(labels, choices):
        lines.append(f"{label}. {str(choice).strip()}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines), labels


def load_mc_examples(task, max_samples):
    if task == "mmlu":
        configs = ["all"]
        if "all" not in safe_configs("cais/mmlu", ["all"]):
            configs = safe_configs("cais/mmlu", [])
        emitted = 0
        for config in configs:
            ds = load_dataset("cais/mmlu", config)
            split = choose_split(ds, ("test", "validation", "dev"))
            for row in limited_iter(ds[split], max_samples):
                if cap_reached(max_samples, emitted):
                    return
                choices = row.get("choices") or [row.get(k) for k in ["A", "B", "C", "D"] if row.get(k) is not None]
                emitted += 1
                yield build_mc_prompt(row.get("question", ""), choices), answer_to_index(row.get("answer"))
        return

    if task == "mmlu_pro":
        ds = load_dataset("TIGER-Lab/MMLU-Pro")
        split = choose_split(ds, ("test", "validation"))
        for row in limited_iter(ds[split], max_samples):
            choices = row.get("options") or row.get("choices") or []
            yield build_mc_prompt(row.get("question", ""), choices), answer_to_index(row.get("answer"))
        return

    if task == "gpqa_diamond":
        try:
            ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
            split = choose_split(ds, ("train", "test", "validation"))
            for row in limited_iter(ds[split], max_samples):
                choices = [
                    row.get("Correct Answer", ""),
                    row.get("Incorrect Answer 1", ""),
                    row.get("Incorrect Answer 2", ""),
                    row.get("Incorrect Answer 3", ""),
                ]
                yield build_mc_prompt(row.get("Question", ""), choices), 0
        except Exception:
            # The original GPQA repository can be gated. This public mirror stores
            # the Diamond questions with choices already embedded in the prompt.
            ds = load_dataset("fingertap/GPQA-Diamond")
            split = choose_split(ds, ("test", "train", "validation"))
            for row in limited_iter(ds[split], max_samples):
                prompt = f"{str(row.get('question', '')).strip()}\n\nAnswer:"
                yield (prompt, ["A", "B", "C", "D"]), answer_to_index(row.get("answer"))
        return

    if task == "arc_challenge":
        ds = load_dataset("ai2_arc", "ARC-Challenge")
        split = choose_split(ds, ("test", "validation"))
        for row in limited_iter(ds[split], max_samples):
            choices_obj = row.get("choices", {})
            choices = choices_obj.get("text", [])
            labels = choices_obj.get("label", [])
            prompt_labels = build_mc_prompt(row.get("question", ""), choices)
            yield prompt_labels, answer_to_index(row.get("answerKey"), labels)
        return

    if task == "hellaswag":
        ds = load_dataset("Rowan/hellaswag")
        split = choose_split(ds, ("validation", "test"))
        for row in limited_iter(ds[split], max_samples):
            prompt = str(row.get("ctx", "")).strip()
            endings = row.get("endings", [])
            yield (prompt, endings), answer_to_index(row.get("label"))
        return

    if task == "cmmlu":
        configs = ["all"]
        available = safe_configs("haonan-li/cmmlu", [])
        if "all" not in available and available:
            configs = available
        emitted = 0
        for config in configs:
            ds = load_dataset("haonan-li/cmmlu", config)
            split = choose_split(ds, ("test", "dev", "validation"))
            for row in limited_iter(ds[split], max_samples):
                if cap_reached(max_samples, emitted):
                    return
                choices = [row.get(k, "") for k in ["A", "B", "C", "D"]]
                question = row.get("Question") or row.get("question") or ""
                answer = row.get("Answer") or row.get("answer")
                emitted += 1
                yield build_mc_prompt(question, choices), answer_to_index(answer)
        return

    if task == "ceval":
        configs = ["all"]
        available = safe_configs("ceval/ceval-exam", [])
        if "all" not in available and available:
            configs = available
        emitted = 0
        for config in configs:
            ds = load_dataset("ceval/ceval-exam", config)
            split = choose_split(ds, ("test", "val", "validation", "dev"))
            for row in limited_iter(ds[split], max_samples):
                if cap_reached(max_samples, emitted):
                    return
                choices = [row.get(k, "") for k in ["A", "B", "C", "D"]]
                question = row.get("question") or row.get("Question") or ""
                answer = row.get("answer") or row.get("Answer")
                emitted += 1
                yield build_mc_prompt(question, choices), answer_to_index(answer)
        return

    raise KeyError(task)


def load_generation_examples(task, max_samples):
    if task == "triviaqa":
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")
        split = choose_split(ds, ("validation", "test"))
        for row in limited_iter(ds[split], max_samples):
            aliases = row.get("answer", {}).get("aliases", [])
            yield f"Question: {row.get('question', '')}\nAnswer:", aliases, "exact_any"
        return

    if task == "gsm8k":
        ds = load_dataset("gsm8k", "main")
        split = choose_split(ds, ("test", "validation"))
        for row in limited_iter(ds[split], max_samples):
            yield f"Question: {row.get('question', '')}\nAnswer:", row.get("answer", ""), "number"
        return

    if task == "mgsm":
        configs = [c for c in ["en", "zh"] if c in safe_configs("juletxara/mgsm", ["en"])] or ["en"]
        emitted = 0
        for config in configs:
            ds = load_dataset("juletxara/mgsm", config)
            split = choose_split(ds, ("test", "validation"))
            for row in limited_iter(ds[split], max_samples):
                if cap_reached(max_samples, emitted):
                    return
                emitted += 1
                yield f"Question: {row.get('question', '')}\nAnswer:", row.get("answer", ""), "number"
        return

    if task == "math":
        ds = load_dataset("hendrycks/competition_math")
        split = choose_split(ds, ("test", "validation"))
        for row in limited_iter(ds[split], max_samples):
            yield f"Problem: {row.get('problem', '')}\nSolution:", row.get("solution", ""), "number"
        return

    if task == "cmath":
        candidates = [
            ("haonan-li/cmath", None),
            ("weitianwen/cmath", None),
            ("meta-math/CMATH", None),
        ]
        last_error = None
        for name, config in candidates:
            try:
                ds = load_dataset(name, config) if config else load_dataset(name)
                split = choose_split(ds, ("test", "validation", "train"))
                for row in limited_iter(ds[split], max_samples):
                    question = row.get("question") or row.get("problem") or row.get("Question") or ""
                    answer = row.get("answer") or row.get("solution") or row.get("Answer") or ""
                    yield f"题目: {question}\n答案:", answer, "number"
                return
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"CMath dataset unavailable: {last_error}")

    if task == "bbh":
        configs = safe_configs("lukaemon/bbh", [])
        if not configs:
            raise RuntimeError("No BBH configs found")
        emitted = 0
        for config in configs:
            ds = load_dataset("lukaemon/bbh", config)
            split = choose_split(ds, ("test", "validation", "train"))
            for row in limited_iter(ds[split], max_samples):
                if cap_reached(max_samples, emitted):
                    return
                emitted += 1
                yield f"{row.get('input', '')}\nAnswer:", row.get("target", ""), "exact"
        return

    raise KeyError(task)


def load_code_examples(task, max_samples):
    if task == "humaneval":
        ds = load_dataset("openai/openai_humaneval")
        split = choose_split(ds, ("test", "validation"))
        for row in limited_iter(ds[split], max_samples):
            yield {
                "prompt": row.get("prompt", ""),
                "tests": row.get("test", ""),
                "entry_point": row.get("entry_point", ""),
            }
        return

    if task == "mbpp":
        try:
            ds = load_dataset("google-research-datasets/mbpp", "sanitized")
        except Exception:
            ds = load_dataset("mbpp")
        split = choose_split(ds, ("test", "validation", "train"))
        for row in limited_iter(ds[split], max_samples):
            tests = row.get("test_list") or row.get("test") or []
            if isinstance(tests, str):
                tests = [tests]
            prompt = row.get("prompt") or row.get("text") or ""
            yield {"prompt": f"# {prompt}\n", "tests": "\n".join(tests), "entry_point": ""}
        return

    raise KeyError(task)


class LocalGPTScorer:
    def __init__(self, cfg, checkpoint, device, precision):
        self.cfg = cfg
        self.device = device
        self.seq_len = int(cfg["data"]["seq_len"])
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["data"].get("tokenizer", "gpt2"), use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = build_model(cfg, self.tokenizer.vocab_size, device)
        state = torch.load(checkpoint, map_location=device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.use_bf16 = precision == "bf16" and device == "cuda" and torch.cuda.is_available()
        if self.use_bf16 and not torch.cuda.is_bf16_supported():
            self.use_bf16 = False

    def autocast(self):
        if self.use_bf16:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def encode(self, text):
        return self.tokenizer.encode(str(text), add_special_tokens=False)

    @torch.no_grad()
    def loglikelihood(self, prompt, continuation):
        cont_ids = self.encode(continuation)
        if not cont_ids:
            return float("-inf")
        prompt_ids = self.encode(prompt)
        max_prompt = self.seq_len - len(cont_ids)
        if max_prompt < 1:
            cont_ids = cont_ids[-(self.seq_len - 1):]
            max_prompt = 1
        prompt_ids = prompt_ids[-max_prompt:]
        ids = prompt_ids + cont_ids
        x = torch.tensor(ids[:-1], dtype=torch.long, device=self.device).unsqueeze(0)
        with self.autocast():
            logits, _ = self.model(x)
        start = max(0, len(prompt_ids) - 1)
        target = torch.tensor(cont_ids, dtype=torch.long, device=self.device)
        logits = logits[0, start : start + len(cont_ids), :]
        if logits.size(0) != target.numel():
            return float("-inf")
        log_probs = F.log_softmax(logits.float(), dim=-1)
        return float(log_probs.gather(1, target.unsqueeze(1)).sum().item())

    @torch.no_grad()
    def greedy(self, prompt, max_new_tokens):
        ids = self.encode(prompt)
        ids = ids[-self.seq_len :]
        eos = self.tokenizer.eos_token_id
        generated = []
        for _ in range(max_new_tokens):
            x_ids = ids[-self.seq_len :]
            x = torch.tensor(x_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            with self.autocast():
                logits, _ = self.model(x)
            next_id = int(torch.argmax(logits[0, -1, :]).item())
            ids.append(next_id)
            generated.append(next_id)
            if eos is not None and next_id == eos:
                break
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff ./%+-]", "", text)
    return text.strip()


def extract_number(text):
    text = str(text)
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        text = boxed[-1]
    marker = re.findall(r"####\s*([-+]?[\d,.]+(?:\.\d+)?)", text)
    if marker:
        return marker[-1].replace(",", "")
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else ""


def score_mc(scorer, task, max_samples):
    scores = []
    n = 0
    for (prompt, choices), gold in load_mc_examples(task, max_samples):
        if gold is None or int(gold) < 0 or int(gold) >= len(choices):
            continue
        if task == "hellaswag":
            lls = [scorer.loglikelihood(prompt, choice) for choice in choices]
        else:
            lls = [scorer.loglikelihood(prompt, f" {label}") for label in choices]
        pred = max(range(len(lls)), key=lambda i: lls[i])
        scores.append(1.0 if pred == int(gold) else 0.0)
        n += 1
    return summarize_scores(scores), n


def score_generation(scorer, task, max_samples):
    scores = []
    n = 0
    max_new = 64 if task in ("triviaqa", "gsm8k", "mgsm", "math", "cmath") else 96
    for prompt, gold, mode in load_generation_examples(task, max_samples):
        pred = scorer.greedy(prompt, max_new)
        if mode == "number":
            pred_val = extract_number(pred)
            gold_val = extract_number(gold)
            ok = pred_val != "" and gold_val != "" and pred_val == gold_val
        elif mode == "exact_any":
            pred_norm = normalize_text(pred)
            aliases = gold if isinstance(gold, list) else [gold]
            ok = any(normalize_text(alias) in pred_norm for alias in aliases if str(alias).strip())
        else:
            ok = normalize_text(pred) == normalize_text(gold)
        scores.append(1.0 if ok else 0.0)
        n += 1
    return summarize_scores(scores), n


def run_python_tests(source, tests, entry_point, timeout_s):
    code = source
    if tests:
        code += "\n\n" + tests
    if entry_point:
        code += f"\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
            cwd=tempfile.gettempdir(),
        )
        return proc.returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def score_code(scorer, task, max_samples, code_timeout_s):
    scores = []
    n = 0
    for row in load_code_examples(task, max_samples):
        completion = scorer.greedy(row["prompt"], max_new_tokens=96)
        source = row["prompt"] + completion
        ok = run_python_tests(source, row["tests"], row.get("entry_point", ""), code_timeout_s)
        scores.append(1.0 if ok else 0.0)
        n += 1
    return summarize_scores(scores), n


def summarize_scores(scores):
    if not scores:
        return float("nan"), float("nan")
    mean = sum(scores) / len(scores)
    if len(scores) <= 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in scores) / (len(scores) - 1)
    return mean, math.sqrt(var / len(scores))


def result_to_row(result):
    return {
        "method": result.method,
        "checkpoint_step": result.checkpoint_step,
        "task": result.task,
        "metric": result.metric,
        "score": f"{result.score:.8f}" if result.score == result.score else "nan",
        "stderr_or_std": f"{result.stderr_or_std:.8f}" if result.stderr_or_std == result.stderr_or_std else "nan",
        "n_examples": result.n_examples,
        "checkpoint_sha256": result.checkpoint_sha256,
        "eval_runtime_s": f"{result.eval_runtime_s:.2f}",
        "task_group": result.task_group,
    }


def write_outputs(out_dir, method, results, failures, missing):
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, f"{method}_benchmark_results_raw.jsonl")
    csv_path = os.path.join(out_dir, f"{method}_benchmark_results_by_task.csv")
    failures_path = os.path.join(out_dir, f"{method}_benchmark_failures.json")
    missing_path = os.path.join(out_dir, f"{method}_benchmark_missing_tasks.json")

    with open(raw_path, "w") as f:
        for result in results:
            f.write(json.dumps(result_to_row(result), sort_keys=True) + "\n")

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
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(result_to_row(result))

    with open(failures_path, "w") as f:
        json.dump(failures, f, indent=2, sort_keys=True)
    with open(missing_path, "w") as f:
        json.dump(missing, f, indent=2, sort_keys=True)
    return {
        "raw_jsonl": raw_path,
        "by_task_csv": csv_path,
        "failures_json": failures_path,
        "missing_json": missing_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--max-samples-per-task", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--code-timeout-s", type=int, default=5)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    tasks = normalize_tasks(args.tasks)
    ckpt_sha = file_sha256(args.checkpoint)
    ckpt_step = load_checkpoint_step(args.checkpoint)
    scorer = LocalGPTScorer(cfg, args.checkpoint, device, args.precision.lower())

    results = []
    failures = []
    missing = []
    for task in tasks:
        t0 = time.time()
        try:
            if task in {"mmlu", "mmlu_pro", "gpqa_diamond", "arc_challenge", "hellaswag", "cmmlu", "ceval"}:
                (score, stderr), n_examples = score_mc(scorer, task, args.max_samples_per_task)
                metric = "accuracy"
            elif task in {"triviaqa", "gsm8k", "mgsm", "math", "cmath", "bbh"}:
                (score, stderr), n_examples = score_generation(scorer, task, args.max_samples_per_task)
                metric = "exact_match"
            elif task in {"humaneval", "mbpp"}:
                (score, stderr), n_examples = score_code(scorer, task, args.max_samples_per_task, args.code_timeout_s)
                metric = "pass@1"
            else:
                missing.append({"task": task, "reason": "unknown task alias"})
                continue
            if n_examples <= 0:
                missing.append({"task": task, "reason": "zero evaluable examples"})
                continue
            elapsed = time.time() - t0
            result = BenchmarkResult(
                method=args.method,
                checkpoint_step=ckpt_step,
                task=task,
                metric=metric,
                score=float(score),
                stderr_or_std=float(stderr),
                n_examples=int(n_examples),
                checkpoint_sha256=ckpt_sha,
                eval_runtime_s=float(elapsed),
                task_group=TASK_GROUPS.get(task, "other"),
            )
            results.append(result)
            print(f"task={task} metric={metric} score={score:.6f} n={n_examples} runtime_s={elapsed:.1f}")
        except Exception as exc:
            failures.append({"task": task, "error": repr(exc)})
            print(f"task_failed task={task} error={repr(exc)}", flush=True)

    paths = write_outputs(args.out_dir, args.method, results, failures, missing)
    print(json.dumps(paths, indent=2, sort_keys=True))
    if failures:
        print(f"benchmark_failures={len(failures)}")
    if missing:
        print(f"benchmark_missing={len(missing)}")


if __name__ == "__main__":
    main()
