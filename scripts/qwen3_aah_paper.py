#!/usr/bin/env python3
"""Qwen3-4B-Base AAH smoke, adaptation, and benchmark runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.benchmark_paper_tasks import (  # noqa: E402
    BenchmarkResult,
    TASK_GROUPS,
    file_sha256,
    result_to_row,
    score_code,
    score_generation,
    score_mc,
    summarize_scores,
    write_outputs,
)
from scripts.qwen3_aah_patch import (  # noqa: E402
    collect_aah_summary,
    config_from_regime,
    freeze_for_aah_adaptation,
    load_aah_adapter,
    patch_model_attention,
    save_aah_adapter,
    write_aah_diagnostics,
)


DEFAULT_MODEL = "Qwen/Qwen3-4B-Base"
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def dtype_from_precision(precision: str):
    precision = precision.lower()
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def load_qwen_model(args, regime: str, adapter_path: Optional[str] = None):
    device = get_device(args.device)
    dtype = dtype_from_precision(args.precision)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype if device == "cuda" else torch.float32,
        device_map=None,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.to(device)
    states = []
    if regime != "full_attention_baseline":
        patch_cfg = config_from_regime(regime)
        states = patch_model_attention(model, patch_cfg)
        if adapter_path and os.path.exists(adapter_path):
            load_aah_adapter(model, adapter_path)
        # AAH controller modules are attached after the initial model.to(...).
        # Move the patched modules as well before the first forward pass.
        model.to(device)
    model.eval()
    return model, tokenizer, states, device


def autocast_ctx(device: str, precision: str):
    if device == "cuda" and precision.lower() == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if device == "cuda" and precision.lower() == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def smoke(args) -> None:
    ensure_dir(args.out_dir)
    model, tokenizer, states, device = load_qwen_model(args, args.regime, args.adapter)
    prompt = "The purpose of adaptive attention is"
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad(), autocast_ctx(device, args.precision):
        outputs = model(**encoded, labels=encoded["input_ids"], use_cache=False)
        loss = float(outputs.loss.detach().float().item())
        logits_finite = bool(torch.isfinite(outputs.logits).all().item())
        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    diag_path = os.path.join(args.out_dir, f"{args.regime}_smoke_heatmap.csv")
    if states:
        write_aah_diagnostics(states, diag_path, args.regime)
    summary = {
        "model": args.model,
        "regime": args.regime,
        "loss": loss,
        "logits_finite": logits_finite,
        "generated_text": text,
        "aah_summary": collect_aah_summary(states) if states else {"mean_ACR": 1.0, "n_layers": 0},
        "diagnostics_csv": diag_path if states else "",
    }
    path = os.path.join(args.out_dir, f"{args.regime}_smoke_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


def iter_token_batches(tokenizer, dataset_name: str, dataset_config: str, split: str, seq_len: int, batch_size: int):
    ds = load_dataset(dataset_name, dataset_config or None, split=split, streaming=True)
    buffer: List[int] = []
    for row in ds:
        text = row.get("text") or row.get("content") or row.get("prompt") or ""
        if not text:
            continue
        buffer.extend(tokenizer.encode(text, add_special_tokens=False))
        while len(buffer) >= batch_size * (seq_len + 1):
            chunk = buffer[: batch_size * (seq_len + 1)]
            buffer = buffer[batch_size * (seq_len + 1) :]
            x = []
            y = []
            for i in range(batch_size):
                sample = chunk[i * (seq_len + 1) : (i + 1) * (seq_len + 1)]
                x.append(sample[:-1])
                y.append(sample[1:])
            yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def adapt(args) -> None:
    if args.regime == "full_attention_baseline":
        raise ValueError("Baseline has no AAH adapter to train.")
    ensure_dir(args.out_dir)
    model, tokenizer, states, device = load_qwen_model(args, args.regime, args.adapter)
    freeze_for_aah_adaptation(model, unfreeze_outputs=args.unfreeze_outputs)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    batches = iter_token_batches(
        tokenizer,
        args.dataset,
        args.dataset_config,
        args.split,
        args.seq_len,
        args.batch_size,
    )
    losses = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, y = next(batches)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(device, args.precision):
            out = model(input_ids=x, labels=y, use_cache=False)
            loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().float().item()))
        if step % args.log_interval == 0 or step == 1:
            aah = collect_aah_summary(states)
            print(f"step {step} | loss {losses[-1]:.4f} | mean_ACR {aah['mean_ACR']:.4f}", flush=True)
    adapter_path = os.path.join(args.out_dir, f"{args.regime}_aah_adapter.pt")
    metadata = {
        "model": args.model,
        "regime": args.regime,
        "steps": args.steps,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "runtime_s": time.time() - t0,
    }
    save_aah_adapter(model, adapter_path, metadata)
    write_aah_diagnostics(states, os.path.join(args.out_dir, f"{args.regime}_heatmap.csv"), args.regime)
    print(json.dumps({"adapter": adapter_path, **metadata}, indent=2, sort_keys=True))


class HFScorer:
    def __init__(self, model, tokenizer, device: str, seq_len: int, precision: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.seq_len = int(seq_len)
        self.precision = precision
        self.model.eval()

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
            cont_ids = cont_ids[-(self.seq_len - 1) :]
            max_prompt = 1
        prompt_ids = prompt_ids[-max_prompt:]
        ids = prompt_ids + cont_ids
        x = torch.tensor(ids[:-1], dtype=torch.long, device=self.device).unsqueeze(0)
        with autocast_ctx(self.device, self.precision):
            logits = self.model(input_ids=x, use_cache=False).logits
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
            x = torch.tensor(ids[-self.seq_len :], dtype=torch.long, device=self.device).unsqueeze(0)
            with autocast_ctx(self.device, self.precision):
                logits = self.model(input_ids=x, use_cache=False).logits
            next_id = int(torch.argmax(logits[0, -1, :]).item())
            ids.append(next_id)
            generated.append(next_id)
            if eos is not None and next_id == eos:
                break
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def benchmark(args) -> None:
    ensure_dir(args.out_dir)
    model, tokenizer, states, device = load_qwen_model(args, args.regime, args.adapter)
    scorer = HFScorer(model, tokenizer, device, args.seq_len, args.precision)
    tasks = [t.strip() for item in args.tasks for t in item.split(",") if t.strip()]
    ckpt_sha = sha256_text(f"{args.model}:{args.regime}:{args.adapter or 'base'}")
    if args.adapter and os.path.exists(args.adapter):
        ckpt_sha = file_sha256(args.adapter)
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
            if n_examples <= 0 or math.isnan(score):
                missing.append({"task": task, "reason": "zero evaluable examples"})
                continue
            results.append(
                BenchmarkResult(
                    method=args.method,
                    checkpoint_step=int(args.checkpoint_step),
                    task=task,
                    metric=metric,
                    score=float(score),
                    stderr_or_std=float(stderr),
                    n_examples=int(n_examples),
                    checkpoint_sha256=ckpt_sha,
                    eval_runtime_s=time.time() - t0,
                    task_group=TASK_GROUPS.get(task, "other"),
                )
            )
            print(f"task={task} metric={metric} score={score:.6f} n={n_examples}", flush=True)
        except Exception as exc:
            failures.append({"task": task, "error": repr(exc)})
            print(f"task_failed task={task} error={repr(exc)}", flush=True)
    paths = write_outputs(args.out_dir, args.method, results, failures, missing)
    if states:
        write_aah_diagnostics(states, os.path.join(args.out_dir, f"{args.method}_heatmap.csv"), args.regime)
    print(json.dumps(paths, indent=2, sort_keys=True))


def add_common(parser):
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--regime", default="full_attention_baseline")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--seq-len", type=int, default=4096)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_smoke = sub.add_parser("smoke")
    add_common(p_smoke)
    p_smoke.add_argument("--max-new-tokens", type=int, default=32)
    p_smoke.set_defaults(func=smoke)

    p_adapt = sub.add_parser("adapt")
    add_common(p_adapt)
    p_adapt.add_argument("--steps", type=int, default=1000)
    p_adapt.add_argument("--batch-size", type=int, default=1)
    p_adapt.add_argument("--lr", type=float, default=1e-4)
    p_adapt.add_argument("--weight-decay", type=float, default=0.0)
    p_adapt.add_argument("--grad-clip", type=float, default=1.0)
    p_adapt.add_argument("--dataset", default="wikitext")
    p_adapt.add_argument("--dataset-config", default="wikitext-103-v1")
    p_adapt.add_argument("--split", default="train")
    p_adapt.add_argument("--log-interval", type=int, default=25)
    p_adapt.add_argument("--unfreeze-outputs", action="store_true")
    p_adapt.set_defaults(func=adapt)

    p_bench = sub.add_parser("benchmark")
    add_common(p_bench)
    p_bench.add_argument("--method", required=True)
    p_bench.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    p_bench.add_argument("--max-samples-per-task", type=int, default=0)
    p_bench.add_argument("--checkpoint-step", type=int, default=0)
    p_bench.add_argument("--code-timeout-s", type=int, default=5)
    p_bench.set_defaults(func=benchmark)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
