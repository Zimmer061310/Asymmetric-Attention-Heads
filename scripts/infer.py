#!/usr/bin/env python3
import argparse
import csv
import glob
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext

import torch
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data import build_dataloaders
from src.models.transformer import GPT, GPTConfig


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def compute_file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return ""


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_pref):
    if device_pref == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_pref


def resolve_checkpoint(exp, ckpt_arg=None, strict=True, allow_fallback=False):
    out_dir = exp.get("out_dir", "experiments")
    exp_name = exp["name"]
    expected = os.path.join(out_dir, f"{exp_name}.pt")

    if ckpt_arg:
        if os.path.exists(ckpt_arg):
            return os.path.abspath(ckpt_arg)
        if strict:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_arg}")
    if (not ckpt_arg) and os.path.exists(expected):
        return os.path.abspath(expected)
    if strict and not allow_fallback:
        missing = ckpt_arg if ckpt_arg else expected
        raise FileNotFoundError(f"Checkpoint not found: {missing}")

    pt_files = sorted(glob.glob(os.path.join(out_dir, "*.pt")), key=os.path.getmtime, reverse=True)
    if not pt_files:
        missing = ckpt_arg if ckpt_arg else expected
        raise FileNotFoundError(f"Checkpoint not found: {missing}; no .pt files in {out_dir}")

    name_matches = [p for p in pt_files if exp_name in os.path.basename(p)]
    if name_matches:
        chosen = name_matches[0]
        print(f"Info: checkpoint not found at expected path, using closest name match: {chosen}")
        return os.path.abspath(chosen)

    chosen = pt_files[0]
    missing = ckpt_arg if ckpt_arg else expected
    print(f"Warning: checkpoint not found: {missing}. Using latest checkpoint in {out_dir}: {chosen}")
    return os.path.abspath(chosen)


def build_model(cfg, vocab_size, device):
    data = cfg["data"]
    model_cfg = cfg["model"]
    gpt_cfg = GPTConfig(
        vocab_size=vocab_size,
        seq_len=data["seq_len"],
        n_layer=model_cfg["n_layer"],
        n_head=model_cfg["n_head"],
        n_embd=model_cfg["n_embd"],
        n_ff=model_cfg["n_ff"],
        dropout=model_cfg["dropout"],
        aah_v2_enabled=model_cfg.get("aah_v2_enabled", False),
        aah_v2_windows=tuple(model_cfg.get("aah_v2_windows", (64, 128, 256, data["seq_len"]))),
        aah_v2_strides=tuple(model_cfg.get("aah_v2_strides", (1, 2, 4))),
        aah_v2_group_size=model_cfg.get("aah_v2_group_size", 1),
        aah_v2_control_dim=model_cfg.get("aah_v2_control_dim", 16),
        aah_v2_temperature=model_cfg.get("aah_v2_temperature", 1.0),
        aah_v2_dynamic_grouping=model_cfg.get("aah_v2_dynamic_grouping", False),
        aah_v2_num_groups=model_cfg.get("aah_v2_num_groups", 4),
        aah_v2_local_chunk=model_cfg.get("aah_v2_local_chunk", 128),
        aah_v2_control_interval=model_cfg.get("aah_v2_control_interval", 1),
        aah_v2_stride_control_enabled=model_cfg.get("aah_v2_stride_control_enabled", True),
        aah_v3_enabled=model_cfg.get("aah_v3_enabled", False),
        aah_v3_windows=tuple(model_cfg.get("aah_v3_windows", (64, 128, 256, data["seq_len"]))),
        aah_v3_control_dim=model_cfg.get("aah_v3_control_dim", 16),
        aah_v3_control_interval=model_cfg.get("aah_v3_control_interval", 100),
        aah_v3_sim_threshold=model_cfg.get("aah_v3_sim_threshold", 0.7),
        aah_v3_super_threshold=model_cfg.get("aah_v3_super_threshold", 0.7),
        aah_v3_max_depth=model_cfg.get("aah_v3_max_depth", 6),
        aah_v3_ema_alpha=model_cfg.get("aah_v3_ema_alpha", 0.9),
        aah_v3_churn_penalty=model_cfg.get("aah_v3_churn_penalty", 0.05),
        aah_v3_min_group_size=model_cfg.get("aah_v3_min_group_size", 1),
        aah_v3_warmup_steps=model_cfg.get("aah_v3_warmup_steps", 0),
        aah_v3_control_enabled=model_cfg.get("aah_v3_control_enabled", True),
        aah_v3_grouping_enabled=model_cfg.get("aah_v3_grouping_enabled", True),
        aah_v3_build_hierarchy=model_cfg.get("aah_v3_build_hierarchy", model_cfg.get("aah_v3_grouping_enabled", True)),
        aah_v3_apply_window_control=model_cfg.get("aah_v3_apply_window_control", model_cfg.get("aah_v3_control_enabled", True)),
        aah_v3_W_min_gpu=model_cfg.get("aah_v3_W_min_gpu", 64),
        aah_v3_mask_cache_size=model_cfg.get("aah_v3_mask_cache_size", 16),
        aah_v3_resolution_ema_alpha=model_cfg.get("aah_v3_resolution_ema_alpha", 0.0),
        aah_v3_resolution_collapse_min_frac=model_cfg.get("aah_v3_resolution_collapse_min_frac", 0.95),
        aah_v3_resolution_collapse_max_frac=model_cfg.get("aah_v3_resolution_collapse_max_frac", 0.95),
        aah_v3_post_warmup_ramp_steps=model_cfg.get("aah_v3_post_warmup_ramp_steps", 0),
        aah_v3_group_feature_mode=model_cfg.get("aah_v3_group_feature_mode", "mean"),
        aah_v3_upper_cluster_metric=model_cfg.get("aah_v3_upper_cluster_metric", "cosine"),
        aah_v3_upper_l2_threshold=model_cfg.get("aah_v3_upper_l2_threshold", 0.0),
        aah_v3_cosine_normdiff_scale=model_cfg.get("aah_v3_cosine_normdiff_scale", 16.0),
        aah_v3_controller_input_mode=model_cfg.get("aah_v3_controller_input_mode", "base"),
        aah_v3_controller_arch=model_cfg.get("aah_v3_controller_arch", "mlp"),
        aah_v3_controller_logit_scale=model_cfg.get("aah_v3_controller_logit_scale", 1.0),
        aah_v3_controller_rng_reference_dim=model_cfg.get("aah_v3_controller_rng_reference_dim", 16),
        aah_v3_controller_choice_mode=model_cfg.get("aah_v3_controller_choice_mode", "learned"),
        aah_v3_controller_pairwise_mode=model_cfg.get("aah_v3_controller_pairwise_mode", "none"),
        aah_v3_pairwise_bias_scale=model_cfg.get("aah_v3_pairwise_bias_scale", 1.0),
        aah_v3_joint_output_scale=model_cfg.get("aah_v3_joint_output_scale", 1.0),
        aah_v3_joint_hidden_dim=model_cfg.get("aah_v3_joint_hidden_dim", 0),
        aah_v3_diagnostic_detail=model_cfg.get("aah_v3_diagnostic_detail", "full"),
        aah_v3_reuse_group_hierarchy=model_cfg.get("aah_v3_reuse_group_hierarchy", False),
        aah_v3_hierarchy_ablation_mode=model_cfg.get("aah_v3_hierarchy_ablation_mode", "adaptive"),
        aah_v3_fixed_hierarchy_seed=model_cfg.get("aah_v3_fixed_hierarchy_seed", cfg["experiment"].get("seed", 1337)),
        aah_v3_parent_constraint=model_cfg.get("aah_v3_parent_constraint", True),
        aah_v3_attention_backend=model_cfg.get("aah_v3_attention_backend", "dense_masked"),
        aah_v3_flex_block_size=model_cfg.get("aah_v3_flex_block_size", 128),
    )
    return GPT(gpt_cfg).to(device)


def estimate_flops(model_cfg, batch_size, seq_len, attn_elements_total=None):
    n_layer = int(model_cfg["n_layer"])
    n_head = int(model_cfg["n_head"])
    n_embd = int(model_cfg["n_embd"])
    n_ff = int(model_cfg["n_ff"])
    head_dim = n_embd // n_head

    attn_full = float(n_layer * (4.0 * batch_size * seq_len * seq_len * n_embd))
    if attn_elements_total is None:
        attn_est = attn_full
    else:
        attn_est = float(4.0 * batch_size * head_dim * attn_elements_total)

    non_attn = float(n_layer * (8.0 * batch_size * seq_len * n_embd * n_embd + 4.0 * batch_size * seq_len * n_embd * n_ff))
    total_est = attn_est + non_attn
    total_full = attn_full + non_attn
    ratio = (total_est / total_full) if total_full > 0 else 1.0
    reduction_pct = (1.0 - ratio) * 100.0
    return attn_est, total_est, ratio, reduction_pct


def evaluate(model, loader, device, max_batches, use_bf16=False):
    model.eval()
    for block in model.blocks:
        if hasattr(block.attn, "set_eval_mode"):
            block.attn.set_eval_mode(True)
    autocast_ctx = nullcontext()
    if use_bf16:
        if device == "cuda":
            autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
        elif device == "cpu":
            autocast_ctx = torch.autocast("cpu", dtype=torch.bfloat16)
    losses = []
    total_tokens = 0
    total_attn_flops = 0.0
    total_flops = 0.0
    total_full_flops = 0.0
    total_attn_elements = 0.0
    total_baseline_elements = 0.0
    total_effective_attn_elements = 0.0
    total_dense_kernel_actual_elements_est = 0.0
    total_backend_realized_elements_est = 0.0
    backend_names = set()
    requested_backends = set()
    backend_bucket_counts = {}
    backend_kernel_calls = 0
    backend_fallback_reasons = set()
    n = 0
    t0 = time.time()
    resolution_per_head_sum = None
    resolution_per_head_count = 0
    branch_usage_sum = {}
    branch_usage_count = 0
    with torch.no_grad(), autocast_ctx:
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
            total_tokens += x.numel()
            attn_elems = None
            if getattr(model.config, "aah_v2_enabled", False) or getattr(model.config, "aah_v3_enabled", False):
                te = 0.0
                be = 0.0
                for block in model.blocks:
                    attn = block.attn
                    if hasattr(attn, "last_stats"):
                        stats = attn.last_stats
                        te += stats.get("total_elements", 0.0)
                        be += stats.get("baseline_elements", 0.0)
                        total_effective_attn_elements += float(stats.get("effective_attn_elements", stats.get("total_elements", 0.0)))
                        total_dense_kernel_actual_elements_est += float(stats.get("dense_kernel_actual_elements_est", stats.get("baseline_elements", 0.0)))
                        total_backend_realized_elements_est += float(stats.get("backend_realized_elements_est", stats.get("baseline_elements", 0.0)))
                        if stats.get("backend_name"):
                            backend_names.add(str(stats.get("backend_name")))
                        if stats.get("requested_backend"):
                            requested_backends.add(str(stats.get("requested_backend")))
                        for k, v in stats.get("backend_bucket_counts", {}).items():
                            ks = str(k)
                            backend_bucket_counts[ks] = backend_bucket_counts.get(ks, 0) + int(v)
                        backend_kernel_calls += int(stats.get("backend_kernel_calls", 0))
                        for reason in stats.get("backend_fallback_reasons", []):
                            backend_fallback_reasons.add(str(reason))
                if be > 0:
                    attn_elems = te
                    total_attn_elements += te
                    total_baseline_elements += be
            else:
                full_elements = float(model.config.n_layer * model.config.n_head * x.size(1) * x.size(1))
                total_attn_elements += full_elements
                total_baseline_elements += full_elements
                total_effective_attn_elements += full_elements
                total_dense_kernel_actual_elements_est += full_elements
                total_backend_realized_elements_est += full_elements
            fa, ft, _, _ = estimate_flops(
                {
                    "n_layer": model.config.n_layer,
                    "n_head": model.config.n_head,
                    "n_embd": model.config.n_embd,
                    "n_ff": model.config.n_ff,
                },
                int(x.size(0)),
                int(x.size(1)),
                attn_elements_total=attn_elems,
            )
            _, ff_total, _, _ = estimate_flops(
                {
                    "n_layer": model.config.n_layer,
                    "n_head": model.config.n_head,
                    "n_embd": model.config.n_embd,
                    "n_ff": model.config.n_ff,
                },
                int(x.size(0)),
                int(x.size(1)),
                attn_elements_total=None,
            )
            total_attn_flops += fa
            total_flops += ft
            total_full_flops += ff_total
            n += 1
            if getattr(model.config, "aah_v2_enabled", False) or getattr(model.config, "aah_v3_enabled", False):
                for block in model.blocks:
                    attn = block.attn
                    if not hasattr(attn, "last_stats"):
                        continue
                    lk = attn.last_stats.get("lk", [])
                    if lk:
                        lk_t = torch.tensor(lk, dtype=torch.float32)
                        if resolution_per_head_sum is None:
                            resolution_per_head_sum = torch.zeros_like(lk_t)
                        if resolution_per_head_sum.shape == lk_t.shape:
                            resolution_per_head_sum += lk_t
                            resolution_per_head_count += 1
                    branch_usage = attn.last_stats.get("branch_usage_freq", {})
                    if branch_usage:
                        for k, v in branch_usage.items():
                            ks = str(k)
                            branch_usage_sum[ks] = branch_usage_sum.get(ks, 0.0) + float(v)
                        branch_usage_count += 1
    elapsed = time.time() - t0
    for block in model.blocks:
        if hasattr(block.attn, "set_eval_mode"):
            block.attn.set_eval_mode(False)
    avg_loss = sum(losses) / max(1, len(losses))
    ppl = math.exp(avg_loss) if losses else float("inf")
    tok_s = total_tokens / max(1e-9, elapsed)
    flops_attn_est = total_attn_flops / max(1, n)
    flops_total_est = total_flops / max(1, n)
    flops_ratio = (total_flops / total_full_flops) if total_full_flops > 0 else 1.0
    flops_reduction_pct = (1.0 - flops_ratio) * 100.0
    attn_ratio = (total_attn_elements / total_baseline_elements) if total_baseline_elements > 0 else 1.0
    backend_metrics = {
        "effective_attn_elements": total_effective_attn_elements,
        "effective_ACR": (total_effective_attn_elements / total_baseline_elements) if total_baseline_elements > 0 else 1.0,
        "dense_kernel_actual_elements_est": total_dense_kernel_actual_elements_est,
        "backend_realized_elements_est": total_backend_realized_elements_est,
        "backend_realized_ACR_est": (
            total_backend_realized_elements_est / total_dense_kernel_actual_elements_est
            if total_dense_kernel_actual_elements_est > 0
            else 1.0
        ),
        "backend_name": next(iter(backend_names)) if len(backend_names) == 1 else ("mixed" if backend_names else "dense_masked"),
        "requested_backend": next(iter(requested_backends)) if len(requested_backends) == 1 else ("mixed" if requested_backends else "dense_masked"),
        "backend_bucket_counts": backend_bucket_counts,
        "backend_kernel_calls": backend_kernel_calls,
        "backend_fallback_reasons": sorted(backend_fallback_reasons),
    }
    resolution_per_head_mean = None
    if resolution_per_head_sum is not None and resolution_per_head_count > 0:
        resolution_per_head_mean = (resolution_per_head_sum / float(resolution_per_head_count)).tolist()
    branch_usage_mean = {}
    if branch_usage_count > 0:
        for k, v in branch_usage_sum.items():
            branch_usage_mean[k] = v / float(branch_usage_count)
    return (
        avg_loss,
        ppl,
        tok_s,
        elapsed,
        flops_attn_est,
        flops_total_est,
        flops_ratio,
        flops_reduction_pct,
        attn_ratio,
        resolution_per_head_mean,
        branch_usage_mean,
        backend_metrics,
    )


def _json_cell(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _window_idx_from_size(size, windows):
    try:
        return list(windows).index(int(size))
    except Exception:
        return ""


def write_mechanism_diagnostics(model, cfg, row, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    exp = cfg["experiment"]
    model_cfg = cfg["model"]
    windows = list(model_cfg.get("aah_v3_windows", [64, 128, 256, cfg["data"]["seq_len"]]))
    context_length = int(cfg["data"]["seq_len"])
    regime = exp.get("variant", exp.get("name", ""))
    seed = int(row.get("seed", exp.get("seed", -1)))
    checkpoint_step = row.get("checkpoint_step")
    run_name = exp.get("name", "")

    safe_name = run_name.replace("/", "_")
    step_label = checkpoint_step if checkpoint_step is not None else "unknown"
    heatmap_path = os.path.join(out_dir, f"{safe_name}_step{step_label}_heatmap.csv")
    sibling_path = os.path.join(out_dir, f"{safe_name}_step{step_label}_siblings.csv")

    heatmap_fields = [
        "run_name",
        "regime",
        "seed",
        "context_length",
        "checkpoint_step",
        "layer_id",
        "head_id",
        "group_id",
        "selected_window_idx",
        "selected_window_size",
        "pre_clamp_window_idx",
        "post_clamp_window_idx",
        "final_window_idx",
        "final_window_size",
        "hierarchy_levels_used",
        "group_counts_per_level",
        "path_mode",
    ]
    sibling_fields = [
        "run_name",
        "regime",
        "seed",
        "context_length",
        "checkpoint_step",
        "layer_id",
        "hierarchy_level",
        "sibling_pair_id",
        "left_group_id",
        "right_group_id",
        "left_raw_window_idx",
        "right_raw_window_idx",
        "left_final_window_idx",
        "right_final_window_idx",
        "same_raw_choice",
        "same_final_choice",
        "abs_window_idx_difference",
        "sibling_choice_entropy",
        "joint_scorer_logits_left",
        "joint_scorer_logits_right",
    ]

    with open(heatmap_path, "w", newline="") as f_heat, open(sibling_path, "w", newline="") as f_sib:
        heat_writer = csv.DictWriter(f_heat, fieldnames=heatmap_fields)
        sib_writer = csv.DictWriter(f_sib, fieldnames=sibling_fields)
        heat_writer.writeheader()
        sib_writer.writeheader()

        for layer_id, block in enumerate(model.blocks):
            attn = block.attn
            stats = getattr(attn, "last_stats", {})
            if not isinstance(stats, dict):
                continue
            group_ids = stats.get("head_groups", []) or []
            pre_idx = stats.get("win_idx_pre_clamp", []) or stats.get("decision_head_idx_before_execution_mapping", []) or []
            post_idx = stats.get("win_idx_post_clamp", []) or stats.get("decision_head_idx_before_execution_mapping", []) or []
            final_idx = stats.get("decision_head_idx_after_execution_mapping", []) or []
            final_sizes = stats.get("resolution_per_head", []) or []
            selected_idx = stats.get("decision_head_idx_before_execution_mapping", []) or post_idx or pre_idx
            n_head = int(getattr(attn, "n_head", len(final_sizes) or len(group_ids) or 0))

            for head_id in range(n_head):
                final_size = final_sizes[head_id] if head_id < len(final_sizes) else ""
                fidx = final_idx[head_id] if head_id < len(final_idx) else _window_idx_from_size(final_size, windows)
                sidx = selected_idx[head_id] if head_id < len(selected_idx) else fidx
                heat_writer.writerow(
                    {
                        "run_name": run_name,
                        "regime": regime,
                        "seed": seed,
                        "context_length": context_length,
                        "checkpoint_step": checkpoint_step,
                        "layer_id": layer_id,
                        "head_id": head_id,
                        "group_id": group_ids[head_id] if head_id < len(group_ids) else "",
                        "selected_window_idx": sidx,
                        "selected_window_size": windows[int(sidx)] if isinstance(sidx, int) and 0 <= int(sidx) < len(windows) else final_size,
                        "pre_clamp_window_idx": pre_idx[head_id] if head_id < len(pre_idx) else "",
                        "post_clamp_window_idx": post_idx[head_id] if head_id < len(post_idx) else "",
                        "final_window_idx": fidx,
                        "final_window_size": final_size,
                        "hierarchy_levels_used": stats.get("hierarchy_levels_used", ""),
                        "group_counts_per_level": _json_cell(stats.get("group_counts_per_level", [])),
                        "path_mode": stats.get("path_mode", ""),
                    }
                )

            raw_levels = stats.get("decision_raw_idx_per_level", []) or []
            final_levels = stats.get("decision_post_parent_idx_per_level", []) or []
            logits_levels = stats.get("decision_logits_per_level", []) or []
            for level, raw in enumerate(raw_levels):
                if not isinstance(raw, list) or len(raw) < 2:
                    continue
                final = final_levels[level] if level < len(final_levels) and isinstance(final_levels[level], list) else raw
                logits = logits_levels[level] if level < len(logits_levels) and isinstance(logits_levels[level], list) else []
                pair_id = 0
                for left in range(0, len(raw) - 1, 2):
                    right = left + 1
                    left_final = final[left] if left < len(final) else raw[left]
                    right_final = final[right] if right < len(final) else raw[right]
                    diff = abs(int(left_final) - int(right_final))
                    if int(left_final) == int(right_final):
                        entropy = 0.0
                    else:
                        entropy = math.log(2.0)
                    sib_writer.writerow(
                        {
                            "run_name": run_name,
                            "regime": regime,
                            "seed": seed,
                            "context_length": context_length,
                            "checkpoint_step": checkpoint_step,
                            "layer_id": layer_id,
                            "hierarchy_level": level,
                            "sibling_pair_id": pair_id,
                            "left_group_id": left,
                            "right_group_id": right,
                            "left_raw_window_idx": raw[left],
                            "right_raw_window_idx": raw[right],
                            "left_final_window_idx": left_final,
                            "right_final_window_idx": right_final,
                            "same_raw_choice": int(raw[left]) == int(raw[right]),
                            "same_final_choice": int(left_final) == int(right_final),
                            "abs_window_idx_difference": diff,
                            "sibling_choice_entropy": entropy,
                            "joint_scorer_logits_left": _json_cell(logits[left] if left < len(logits) else ""),
                            "joint_scorer_logits_right": _json_cell(logits[right] if right < len(logits) else ""),
                        }
                    )
                    pair_id += 1
    return heatmap_path, sibling_path


def parse_step_from_checkpoint_name(path):
    base = os.path.basename(path)
    if "_step" not in base:
        return None
    tail = base.split("_step", 1)[1]
    digits = "".join(ch for ch in tail if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except Exception:
        return None


def load_checkpoint_metadata(ckpt_path):
    meta_path = f"{ckpt_path}.meta.json"
    if not os.path.exists(meta_path):
        return None, meta_path
    with open(meta_path, "r") as f:
        return json.load(f), meta_path


def validate_checkpoint_metadata(meta, expected_run_name, expected_config_hash, expected_seed):
    mismatches = []
    if str(meta.get("run_name", "")) != str(expected_run_name):
        mismatches.append(f"run_name meta={meta.get('run_name')} expected={expected_run_name}")
    if str(meta.get("config_hash", "")) != str(expected_config_hash):
        mismatches.append(f"config_hash meta={meta.get('config_hash')} expected={expected_config_hash}")
    if int(meta.get("seed", -1)) != int(expected_seed):
        mismatches.append(f"seed meta={meta.get('seed')} expected={expected_seed}")
    return mismatches


def mean_std(values):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return sum(vals) / len(vals), statistics.pstdev(vals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--checkpoints", nargs="+", default=None, help="Explicit checkpoint paths for multi-checkpoint evaluation")
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--strict-checkpoint", dest="strict_checkpoint", action="store_true", help="Fail if checkpoint path is missing or metadata mismatches")
    parser.add_argument("--no-strict-checkpoint", dest="strict_checkpoint", action="store_false", help="Allow non-strict checkpoint behavior")
    parser.add_argument("--allow-fallback", action="store_true", help="Allow fallback checkpoint search when strict mode is off")
    parser.add_argument("--allow-missing-metadata", action="store_true", help="Allow checkpoint loading without sidecar metadata")
    parser.add_argument("--allow-metadata-mismatch", action="store_true", help="Allow config/run/seed mismatch between checkpoint metadata and current config")
    parser.add_argument("--deterministic-eval", dest="deterministic_eval", action="store_true", help="Enable deterministic evaluation controls")
    parser.add_argument("--no-deterministic-eval", dest="deterministic_eval", action="store_false", help="Disable deterministic evaluation controls")
    parser.add_argument("--seed", type=int, default=None, help="Override eval seed")
    parser.add_argument("--summary-json", default=None, help="Write per-checkpoint and aggregate summary JSON")
    parser.add_argument("--diagnostics-dir", default=None, help="Write per-layer/head heatmap and sibling diagnostic CSVs")
    parser.set_defaults(strict_checkpoint=True, deterministic_eval=True)
    args = parser.parse_args()

    if args.checkpoint and args.checkpoints:
        raise ValueError("Use either --checkpoint or --checkpoints, not both.")

    config_path = os.path.abspath(args.config)
    cfg = load_config(config_path)
    exp = cfg["experiment"]
    train = cfg["train"]
    data = cfg["data"]
    use_wandb = train.get("use_wandb", False)
    config_hash = compute_file_sha256(config_path)
    expected_seed = int(args.seed) if args.seed is not None else int(exp["seed"])
    expected_run_name = exp["name"]
    current_commit = get_git_commit()

    if args.checkpoints:
        requested_checkpoints = args.checkpoints
    else:
        requested_checkpoints = [args.checkpoint]
    checkpoint_paths = [
        resolve_checkpoint(
            exp,
            ckpt_arg=ckpt_arg,
            strict=args.strict_checkpoint,
            allow_fallback=args.allow_fallback,
        )
        for ckpt_arg in requested_checkpoints
    ]

    device = get_device(train.get("device", "auto"))
    precision = train.get("precision", "fp32").lower()
    use_bf16 = precision == "bf16" and device in ("cuda", "cpu")
    if use_bf16 and device == "cuda" and not torch.cuda.is_bf16_supported():
        use_bf16 = False
    if args.deterministic_eval:
        seed_everything(expected_seed)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        if torch.cuda.is_available() and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    eval_num_workers = 0 if args.deterministic_eval else data["num_workers"]

    _, val_loader, vocab_size = build_dataloaders(
        data["dataset"], data["tokenizer"], data["seq_len"], train["batch_size"], eval_num_workers
    )
    model = build_model(cfg, vocab_size, device)

    run = None
    if use_wandb:
        try:
            import wandb
            run = wandb.init(
                project="ENA-AAH",
                name=f"{exp['name']}-infer" if len(checkpoint_paths) == 1 else f"{exp['name']}-infer-multi",
                config=cfg,
                job_type="inference",
                reinit=True,
            )
        except Exception as exc:
            print(f"wandb init failed: {exc}")
            run = None

    eval_batches = args.eval_batches if args.eval_batches is not None else train.get("eval_batches", 50)
    print(
        f"run_name={expected_run_name} config_path={config_path} config_hash={config_hash} "
        f"git_commit={current_commit or 'unknown'} seed={expected_seed} strict={args.strict_checkpoint}"
    )

    results = []
    for idx, ckpt in enumerate(checkpoint_paths):
        meta, meta_path = load_checkpoint_metadata(ckpt)
        if meta is None and not args.allow_missing_metadata:
            raise FileNotFoundError(f"Checkpoint metadata missing: {meta_path}")
        if meta is not None and not args.allow_metadata_mismatch:
            mismatches = validate_checkpoint_metadata(
                meta,
                expected_run_name=expected_run_name,
                expected_config_hash=config_hash,
                expected_seed=expected_seed,
            )
            if mismatches:
                raise RuntimeError(
                    f"Checkpoint metadata mismatch for {ckpt}: " + "; ".join(mismatches)
                )

        if args.deterministic_eval:
            seed_everything(expected_seed)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state, strict=True)
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        (
            val_loss,
            val_ppl,
            tok_s,
            elapsed,
            flops_attn_est,
            flops_total_est,
            flops_ratio,
            flops_reduction_pct,
            attn_ratio,
            resolution_per_head_mean,
            branch_usage_mean,
            backend_metrics,
        ) = evaluate(
            model,
            val_loader,
            device,
            eval_batches,
            use_bf16=use_bf16,
        )
        peak_memory_mb = None
        if device == "cuda" and torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        ckpt_step = None
        if meta is not None:
            try:
                ckpt_step = int(meta.get("step"))
            except Exception:
                ckpt_step = None
        if ckpt_step is None:
            ckpt_step = parse_step_from_checkpoint_name(ckpt)

        row = {
            "checkpoint_index": idx,
            "checkpoint_path": ckpt,
            "checkpoint_step": ckpt_step,
            "val_loss": float(val_loss),
            "val_ppl": float(val_ppl),
            "tok_s": float(tok_s),
            "elapsed_s": float(elapsed),
            "flops_attn_est": float(flops_attn_est),
            "flops_total_est": float(flops_total_est),
            "flops_ratio": float(flops_ratio),
            "flops_reduction_pct": float(flops_reduction_pct),
            "ACR": float(attn_ratio),
            "effective_attn_elements": float(backend_metrics["effective_attn_elements"]),
            "effective_ACR": float(backend_metrics["effective_ACR"]),
            "dense_kernel_actual_elements_est": float(backend_metrics["dense_kernel_actual_elements_est"]),
            "backend_realized_elements_est": float(backend_metrics["backend_realized_elements_est"]),
            "backend_realized_ACR_est": float(backend_metrics["backend_realized_ACR_est"]),
            "backend_name": backend_metrics["backend_name"],
            "requested_backend": backend_metrics["requested_backend"],
            "backend_bucket_counts": backend_metrics["backend_bucket_counts"],
            "backend_kernel_calls": int(backend_metrics["backend_kernel_calls"]),
            "backend_fallback_reasons": backend_metrics["backend_fallback_reasons"],
            "peak_memory_mb": peak_memory_mb,
            "resolution_per_head_mean": resolution_per_head_mean,
            "branch_usage_freq": branch_usage_mean,
            "run_name": meta.get("run_name") if meta else expected_run_name,
            "seed": int(meta.get("seed")) if meta and "seed" in meta else expected_seed,
            "config_hash": meta.get("config_hash") if meta else config_hash,
            "git_commit": meta.get("git_commit") if meta else current_commit,
            "config_path": config_path,
            "metadata_path": meta_path if meta else "",
        }
        results.append(row)
        if args.diagnostics_dir:
            heatmap_path, sibling_path = write_mechanism_diagnostics(model, cfg, row, args.diagnostics_dir)
            row["heatmap_csv"] = heatmap_path
            row["sibling_csv"] = sibling_path
            print(f"diagnostics heatmap_csv={heatmap_path} sibling_csv={sibling_path}")
        print(
            f"[{idx+1}/{len(checkpoint_paths)}] checkpoint={ckpt} step={ckpt_step} "
            f"val_loss={val_loss:.6f} val_ppl={val_ppl:.4f} tok_s={tok_s:.2f} "
            f"ACR={attn_ratio:.6f} backend_realized_ACR={row['backend_realized_ACR_est']:.6f} "
            f"backend={row['backend_name']} flops_ratio={flops_ratio:.6f}"
        )
        if run is not None:
            try:
                run.log(
                    {
                        "infer/checkpoint_index": idx,
                        "infer/checkpoint": ckpt,
                        "infer/checkpoint_step": ckpt_step if ckpt_step is not None else -1,
                        "infer/val_loss": val_loss,
                        "infer/val_ppl": val_ppl,
                        "infer/tok_s": tok_s,
                        "infer/elapsed_s": elapsed,
                        "infer/eval_batches": eval_batches,
                        "infer/config": config_path,
                        "infer/run_name": row["run_name"],
                        "infer/seed": row["seed"],
                        "infer/git_commit": row["git_commit"] or "unknown",
                        "aah/flops_attn_est": flops_attn_est,
                        "aah/flops_total_est": flops_total_est,
                        "aah/flops_ratio": flops_ratio,
                        "aah/flops_reduction_%": flops_reduction_pct,
                        "aah/ACR": attn_ratio,
                        "aah/effective_attn_elements": row["effective_attn_elements"],
                        "aah/effective_ACR": row["effective_ACR"],
                        "aah/dense_kernel_actual_elements_est": row["dense_kernel_actual_elements_est"],
                        "aah/backend_realized_elements_est": row["backend_realized_elements_est"],
                        "aah/backend_realized_ACR_est": row["backend_realized_ACR_est"],
                        "aah/backend_name": row["backend_name"],
                        "aah/requested_backend": row["requested_backend"],
                        "aah/backend_bucket_counts": row["backend_bucket_counts"],
                        "aah/backend_kernel_calls": row["backend_kernel_calls"],
                        "aah/backend_fallback_reasons": row["backend_fallback_reasons"],
                        "aah/resolution_per_head_mean": resolution_per_head_mean,
                        "aah/branch_usage_freq": branch_usage_mean,
                    }
                )
            except Exception as exc:
                print(f"wandb logging failed: {exc}")

    val_loss_mean, val_loss_std = mean_std([r["val_loss"] for r in results])
    val_ppl_mean, val_ppl_std = mean_std([r["val_ppl"] for r in results])
    flops_ratio_mean, flops_ratio_std = mean_std([r["flops_ratio"] for r in results])
    effective_ACR_mean, effective_ACR_std = mean_std([r["effective_ACR"] for r in results])
    backend_realized_ACR_mean, backend_realized_ACR_std = mean_std([r["backend_realized_ACR_est"] for r in results])
    tok_s_mean, tok_s_std = mean_std([r["tok_s"] for r in results])
    best_row = min(results, key=lambda r: r["val_ppl"])
    last_row = results[-1]
    ppl_values = [r["val_ppl"] for r in results]
    loss_values = [r["val_loss"] for r in results]
    summary = {
        "n_checkpoints": len(results),
        "val_loss_mean": val_loss_mean,
        "val_loss_std": val_loss_std,
        "val_ppl_mean": val_ppl_mean,
        "val_ppl_std": val_ppl_std,
        "flops_ratio_mean": flops_ratio_mean,
        "flops_ratio_std": flops_ratio_std,
        "effective_ACR_mean": effective_ACR_mean,
        "effective_ACR_std": effective_ACR_std,
        "backend_realized_ACR_mean": backend_realized_ACR_mean,
        "backend_realized_ACR_std": backend_realized_ACR_std,
        "tok_s_mean": tok_s_mean,
        "tok_s_std": tok_s_std,
        "best_checkpoint": best_row["checkpoint_path"],
        "best_checkpoint_step": best_row["checkpoint_step"],
        "best_val_ppl": best_row["val_ppl"],
        "best_val_loss": best_row["val_loss"],
        "last_checkpoint": last_row["checkpoint_path"],
        "last_checkpoint_step": last_row["checkpoint_step"],
        "last_val_ppl": last_row["val_ppl"],
        "last_val_loss": last_row["val_loss"],
        "checkpoint_sensitivity_val_ppl": max(ppl_values) - min(ppl_values),
        "checkpoint_sensitivity_val_loss": max(loss_values) - min(loss_values),
    }

    print(
        "summary "
        f"n={summary['n_checkpoints']} "
        f"val_ppl_mean={summary['val_ppl_mean']:.4f} val_ppl_std={summary['val_ppl_std']:.4f} "
        f"val_loss_mean={summary['val_loss_mean']:.6f} val_loss_std={summary['val_loss_std']:.6f} "
        f"flops_ratio_mean={summary['flops_ratio_mean']:.6f} flops_ratio_std={summary['flops_ratio_std']:.6f} "
        f"backend_realized_ACR_mean={summary['backend_realized_ACR_mean']:.6f} "
        f"best_val_ppl={summary['best_val_ppl']:.4f} last_val_ppl={summary['last_val_ppl']:.4f} "
        f"delta_ppl={summary['checkpoint_sensitivity_val_ppl']:.4f} "
        f"delta_loss={summary['checkpoint_sensitivity_val_loss']:.6f}"
    )

    if run is not None:
        try:
            run.log(
                {
                    "infer_multi/n_checkpoints": summary["n_checkpoints"],
                    "infer_multi/val_ppl_mean": summary["val_ppl_mean"],
                    "infer_multi/val_ppl_std": summary["val_ppl_std"],
                    "infer_multi/val_loss_mean": summary["val_loss_mean"],
                    "infer_multi/val_loss_std": summary["val_loss_std"],
                    "infer_multi/flops_ratio_mean": summary["flops_ratio_mean"],
                    "infer_multi/flops_ratio_std": summary["flops_ratio_std"],
                    "infer_multi/effective_ACR_mean": summary["effective_ACR_mean"],
                    "infer_multi/effective_ACR_std": summary["effective_ACR_std"],
                    "infer_multi/backend_realized_ACR_mean": summary["backend_realized_ACR_mean"],
                    "infer_multi/backend_realized_ACR_std": summary["backend_realized_ACR_std"],
                    "infer_multi/tok_s_mean": summary["tok_s_mean"],
                    "infer_multi/tok_s_std": summary["tok_s_std"],
                    "infer_multi/best_val_ppl": summary["best_val_ppl"],
                    "infer_multi/last_val_ppl": summary["last_val_ppl"],
                    "infer_multi/checkpoint_sensitivity_val_ppl": summary["checkpoint_sensitivity_val_ppl"],
                    "infer_multi/checkpoint_sensitivity_val_loss": summary["checkpoint_sensitivity_val_loss"],
                }
            )
            run.finish()
        except Exception as exc:
            print(f"wandb logging failed: {exc}")

    if len(results) == 1:
        r = results[0]
        print(f"config={config_path}")
        print(f"checkpoint={r['checkpoint_path']}")
        print(f"device={device} precision={precision}")
        print(
            f"eval_batches={eval_batches} val_loss={r['val_loss']:.6f} "
            f"val_ppl={r['val_ppl']:.4f} tok_s={r['tok_s']:.2f} elapsed_s={r['elapsed_s']:.2f}"
        )
        print(
            f"aah/flops_attn_est={r['flops_attn_est']:.2f} aah/flops_total_est={r['flops_total_est']:.2f} "
            f"aah/ACR={r['ACR']:.6f} aah/flops_ratio={r['flops_ratio']:.6f} "
            f"aah/flops_reduction_%={r['flops_reduction_pct']:.4f} "
            f"aah/backend_realized_ACR_est={r['backend_realized_ACR_est']:.6f} "
            f"aah/backend={r['backend_name']}"
        )

    if args.summary_json:
        out_path = os.path.abspath(args.summary_json)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {
            "meta": {
                "config_path": config_path,
                "config_hash": config_hash,
                "run_name": expected_run_name,
                "seed": expected_seed,
                "git_commit": current_commit,
                "eval_batches": int(eval_batches),
                "device": device,
                "precision": precision,
                "strict_checkpoint": bool(args.strict_checkpoint),
                "deterministic_eval": bool(args.deterministic_eval),
            },
            "checkpoints": checkpoint_paths,
            "results": results,
            "summary": summary,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"summary_json={out_path}")


if __name__ == "__main__":
    main()
