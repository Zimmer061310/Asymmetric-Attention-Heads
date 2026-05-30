import os
import time
import math
import csv
import sys
import argparse
import json
import hashlib
import subprocess
import yaml
import platform
import resource
import traceback
from datetime import datetime, timezone
from collections import deque
from contextlib import nullcontext
import importlib
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
try:
    import psutil
except Exception:
    psutil = None

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data import build_dataloaders

TRANSFORMER_MODULE = os.environ.get(
    "AAH_BACKEND_TRANSFORMER_MODULE",
    "experiments.backend_realized_local_attention._common.aah_backend_transformer",
)
_transformer_mod = importlib.import_module(TRANSFORMER_MODULE)
GPT = _transformer_mod.GPT
GPTConfig = _transformer_mod.GPTConfig


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device(device_pref):
    if device_pref == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_pref


def linear_warmup_cosine(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def evaluate(model, loader, device, max_batches, log_progress=False, use_bf16=False):
    model.eval()
    for block in model.blocks:
        attn = block.attn
        if hasattr(attn, "set_eval_mode"):
            attn.set_eval_mode(True)
    losses = []
    t0 = time.time()
    autocast_ctx = nullcontext()
    if use_bf16:
        if device == "cuda":
            autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
        elif device == "cpu":
            autocast_ctx = torch.autocast("cpu", dtype=torch.bfloat16)
    with torch.no_grad(), autocast_ctx:
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            if log_progress and i % 5 == 0:
                print(f"eval batch {i}/{max_batches}")
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    for block in model.blocks:
        attn = block.attn
        if hasattr(attn, "set_eval_mode"):
            attn.set_eval_mode(False)
    if not losses:
        return float("inf"), float("inf"), 0.0
    avg_loss = sum(losses) / len(losses)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl, time.time() - t0


def get_memory_stats():
    gpu_alloc = None
    gpu_reserved = None
    gpu_alloc_max = None
    gpu_reserved_max = None
    if torch.cuda.is_available():
        gpu_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        gpu_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        gpu_alloc_max = torch.cuda.max_memory_allocated() / (1024 ** 2)
        gpu_reserved_max = torch.cuda.max_memory_reserved() / (1024 ** 2)
    cpu_rss = None
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            cpu_rss = rss / (1024 ** 2)
        else:
            cpu_rss = rss / 1024
    except Exception:
        pass
    return gpu_alloc, gpu_reserved, gpu_alloc_max, gpu_reserved_max, cpu_rss


def get_psutil_memory_mb():
    if psutil is None:
        return {}
    try:
        proc = psutil.Process(os.getpid())
        info = proc.memory_info()
        full = proc.memory_full_info()
        vm = psutil.virtual_memory()
        def mb(x):
            return x / (1024 ** 2) if x is not None else None
        return {
            "psutil_rss_mb": mb(getattr(info, "rss", None)),
            "psutil_vms_mb": mb(getattr(info, "vms", None)),
            "psutil_shared_mb": mb(getattr(info, "shared", None)),
            "psutil_text_mb": mb(getattr(info, "text", None)),
            "psutil_data_mb": mb(getattr(info, "data", None)),
            "psutil_uss_mb": mb(getattr(full, "uss", None)),
            "psutil_pss_mb": mb(getattr(full, "pss", None)),
            "psutil_swap_mb": mb(getattr(full, "swap", None)),
            "psutil_ram_used_mb": mb(getattr(vm, "used", None)),
            "psutil_ram_total_mb": mb(getattr(vm, "total", None)),
        }
    except Exception:
        return {}
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


def default_checkpoint_steps(max_steps):
    if max_steps <= 1:
        return [max_steps]
    s1 = max(1, max_steps - 1000)
    s2 = max(1, max_steps - 500)
    return sorted(set([s1, s2, max_steps]))


def save_checkpoint_with_metadata(model, ckpt_path, metadata):
    torch.save(model.state_dict(), ckpt_path)
    meta_path = f"{ckpt_path}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def get_group_count_metrics(model):
    per_level_sums = []
    per_level_counts = []
    for block in model.blocks:
        attn = block.attn
        if not hasattr(attn, "last_stats"):
            continue
        counts = attn.last_stats.get("group_counts_per_level")
        if counts is None:
            continue
        if not isinstance(counts, (list, tuple)):
            try:
                counts = list(counts)
            except TypeError:
                continue
        if len(counts) > 0:
            vals = [float(v) for v in counts]
            if len(per_level_sums) < len(vals):
                need = len(vals) - len(per_level_sums)
                per_level_sums.extend([0.0] * need)
                per_level_counts.extend([0] * need)
            for i, v in enumerate(vals):
                per_level_sums[i] += v
                per_level_counts[i] += 1
    if not per_level_sums:
        return None, None, []
    per_level_means = []
    for s, c in zip(per_level_sums, per_level_counts):
        per_level_means.append((s / c) if c > 0 else 0.0)
    total_mean = sum(per_level_means)
    level0_mean = per_level_means[0] if per_level_means else None
    return total_mean, level0_mean, per_level_means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/small.yaml")
    args = parser.parse_args()
    config_path = os.path.abspath(args.config)
    cfg = load_config(config_path)
    exp = cfg["experiment"]
    data = cfg["data"]
    model_cfg = cfg["model"]
    train = cfg["train"]
    config_hash = compute_file_sha256(config_path)
    git_commit = get_git_commit()

    seed = int(exp["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = get_device(train["device"])
    precision = train.get("precision", "fp32").lower()
    use_bf16 = precision == "bf16"
    if use_bf16:
        if device == "cuda" and not torch.cuda.is_bf16_supported():
            print("Warning: BF16 requested but not supported on this CUDA device. Falling back to FP32.")
            use_bf16 = False
        if device == "mps":
            print("Warning: BF16 autocast not supported on MPS. Falling back to FP32.")
            use_bf16 = False

    train_loader, val_loader, vocab_size = build_dataloaders(
        data["dataset"],
        data["tokenizer"],
        data["seq_len"],
        train["batch_size"],
        data["num_workers"],
    )

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
        aah_v3_fixed_hierarchy_seed=model_cfg.get("aah_v3_fixed_hierarchy_seed", exp.get("seed", 1337)),
        aah_v3_parent_constraint=model_cfg.get("aah_v3_parent_constraint", True),
        aah_v3_attention_backend=model_cfg.get("aah_v3_attention_backend", model_cfg.get("attention_backend", "dense_masked")),
        aah_v3_flex_block_size=model_cfg.get("aah_v3_flex_block_size", model_cfg.get("flex_block_size", 128)),
    )
    model = GPT(gpt_cfg).to(device)

    opt = AdamW(model.parameters(), lr=train["lr"], weight_decay=train["weight_decay"])
    scheduler = LambdaLR(opt, lambda s: linear_warmup_cosine(s, train["warmup_steps"], train["max_steps"]))

    use_wandb = train.get("use_wandb", False)
    log_csv = train.get("log_csv", False)
    cfg_log_interval = int(train.get("log_interval", 50))
    effective_log_interval = max(1, cfg_log_interval)
    out_dir = exp.get("out_dir", "experiments")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{exp['name']}_{exp.get('variant','run')}.csv")
    print(
        f"run_name={exp['name']} config_path={config_path} "
        f"config_hash={config_hash} git_commit={git_commit or 'unknown'} seed={seed}"
    )
    save_checkpoints = bool(train.get("save_checkpoints", True))
    checkpoint_steps_cfg = train.get("checkpoint_steps")
    if not save_checkpoints:
        checkpoint_steps = []
    elif checkpoint_steps_cfg is None:
        checkpoint_steps = default_checkpoint_steps(int(train["max_steps"]))
    else:
        checkpoint_steps = [int(s) for s in checkpoint_steps_cfg]
    checkpoint_steps = sorted(
        set(
            s
            for s in checkpoint_steps
            if 0 < s <= int(train["max_steps"])
        )
    )
    if save_checkpoints and int(train["max_steps"]) not in checkpoint_steps:
        checkpoint_steps.append(int(train["max_steps"]))
    checkpoint_steps_set = set(checkpoint_steps)
    print(f"checkpoint_steps={checkpoint_steps}")

    checkpoint_meta_base = {
        "run_name": exp["name"],
        "variant": exp.get("variant", "run"),
        "config_path": config_path,
        "config_hash": config_hash,
        "seed": seed,
        "git_commit": git_commit,
        "max_steps": int(train["max_steps"]),
        "saved_at_utc": "",
        "step": -1,
        "checkpoint_path": "",
        "checkpoint_role": "",
    }

    def persist_checkpoint(step_now, role):
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        step_ckpt = os.path.join(out_dir, f"{exp['name']}_step{step_now}.pt")
        step_meta = dict(checkpoint_meta_base)
        step_meta["saved_at_utc"] = timestamp_utc
        step_meta["step"] = int(step_now)
        step_meta["checkpoint_path"] = os.path.abspath(step_ckpt)
        step_meta["checkpoint_role"] = role
        save_checkpoint_with_metadata(model, step_ckpt, step_meta)
        print(
            f"saved_checkpoint step={step_now} role={role} "
            f"path={step_meta['checkpoint_path']}"
        )
        canonical_ckpt = os.path.join(out_dir, f"{exp['name']}.pt")
        if int(step_now) == int(train["max_steps"]):
            final_meta = dict(step_meta)
            final_meta["checkpoint_path"] = os.path.abspath(canonical_ckpt)
            final_meta["checkpoint_role"] = "final"
            save_checkpoint_with_metadata(model, canonical_ckpt, final_meta)
            print(
                f"saved_checkpoint step={step_now} role=final "
                f"path={final_meta['checkpoint_path']}"
            )
    csv_file = None
    csv_writer = None
    csv_headers = [
        "step",
        "train_loss",
        "tok_s",
        "mem_mb",
        "gpu_alloc_mb",
        "gpu_reserved_mb",
        "gpu_alloc_max_mb",
        "gpu_reserved_max_mb",
        "cpu_rss_mb",
        "psutil_rss_mb",
        "psutil_vms_mb",
        "psutil_shared_mb",
        "psutil_text_mb",
        "psutil_data_mb",
        "psutil_uss_mb",
        "psutil_pss_mb",
        "psutil_swap_mb",
        "psutil_ram_used_mb",
        "psutil_ram_total_mb",
        "val_loss",
        "val_ppl",
        "attn_elems",
        "attn_ratio",
        "attn_reduction",
        "effective_attn_elements",
        "effective_ACR",
        "dense_kernel_actual_elements_est",
        "backend_realized_elements_est",
        "backend_realized_ACR_est",
        "backend_name",
        "requested_backend",
        "backend_bucket_counts",
        "backend_kernel_calls",
        "backend_time_ms",
        "backend_fallback_reasons",
        "analytic_flops_attn_est",
        "analytic_flops_total_est",
        "analytic_flops_ratio",
        "analytic_flops_reduction_pct",
        "attn_lq",
        "attn_lk_per_layer",
        "head_entropy",
        "group_change_rate",
        "avg_window",
        "group_overlap",
        "head_reassign_rate",
        "group_lifespan_ema",
        "head_groups",
        "shadow_win_idx",
        "shadow_logit_mean",
        "group_heads",
        "group_ratios",
        "resolution_mean",
        "resolution_std",
        "resolution_min_frac",
        "resolution_max_frac",
        "resolution_collapse_min",
        "resolution_collapse_max",
        "resolution_delta",
        "branch_usage_freq",
        "attn_ratio_std_ema",
        "analytic_flops_ratio_std_ema",
        "lk_mean",
        "lk_p90",
        "w_mean",
        "w_min",
        "w_max",
        "control_time_ms",
        "control_feature_time_ms",
        "hierarchy_time_ms",
        "window_select_time_ms",
        "control_mapping_time_ms",
        "diagnostics_pack_time_ms",
        "attn_time_ms",
        "mask_time_ms",
        "overhead_time_ms",
        "step_time_ms",
        "eval_time_s",
        "val_group_count_total",
        "val_group_count_level0",
        "val_group_count_per_level",
        "path_mode_freq",
        "train_group_counts_per_level",
        "controller_logits_std_per_level",
        "win_idx_pre_clamp",
        "win_idx_post_clamp",
        "decision_logits_per_level",
        "decision_logits_var_per_level",
        "controller_input_per_level",
        "controller_input_cos_sim_mean_per_level",
        "controller_input_cos_sim_min_per_level",
        "controller_input_l2_dist_mean_per_level",
        "controller_input_dim_var_mean_per_level",
        "decision_logits_margin_mean_per_level",
        "decision_logits_margin_min_per_level",
        "decision_argmax_diversity_frac_per_level",
        "sibling_feature_delta_norm_mean_per_level",
        "sibling_feature_delta_norm_min_per_level",
        "sibling_feature_delta_norm_max_per_level",
        "sibling_feature_cos_mean_per_level",
        "sibling_feature_cos_min_per_level",
        "sibling_logit_delta_l2_mean_per_level",
        "sibling_logit_delta_l2_max_per_level",
        "sibling_logit_delta_abs_mean_per_level",
        "sibling_logit_delta_abs_max_per_level",
        "sibling_ranking_diff_frac_per_level",
        "sibling_top1_differ_frac_per_level",
        "sibling_top1_ids_per_level",
        "pairwise_bias_l2_mean_per_level",
        "pairwise_bias_l2_max_per_level",
        "pairwise_bias_abs_mean_per_level",
        "pairwise_bias_abs_max_per_level",
        "pairwise_bias_top1_changed_frac_per_level",
        "pairwise_base_top1_ids_per_level",
        "joint_pair_count_per_level",
        "joint_output_delta_l2_mean_per_level",
        "joint_output_delta_l2_max_per_level",
        "joint_output_abs_delta_mean_per_level",
        "joint_output_abs_delta_max_per_level",
        "joint_top1_changed_frac_per_level",
        "joint_base_top1_ids_per_level",
        "joint_output_scale_per_level",
        "decision_raw_idx_per_level",
        "decision_parent_idx_per_level",
        "decision_post_parent_idx_per_level",
        "decision_differ_from_parent_frac_per_level",
        "decision_unique_raw_idx_per_level",
        "decision_unique_post_parent_idx_per_level",
        "decision_non_one_raw_count_per_level",
        "decision_non_one_post_parent_count_per_level",
        "decision_head_idx_before_execution_mapping",
        "decision_head_idx_after_execution_mapping",
        "decision_unique_head_idx_before_execution_mapping",
        "decision_unique_head_idx_after_execution_mapping",
        "decision_head_idx_changed_by_execution_mapping_frac",
        "decision_all_layers_unique_head_idx_before_execution_mapping",
        "decision_all_layers_unique_head_idx_after_execution_mapping",
        "decision_all_layers_head_idx_changed_by_execution_mapping_frac",
        "decision_all_layers_unique_raw_idx_per_level",
        "decision_all_layers_unique_post_parent_idx_per_level",
        "all_layers_sibling_ranking_diff_frac_per_level",
        "all_layers_sibling_top1_differ_frac_per_level",
        "all_layers_sibling_top1_ids_per_level",
        "all_layers_joint_pair_count_per_level",
        "all_layers_joint_output_delta_l2_mean_per_level",
        "all_layers_joint_output_abs_delta_mean_per_level",
        "all_layers_joint_top1_changed_frac_per_level",
        "all_layers_joint_base_top1_ids_per_level",
        "hierarchy_head_group_map_per_level",
        "hierarchy_group_members_per_level",
        "cluster_metric_per_level",
        "cluster_threshold_kind_per_level",
        "cluster_threshold_per_level",
        "cluster_item_count_per_level",
        "cluster_groups_before_merge_per_level",
        "cluster_groups_after_merge_per_level",
        "cluster_groups_merged_per_level",
        "cluster_sim_min_per_level",
        "cluster_sim_mean_per_level",
        "cluster_sim_max_per_level",
        "cluster_sim_std_per_level",
        "cluster_forced_bipartition_per_level",
        "cluster_force_split_anchor_similarity_per_level",
        "cluster_origin_per_level",
        "cluster_forced_bipartition_allowed_per_level",
        "cluster_groups_before_force_per_level",
        "cluster_feature_norm_mean_per_level",
        "cluster_feature_norm_std_per_level",
        "cluster_feature_dim_var_mean_per_level",
        "cluster_feature_dim_var_std_per_level",
        "cluster_feature_l2_dist_mean_per_level",
        "cluster_feature_l2_dist_std_per_level",
        "cluster_feature_top_singular_ratio_per_level",
        "hierarchy_level_added_per_level",
        "hierarchy_growth_stopped_per_level",
        "hierarchy_stop_reason_per_level",
        "cluster_forced_bipartition_level0",
        "cluster_force_split_anchor_similarity_level0",
    ]
    csv_idx = {k: i for i, k in enumerate(csv_headers)}
    wandb_mod = None
    if use_wandb:
        try:
            import wandb
            wandb_mod = wandb
            wandb_mod.init(project="ENA-AAH", name=exp["name"], config=cfg)
            wandb_mod.log(
                {
                    "run/started": 1,
                    "step": 0,
                    "run/config_path": config_path,
                    "run/config_hash": config_hash,
                    "run/git_commit": git_commit or "unknown",
                    "run/seed": seed,
                }
            )
        except Exception:
            use_wandb = False
    if log_csv:
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_headers)

    step = 0
    prev_head_groups = None
    lifespan_ema = None
    ratio_window = deque(maxlen=20)
    analytic_flops_ratio_window = deque(maxlen=20)
    model.train()
    t0 = time.time()
    train_iter = iter(train_loader)
    last_eval_time_s = ""
    saved_checkpoint_steps = set()
    crash_log_path = os.path.join(out_dir, f"{exp['name']}_crash.log")
    try:
        while step < train["max_steps"]:
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(device), y.to(device)
            if step == 0 and device == "cuda":
                try:
                    print(f"autocast gpu dtype: {torch.get_autocast_gpu_dtype()}")
                except Exception as exc:
                    print(f"autocast gpu dtype: unavailable ({exc})")
            aah_v2_enabled = model_cfg.get("aah_v2_enabled", False)
            aah_v3_enabled = model_cfg.get("aah_v3_enabled", False)
            attention_stats_enabled = (
                aah_v2_enabled
                or aah_v3_enabled
                or model_cfg.get("aah_v3_attention_backend") in {"flex_attention", "flash_attn"}
                or model_cfg.get("attention_backend") in {"flex_attention", "flash_attn"}
            )
            warmup_steps = model_cfg.get("aah_v2_warmup_steps", 0)
            activation_steps = model_cfg.get("aah_v2_activation_steps", 0)
            compute_lambda = model_cfg.get("aah_v2_compute_lambda", 0.0)
            min_head_norm = model_cfg.get("aah_v2_min_head_norm", 0.0)
            norm_lambda = model_cfg.get("aah_v2_norm_lambda", 0.0)
            min_head_entropy = model_cfg.get("aah_v2_min_head_entropy", 0.0)
            ent_lambda = model_cfg.get("aah_v2_entropy_lambda", 0.0)
            if aah_v2_enabled:
                if step < warmup_steps:
                    control_enabled = False
                    lambda_now = 0.0
                    guardrails_enabled = False
                elif step < (warmup_steps + activation_steps):
                    control_enabled = True
                    lambda_now = 0.0
                    guardrails_enabled = True
                else:
                    control_enabled = True
                    lambda_now = compute_lambda
                    guardrails_enabled = False
                if step == warmup_steps:
                    for block in model.blocks:
                        attn = block.attn
                        if hasattr(attn, "reset_cache"):
                            attn.reset_cache()
                for block in model.blocks:
                    attn = block.attn
                    if hasattr(attn, "set_control"):
                        attn.set_control(control_enabled)
                    if hasattr(attn, "set_step"):
                        attn.set_step(step)
            if aah_v3_enabled:
                for block in model.blocks:
                    attn = block.attn
                    if hasattr(attn, "set_control"):
                        attn.set_control(model_cfg.get("aah_v3_control_enabled", True))
                    if hasattr(attn, "set_step"):
                        attn.set_step(step)
            step_t0 = time.time()
            autocast_ctx = nullcontext()
            if use_bf16:
                if device == "cuda":
                    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
                elif device == "cpu":
                    autocast_ctx = torch.autocast("cpu", dtype=torch.bfloat16)
            with autocast_ctx:
                logits, loss = model(x, y)
            if model_cfg.get("aah_v2_enabled", False):
                if lambda_now > 0:
                    total_elements = 0.0
                    baseline_elements = 0.0
                    lq = None
                    lk_layers = []
                    for block in model.blocks:
                        attn = block.attn
                        if hasattr(attn, "last_stats"):
                            total_elements += attn.last_stats.get("total_elements", 0.0)
                            baseline_elements += attn.last_stats.get("baseline_elements", 0.0)
                            if lq is None:
                                lq = attn.last_stats.get("lq")
                            lk_layers.append(attn.last_stats.get("lk", []))
                    if baseline_elements > 0:
                        attn_ratio = total_elements / baseline_elements
                        loss = loss + (lambda_now * torch.tensor(attn_ratio, device=loss.device))
                if guardrails_enabled:
                    if min_head_norm > 0 and norm_lambda > 0:
                        norms = []
                        for block in model.blocks:
                            attn = block.attn
                            if hasattr(attn, "last_stats"):
                                norms.extend(attn.last_stats.get("head_norms", []))
                        if norms:
                            norms_t = torch.tensor(norms, device=loss.device)
                            shortfall = (min_head_norm - norms_t).clamp_min(0.0).mean()
                            loss = loss + (norm_lambda * shortfall)
                    if min_head_entropy > 0 and ent_lambda > 0:
                        ents = []
                        for block in model.blocks:
                            attn = block.attn
                            if hasattr(attn, "last_stats"):
                                ents.extend(attn.last_stats.get("head_entropy", []))
                        if ents:
                            ents_t = torch.tensor(ents, device=loss.device)
                            shortfall = (min_head_entropy - ents_t).clamp_min(0.0).mean()
                            loss = loss + (ent_lambda * shortfall)
                        loss = loss + (ent_lambda * shortfall)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            scheduler.step()
            step_time_ms = (time.time() - step_t0) * 1000.0

            step += 1
            if step in checkpoint_steps_set and step not in saved_checkpoint_steps:
                persist_checkpoint(step, role="near_end")
                saved_checkpoint_steps.add(step)

            if step % effective_log_interval == 0:
                elapsed = time.time() - t0
                tokens = train["batch_size"] * data["seq_len"] * effective_log_interval
                tok_per_sec = tokens / max(1e-9, elapsed)
                gpu_alloc, gpu_reserved, gpu_alloc_max, gpu_reserved_max, cpu_rss = get_memory_stats()
                ps_mem = get_psutil_memory_mb()
                if gpu_alloc is not None:
                    mem = gpu_alloc
                elif cpu_rss is not None:
                    mem = cpu_rss
                else:
                    mem = 0.0
                attn_elems = None
                attn_ratio = None
                attn_reduction = None
                analytic_flops_attn_est = None
                analytic_flops_total_est = None
                analytic_flops_ratio = None
                analytic_flops_reduction_pct = None
                lq = None
                lk_layers = []
                head_entropy = []
                group_change_rates = []
                avg_windows = []
                head_groups = []
                shadow_win_idx = []
                shadow_logit_mean = []
                group_heads = []
                group_ratios = []
                resolution_means = []
                resolution_stds = []
                resolution_min_fracs = []
                resolution_max_fracs = []
                resolution_collapse_mins = []
                resolution_collapse_maxs = []
                resolution_deltas = []
                branch_usage_freqs = []
                effective_attn_elements_vals = []
                effective_ACR_vals = []
                dense_kernel_actual_elements_vals = []
                backend_realized_elements_vals = []
                backend_realized_ACR_vals = []
                backend_names = []
                requested_backends = []
                backend_bucket_counts_vals = []
                backend_kernel_calls_vals = []
                backend_time_vals = []
                backend_fallback_reasons_vals = []
                hierarchy_levels_useds = []
                group_counts_per_levels = []
                controller_logits_std_per_levels = []
                path_modes = []
                win_idx_pre_parent_clamps = []
                win_idx_post_parent_clamps = []
                decision_logits_per_levels = []
                decision_logits_var_per_levels = []
                controller_input_per_levels = []
                controller_input_cos_sim_mean_per_levels = []
                controller_input_cos_sim_min_per_levels = []
                controller_input_l2_dist_mean_per_levels = []
                controller_input_dim_var_mean_per_levels = []
                decision_logits_margin_mean_per_levels = []
                decision_logits_margin_min_per_levels = []
                decision_argmax_diversity_frac_per_levels = []
                sibling_feature_delta_norm_mean_per_levels = []
                sibling_feature_delta_norm_min_per_levels = []
                sibling_feature_delta_norm_max_per_levels = []
                sibling_feature_cos_mean_per_levels = []
                sibling_feature_cos_min_per_levels = []
                sibling_logit_delta_l2_mean_per_levels = []
                sibling_logit_delta_l2_max_per_levels = []
                sibling_logit_delta_abs_mean_per_levels = []
                sibling_logit_delta_abs_max_per_levels = []
                sibling_ranking_diff_frac_per_levels = []
                sibling_top1_differ_frac_per_levels = []
                sibling_top1_ids_per_levels = []
                pairwise_bias_l2_mean_per_levels = []
                pairwise_bias_l2_max_per_levels = []
                pairwise_bias_abs_mean_per_levels = []
                pairwise_bias_abs_max_per_levels = []
                pairwise_bias_top1_changed_frac_per_levels = []
                pairwise_base_top1_ids_per_levels = []
                joint_pair_count_per_levels = []
                joint_output_delta_l2_mean_per_levels = []
                joint_output_delta_l2_max_per_levels = []
                joint_output_abs_delta_mean_per_levels = []
                joint_output_abs_delta_max_per_levels = []
                joint_top1_changed_frac_per_levels = []
                joint_base_top1_ids_per_levels = []
                joint_output_scale_per_levels = []
                decision_raw_idx_per_levels = []
                decision_parent_idx_per_levels = []
                decision_post_parent_idx_per_levels = []
                decision_differ_from_parent_frac_per_levels = []
                decision_unique_raw_idx_per_levels = []
                decision_unique_post_parent_idx_per_levels = []
                decision_non_one_raw_count_per_levels = []
                decision_non_one_post_parent_count_per_levels = []
                decision_head_idx_before_execution_mappings = []
                decision_head_idx_after_execution_mappings = []
                decision_unique_head_idx_before_execution_mappings = []
                decision_unique_head_idx_after_execution_mappings = []
                decision_head_idx_changed_by_execution_mapping_fracs = []
                hierarchy_head_group_map_per_levels = []
                hierarchy_group_members_per_levels = []
                cluster_metric_per_levels = []
                cluster_threshold_kind_per_levels = []
                cluster_threshold_per_levels = []
                cluster_item_count_per_levels = []
                cluster_groups_before_merge_per_levels = []
                cluster_groups_after_merge_per_levels = []
                cluster_groups_merged_per_levels = []
                cluster_small_groups_before_merge_per_levels = []
                cluster_singletons_before_merge_per_levels = []
                cluster_sim_mean_per_levels = []
                cluster_sim_std_per_levels = []
                cluster_sim_min_per_levels = []
                cluster_sim_max_per_levels = []
                cluster_min_group_sizes = []
                cluster_sim_thresholds = []
                cluster_super_thresholds = []
                cluster_forced_bipartition_per_levels = []
                cluster_force_split_anchor_similarity_per_levels = []
                cluster_origin_per_levels = []
                cluster_forced_bipartition_allowed_per_levels = []
                cluster_groups_before_force_per_levels = []
                cluster_feature_norm_mean_per_levels = []
                cluster_feature_norm_std_per_levels = []
                cluster_feature_dim_var_mean_per_levels = []
                cluster_feature_dim_var_std_per_levels = []
                cluster_feature_l2_dist_mean_per_levels = []
                cluster_feature_l2_dist_std_per_levels = []
                cluster_feature_top_singular_ratio_per_levels = []
                hierarchy_level_added_per_levels = []
                hierarchy_growth_stopped_per_levels = []
                hierarchy_stop_reason_per_levels = []
                feature_dim_var_means = []
                feature_dim_var_stds = []
                feature_dim_var_mins = []
                feature_dim_var_maxs = []
                feature_cos_sim_means = []
                feature_cos_sim_stds = []
                feature_cos_sim_mins = []
                feature_cos_sim_maxs = []
                feature_l2_dist_means = []
                feature_l2_dist_stds = []
                feature_l2_dist_mins = []
                feature_l2_dist_maxs = []
                feature_norm_means = []
                feature_norm_stds = []
                feature_top_singular_ratios = []
                lk_means = []
                lk_p90s = []
                w_means = []
                w_mins = []
                w_maxs = []
                control_times = []
                control_feature_times = []
                hierarchy_times = []
                window_select_times = []
                control_mapping_times = []
                diagnostics_pack_times = []
                attn_times = []
                mask_times = []
                overhead_times = []
                if attention_stats_enabled:
                    total_elements = 0.0
                    baseline_elements = 0.0
                    for block in model.blocks:
                        attn = block.attn
                        if hasattr(attn, "last_stats"):
                            total_elements += attn.last_stats.get("total_elements", 0.0)
                            baseline_elements += attn.last_stats.get("baseline_elements", 0.0)
                            if lq is None:
                                lq = attn.last_stats.get("lq")
                            lk_layers.append(attn.last_stats.get("lk", []))
                            head_entropy.extend(attn.last_stats.get("head_entropy", []))
                            if "group_change_rate" in attn.last_stats:
                                group_change_rates.append(attn.last_stats.get("group_change_rate"))
                            if "avg_window" in attn.last_stats:
                                avg_windows.append(attn.last_stats.get("avg_window"))
                            if "head_groups" in attn.last_stats:
                                head_groups.append(attn.last_stats.get("head_groups"))
                            if "shadow_win_idx" in attn.last_stats:
                                shadow_win_idx.append(attn.last_stats.get("shadow_win_idx"))
                            if "shadow_logit_mean" in attn.last_stats:
                                shadow_logit_mean.append(attn.last_stats.get("shadow_logit_mean"))
                            if "group_heads" in attn.last_stats:
                                group_heads.append(attn.last_stats.get("group_heads"))
                            if "group_ratios" in attn.last_stats:
                                group_ratios.append(attn.last_stats.get("group_ratios"))
                            if "resolution_mean" in attn.last_stats:
                                resolution_means.append(attn.last_stats.get("resolution_mean"))
                            if "resolution_std" in attn.last_stats:
                                resolution_stds.append(attn.last_stats.get("resolution_std"))
                            if "resolution_min_frac" in attn.last_stats:
                                resolution_min_fracs.append(attn.last_stats.get("resolution_min_frac"))
                            if "resolution_max_frac" in attn.last_stats:
                                resolution_max_fracs.append(attn.last_stats.get("resolution_max_frac"))
                            if "resolution_collapse_min" in attn.last_stats:
                                resolution_collapse_mins.append(1.0 if attn.last_stats.get("resolution_collapse_min") else 0.0)
                            if "resolution_collapse_max" in attn.last_stats:
                                resolution_collapse_maxs.append(1.0 if attn.last_stats.get("resolution_collapse_max") else 0.0)
                            if "resolution_delta" in attn.last_stats and attn.last_stats.get("resolution_delta") is not None:
                                resolution_deltas.append(attn.last_stats.get("resolution_delta"))
                            if "branch_usage_freq" in attn.last_stats:
                                branch_usage_freqs.append(attn.last_stats.get("branch_usage_freq"))
                            if "effective_attn_elements" in attn.last_stats:
                                effective_attn_elements_vals.append(attn.last_stats.get("effective_attn_elements"))
                            if "effective_ACR" in attn.last_stats:
                                effective_ACR_vals.append(attn.last_stats.get("effective_ACR"))
                            if "dense_kernel_actual_elements_est" in attn.last_stats:
                                dense_kernel_actual_elements_vals.append(attn.last_stats.get("dense_kernel_actual_elements_est"))
                            if "backend_realized_elements_est" in attn.last_stats:
                                backend_realized_elements_vals.append(attn.last_stats.get("backend_realized_elements_est"))
                            if "backend_realized_ACR_est" in attn.last_stats:
                                backend_realized_ACR_vals.append(attn.last_stats.get("backend_realized_ACR_est"))
                            if "backend_name" in attn.last_stats:
                                backend_names.append(attn.last_stats.get("backend_name"))
                            if "requested_backend" in attn.last_stats:
                                requested_backends.append(attn.last_stats.get("requested_backend"))
                            if "backend_bucket_counts" in attn.last_stats:
                                backend_bucket_counts_vals.append(attn.last_stats.get("backend_bucket_counts"))
                            if "backend_kernel_calls" in attn.last_stats:
                                backend_kernel_calls_vals.append(attn.last_stats.get("backend_kernel_calls"))
                            if "backend_time_ms" in attn.last_stats:
                                backend_time_vals.append(attn.last_stats.get("backend_time_ms"))
                            if "backend_fallback_reasons" in attn.last_stats:
                                backend_fallback_reasons_vals.append(attn.last_stats.get("backend_fallback_reasons"))
                            if "hierarchy_levels_used" in attn.last_stats:
                                hierarchy_levels_useds.append(attn.last_stats.get("hierarchy_levels_used"))
                            if "group_counts_per_level" in attn.last_stats:
                                group_counts_per_levels.append(attn.last_stats.get("group_counts_per_level"))
                            if "controller_logits_std_per_level" in attn.last_stats:
                                controller_logits_std_per_levels.append(attn.last_stats.get("controller_logits_std_per_level"))
                            if "path_mode" in attn.last_stats:
                                path_modes.append(attn.last_stats.get("path_mode"))
                            if "win_idx_pre_clamp" in attn.last_stats:
                                win_idx_pre_parent_clamps.append(attn.last_stats.get("win_idx_pre_clamp"))
                            if "win_idx_post_clamp" in attn.last_stats:
                                win_idx_post_parent_clamps.append(attn.last_stats.get("win_idx_post_clamp"))
                            if "decision_logits_per_level" in attn.last_stats:
                                decision_logits_per_levels.append(attn.last_stats.get("decision_logits_per_level"))
                            if "decision_logits_var_per_level" in attn.last_stats:
                                decision_logits_var_per_levels.append(attn.last_stats.get("decision_logits_var_per_level"))
                            if "controller_input_per_level" in attn.last_stats:
                                controller_input_per_levels.append(attn.last_stats.get("controller_input_per_level"))
                            if "controller_input_cos_sim_mean_per_level" in attn.last_stats:
                                controller_input_cos_sim_mean_per_levels.append(attn.last_stats.get("controller_input_cos_sim_mean_per_level"))
                            if "controller_input_cos_sim_min_per_level" in attn.last_stats:
                                controller_input_cos_sim_min_per_levels.append(attn.last_stats.get("controller_input_cos_sim_min_per_level"))
                            if "controller_input_l2_dist_mean_per_level" in attn.last_stats:
                                controller_input_l2_dist_mean_per_levels.append(attn.last_stats.get("controller_input_l2_dist_mean_per_level"))
                            if "controller_input_dim_var_mean_per_level" in attn.last_stats:
                                controller_input_dim_var_mean_per_levels.append(attn.last_stats.get("controller_input_dim_var_mean_per_level"))
                            if "decision_logits_margin_mean_per_level" in attn.last_stats:
                                decision_logits_margin_mean_per_levels.append(attn.last_stats.get("decision_logits_margin_mean_per_level"))
                            if "decision_logits_margin_min_per_level" in attn.last_stats:
                                decision_logits_margin_min_per_levels.append(attn.last_stats.get("decision_logits_margin_min_per_level"))
                            if "decision_argmax_diversity_frac_per_level" in attn.last_stats:
                                decision_argmax_diversity_frac_per_levels.append(attn.last_stats.get("decision_argmax_diversity_frac_per_level"))
                            if "sibling_feature_delta_norm_mean_per_level" in attn.last_stats:
                                sibling_feature_delta_norm_mean_per_levels.append(attn.last_stats.get("sibling_feature_delta_norm_mean_per_level"))
                            if "sibling_feature_delta_norm_min_per_level" in attn.last_stats:
                                sibling_feature_delta_norm_min_per_levels.append(attn.last_stats.get("sibling_feature_delta_norm_min_per_level"))
                            if "sibling_feature_delta_norm_max_per_level" in attn.last_stats:
                                sibling_feature_delta_norm_max_per_levels.append(attn.last_stats.get("sibling_feature_delta_norm_max_per_level"))
                            if "sibling_feature_cos_mean_per_level" in attn.last_stats:
                                sibling_feature_cos_mean_per_levels.append(attn.last_stats.get("sibling_feature_cos_mean_per_level"))
                            if "sibling_feature_cos_min_per_level" in attn.last_stats:
                                sibling_feature_cos_min_per_levels.append(attn.last_stats.get("sibling_feature_cos_min_per_level"))
                            if "sibling_logit_delta_l2_mean_per_level" in attn.last_stats:
                                sibling_logit_delta_l2_mean_per_levels.append(attn.last_stats.get("sibling_logit_delta_l2_mean_per_level"))
                            if "sibling_logit_delta_l2_max_per_level" in attn.last_stats:
                                sibling_logit_delta_l2_max_per_levels.append(attn.last_stats.get("sibling_logit_delta_l2_max_per_level"))
                            if "sibling_logit_delta_abs_mean_per_level" in attn.last_stats:
                                sibling_logit_delta_abs_mean_per_levels.append(attn.last_stats.get("sibling_logit_delta_abs_mean_per_level"))
                            if "sibling_logit_delta_abs_max_per_level" in attn.last_stats:
                                sibling_logit_delta_abs_max_per_levels.append(attn.last_stats.get("sibling_logit_delta_abs_max_per_level"))
                            if "sibling_ranking_diff_frac_per_level" in attn.last_stats:
                                sibling_ranking_diff_frac_per_levels.append(attn.last_stats.get("sibling_ranking_diff_frac_per_level"))
                            if "sibling_top1_differ_frac_per_level" in attn.last_stats:
                                sibling_top1_differ_frac_per_levels.append(attn.last_stats.get("sibling_top1_differ_frac_per_level"))
                            if "sibling_top1_ids_per_level" in attn.last_stats:
                                sibling_top1_ids_per_levels.append(attn.last_stats.get("sibling_top1_ids_per_level"))
                            if "pairwise_bias_l2_mean_per_level" in attn.last_stats:
                                pairwise_bias_l2_mean_per_levels.append(attn.last_stats.get("pairwise_bias_l2_mean_per_level"))
                            if "pairwise_bias_l2_max_per_level" in attn.last_stats:
                                pairwise_bias_l2_max_per_levels.append(attn.last_stats.get("pairwise_bias_l2_max_per_level"))
                            if "pairwise_bias_abs_mean_per_level" in attn.last_stats:
                                pairwise_bias_abs_mean_per_levels.append(attn.last_stats.get("pairwise_bias_abs_mean_per_level"))
                            if "pairwise_bias_abs_max_per_level" in attn.last_stats:
                                pairwise_bias_abs_max_per_levels.append(attn.last_stats.get("pairwise_bias_abs_max_per_level"))
                            if "pairwise_bias_top1_changed_frac_per_level" in attn.last_stats:
                                pairwise_bias_top1_changed_frac_per_levels.append(attn.last_stats.get("pairwise_bias_top1_changed_frac_per_level"))
                            if "pairwise_base_top1_ids_per_level" in attn.last_stats:
                                pairwise_base_top1_ids_per_levels.append(attn.last_stats.get("pairwise_base_top1_ids_per_level"))
                            if "joint_pair_count_per_level" in attn.last_stats:
                                joint_pair_count_per_levels.append(attn.last_stats.get("joint_pair_count_per_level"))
                            if "joint_output_delta_l2_mean_per_level" in attn.last_stats:
                                joint_output_delta_l2_mean_per_levels.append(attn.last_stats.get("joint_output_delta_l2_mean_per_level"))
                            if "joint_output_delta_l2_max_per_level" in attn.last_stats:
                                joint_output_delta_l2_max_per_levels.append(attn.last_stats.get("joint_output_delta_l2_max_per_level"))
                            if "joint_output_abs_delta_mean_per_level" in attn.last_stats:
                                joint_output_abs_delta_mean_per_levels.append(attn.last_stats.get("joint_output_abs_delta_mean_per_level"))
                            if "joint_output_abs_delta_max_per_level" in attn.last_stats:
                                joint_output_abs_delta_max_per_levels.append(attn.last_stats.get("joint_output_abs_delta_max_per_level"))
                            if "joint_top1_changed_frac_per_level" in attn.last_stats:
                                joint_top1_changed_frac_per_levels.append(attn.last_stats.get("joint_top1_changed_frac_per_level"))
                            if "joint_base_top1_ids_per_level" in attn.last_stats:
                                joint_base_top1_ids_per_levels.append(attn.last_stats.get("joint_base_top1_ids_per_level"))
                            if "joint_output_scale_per_level" in attn.last_stats:
                                joint_output_scale_per_levels.append(attn.last_stats.get("joint_output_scale_per_level"))
                            if "decision_raw_idx_per_level" in attn.last_stats:
                                decision_raw_idx_per_levels.append(attn.last_stats.get("decision_raw_idx_per_level"))
                            if "decision_parent_idx_per_level" in attn.last_stats:
                                decision_parent_idx_per_levels.append(attn.last_stats.get("decision_parent_idx_per_level"))
                            if "decision_post_parent_idx_per_level" in attn.last_stats:
                                decision_post_parent_idx_per_levels.append(attn.last_stats.get("decision_post_parent_idx_per_level"))
                            if "decision_differ_from_parent_frac_per_level" in attn.last_stats:
                                decision_differ_from_parent_frac_per_levels.append(attn.last_stats.get("decision_differ_from_parent_frac_per_level"))
                            if "decision_unique_raw_idx_per_level" in attn.last_stats:
                                decision_unique_raw_idx_per_levels.append(attn.last_stats.get("decision_unique_raw_idx_per_level"))
                            if "decision_unique_post_parent_idx_per_level" in attn.last_stats:
                                decision_unique_post_parent_idx_per_levels.append(attn.last_stats.get("decision_unique_post_parent_idx_per_level"))
                            if "decision_non_one_raw_count_per_level" in attn.last_stats:
                                decision_non_one_raw_count_per_levels.append(attn.last_stats.get("decision_non_one_raw_count_per_level"))
                            if "decision_non_one_post_parent_count_per_level" in attn.last_stats:
                                decision_non_one_post_parent_count_per_levels.append(attn.last_stats.get("decision_non_one_post_parent_count_per_level"))
                            if "decision_head_idx_before_execution_mapping" in attn.last_stats:
                                decision_head_idx_before_execution_mappings.append(attn.last_stats.get("decision_head_idx_before_execution_mapping"))
                            if "decision_head_idx_after_execution_mapping" in attn.last_stats:
                                decision_head_idx_after_execution_mappings.append(attn.last_stats.get("decision_head_idx_after_execution_mapping"))
                            if "decision_unique_head_idx_before_execution_mapping" in attn.last_stats:
                                decision_unique_head_idx_before_execution_mappings.append(attn.last_stats.get("decision_unique_head_idx_before_execution_mapping"))
                            if "decision_unique_head_idx_after_execution_mapping" in attn.last_stats:
                                decision_unique_head_idx_after_execution_mappings.append(attn.last_stats.get("decision_unique_head_idx_after_execution_mapping"))
                            if "decision_head_idx_changed_by_execution_mapping_frac" in attn.last_stats:
                                decision_head_idx_changed_by_execution_mapping_fracs.append(attn.last_stats.get("decision_head_idx_changed_by_execution_mapping_frac"))
                            if "hierarchy_head_group_map_per_level" in attn.last_stats:
                                hierarchy_head_group_map_per_levels.append(attn.last_stats.get("hierarchy_head_group_map_per_level"))
                            if "hierarchy_group_members_per_level" in attn.last_stats:
                                hierarchy_group_members_per_levels.append(attn.last_stats.get("hierarchy_group_members_per_level"))
                            if "cluster_metric_per_level" in attn.last_stats:
                                cluster_metric_per_levels.append(attn.last_stats.get("cluster_metric_per_level"))
                            if "cluster_threshold_kind_per_level" in attn.last_stats:
                                cluster_threshold_kind_per_levels.append(attn.last_stats.get("cluster_threshold_kind_per_level"))
                            if "cluster_threshold_per_level" in attn.last_stats:
                                cluster_threshold_per_levels.append(attn.last_stats.get("cluster_threshold_per_level"))
                            if "cluster_item_count_per_level" in attn.last_stats:
                                cluster_item_count_per_levels.append(attn.last_stats.get("cluster_item_count_per_level"))
                            if "cluster_groups_before_merge_per_level" in attn.last_stats:
                                cluster_groups_before_merge_per_levels.append(attn.last_stats.get("cluster_groups_before_merge_per_level"))
                            if "cluster_groups_after_merge_per_level" in attn.last_stats:
                                cluster_groups_after_merge_per_levels.append(attn.last_stats.get("cluster_groups_after_merge_per_level"))
                            if "cluster_groups_merged_per_level" in attn.last_stats:
                                cluster_groups_merged_per_levels.append(attn.last_stats.get("cluster_groups_merged_per_level"))
                            if "cluster_small_groups_before_merge_per_level" in attn.last_stats:
                                cluster_small_groups_before_merge_per_levels.append(attn.last_stats.get("cluster_small_groups_before_merge_per_level"))
                            if "cluster_singletons_before_merge_per_level" in attn.last_stats:
                                cluster_singletons_before_merge_per_levels.append(attn.last_stats.get("cluster_singletons_before_merge_per_level"))
                            if "cluster_sim_mean_per_level" in attn.last_stats:
                                cluster_sim_mean_per_levels.append(attn.last_stats.get("cluster_sim_mean_per_level"))
                            if "cluster_sim_std_per_level" in attn.last_stats:
                                cluster_sim_std_per_levels.append(attn.last_stats.get("cluster_sim_std_per_level"))
                            if "cluster_sim_min_per_level" in attn.last_stats:
                                cluster_sim_min_per_levels.append(attn.last_stats.get("cluster_sim_min_per_level"))
                            if "cluster_sim_max_per_level" in attn.last_stats:
                                cluster_sim_max_per_levels.append(attn.last_stats.get("cluster_sim_max_per_level"))
                            if "cluster_min_group_size" in attn.last_stats:
                                cluster_min_group_sizes.append(attn.last_stats.get("cluster_min_group_size"))
                            if "cluster_sim_threshold" in attn.last_stats:
                                cluster_sim_thresholds.append(attn.last_stats.get("cluster_sim_threshold"))
                            if "cluster_super_threshold" in attn.last_stats:
                                cluster_super_thresholds.append(attn.last_stats.get("cluster_super_threshold"))
                            if "cluster_forced_bipartition_per_level" in attn.last_stats:
                                cluster_forced_bipartition_per_levels.append(attn.last_stats.get("cluster_forced_bipartition_per_level"))
                            if "cluster_force_split_anchor_similarity_per_level" in attn.last_stats:
                                cluster_force_split_anchor_similarity_per_levels.append(attn.last_stats.get("cluster_force_split_anchor_similarity_per_level"))
                            if "cluster_origin_per_level" in attn.last_stats:
                                cluster_origin_per_levels.append(attn.last_stats.get("cluster_origin_per_level"))
                            if "cluster_forced_bipartition_allowed_per_level" in attn.last_stats:
                                cluster_forced_bipartition_allowed_per_levels.append(attn.last_stats.get("cluster_forced_bipartition_allowed_per_level"))
                            if "cluster_groups_before_force_per_level" in attn.last_stats:
                                cluster_groups_before_force_per_levels.append(attn.last_stats.get("cluster_groups_before_force_per_level"))
                            if "cluster_feature_norm_mean_per_level" in attn.last_stats:
                                cluster_feature_norm_mean_per_levels.append(attn.last_stats.get("cluster_feature_norm_mean_per_level"))
                            if "cluster_feature_norm_std_per_level" in attn.last_stats:
                                cluster_feature_norm_std_per_levels.append(attn.last_stats.get("cluster_feature_norm_std_per_level"))
                            if "cluster_feature_dim_var_mean_per_level" in attn.last_stats:
                                cluster_feature_dim_var_mean_per_levels.append(attn.last_stats.get("cluster_feature_dim_var_mean_per_level"))
                            if "cluster_feature_dim_var_std_per_level" in attn.last_stats:
                                cluster_feature_dim_var_std_per_levels.append(attn.last_stats.get("cluster_feature_dim_var_std_per_level"))
                            if "cluster_feature_l2_dist_mean_per_level" in attn.last_stats:
                                cluster_feature_l2_dist_mean_per_levels.append(attn.last_stats.get("cluster_feature_l2_dist_mean_per_level"))
                            if "cluster_feature_l2_dist_std_per_level" in attn.last_stats:
                                cluster_feature_l2_dist_std_per_levels.append(attn.last_stats.get("cluster_feature_l2_dist_std_per_level"))
                            if "cluster_feature_top_singular_ratio_per_level" in attn.last_stats:
                                cluster_feature_top_singular_ratio_per_levels.append(attn.last_stats.get("cluster_feature_top_singular_ratio_per_level"))
                            if "hierarchy_level_added_per_level" in attn.last_stats:
                                hierarchy_level_added_per_levels.append(attn.last_stats.get("hierarchy_level_added_per_level"))
                            if "hierarchy_growth_stopped_per_level" in attn.last_stats:
                                hierarchy_growth_stopped_per_levels.append(attn.last_stats.get("hierarchy_growth_stopped_per_level"))
                            if "hierarchy_stop_reason_per_level" in attn.last_stats:
                                hierarchy_stop_reason_per_levels.append(attn.last_stats.get("hierarchy_stop_reason_per_level"))
                            if "feature_dim_var_mean" in attn.last_stats:
                                feature_dim_var_means.append(attn.last_stats.get("feature_dim_var_mean"))
                            if "feature_dim_var_std" in attn.last_stats:
                                feature_dim_var_stds.append(attn.last_stats.get("feature_dim_var_std"))
                            if "feature_dim_var_min" in attn.last_stats:
                                feature_dim_var_mins.append(attn.last_stats.get("feature_dim_var_min"))
                            if "feature_dim_var_max" in attn.last_stats:
                                feature_dim_var_maxs.append(attn.last_stats.get("feature_dim_var_max"))
                            if "feature_cos_sim_mean" in attn.last_stats:
                                feature_cos_sim_means.append(attn.last_stats.get("feature_cos_sim_mean"))
                            if "feature_cos_sim_std" in attn.last_stats:
                                feature_cos_sim_stds.append(attn.last_stats.get("feature_cos_sim_std"))
                            if "feature_cos_sim_min" in attn.last_stats:
                                feature_cos_sim_mins.append(attn.last_stats.get("feature_cos_sim_min"))
                            if "feature_cos_sim_max" in attn.last_stats:
                                feature_cos_sim_maxs.append(attn.last_stats.get("feature_cos_sim_max"))
                            if "feature_l2_dist_mean" in attn.last_stats:
                                feature_l2_dist_means.append(attn.last_stats.get("feature_l2_dist_mean"))
                            if "feature_l2_dist_std" in attn.last_stats:
                                feature_l2_dist_stds.append(attn.last_stats.get("feature_l2_dist_std"))
                            if "feature_l2_dist_min" in attn.last_stats:
                                feature_l2_dist_mins.append(attn.last_stats.get("feature_l2_dist_min"))
                            if "feature_l2_dist_max" in attn.last_stats:
                                feature_l2_dist_maxs.append(attn.last_stats.get("feature_l2_dist_max"))
                            if "feature_norm_mean" in attn.last_stats:
                                feature_norm_means.append(attn.last_stats.get("feature_norm_mean"))
                            if "feature_norm_std" in attn.last_stats:
                                feature_norm_stds.append(attn.last_stats.get("feature_norm_std"))
                            if "feature_top_singular_ratio" in attn.last_stats:
                                feature_top_singular_ratios.append(attn.last_stats.get("feature_top_singular_ratio"))
                            if "control_time_ms" in attn.last_stats:
                                control_times.append(attn.last_stats.get("control_time_ms"))
                            if "control_feature_time_ms" in attn.last_stats:
                                control_feature_times.append(attn.last_stats.get("control_feature_time_ms"))
                            if "hierarchy_time_ms" in attn.last_stats:
                                hierarchy_times.append(attn.last_stats.get("hierarchy_time_ms"))
                            if "window_select_time_ms" in attn.last_stats:
                                window_select_times.append(attn.last_stats.get("window_select_time_ms"))
                            if "control_mapping_time_ms" in attn.last_stats:
                                control_mapping_times.append(attn.last_stats.get("control_mapping_time_ms"))
                            if "diagnostics_pack_time_ms" in attn.last_stats:
                                diagnostics_pack_times.append(attn.last_stats.get("diagnostics_pack_time_ms"))
                            if "attn_time_ms" in attn.last_stats:
                                attn_times.append(attn.last_stats.get("attn_time_ms"))
                            if "mask_time_ms" in attn.last_stats:
                                mask_times.append(attn.last_stats.get("mask_time_ms"))
                            if "overhead_time_ms" in attn.last_stats:
                                overhead_times.append(attn.last_stats.get("overhead_time_ms"))
                            if "lk_mean" in attn.last_stats:
                                lk_means.append(attn.last_stats.get("lk_mean"))
                            if "lk_p90" in attn.last_stats:
                                lk_p90s.append(attn.last_stats.get("lk_p90"))
                            if "w_mean" in attn.last_stats:
                                w_means.append(attn.last_stats.get("w_mean"))
                            if "w_min" in attn.last_stats:
                                w_mins.append(attn.last_stats.get("w_min"))
                            if "w_max" in attn.last_stats:
                                w_maxs.append(attn.last_stats.get("w_max"))
                    if baseline_elements > 0:
                        attn_elems = total_elements
                        attn_ratio = total_elements / baseline_elements
                        attn_reduction = 1.0 - attn_ratio
                b_cur = int(x.size(0))
                t_cur = int(x.size(1))
                analytic_flops_attn_est, analytic_flops_total_est, analytic_flops_ratio, analytic_flops_reduction_pct = estimate_flops(
                    model_cfg,
                    b_cur,
                    t_cur,
                    attn_elements_total=attn_elems,
                )
                group_change_rates = [v for v in group_change_rates if v is not None]
                group_change_rate = sum(group_change_rates) / len(group_change_rates) if group_change_rates else None
                resolution_mean = sum(resolution_means) / len(resolution_means) if resolution_means else None
                resolution_std = sum(resolution_stds) / len(resolution_stds) if resolution_stds else None
                resolution_min_frac = sum(resolution_min_fracs) / len(resolution_min_fracs) if resolution_min_fracs else None
                resolution_max_frac = sum(resolution_max_fracs) / len(resolution_max_fracs) if resolution_max_fracs else None
                resolution_collapse_min = (sum(resolution_collapse_mins) / len(resolution_collapse_mins)) if resolution_collapse_mins else None
                resolution_collapse_max = (sum(resolution_collapse_maxs) / len(resolution_collapse_maxs)) if resolution_collapse_maxs else None
                resolution_delta = sum(resolution_deltas) / len(resolution_deltas) if resolution_deltas else None
                hierarchy_levels_used = (sum(hierarchy_levels_useds) / len(hierarchy_levels_useds)) if hierarchy_levels_useds else None
                cluster_sim_mean_level0 = None
                cluster_sim_std_level0 = None
                cluster_sim_min_level0 = None
                cluster_sim_max_level0 = None
                cluster_item_count_level0 = None
                cluster_groups_before_merge_level0 = None
                cluster_groups_after_merge_level0 = None
                cluster_groups_merged_level0 = None
                cluster_small_groups_before_merge_level0 = None
                cluster_singletons_before_merge_level0 = None
                cluster_forced_bipartition_level0 = None
                cluster_force_split_anchor_similarity_level0 = None
                if cluster_sim_mean_per_levels:
                    vals = [x[0] for x in cluster_sim_mean_per_levels if isinstance(x, list) and len(x) > 0]
                    cluster_sim_mean_level0 = (sum(vals) / len(vals)) if vals else None
                if cluster_sim_std_per_levels:
                    vals = [x[0] for x in cluster_sim_std_per_levels if isinstance(x, list) and len(x) > 0]
                    cluster_sim_std_level0 = (sum(vals) / len(vals)) if vals else None
                if cluster_sim_min_per_levels:
                    vals = [x[0] for x in cluster_sim_min_per_levels if isinstance(x, list) and len(x) > 0]
                    cluster_sim_min_level0 = (sum(vals) / len(vals)) if vals else None
                if cluster_sim_max_per_levels:
                    vals = [x[0] for x in cluster_sim_max_per_levels if isinstance(x, list) and len(x) > 0]
                    cluster_sim_max_level0 = (sum(vals) / len(vals)) if vals else None
                if cluster_item_count_per_levels:
                    vals = [x[0] for x in cluster_item_count_per_levels if isinstance(x, list) and len(x) > 0]
                    cluster_item_count_level0 = (sum(vals) / len(vals)) if vals else None
                if cluster_groups_before_merge_per_levels:
                    vals = [x[0] for x in cluster_groups_before_merge_per_levels if isinstance(x, list) and len(x) > 0]
                    cluster_groups_before_merge_level0 = (sum(vals) / len(vals)) if vals else None
                if cluster_groups_after_merge_per_levels:
                    vals = [x[0] for x in cluster_groups_after_merge_per_levels if isinstance(x, list) and len(x) > 0]
                    cluster_groups_after_merge_level0 = (sum(vals) / len(vals)) if vals else None
                if cluster_groups_merged_per_levels:
                    vals = [x[0] for x in cluster_groups_merged_per_levels if isinstance(x, list) and len(x) > 0]
                    cluster_groups_merged_level0 = (sum(vals) / len(vals)) if vals else None
                if cluster_small_groups_before_merge_per_levels:
                    vals = [x[0] for x in cluster_small_groups_before_merge_per_levels if isinstance(x, list) and len(x) > 0]
                    cluster_small_groups_before_merge_level0 = (sum(vals) / len(vals)) if vals else None
                if cluster_singletons_before_merge_per_levels:
                    vals = [x[0] for x in cluster_singletons_before_merge_per_levels if isinstance(x, list) and len(x) > 0]
                    cluster_singletons_before_merge_level0 = (sum(vals) / len(vals)) if vals else None
                if cluster_forced_bipartition_per_levels:
                    vals = [1.0 if bool(x[0]) else 0.0 for x in cluster_forced_bipartition_per_levels if isinstance(x, list) and len(x) > 0]
                    cluster_forced_bipartition_level0 = (sum(vals) / len(vals)) if vals else None
                if cluster_force_split_anchor_similarity_per_levels:
                    vals = [float(x[0]) for x in cluster_force_split_anchor_similarity_per_levels if isinstance(x, list) and len(x) > 0 and x[0] is not None]
                    cluster_force_split_anchor_similarity_level0 = (sum(vals) / len(vals)) if vals else None
                feature_dim_var_mean = sum(feature_dim_var_means) / len(feature_dim_var_means) if feature_dim_var_means else None
                feature_dim_var_std = sum(feature_dim_var_stds) / len(feature_dim_var_stds) if feature_dim_var_stds else None
                feature_dim_var_min = sum(feature_dim_var_mins) / len(feature_dim_var_mins) if feature_dim_var_mins else None
                feature_dim_var_max = sum(feature_dim_var_maxs) / len(feature_dim_var_maxs) if feature_dim_var_maxs else None
                feature_cos_sim_mean = sum(feature_cos_sim_means) / len(feature_cos_sim_means) if feature_cos_sim_means else None
                feature_cos_sim_std = sum(feature_cos_sim_stds) / len(feature_cos_sim_stds) if feature_cos_sim_stds else None
                feature_cos_sim_min = sum(feature_cos_sim_mins) / len(feature_cos_sim_mins) if feature_cos_sim_mins else None
                feature_cos_sim_max = sum(feature_cos_sim_maxs) / len(feature_cos_sim_maxs) if feature_cos_sim_maxs else None
                feature_l2_dist_mean = sum(feature_l2_dist_means) / len(feature_l2_dist_means) if feature_l2_dist_means else None
                feature_l2_dist_std = sum(feature_l2_dist_stds) / len(feature_l2_dist_stds) if feature_l2_dist_stds else None
                feature_l2_dist_min = sum(feature_l2_dist_mins) / len(feature_l2_dist_mins) if feature_l2_dist_mins else None
                feature_l2_dist_max = sum(feature_l2_dist_maxs) / len(feature_l2_dist_maxs) if feature_l2_dist_maxs else None
                feature_norm_mean = sum(feature_norm_means) / len(feature_norm_means) if feature_norm_means else None
                feature_norm_std = sum(feature_norm_stds) / len(feature_norm_stds) if feature_norm_stds else None
                feature_top_singular_ratio = sum(feature_top_singular_ratios) / len(feature_top_singular_ratios) if feature_top_singular_ratios else None
                branch_usage_agg = {}
                if branch_usage_freqs:
                    for freq_dict in branch_usage_freqs:
                        for k, v in freq_dict.items():
                            ks = str(k)
                            branch_usage_agg[ks] = branch_usage_agg.get(ks, 0.0) + float(v)
                    denom = float(len(branch_usage_freqs))
                    for k in list(branch_usage_agg.keys()):
                        branch_usage_agg[k] = branch_usage_agg[k] / denom
                effective_attn_elements = sum(float(v) for v in effective_attn_elements_vals) if effective_attn_elements_vals else None
                effective_ACR = (
                    effective_attn_elements / baseline_elements
                    if effective_attn_elements is not None and baseline_elements > 0
                    else None
                )
                dense_kernel_actual_elements_est = (
                    sum(float(v) for v in dense_kernel_actual_elements_vals)
                    if dense_kernel_actual_elements_vals
                    else None
                )
                backend_realized_elements_est = (
                    sum(float(v) for v in backend_realized_elements_vals)
                    if backend_realized_elements_vals
                    else None
                )
                backend_realized_ACR_est = (
                    backend_realized_elements_est / dense_kernel_actual_elements_est
                    if backend_realized_elements_est is not None and dense_kernel_actual_elements_est and dense_kernel_actual_elements_est > 0
                    else None
                )
                backend_name = ""
                if backend_names:
                    unique_backend_names = sorted(set(str(v) for v in backend_names))
                    backend_name = unique_backend_names[0] if len(unique_backend_names) == 1 else "mixed"
                requested_backend = ""
                if requested_backends:
                    unique_requested_backends = sorted(set(str(v) for v in requested_backends))
                    requested_backend = unique_requested_backends[0] if len(unique_requested_backends) == 1 else "mixed"
                backend_bucket_counts = {}
                for counts in backend_bucket_counts_vals:
                    if not isinstance(counts, dict):
                        continue
                    for k, v in counts.items():
                        ks = str(k)
                        backend_bucket_counts[ks] = backend_bucket_counts.get(ks, 0) + int(v)
                backend_kernel_calls = sum(int(v) for v in backend_kernel_calls_vals) if backend_kernel_calls_vals else None
                backend_time_ms = sum(float(v) for v in backend_time_vals) / len(backend_time_vals) if backend_time_vals else None
                backend_fallback_reasons = sorted(
                    {
                        str(reason)
                        for reasons in backend_fallback_reasons_vals
                        if isinstance(reasons, list)
                        for reason in reasons
                    }
                )
                path_mode_freq = {}
                if path_modes:
                    for mode in path_modes:
                        ms = str(mode)
                        path_mode_freq[ms] = path_mode_freq.get(ms, 0.0) + 1.0
                    denom = float(len(path_modes))
                    for k in list(path_mode_freq.keys()):
                        path_mode_freq[k] = path_mode_freq[k] / denom
                avg_window = sum(avg_windows) / len(avg_windows) if avg_windows else None
                lk_mean = sum(lk_means) / len(lk_means) if lk_means else None
                lk_p90 = sum(lk_p90s) / len(lk_p90s) if lk_p90s else None
                w_mean = sum(w_means) / len(w_means) if w_means else None
                w_min = min(w_mins) if w_mins else None
                w_max = max(w_maxs) if w_maxs else None
                control_time_ms = sum(control_times) / len(control_times) if control_times else None
                control_feature_time_ms = sum(control_feature_times) / len(control_feature_times) if control_feature_times else None
                hierarchy_time_ms = sum(hierarchy_times) / len(hierarchy_times) if hierarchy_times else None
                window_select_time_ms = sum(window_select_times) / len(window_select_times) if window_select_times else None
                control_mapping_time_ms = sum(control_mapping_times) / len(control_mapping_times) if control_mapping_times else None
                diagnostics_pack_time_ms = sum(diagnostics_pack_times) / len(diagnostics_pack_times) if diagnostics_pack_times else None
                attn_time_ms = sum(attn_times) / len(attn_times) if attn_times else None
                mask_time_ms = sum(mask_times) / len(mask_times) if mask_times else None
                overhead_time_ms = sum(overhead_times) / len(overhead_times) if overhead_times else None
                overlap_rates = []
                reassign_rates = []
                if head_groups and prev_head_groups is not None:
                    for curr, prev in zip(head_groups, prev_head_groups):
                        if not curr or not prev:
                            continue
                        n = min(len(curr), len(prev))
                        curr = curr[:n]
                        prev = prev[:n]
                        reassign = sum(1 for i in range(n) if curr[i] != prev[i]) / max(1, n)
                        reassign_rates.append(reassign)
                        curr_pairs = {(i, j) for i in range(n) for j in range(i + 1, n) if curr[i] == curr[j]}
                        prev_pairs = {(i, j) for i in range(n) for j in range(i + 1, n) if prev[i] == prev[j]}
                        if curr_pairs or prev_pairs:
                            jaccard = len(curr_pairs & prev_pairs) / len(curr_pairs | prev_pairs)
                            overlap_rates.append(jaccard)
                avg_overlap = sum(overlap_rates) / len(overlap_rates) if overlap_rates else None
                avg_reassign = sum(reassign_rates) / len(reassign_rates) if reassign_rates else None
                if avg_reassign is not None:
                    if lifespan_ema is None:
                        lifespan_ema = 1.0 - avg_reassign
                    else:
                        lifespan_ema = 0.9 * lifespan_ema + 0.1 * (1.0 - avg_reassign)
                if attn_ratio is not None:
                    ratio_window.append(float(attn_ratio))
                if analytic_flops_ratio is not None:
                    analytic_flops_ratio_window.append(float(analytic_flops_ratio))
                attn_ratio_std_ema = float(torch.tensor(list(ratio_window)).std(unbiased=False).item()) if len(ratio_window) >= 2 else 0.0
                analytic_flops_ratio_std_ema = float(torch.tensor(list(analytic_flops_ratio_window)).std(unbiased=False).item()) if len(analytic_flops_ratio_window) >= 2 else 0.0
                msg = f"step {step} | loss {loss.item():.4f} | tok/s {tok_per_sec:.1f}"
                if attn_ratio is not None:
                    msg += f" | attn_elems {attn_elems:.0f} | attn_ratio {attn_ratio:.3f}"
                if resolution_std is not None:
                    msg += f" | res_std {resolution_std:.2f}"
                print(msg)
                if use_wandb:
                    payload = {"train/loss": loss.item(), "perf/tok_s": tok_per_sec, "perf/mem_mb": mem, "step": step}
                    if gpu_alloc is not None:
                        payload["perf/gpu_alloc_mb"] = gpu_alloc
                    if gpu_reserved is not None:
                        payload["perf/gpu_reserved_mb"] = gpu_reserved
                    if gpu_alloc_max is not None:
                        payload["perf/gpu_alloc_max_mb"] = gpu_alloc_max
                    if gpu_reserved_max is not None:
                        payload["perf/gpu_reserved_max_mb"] = gpu_reserved_max
                    if cpu_rss is not None:
                        payload["perf/cpu_rss_mb"] = cpu_rss
                    if ps_mem:
                        payload.update(ps_mem)
                    if attn_ratio is not None:
                        payload["perf/attn_elems"] = attn_elems
                        payload["perf/attn_ratio"] = attn_ratio
                        payload["aah/attn_ratio"] = attn_ratio
                    if effective_attn_elements is not None:
                        payload["aah/effective_attn_elements"] = effective_attn_elements
                    if effective_ACR is not None:
                        payload["aah/effective_ACR"] = effective_ACR
                    if dense_kernel_actual_elements_est is not None:
                        payload["aah/dense_kernel_actual_elements_est"] = dense_kernel_actual_elements_est
                    if backend_realized_elements_est is not None:
                        payload["aah/backend_realized_elements_est"] = backend_realized_elements_est
                    if backend_realized_ACR_est is not None:
                        payload["aah/backend_realized_ACR_est"] = backend_realized_ACR_est
                    if backend_name:
                        payload["aah/backend_name"] = backend_name
                    if requested_backend:
                        payload["aah/requested_backend"] = requested_backend
                    if backend_bucket_counts:
                        payload["aah/backend_bucket_counts"] = backend_bucket_counts
                    if backend_kernel_calls is not None:
                        payload["aah/backend_kernel_calls"] = backend_kernel_calls
                    if backend_time_ms is not None:
                        payload["aah/backend_time_ms"] = backend_time_ms
                    if backend_fallback_reasons:
                        payload["aah/backend_fallback_reasons"] = backend_fallback_reasons
                    if attn_reduction is not None:
                        payload["perf/attn_reduction"] = attn_reduction
                    if analytic_flops_attn_est is not None:
                        payload["aah/analytic_flops_attn_est"] = analytic_flops_attn_est
                    if analytic_flops_total_est is not None:
                        payload["aah/analytic_flops_total_est"] = analytic_flops_total_est
                    if analytic_flops_ratio is not None:
                        payload["aah/analytic_flops_ratio"] = analytic_flops_ratio
                        payload["aah/analytic_flops_ratio_std_ema"] = analytic_flops_ratio_std_ema
                    if attn_ratio is not None:
                        payload["aah/attn_ratio_std_ema"] = attn_ratio_std_ema
                    if analytic_flops_reduction_pct is not None:
                        payload["aah/analytic_flops_reduction_%"] = analytic_flops_reduction_pct
                    if group_change_rate is not None:
                        payload["aah/group_change_rate"] = group_change_rate
                    if avg_window is not None:
                        payload["aah/avg_window"] = avg_window
                    if lk_mean is not None:
                        payload["aah/Lk_mean"] = lk_mean
                    if lk_p90 is not None:
                        payload["aah/Lk_p90"] = lk_p90
                    if w_mean is not None:
                        payload["aah/W_mean"] = w_mean
                    if w_min is not None:
                        payload["aah/W_min"] = w_min
                    if w_max is not None:
                        payload["aah/W_max"] = w_max
                    if control_time_ms is not None:
                        payload["aah/time/control_ms"] = control_time_ms
                    if control_feature_time_ms is not None:
                        payload["aah/time/control_feature_ms"] = control_feature_time_ms
                    if hierarchy_time_ms is not None:
                        payload["aah/time/hierarchy_ms"] = hierarchy_time_ms
                    if window_select_time_ms is not None:
                        payload["aah/time/window_select_ms"] = window_select_time_ms
                    if control_mapping_time_ms is not None:
                        payload["aah/time/control_mapping_ms"] = control_mapping_time_ms
                    if diagnostics_pack_time_ms is not None:
                        payload["aah/time/diagnostics_pack_ms"] = diagnostics_pack_time_ms
                    if attn_time_ms is not None:
                        payload["aah/time/attention_ms"] = attn_time_ms
                    if mask_time_ms is not None:
                        payload["aah/time/mask_ms"] = mask_time_ms
                    if overhead_time_ms is not None:
                        payload["aah/time/overhead_ms"] = overhead_time_ms
                    if shadow_logit_mean:
                        payload["aah/shadow_logit_mean"] = shadow_logit_mean[0]
                    if group_ratios:
                        payload["aah/group_ratios"] = group_ratios[0]
                    if resolution_mean is not None:
                        payload["aah/resolution_mean"] = resolution_mean
                    if resolution_std is not None:
                        payload["aah/resolution_std"] = resolution_std
                    if resolution_min_frac is not None:
                        payload["aah/resolution_min_frac"] = resolution_min_frac
                    if resolution_max_frac is not None:
                        payload["aah/resolution_max_frac"] = resolution_max_frac
                    if resolution_collapse_min is not None:
                        payload["aah/resolution_collapse_min"] = resolution_collapse_min
                    if resolution_collapse_max is not None:
                        payload["aah/resolution_collapse_max"] = resolution_collapse_max
                    if resolution_delta is not None:
                        payload["aah/resolution_delta"] = resolution_delta
                    if hierarchy_levels_used is not None:
                        payload["aah/hierarchy_levels_used"] = hierarchy_levels_used
                    if group_counts_per_levels:
                        payload["aah/group_counts_per_level"] = group_counts_per_levels[0]
                    if controller_logits_std_per_levels:
                        payload["aah/controller_logits_std_per_level"] = controller_logits_std_per_levels[0]
                    if win_idx_pre_parent_clamps:
                        payload["aah/win_idx_pre_parent_clamp"] = win_idx_pre_parent_clamps[0]
                    if win_idx_post_parent_clamps:
                        payload["aah/win_idx_post_parent_clamp"] = win_idx_post_parent_clamps[0]
                    if hierarchy_head_group_map_per_levels:
                        payload["aah/hierarchy_head_group_map_per_level"] = hierarchy_head_group_map_per_levels[0]
                    if hierarchy_group_members_per_levels:
                        payload["aah/hierarchy_group_members_per_level"] = hierarchy_group_members_per_levels[0]
                    if cluster_metric_per_levels:
                        payload["aah/cluster_metric_per_level"] = cluster_metric_per_levels[0]
                    if cluster_threshold_kind_per_levels:
                        payload["aah/cluster_threshold_kind_per_level"] = cluster_threshold_kind_per_levels[0]
                    if cluster_threshold_per_levels:
                        payload["aah/cluster_threshold_per_level"] = cluster_threshold_per_levels[0]
                    if cluster_item_count_per_levels:
                        payload["aah/cluster_item_count_per_level"] = cluster_item_count_per_levels[0]
                    if cluster_groups_before_merge_per_levels:
                        payload["aah/cluster_groups_before_merge_per_level"] = cluster_groups_before_merge_per_levels[0]
                    if cluster_groups_after_merge_per_levels:
                        payload["aah/cluster_groups_after_merge_per_level"] = cluster_groups_after_merge_per_levels[0]
                    if cluster_groups_merged_per_levels:
                        payload["aah/cluster_groups_merged_per_level"] = cluster_groups_merged_per_levels[0]
                    if cluster_small_groups_before_merge_per_levels:
                        payload["aah/cluster_small_groups_before_merge_per_level"] = cluster_small_groups_before_merge_per_levels[0]
                    if cluster_singletons_before_merge_per_levels:
                        payload["aah/cluster_singletons_before_merge_per_level"] = cluster_singletons_before_merge_per_levels[0]
                    if cluster_sim_mean_per_levels:
                        payload["aah/cluster_sim_mean_per_level"] = cluster_sim_mean_per_levels[0]
                    if cluster_sim_std_per_levels:
                        payload["aah/cluster_sim_std_per_level"] = cluster_sim_std_per_levels[0]
                    if cluster_sim_min_per_levels:
                        payload["aah/cluster_sim_min_per_level"] = cluster_sim_min_per_levels[0]
                    if cluster_sim_max_per_levels:
                        payload["aah/cluster_sim_max_per_level"] = cluster_sim_max_per_levels[0]
                    if cluster_min_group_sizes:
                        payload["aah/cluster_min_group_size"] = cluster_min_group_sizes[0]
                    if cluster_sim_thresholds:
                        payload["aah/cluster_sim_threshold"] = cluster_sim_thresholds[0]
                    if cluster_super_thresholds:
                        payload["aah/cluster_super_threshold"] = cluster_super_thresholds[0]
                    if cluster_forced_bipartition_per_levels:
                        payload["aah/cluster_forced_bipartition_per_level"] = cluster_forced_bipartition_per_levels[0]
                    if cluster_force_split_anchor_similarity_per_levels:
                        payload["aah/cluster_force_split_anchor_similarity_per_level"] = cluster_force_split_anchor_similarity_per_levels[0]
                    if cluster_origin_per_levels:
                        payload["aah/cluster_origin_per_level"] = cluster_origin_per_levels[0]
                    if cluster_forced_bipartition_allowed_per_levels:
                        payload["aah/cluster_forced_bipartition_allowed_per_level"] = cluster_forced_bipartition_allowed_per_levels[0]
                    if cluster_groups_before_force_per_levels:
                        payload["aah/cluster_groups_before_force_per_level"] = cluster_groups_before_force_per_levels[0]
                    if cluster_feature_norm_mean_per_levels:
                        payload["aah/cluster_feature_norm_mean_per_level"] = cluster_feature_norm_mean_per_levels[0]
                    if cluster_feature_norm_std_per_levels:
                        payload["aah/cluster_feature_norm_std_per_level"] = cluster_feature_norm_std_per_levels[0]
                    if cluster_feature_dim_var_mean_per_levels:
                        payload["aah/cluster_feature_dim_var_mean_per_level"] = cluster_feature_dim_var_mean_per_levels[0]
                    if cluster_feature_dim_var_std_per_levels:
                        payload["aah/cluster_feature_dim_var_std_per_level"] = cluster_feature_dim_var_std_per_levels[0]
                    if cluster_feature_l2_dist_mean_per_levels:
                        payload["aah/cluster_feature_l2_dist_mean_per_level"] = cluster_feature_l2_dist_mean_per_levels[0]
                    if cluster_feature_l2_dist_std_per_levels:
                        payload["aah/cluster_feature_l2_dist_std_per_level"] = cluster_feature_l2_dist_std_per_levels[0]
                    if cluster_feature_top_singular_ratio_per_levels:
                        payload["aah/cluster_feature_top_singular_ratio_per_level"] = cluster_feature_top_singular_ratio_per_levels[0]
                    if hierarchy_level_added_per_levels:
                        payload["aah/hierarchy_level_added_per_level"] = hierarchy_level_added_per_levels[0]
                    if hierarchy_growth_stopped_per_levels:
                        payload["aah/hierarchy_growth_stopped_per_level"] = hierarchy_growth_stopped_per_levels[0]
                    if hierarchy_stop_reason_per_levels:
                        payload["aah/hierarchy_stop_reason_per_level"] = hierarchy_stop_reason_per_levels[0]
                    if cluster_forced_bipartition_level0 is not None:
                        payload["aah/cluster_forced_bipartition_level0"] = cluster_forced_bipartition_level0
                        payload["aah/forced_bipartition"] = cluster_forced_bipartition_level0
                    if cluster_force_split_anchor_similarity_level0 is not None:
                        payload["aah/cluster_force_split_anchor_similarity_level0"] = cluster_force_split_anchor_similarity_level0
                        payload["aah/force_split_anchor_similarity"] = cluster_force_split_anchor_similarity_level0
                    if cluster_sim_mean_level0 is not None:
                        payload["aah/cluster_sim_mean_level0"] = cluster_sim_mean_level0
                    if cluster_sim_std_level0 is not None:
                        payload["aah/cluster_sim_std_level0"] = cluster_sim_std_level0
                    if cluster_sim_min_level0 is not None:
                        payload["aah/cluster_sim_min_level0"] = cluster_sim_min_level0
                    if cluster_sim_max_level0 is not None:
                        payload["aah/cluster_sim_max_level0"] = cluster_sim_max_level0
                    if cluster_item_count_level0 is not None:
                        payload["aah/cluster_item_count_level0"] = cluster_item_count_level0
                    if cluster_groups_before_merge_level0 is not None:
                        payload["aah/cluster_groups_before_merge_level0"] = cluster_groups_before_merge_level0
                    if cluster_groups_after_merge_level0 is not None:
                        payload["aah/cluster_groups_after_merge_level0"] = cluster_groups_after_merge_level0
                    if cluster_groups_merged_level0 is not None:
                        payload["aah/cluster_groups_merged_level0"] = cluster_groups_merged_level0
                    if cluster_small_groups_before_merge_level0 is not None:
                        payload["aah/cluster_small_groups_before_merge_level0"] = cluster_small_groups_before_merge_level0
                    if cluster_singletons_before_merge_level0 is not None:
                        payload["aah/cluster_singletons_before_merge_level0"] = cluster_singletons_before_merge_level0
                    if feature_dim_var_mean is not None:
                        payload["aah/feature_dim_var_mean"] = feature_dim_var_mean
                    if feature_dim_var_std is not None:
                        payload["aah/feature_dim_var_std"] = feature_dim_var_std
                    if feature_dim_var_min is not None:
                        payload["aah/feature_dim_var_min"] = feature_dim_var_min
                    if feature_dim_var_max is not None:
                        payload["aah/feature_dim_var_max"] = feature_dim_var_max
                    if feature_cos_sim_mean is not None:
                        payload["aah/feature_cos_sim_mean"] = feature_cos_sim_mean
                    if feature_cos_sim_std is not None:
                        payload["aah/feature_cos_sim_std"] = feature_cos_sim_std
                    if feature_cos_sim_min is not None:
                        payload["aah/feature_cos_sim_min"] = feature_cos_sim_min
                    if feature_cos_sim_max is not None:
                        payload["aah/feature_cos_sim_max"] = feature_cos_sim_max
                    if feature_l2_dist_mean is not None:
                        payload["aah/feature_l2_dist_mean"] = feature_l2_dist_mean
                    if feature_l2_dist_std is not None:
                        payload["aah/feature_l2_dist_std"] = feature_l2_dist_std
                    if feature_l2_dist_min is not None:
                        payload["aah/feature_l2_dist_min"] = feature_l2_dist_min
                    if feature_l2_dist_max is not None:
                        payload["aah/feature_l2_dist_max"] = feature_l2_dist_max
                    if feature_norm_mean is not None:
                        payload["aah/feature_norm_mean"] = feature_norm_mean
                    if feature_norm_std is not None:
                        payload["aah/feature_norm_std"] = feature_norm_std
                    if feature_top_singular_ratio is not None:
                        payload["aah/feature_top_singular_ratio"] = feature_top_singular_ratio
                    if branch_usage_agg:
                        payload["aah/branch_usage_freq"] = branch_usage_agg
                    if path_mode_freq:
                        for mode, freq in path_mode_freq.items():
                            payload[f"aah/path_mode_freq/{mode}"] = freq
                    if avg_overlap is not None:
                        payload["aah/group_overlap"] = avg_overlap
                    if avg_reassign is not None:
                        payload["aah/head_reassign_rate"] = avg_reassign
                    if lifespan_ema is not None:
                        payload["aah/group_lifespan_ema"] = lifespan_ema
                    wandb.log(payload)
                if csv_writer:
                    lk_serialized = ""
                    if lk_layers:
                        lk_serialized = "|".join([",".join(map(str, layer)) for layer in lk_layers])
                    ent_serialized = ""
                    if head_entropy:
                        ent_serialized = ",".join([f"{v:.4f}" for v in head_entropy])
                    def fmt(v):
                        return f"{v:.2f}" if v is not None else ""
                    csv_writer.writerow([
                        step,
                        f"{loss.item():.6f}",
                        f"{tok_per_sec:.2f}",
                        f"{mem:.2f}",
                        f"{gpu_alloc:.2f}" if gpu_alloc is not None else "",
                        f"{gpu_reserved:.2f}" if gpu_reserved is not None else "",
                        f"{gpu_alloc_max:.2f}" if gpu_alloc_max is not None else "",
                        f"{gpu_reserved_max:.2f}" if gpu_reserved_max is not None else "",
                        f"{cpu_rss:.2f}" if cpu_rss is not None else "",
                        fmt(ps_mem.get("psutil_rss_mb")) if ps_mem else "",
                        fmt(ps_mem.get("psutil_vms_mb")) if ps_mem else "",
                        fmt(ps_mem.get("psutil_shared_mb")) if ps_mem else "",
                        fmt(ps_mem.get("psutil_text_mb")) if ps_mem else "",
                        fmt(ps_mem.get("psutil_data_mb")) if ps_mem else "",
                        fmt(ps_mem.get("psutil_uss_mb")) if ps_mem else "",
                        fmt(ps_mem.get("psutil_pss_mb")) if ps_mem else "",
                        fmt(ps_mem.get("psutil_swap_mb")) if ps_mem else "",
                        fmt(ps_mem.get("psutil_ram_used_mb")) if ps_mem else "",
                        fmt(ps_mem.get("psutil_ram_total_mb")) if ps_mem else "",
                        "",
                        "",
                        f"{attn_elems:.2f}" if attn_elems is not None else "",
                        f"{attn_ratio:.6f}" if attn_ratio is not None else "",
                        f"{attn_reduction:.6f}" if attn_reduction is not None else "",
                        f"{effective_attn_elements:.2f}" if effective_attn_elements is not None else "",
                        f"{effective_ACR:.6f}" if effective_ACR is not None else "",
                        f"{dense_kernel_actual_elements_est:.2f}" if dense_kernel_actual_elements_est is not None else "",
                        f"{backend_realized_elements_est:.2f}" if backend_realized_elements_est is not None else "",
                        f"{backend_realized_ACR_est:.6f}" if backend_realized_ACR_est is not None else "",
                        backend_name,
                        requested_backend,
                        str(backend_bucket_counts) if backend_bucket_counts else "",
                        str(backend_kernel_calls) if backend_kernel_calls is not None else "",
                        fmt(backend_time_ms),
                        str(backend_fallback_reasons) if backend_fallback_reasons else "",
                        f"{analytic_flops_attn_est:.2f}" if analytic_flops_attn_est is not None else "",
                        f"{analytic_flops_total_est:.2f}" if analytic_flops_total_est is not None else "",
                        f"{analytic_flops_ratio:.6f}" if analytic_flops_ratio is not None else "",
                        f"{analytic_flops_reduction_pct:.4f}" if analytic_flops_reduction_pct is not None else "",
                        str(lq) if lq is not None else "",
                        lk_serialized,
                        ent_serialized,
                        f"{group_change_rate:.6f}" if group_change_rate is not None else "",
                        f"{avg_window:.2f}" if avg_window is not None else "",
                        f"{avg_overlap:.6f}" if avg_overlap is not None else "",
                        f"{avg_reassign:.6f}" if avg_reassign is not None else "",
                        f"{lifespan_ema:.6f}" if lifespan_ema is not None else "",
                        "|".join([",".join(map(str, g)) for g in head_groups]) if head_groups else "",
                        "|".join([",".join(map(str, g)) for g in shadow_win_idx]) if shadow_win_idx else "",
                        "|".join([",".join(f"{v:.6f}" for v in g) for g in shadow_logit_mean]) if shadow_logit_mean else "",
                        str(group_heads[0]) if group_heads else "",
                        str(group_ratios[0]) if group_ratios else "",
                        fmt(resolution_mean),
                        fmt(resolution_std),
                        fmt(resolution_min_frac),
                        fmt(resolution_max_frac),
                        fmt(resolution_collapse_min),
                        fmt(resolution_collapse_max),
                        fmt(resolution_delta),
                        str(branch_usage_agg) if branch_usage_agg else "",
                        fmt(attn_ratio_std_ema),
                        fmt(analytic_flops_ratio_std_ema),
                        fmt(lk_mean),
                        fmt(lk_p90),
                        fmt(w_mean),
                        fmt(w_min),
                        fmt(w_max),
                        fmt(control_time_ms),
                        fmt(control_feature_time_ms),
                        fmt(hierarchy_time_ms),
                        fmt(window_select_time_ms),
                        fmt(control_mapping_time_ms),
                        fmt(diagnostics_pack_time_ms),
                        fmt(attn_time_ms),
                        fmt(mask_time_ms),
                        fmt(overhead_time_ms),
                        f"{step_time_ms:.2f}",
                        last_eval_time_s,
                        "",
                        "",
                        "",
                        str(path_mode_freq) if path_mode_freq else "",
                        str(group_counts_per_levels[0]) if group_counts_per_levels else "",
                        str(controller_logits_std_per_levels[0]) if controller_logits_std_per_levels else "",
                        str(win_idx_pre_parent_clamps[0]) if win_idx_pre_parent_clamps else "",
                        str(win_idx_post_parent_clamps[0]) if win_idx_post_parent_clamps else "",
                        str(decision_logits_per_levels[0]) if decision_logits_per_levels else "",
                        str(decision_logits_var_per_levels[0]) if decision_logits_var_per_levels else "",
                        str(controller_input_per_levels[0]) if controller_input_per_levels else "",
                        str(controller_input_cos_sim_mean_per_levels[0]) if controller_input_cos_sim_mean_per_levels else "",
                        str(controller_input_cos_sim_min_per_levels[0]) if controller_input_cos_sim_min_per_levels else "",
                        str(controller_input_l2_dist_mean_per_levels[0]) if controller_input_l2_dist_mean_per_levels else "",
                        str(controller_input_dim_var_mean_per_levels[0]) if controller_input_dim_var_mean_per_levels else "",
                        str(decision_logits_margin_mean_per_levels[0]) if decision_logits_margin_mean_per_levels else "",
                        str(decision_logits_margin_min_per_levels[0]) if decision_logits_margin_min_per_levels else "",
                        str(decision_argmax_diversity_frac_per_levels[0]) if decision_argmax_diversity_frac_per_levels else "",
                        str(sibling_feature_delta_norm_mean_per_levels[0]) if sibling_feature_delta_norm_mean_per_levels else "",
                        str(sibling_feature_delta_norm_min_per_levels[0]) if sibling_feature_delta_norm_min_per_levels else "",
                        str(sibling_feature_delta_norm_max_per_levels[0]) if sibling_feature_delta_norm_max_per_levels else "",
                        str(sibling_feature_cos_mean_per_levels[0]) if sibling_feature_cos_mean_per_levels else "",
                        str(sibling_feature_cos_min_per_levels[0]) if sibling_feature_cos_min_per_levels else "",
                        str(sibling_logit_delta_l2_mean_per_levels[0]) if sibling_logit_delta_l2_mean_per_levels else "",
                        str(sibling_logit_delta_l2_max_per_levels[0]) if sibling_logit_delta_l2_max_per_levels else "",
                        str(sibling_logit_delta_abs_mean_per_levels[0]) if sibling_logit_delta_abs_mean_per_levels else "",
                        str(sibling_logit_delta_abs_max_per_levels[0]) if sibling_logit_delta_abs_max_per_levels else "",
                        str(sibling_ranking_diff_frac_per_levels[0]) if sibling_ranking_diff_frac_per_levels else "",
                        str(sibling_top1_differ_frac_per_levels[0]) if sibling_top1_differ_frac_per_levels else "",
                        str(sibling_top1_ids_per_levels[0]) if sibling_top1_ids_per_levels else "",
                        str(pairwise_bias_l2_mean_per_levels[0]) if pairwise_bias_l2_mean_per_levels else "",
                        str(pairwise_bias_l2_max_per_levels[0]) if pairwise_bias_l2_max_per_levels else "",
                        str(pairwise_bias_abs_mean_per_levels[0]) if pairwise_bias_abs_mean_per_levels else "",
                        str(pairwise_bias_abs_max_per_levels[0]) if pairwise_bias_abs_max_per_levels else "",
                        str(pairwise_bias_top1_changed_frac_per_levels[0]) if pairwise_bias_top1_changed_frac_per_levels else "",
                        str(pairwise_base_top1_ids_per_levels[0]) if pairwise_base_top1_ids_per_levels else "",
                        str(joint_pair_count_per_levels[0]) if joint_pair_count_per_levels else "",
                        str(joint_output_delta_l2_mean_per_levels[0]) if joint_output_delta_l2_mean_per_levels else "",
                        str(joint_output_delta_l2_max_per_levels[0]) if joint_output_delta_l2_max_per_levels else "",
                        str(joint_output_abs_delta_mean_per_levels[0]) if joint_output_abs_delta_mean_per_levels else "",
                        str(joint_output_abs_delta_max_per_levels[0]) if joint_output_abs_delta_max_per_levels else "",
                        str(joint_top1_changed_frac_per_levels[0]) if joint_top1_changed_frac_per_levels else "",
                        str(joint_base_top1_ids_per_levels[0]) if joint_base_top1_ids_per_levels else "",
                        str(joint_output_scale_per_levels[0]) if joint_output_scale_per_levels else "",
                        str(decision_raw_idx_per_levels[0]) if decision_raw_idx_per_levels else "",
                        str(decision_parent_idx_per_levels[0]) if decision_parent_idx_per_levels else "",
                        str(decision_post_parent_idx_per_levels[0]) if decision_post_parent_idx_per_levels else "",
                        str(decision_differ_from_parent_frac_per_levels[0]) if decision_differ_from_parent_frac_per_levels else "",
                        str(decision_unique_raw_idx_per_levels[0]) if decision_unique_raw_idx_per_levels else "",
                        str(decision_unique_post_parent_idx_per_levels[0]) if decision_unique_post_parent_idx_per_levels else "",
                        str(decision_non_one_raw_count_per_levels[0]) if decision_non_one_raw_count_per_levels else "",
                        str(decision_non_one_post_parent_count_per_levels[0]) if decision_non_one_post_parent_count_per_levels else "",
                        str(decision_head_idx_before_execution_mappings[0]) if decision_head_idx_before_execution_mappings else "",
                        str(decision_head_idx_after_execution_mappings[0]) if decision_head_idx_after_execution_mappings else "",
                        str(decision_unique_head_idx_before_execution_mappings[0]) if decision_unique_head_idx_before_execution_mappings else "",
                        str(decision_unique_head_idx_after_execution_mappings[0]) if decision_unique_head_idx_after_execution_mappings else "",
                        str(decision_head_idx_changed_by_execution_mapping_fracs[0]) if decision_head_idx_changed_by_execution_mapping_fracs else "",
                        str([sorted(set(v)) for v in decision_head_idx_before_execution_mappings]) if decision_head_idx_before_execution_mappings else "",
                        str([sorted(set(v)) for v in decision_head_idx_after_execution_mappings]) if decision_head_idx_after_execution_mappings else "",
                        str(decision_head_idx_changed_by_execution_mapping_fracs) if decision_head_idx_changed_by_execution_mapping_fracs else "",
                        str(decision_unique_raw_idx_per_levels) if decision_unique_raw_idx_per_levels else "",
                        str(decision_unique_post_parent_idx_per_levels) if decision_unique_post_parent_idx_per_levels else "",
                        str(sibling_ranking_diff_frac_per_levels) if sibling_ranking_diff_frac_per_levels else "",
                        str(sibling_top1_differ_frac_per_levels) if sibling_top1_differ_frac_per_levels else "",
                        str(sibling_top1_ids_per_levels) if sibling_top1_ids_per_levels else "",
                        str(joint_pair_count_per_levels) if joint_pair_count_per_levels else "",
                        str(joint_output_delta_l2_mean_per_levels) if joint_output_delta_l2_mean_per_levels else "",
                        str(joint_output_abs_delta_mean_per_levels) if joint_output_abs_delta_mean_per_levels else "",
                        str(joint_top1_changed_frac_per_levels) if joint_top1_changed_frac_per_levels else "",
                        str(joint_base_top1_ids_per_levels) if joint_base_top1_ids_per_levels else "",
                        str(hierarchy_head_group_map_per_levels[0]) if hierarchy_head_group_map_per_levels else "",
                        str(hierarchy_group_members_per_levels[0]) if hierarchy_group_members_per_levels else "",
                        str(cluster_metric_per_levels[0]) if cluster_metric_per_levels else "",
                        str(cluster_threshold_kind_per_levels[0]) if cluster_threshold_kind_per_levels else "",
                        str(cluster_threshold_per_levels[0]) if cluster_threshold_per_levels else "",
                        str(cluster_item_count_per_levels[0]) if cluster_item_count_per_levels else "",
                        str(cluster_groups_before_merge_per_levels[0]) if cluster_groups_before_merge_per_levels else "",
                        str(cluster_groups_after_merge_per_levels[0]) if cluster_groups_after_merge_per_levels else "",
                        str(cluster_groups_merged_per_levels[0]) if cluster_groups_merged_per_levels else "",
                        str(cluster_sim_min_per_levels[0]) if cluster_sim_min_per_levels else "",
                        str(cluster_sim_mean_per_levels[0]) if cluster_sim_mean_per_levels else "",
                        str(cluster_sim_max_per_levels[0]) if cluster_sim_max_per_levels else "",
                        str(cluster_sim_std_per_levels[0]) if cluster_sim_std_per_levels else "",
                        str(cluster_forced_bipartition_per_levels[0]) if cluster_forced_bipartition_per_levels else "",
                        str(cluster_force_split_anchor_similarity_per_levels[0]) if cluster_force_split_anchor_similarity_per_levels else "",
                        str(cluster_origin_per_levels[0]) if cluster_origin_per_levels else "",
                        str(cluster_forced_bipartition_allowed_per_levels[0]) if cluster_forced_bipartition_allowed_per_levels else "",
                        str(cluster_groups_before_force_per_levels[0]) if cluster_groups_before_force_per_levels else "",
                        str(cluster_feature_norm_mean_per_levels[0]) if cluster_feature_norm_mean_per_levels else "",
                        str(cluster_feature_norm_std_per_levels[0]) if cluster_feature_norm_std_per_levels else "",
                        str(cluster_feature_dim_var_mean_per_levels[0]) if cluster_feature_dim_var_mean_per_levels else "",
                        str(cluster_feature_dim_var_std_per_levels[0]) if cluster_feature_dim_var_std_per_levels else "",
                        str(cluster_feature_l2_dist_mean_per_levels[0]) if cluster_feature_l2_dist_mean_per_levels else "",
                        str(cluster_feature_l2_dist_std_per_levels[0]) if cluster_feature_l2_dist_std_per_levels else "",
                        str(cluster_feature_top_singular_ratio_per_levels[0]) if cluster_feature_top_singular_ratio_per_levels else "",
                        str(hierarchy_level_added_per_levels[0]) if hierarchy_level_added_per_levels else "",
                        str(hierarchy_growth_stopped_per_levels[0]) if hierarchy_growth_stopped_per_levels else "",
                        str(hierarchy_stop_reason_per_levels[0]) if hierarchy_stop_reason_per_levels else "",
                        str(cluster_forced_bipartition_level0) if cluster_forced_bipartition_level0 is not None else "",
                        f"{cluster_force_split_anchor_similarity_level0:.6f}" if cluster_force_split_anchor_similarity_level0 is not None else "",
                    ])
                if head_groups:
                    prev_head_groups = [list(g) for g in head_groups]
                t0 = time.time()

            eval_time_s = ""
            if step % train["eval_interval"] == 0:
                val_loss, val_ppl, eval_time = evaluate(
                    model,
                    val_loader,
                    device,
                    train["eval_batches"],
                    log_progress=train.get("eval_log_progress", False),
                    use_bf16=use_bf16,
                )
                last_eval_time_s = f"{eval_time:.2f}"
                print(f"eval step {step} | loss {val_loss:.4f} | ppl {val_ppl:.2f}")
                val_group_count_total, val_group_count_level0, val_group_count_per_level = get_group_count_metrics(model)
                if use_wandb:
                    payload = {"val/loss": val_loss, "val/ppl": val_ppl, "step": step, "perf/eval_time_s": eval_time}
                    if val_group_count_total is not None:
                        payload["aah/group_count_total"] = val_group_count_total
                    if val_group_count_level0 is not None:
                        payload["aah/group_count_level0"] = val_group_count_level0
                    if val_group_count_per_level:
                        payload["aah/group_count_per_level"] = val_group_count_per_level
                        for i, v in enumerate(val_group_count_per_level):
                            payload[f"aah/group_count_level{i}"] = v
                    wandb.log(payload)
                if csv_writer:
                    row = [""] * len(csv_headers)
                    row[csv_idx["step"]] = str(step)
                    row[csv_idx["val_loss"]] = f"{val_loss:.6f}"
                    row[csv_idx["val_ppl"]] = f"{val_ppl:.4f}"
                    row[csv_idx["eval_time_s"]] = f"{eval_time:.2f}"
                    if val_group_count_total is not None:
                        row[csv_idx["val_group_count_total"]] = f"{val_group_count_total:.6f}"
                    if val_group_count_level0 is not None:
                        row[csv_idx["val_group_count_level0"]] = f"{val_group_count_level0:.6f}"
                    if val_group_count_per_level:
                        row[csv_idx["val_group_count_per_level"]] = ",".join(f"{v:.6f}" for v in val_group_count_per_level)
                    csv_writer.writerow(row)

                if step >= train["max_steps"]:
                    break
        if save_checkpoints:
            if int(train["max_steps"]) not in saved_checkpoint_steps:
                persist_checkpoint(int(train["max_steps"]), role="final")
                saved_checkpoint_steps.add(int(train["max_steps"]))
            torch.save(model.state_dict(), os.path.join(out_dir, f"{exp['name']}.pt"))
    except Exception as exc:
        tb = traceback.format_exc()
        print(tb)
        try:
            with open(crash_log_path, "w") as f:
                f.write(tb)
        except Exception:
            pass
        if use_wandb and wandb_mod is not None:
            try:
                wandb_mod.log({"run/crashed": 1, "run/error": str(exc)[:500], "step": step})
            except Exception:
                pass
        raise
    finally:
        if csv_file:
            csv_file.close()
        if use_wandb and wandb_mod is not None:
            try:
                wandb_mod.finish()
            except Exception:
                pass


if __name__ == "__main__":
    main()
