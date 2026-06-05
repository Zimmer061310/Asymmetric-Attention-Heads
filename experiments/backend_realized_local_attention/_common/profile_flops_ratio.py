"""Legacy Torch-profiler diagnostic for backend-local attention experiments.

This script is intentionally scoped to experiments/backend_realized_local_attention.
It does not import or modify the top-level src/ Transformer.

Do not use this script's `measured_*_flops_ratio` fields as paper FLOPs/FLOPs
evidence. The paper metric is `gpu_flops_total_ratio_ncu` from
`profile_gpu_flops_ncu.py`.
"""

import argparse
import importlib
import json
import os
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path

import torch
import yaml
from torch.profiler import ProfilerActivity, profile, record_function


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODULES = {
    "pure": "experiments.backend_realized_local_attention._common.pure_backend_transformer",
    "aah": "experiments.backend_realized_local_attention._common.aah_backend_transformer",
}

ATTN_SCOPES = {
    "attn_qkv",
    "attn_matmul_qk",
    "attn_matmul_av",
    "attn_flash_backend",
    "attn_flex_backend",
    "attn_dense_masked_backend",
}


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device(name):
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_context(device, precision):
    precision = str(precision).lower()
    if device.type == "cuda" and precision == "bf16" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device.type == "cuda" and precision in {"fp16", "float16"}:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type=device.type, enabled=False)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def build_gpt_config(GPTConfig, cfg):
    data = cfg["data"]
    model_cfg = cfg["model"]
    lab_cfg = cfg.get("lab", {}) or {}
    bucket_policy = lab_cfg.get("bucket_policy", {}) or {}

    def model_or_lab(model_key, lab_key=None, default=None):
        if model_key in model_cfg:
            return model_cfg.get(model_key)
        if lab_key and lab_key in lab_cfg:
            return lab_cfg.get(lab_key)
        return default

    params = {
        "vocab_size": int(model_cfg.get("vocab_size", 50257)),
        "seq_len": int(data["seq_len"]),
        "n_layer": int(model_cfg["n_layer"]),
        "n_head": int(model_cfg["n_head"]),
        "n_embd": int(model_cfg["n_embd"]),
        "n_ff": int(model_cfg["n_ff"]),
        "dropout": float(model_cfg.get("dropout", 0.0)),
        "aah_v2_enabled": bool(model_cfg.get("aah_v2_enabled", False)),
        "aah_v2_windows": tuple(model_cfg.get("aah_v2_windows", (64, 128, 256, int(data["seq_len"])))),
        "aah_v2_strides": tuple(model_cfg.get("aah_v2_strides", (1, 2, 4))),
        "aah_v2_group_size": int(model_cfg.get("aah_v2_group_size", 1)),
        "aah_v2_control_dim": int(model_cfg.get("aah_v2_control_dim", 16)),
        "aah_v2_temperature": float(model_cfg.get("aah_v2_temperature", 1.0)),
        "aah_v2_dynamic_grouping": bool(model_cfg.get("aah_v2_dynamic_grouping", False)),
        "aah_v2_num_groups": int(model_cfg.get("aah_v2_num_groups", 4)),
        "aah_v2_local_chunk": int(model_cfg.get("aah_v2_local_chunk", 128)),
        "aah_v2_control_interval": int(model_cfg.get("aah_v2_control_interval", 1)),
        "aah_v2_stride_control_enabled": bool(model_cfg.get("aah_v2_stride_control_enabled", True)),
        "aah_v3_enabled": bool(model_cfg.get("aah_v3_enabled", False)),
        "aah_v3_windows": tuple(model_cfg.get("aah_v3_windows", (64, 128, 256, int(data["seq_len"])))),
        "aah_v3_control_dim": int(model_cfg.get("aah_v3_control_dim", 16)),
        "aah_v3_control_interval": int(model_cfg.get("aah_v3_control_interval", 100)),
        "aah_v3_sim_threshold": float(model_cfg.get("aah_v3_sim_threshold", 0.7)),
        "aah_v3_super_threshold": float(model_cfg.get("aah_v3_super_threshold", 0.7)),
        "aah_v3_max_depth": int(model_cfg.get("aah_v3_max_depth", 6)),
        "aah_v3_ema_alpha": float(model_cfg.get("aah_v3_ema_alpha", 0.9)),
        "aah_v3_churn_penalty": float(model_cfg.get("aah_v3_churn_penalty", 0.05)),
        "aah_v3_min_group_size": int(model_cfg.get("aah_v3_min_group_size", 1)),
        "aah_v3_warmup_steps": int(model_cfg.get("aah_v3_warmup_steps", 0)),
        "aah_v3_control_enabled": bool(model_cfg.get("aah_v3_control_enabled", True)),
        "aah_v3_grouping_enabled": bool(model_cfg.get("aah_v3_grouping_enabled", True)),
        "aah_v3_build_hierarchy": bool(model_cfg.get("aah_v3_build_hierarchy", model_cfg.get("aah_v3_grouping_enabled", True))),
        "aah_v3_apply_window_control": bool(model_cfg.get("aah_v3_apply_window_control", model_cfg.get("aah_v3_control_enabled", True))),
        "aah_v3_W_min_gpu": int(model_cfg.get("aah_v3_W_min_gpu", 64)),
        "aah_v3_mask_cache_size": int(model_cfg.get("aah_v3_mask_cache_size", 16)),
        "aah_v3_resolution_ema_alpha": float(model_cfg.get("aah_v3_resolution_ema_alpha", 0.0)),
        "aah_v3_resolution_collapse_min_frac": float(model_cfg.get("aah_v3_resolution_collapse_min_frac", 0.95)),
        "aah_v3_resolution_collapse_max_frac": float(model_cfg.get("aah_v3_resolution_collapse_max_frac", 0.95)),
        "aah_v3_post_warmup_ramp_steps": int(model_cfg.get("aah_v3_post_warmup_ramp_steps", 0)),
        "aah_v3_group_feature_mode": str(model_cfg.get("aah_v3_group_feature_mode", "mean")),
        "aah_v3_upper_cluster_metric": str(model_cfg.get("aah_v3_upper_cluster_metric", "cosine")),
        "aah_v3_upper_l2_threshold": float(model_cfg.get("aah_v3_upper_l2_threshold", 0.0)),
        "aah_v3_cosine_normdiff_scale": float(model_cfg.get("aah_v3_cosine_normdiff_scale", 16.0)),
        "aah_v3_controller_input_mode": str(model_cfg.get("aah_v3_controller_input_mode", "base")),
        "aah_v3_controller_arch": str(model_cfg.get("aah_v3_controller_arch", "mlp")),
        "aah_v3_controller_logit_scale": float(model_cfg.get("aah_v3_controller_logit_scale", 1.0)),
        "aah_v3_controller_rng_reference_dim": int(model_cfg.get("aah_v3_controller_rng_reference_dim", 16)),
        "aah_v3_controller_choice_mode": str(model_cfg.get("aah_v3_controller_choice_mode", "learned")),
        "aah_v3_controller_pairwise_mode": str(model_cfg.get("aah_v3_controller_pairwise_mode", "none")),
        "aah_v3_pairwise_bias_scale": float(model_cfg.get("aah_v3_pairwise_bias_scale", 1.0)),
        "aah_v3_joint_output_scale": float(model_cfg.get("aah_v3_joint_output_scale", 1.0)),
        "aah_v3_joint_hidden_dim": int(model_cfg.get("aah_v3_joint_hidden_dim", 0)),
        "aah_v3_diagnostic_detail": str(model_cfg.get("aah_v3_diagnostic_detail", "light")),
        "aah_v3_reuse_group_hierarchy": bool(model_cfg.get("aah_v3_reuse_group_hierarchy", False)),
        "aah_v3_hierarchy_ablation_mode": str(model_cfg.get("aah_v3_hierarchy_ablation_mode", "adaptive")),
        "aah_v3_fixed_hierarchy_seed": int(model_cfg.get("aah_v3_fixed_hierarchy_seed", cfg["experiment"].get("seed", 1337))),
        "aah_v3_parent_constraint": bool(model_cfg.get("aah_v3_parent_constraint", True)),
        "aah_v3_attention_backend": str(model_cfg.get("aah_v3_attention_backend", model_cfg.get("attention_backend", "dense_masked"))),
        "aah_v3_flex_block_size": int(model_cfg.get("aah_v3_flex_block_size", model_cfg.get("flex_block_size", 128))),
        "aah_flopslab_enabled": bool(model_or_lab("aah_flopslab_enabled", "enabled", False)),
        "aah_flopslab_mode": str(model_or_lab("aah_flopslab_mode", "mode", "")),
        "aah_flopslab_variant": str(model_or_lab("aah_flopslab_variant", "variant", "")),
        "aah_flopslab_plan_path": str(model_or_lab("aah_flopslab_plan_path", "plan_path", "")),
        "aah_flopslab_bucket_policy_kind": str(model_cfg.get("aah_flopslab_bucket_policy_kind", bucket_policy.get("kind", ""))),
        "aah_flopslab_bucket_windows": tuple(model_cfg.get("aah_flopslab_bucket_windows", bucket_policy.get("windows", ()))),
        "aah_flopslab_bucket_threshold": int(model_cfg.get("aah_flopslab_bucket_threshold", bucket_policy.get("threshold", 0) or 0)),
        "aah_flopslab_minimal_runtime": bool(model_or_lab("aah_flopslab_minimal_runtime", "minimal_runtime", False)),
        "aah_flopslab_assume_preordered_heads": bool(model_or_lab("aah_flopslab_assume_preordered_heads", "assume_preordered_heads", False)),
    }
    if is_dataclass(GPTConfig):
        allowed = {f.name for f in fields(GPTConfig)}
        params = {k: v for k, v in params.items() if k in allowed}
    return GPTConfig(**params)


def find_checkpoint(cfg):
    exp = cfg["experiment"]
    out_dir = Path(exp.get("out_dir", "experiments"))
    ckpt = out_dir / f"{exp['name']}.pt"
    return ckpt if ckpt.exists() else None


def load_model(cfg, module_key, device, checkpoint=None):
    module = importlib.import_module(MODULES[module_key])
    gpt_cfg = build_gpt_config(module.GPTConfig, cfg)
    model = module.GPT(gpt_cfg).to(device)
    checkpoint_loaded = False
    ckpt = Path(checkpoint) if checkpoint else find_checkpoint(cfg)
    if ckpt and ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state, strict=False)
        checkpoint_loaded = True
    model.eval()
    return model, gpt_cfg, checkpoint_loaded, str(ckpt) if ckpt else ""


def event_has_attention_parent(evt):
    cur = evt
    while cur is not None:
        name = getattr(cur, "name", getattr(cur, "key", ""))
        if name in ATTN_SCOPES or str(name).startswith("attn_"):
            return True
        cur = getattr(cur, "cpu_parent", None)
    return False


def sum_profiler_flops(prof):
    total = 0.0
    attention = 0.0
    for evt in prof.events():
        flops = float(getattr(evt, "flops", 0) or 0)
        if flops <= 0:
            continue
        total += flops
        if event_has_attention_parent(evt):
            attention += flops
    return total, attention


def collect_backend_stats(model, batch_size, head_dim):
    realized_elements = 0.0
    dense_elements = 0.0
    fallback_reasons = []
    backend_names = set()
    for block in getattr(model, "blocks", []):
        stats = getattr(getattr(block, "attn", None), "last_stats", {}) or {}
        realized_elements += float(stats.get("backend_realized_elements_est", stats.get("effective_attn_elements", 0.0)) or 0.0)
        dense_elements += float(stats.get("dense_kernel_actual_elements_est", stats.get("baseline_elements", 0.0)) or 0.0)
        backend = stats.get("backend_name")
        if backend:
            backend_names.add(str(backend))
        fallback_reasons.extend(str(x) for x in stats.get("backend_fallback_reasons", []) if x)
    realized_attention_flops = 4.0 * float(batch_size) * float(head_dim) * realized_elements
    dense_attention_flops = 4.0 * float(batch_size) * float(head_dim) * dense_elements
    return {
        "backend_realized_elements_est": realized_elements,
        "dense_kernel_actual_elements_est": dense_elements,
        "realized_attention_flops_formula": realized_attention_flops,
        "dense_attention_flops_formula": dense_attention_flops,
        "backend_names": sorted(backend_names),
        "backend_fallback_reasons": sorted(set(fallback_reasons)),
    }


def profile_config(config_path, module_key, device_name, checkpoint=None, warmup=1, steps=1):
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

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
        torch.cuda.reset_peak_memory_stats()

    with profile(activities=activities, record_shapes=True, with_flops=True) as prof:
        with torch.no_grad():
            for _ in range(max(1, int(steps))):
                with record_function("profile_total_forward"):
                    with autocast_context(device, precision):
                        model(idx)
                sync(device)

    total_flops, attention_flops = sum_profiler_flops(prof)
    stats = collect_backend_stats(model, batch_size=batch_size, head_dim=int(gpt_cfg.n_embd) // int(gpt_cfg.n_head))
    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    measured_attention = attention_flops if attention_flops > 0 else stats["realized_attention_flops_formula"]
    return {
        "config_path": os.path.abspath(config_path),
        "module": module_key,
        "checkpoint_loaded": checkpoint_loaded,
        "checkpoint_path": checkpoint_path,
        "device": str(device),
        "precision": precision,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "profiler_total_flops": total_flops,
        "profiler_attention_flops": attention_flops,
        "measured_attention_flops": measured_attention,
        "measured_total_flops": total_flops,
        "peak_memory_mb": peak_memory_mb,
        **stats,
    }


def add_ratios(result, baseline):
    for key in ("measured_attention_flops", "measured_total_flops", "profiler_total_flops", "realized_attention_flops_formula"):
        denom = float(baseline.get(key, 0.0) or 0.0)
        result[f"{key}_ratio"] = (float(result.get(key, 0.0) or 0.0) / denom) if denom > 0 else None
    result["measured_attention_flops_ratio"] = result.get("measured_attention_flops_ratio")
    result["measured_total_flops_ratio"] = result.get("measured_total_flops_ratio")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--module", choices=sorted(MODULES), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--baseline-json")
    args = parser.parse_args()

    result = profile_config(args.config, args.module, args.device, args.checkpoint, args.warmup, args.steps)
    if args.baseline_json:
        with open(args.baseline_json, "r") as f:
            baseline = json.load(f)
        result = add_ratios(result, baseline)
    else:
        result["measured_attention_flops_ratio"] = 1.0
        result["measured_total_flops_ratio"] = 1.0

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"wrote_flops_profile {out}")
    print(
        "measured_attention_flops_ratio="
        f"{result.get('measured_attention_flops_ratio')} "
        "measured_total_flops_ratio="
        f"{result.get('measured_total_flops_ratio')}"
    )


if __name__ == "__main__":
    main()
