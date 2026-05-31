"""Profile backend-local attention with DeepSpeed FLOPs Profiler.

DeepSpeed FLOPs Profiler is a software/model profiler, not a hardware-counter
profiler. For FlashAttention/FlexAttention custom kernels it may not directly
see the local-window math, so this script records both raw DeepSpeed FLOPs and
an adjusted estimate that combines DeepSpeed non-attention FLOPs with the
backend-realized attention formula already logged by the model.
"""

import argparse
import json
import os
import sys
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


def write_json(path, payload):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def adjusted_total_flops(raw_total, dense_attention, realized_attention):
    raw_total = float(raw_total or 0.0)
    dense_attention = float(dense_attention or 0.0)
    realized_attention = float(realized_attention or 0.0)
    if raw_total <= 0.0:
        return None, "raw_deepspeed_flops_missing"
    if dense_attention <= 0.0:
        return raw_total, "no_attention_formula_available"

    # If raw DeepSpeed already looks larger than the dense attention formula, it
    # likely counted dense attention-shaped work. Replace that component with
    # backend-realized local attention. If not, assume the custom backend was not
    # counted and add the backend-realized attention formula separately.
    if raw_total >= 1.05 * dense_attention:
        return max(0.0, raw_total - dense_attention + realized_attention), "replace_dense_attention_component"
    return raw_total + realized_attention, "add_backend_attention_component"


def profile_config(config_path, module_key, device_name=None, checkpoint=None, warmup=3, detailed=False, output_file=None):
    try:
        from deepspeed.profiling.flops_profiler import get_model_profile
    except Exception as exc:
        return {
            "ok": False,
            "error_kind": "deepspeed_import_failed",
            "error": str(exc),
            "install_hint": "pip install deepspeed",
        }

    cfg = load_config(config_path)
    train = cfg.get("train", {})
    device = get_device(device_name or train.get("device", "cuda"))
    precision = train.get("precision", "bf16")
    model, gpt_cfg, checkpoint_loaded, checkpoint_path = load_model(cfg, module_key, device, checkpoint)
    batch_size = int(train.get("batch_size", 1))
    seq_len = int(cfg["data"]["seq_len"])
    torch.manual_seed(int(cfg.get("experiment", {}).get("seed", 0)))
    idx = torch.randint(0, int(gpt_cfg.vocab_size), (batch_size, seq_len), device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    with torch.no_grad():
        with autocast_context(device, precision):
            flops, macs, params = get_model_profile(
                model=model,
                args=[idx],
                print_profile=bool(output_file),
                detailed=bool(detailed),
                module_depth=-1 if detailed else 2,
                top_modules=10,
                warm_up=max(0, int(warmup)),
                as_string=False,
                output_file=output_file,
                mode="forward",
            )
        sync(device)
    elapsed = time.time() - started

    stats = collect_backend_stats(model, batch_size=batch_size, head_dim=int(gpt_cfg.n_embd) // int(gpt_cfg.n_head))
    adjusted, adjusted_mode = adjusted_total_flops(
        raw_total=flops,
        dense_attention=stats["dense_attention_flops_formula"],
        realized_attention=stats["realized_attention_flops_formula"],
    )
    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        "ok": True,
        "config_path": os.path.abspath(config_path),
        "module": module_key,
        "checkpoint_loaded": checkpoint_loaded,
        "checkpoint_path": checkpoint_path,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "precision": precision,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "deepspeed_raw_total_flops": float(flops),
        "deepspeed_raw_macs": float(macs),
        "deepspeed_params": float(params),
        "deepspeed_adjusted_total_flops_est": adjusted,
        "deepspeed_adjusted_attention_mode": adjusted_mode,
        "deepspeed_elapsed_s": elapsed,
        "peak_memory_mb": peak_memory_mb,
        "metric_source": "DeepSpeed Flops Profiler plus backend-realized attention formula estimate",
        "hardware_counter_metric": False,
        **stats,
    }


def add_ratios(result, baseline_json):
    if not result.get("ok"):
        return result
    if not baseline_json:
        result["deepspeed_raw_total_flops_ratio"] = 1.0
        result["deepspeed_adjusted_total_flops_ratio_est"] = 1.0
        return result
    with open(baseline_json, "r") as f:
        baseline = json.load(f)
    raw_denom = float(baseline.get("deepspeed_raw_total_flops") or 0.0)
    adj_denom = float(baseline.get("deepspeed_adjusted_total_flops_est") or 0.0)
    result["baseline_json"] = os.path.abspath(baseline_json)
    result["deepspeed_raw_total_flops_ratio"] = (
        float(result.get("deepspeed_raw_total_flops") or 0.0) / raw_denom if raw_denom > 0 else None
    )
    result["deepspeed_adjusted_total_flops_ratio_est"] = (
        float(result.get("deepspeed_adjusted_total_flops_est") or 0.0) / adj_denom if adj_denom > 0 else None
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--module", choices=("pure", "aah"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--baseline-json")
    parser.add_argument("--detailed", action="store_true")
    parser.add_argument("--profile-output")
    args = parser.parse_args()

    result = profile_config(
        args.config,
        args.module,
        device_name=args.device,
        checkpoint=args.checkpoint,
        warmup=args.warmup,
        detailed=args.detailed,
        output_file=args.profile_output,
    )
    result = add_ratios(result, args.baseline_json)
    write_json(args.output, result)
    if not result.get("ok"):
        print(f"deepspeed_flops_profile_failed {result.get('error_kind')} wrote {args.output}")
        raise SystemExit(2)
    print(
        "wrote_deepspeed_flops_profile "
        f"{args.output} raw_ratio={result.get('deepspeed_raw_total_flops_ratio')} "
        f"adjusted_ratio={result.get('deepspeed_adjusted_total_flops_ratio_est')}"
    )


if __name__ == "__main__":
    main()
