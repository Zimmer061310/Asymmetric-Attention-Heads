"""Generate FLOPs lab config files from the Flash AAH template."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from experiments.aah_flops_reduction_lab._common.naming import (
    AAH_FLASH_TEMPLATE,
    PURE_FLASH_BASELINE,
    RUN_ROOT,
    VARIANTS,
    hypothesis_dir,
)


def lab_mode(axis: str) -> str:
    if axis == "static-plan":
        return "static_compiled_plan"
    if axis == "quantized":
        return "quantized_execution"
    if axis == "noscatter":
        return "noscatter_prototype"
    if axis == "fixed":
        return "fixed_plan"
    if axis == "slow-update":
        # Nsight profiles one forward pass between updates, so the profile-time
        # execution path is a compiled fixed plan plus interval metadata.
        return "fixed_plan"
    return axis.replace("-", "_")


def variant_bucket_policy(variant: str) -> dict:
    if variant == "single-1024":
        return {"kind": "single", "windows": [1024]}
    if variant == "single-2048":
        return {"kind": "single", "windows": [2048]}
    if variant == "two-bucket-1024-4096":
        return {"kind": "two_bucket", "windows": [1024, 4096], "threshold": 2048}
    if variant == "two-bucket-2048-4096":
        return {"kind": "two_bucket", "windows": [2048, 4096], "threshold": 4096}
    if variant in {"contiguous-1024-4096", "scatter-control-matched"}:
        return {"kind": "two_bucket", "windows": [1024, 4096], "threshold": 2048}
    return {"kind": "from_plan"}


def build_config(template: dict, variant) -> dict:
    cfg = yaml.safe_load(yaml.safe_dump(template))
    bucket_policy = variant_bucket_policy(variant.variant)
    bucket_windows = list(bucket_policy.get("windows", []))
    lab_bucket_policy = dict(bucket_policy)
    if bucket_windows:
        lab_bucket_policy["windows"] = list(bucket_windows)
    cfg["experiment"]["name"] = variant.name
    cfg["experiment"]["variant"] = f"{variant.axis}_{variant.variant}_{variant.backend}".replace("-", "_")
    cfg["experiment"]["out_dir"] = f"experiments/aah_flops_reduction_lab/{variant.hypothesis}/results"
    model_cfg = cfg.setdefault("model", {})
    model_cfg["aah_v3_attention_backend"] = "flash_attn"
    model_cfg["aah_flopslab_enabled"] = True
    model_cfg["aah_flopslab_mode"] = lab_mode(variant.axis)
    model_cfg["aah_flopslab_variant"] = variant.variant
    model_cfg["aah_flopslab_plan_path"] = f"{RUN_ROOT}/plans/{variant.name}.json" if variant.requires_plan else ""
    model_cfg["aah_flopslab_bucket_policy_kind"] = bucket_policy.get("kind", "")
    model_cfg["aah_flopslab_bucket_windows"] = list(bucket_windows)
    model_cfg["aah_flopslab_bucket_threshold"] = int(bucket_policy.get("threshold", 0) or 0)
    cfg.setdefault("train", {})["batch_size"] = 1
    cfg["train"]["precision"] = "bf16"
    cfg["train"]["eval_batches"] = 20
    cfg["train"]["eval_interval"] = 200
    cfg.setdefault("profiling", {})["baseline_config"] = PURE_FLASH_BASELINE
    cfg["profiling"]["ncu_gpu_flops_ratio_required"] = True
    cfg["profiling"]["notes"] = (
        "FLOPs lab variant. gpu_flops_total_ratio_ncu must be Nsight GPU FP ops "
        "divided by the matched pure FlashAttention baseline."
    )
    cfg["lab"] = {
        "enabled": True,
        "hypothesis": variant.hypothesis,
        "axis": variant.axis,
        "variant": variant.variant,
        "mode": lab_mode(variant.axis),
        "description": variant.description,
        "requires_plan": variant.requires_plan,
        "plan_path": f"{RUN_ROOT}/plans/{variant.name}.json" if variant.requires_plan else "",
        "bucket_policy": lab_bucket_policy,
        "baseline_config": PURE_FLASH_BASELINE,
        "primary_metric": "gpu_flops_total_ratio_ncu",
        "success_threshold": 1.0,
    }
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default=AAH_FLASH_TEMPLATE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.template, "r") as f:
        template = yaml.safe_load(f)

    for variant in VARIANTS:
        out_dir = Path(hypothesis_dir(variant)) / "configs"
        out_path = out_dir / variant.yaml_name
        cfg = build_config(template, variant)
        print(out_path)
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)


if __name__ == "__main__":
    main()
