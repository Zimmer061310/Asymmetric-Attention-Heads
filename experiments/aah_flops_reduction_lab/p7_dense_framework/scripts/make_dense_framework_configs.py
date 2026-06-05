"""Generate dense-framework configs for the AAH FLOPs lab.

The dense framework applies the same lab hypotheses that were first tested
against pure FlashAttention, but changes the denominator to a standard dense
MHA path and sets AAH execution to ``dense_masked``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import yaml


RUN_ROOT = "paper_results/aah_flops_reduction_lab"
P7_ROOT = "experiments/aah_flops_reduction_lab/p7_dense_framework"
P7_CONFIG_DIR = f"{P7_ROOT}/configs"
P7_RESULT_DIR = f"{P7_ROOT}/results"
DENSE_BASELINE_TEMPLATE = (
    "experiments/backend_realized_local_attention/"
    "DenseMasked/memory_sanity/configs/backend_4096_dense_memory_sanity_seed0.yaml"
)
AAH_TEMPLATE = (
    "experiments/backend_realized_local_attention/"
    "FlashAttention/aah_modified/configs/backend_4096_full_adaptive_flash_seed0.yaml"
)
FLASH_PLAN_ROOT = f"{RUN_ROOT}/plans"


@dataclass(frozen=True)
class DenseVariant:
    axis: str
    variant: str
    mode: str
    hypothesis: str
    description: str
    requires_plan: bool = False
    plan_source_name: str = ""
    bucket_policy: dict | None = None
    minimal_runtime: bool = False
    assume_preordered_heads: bool = False
    same_codepath_full: bool = False
    module: str = "aah"

    @property
    def name(self) -> str:
        return f"flopslab-4096-{self.axis}-{self.variant}-dense-seed0"

    @property
    def yaml_name(self) -> str:
        return f"{self.name}.yaml"


def bucket_policy(variant: str) -> dict:
    if variant == "single-1024":
        return {"kind": "single", "windows": [1024]}
    if variant == "single-2048":
        return {"kind": "single", "windows": [2048]}
    if variant == "two-bucket-1024-4096":
        return {"kind": "two_bucket", "windows": [1024, 4096], "threshold": 2048}
    if variant == "two-bucket-2048-4096":
        return {"kind": "two_bucket", "windows": [2048, 4096], "threshold": 4096}
    if variant in {"contiguous-1024-4096", "scatter-control-matched", "minruntime-noscatter-1024-4096", "lowerbound-1024-4096"}:
        return {"kind": "two_bucket", "windows": [1024, 4096], "threshold": 2048}
    return {"kind": "from_plan"}


DENSE_VARIANTS = (
    DenseVariant(
        axis="same-codepath",
        variant="full",
        mode="same_codepath_full",
        hypothesis="p1_same_codepath_full_baseline",
        description="AAH dense backend full-window path with hierarchy/control disabled.",
        same_codepath_full=True,
    ),
    DenseVariant(
        axis="static-plan",
        variant="per-layer",
        mode="static_compiled_plan",
        hypothesis="h1_static_compiled_plan",
        description="dense backend with one exported window vector per layer",
        requires_plan=True,
        plan_source_name="flopslab-4096-static-plan-per-layer-flash-seed0",
    ),
    DenseVariant(
        axis="static-plan",
        variant="per-layer-head",
        mode="static_compiled_plan",
        hypothesis="h1_static_compiled_plan",
        description="dense backend with one exported window per layer/head",
        requires_plan=True,
        plan_source_name="flopslab-4096-static-plan-per-layer-head-flash-seed0",
    ),
    DenseVariant(
        axis="static-plan",
        variant="majority",
        mode="static_compiled_plan",
        hypothesis="h1_static_compiled_plan",
        description="dense backend with majority window from calibration batches",
        requires_plan=True,
        plan_source_name="flopslab-4096-static-plan-majority-flash-seed0",
    ),
    DenseVariant(
        axis="quantized",
        variant="single-1024",
        mode="quantized_execution",
        hypothesis="h2_quantized_execution",
        description="dense backend with all local execution collapsed to W=1024",
        bucket_policy=bucket_policy("single-1024"),
    ),
    DenseVariant(
        axis="quantized",
        variant="single-2048",
        mode="quantized_execution",
        hypothesis="h2_quantized_execution",
        description="dense backend with all local execution collapsed to W=2048",
        bucket_policy=bucket_policy("single-2048"),
    ),
    DenseVariant(
        axis="quantized",
        variant="two-bucket-1024-4096",
        mode="quantized_execution",
        hypothesis="h2_quantized_execution",
        description="dense backend with short W=1024 and full W=4096 buckets",
        bucket_policy=bucket_policy("two-bucket-1024-4096"),
    ),
    DenseVariant(
        axis="quantized",
        variant="two-bucket-2048-4096",
        mode="quantized_execution",
        hypothesis="h2_quantized_execution",
        description="dense backend with short W=2048 and full W=4096 buckets",
        bucket_policy=bucket_policy("two-bucket-2048-4096"),
    ),
    DenseVariant(
        axis="noscatter",
        variant="contiguous-1024-4096",
        mode="noscatter_prototype",
        hypothesis="h3_noscatter_prototype",
        description="dense backend contiguous-head prototype for W=1024/full",
        requires_plan=True,
        plan_source_name="flopslab-4096-noscatter-contiguous-1024-4096-flash-seed0",
        bucket_policy=bucket_policy("contiguous-1024-4096"),
    ),
    DenseVariant(
        axis="noscatter",
        variant="contiguous-layer-plan",
        mode="noscatter_prototype",
        hypothesis="h3_noscatter_prototype",
        description="dense backend contiguous-head prototype from exported layer plan",
        requires_plan=True,
        plan_source_name="flopslab-4096-noscatter-contiguous-layer-plan-flash-seed0",
    ),
    DenseVariant(
        axis="noscatter",
        variant="scatter-control-matched",
        mode="noscatter_prototype",
        hypothesis="h3_noscatter_prototype",
        description="dense backend matched scatter control for no-scatter variants",
        requires_plan=True,
        plan_source_name="flopslab-4096-noscatter-scatter-control-matched-flash-seed0",
        bucket_policy=bucket_policy("scatter-control-matched"),
    ),
    DenseVariant(
        axis="fixed",
        variant="per-layer",
        mode="fixed_plan",
        hypothesis="h4_fixed_plan_granularity",
        description="dense backend with one fixed window per layer",
        requires_plan=True,
        plan_source_name="flopslab-4096-fixed-per-layer-flash-seed0",
    ),
    DenseVariant(
        axis="fixed",
        variant="per-state",
        mode="fixed_plan",
        hypothesis="h4_fixed_plan_granularity",
        description="dense backend with one fixed plan per cheap state bucket",
        requires_plan=True,
        plan_source_name="flopslab-4096-fixed-per-state-flash-seed0",
    ),
    DenseVariant(
        axis="fixed",
        variant="per-head-group",
        mode="fixed_plan",
        hypothesis="h4_fixed_plan_granularity",
        description="dense backend with one fixed window per layer/group",
        requires_plan=True,
        plan_source_name="flopslab-4096-fixed-per-head-group-flash-seed0",
    ),
    DenseVariant(
        axis="fixed",
        variant="per-head",
        mode="fixed_plan",
        hypothesis="h4_fixed_plan_granularity",
        description="dense backend with one fixed window per layer/head",
        requires_plan=True,
        plan_source_name="flopslab-4096-fixed-per-head-flash-seed0",
    ),
    DenseVariant(
        axis="slow-update",
        variant="N200",
        mode="fixed_plan",
        hypothesis="h4_fixed_plan_granularity",
        description="dense backend with plan recompute interval metadata N=200",
        requires_plan=True,
        plan_source_name="flopslab-4096-slow-update-N200-flash-seed0",
    ),
    DenseVariant(
        axis="slow-update",
        variant="N1000",
        mode="fixed_plan",
        hypothesis="h4_fixed_plan_granularity",
        description="dense backend with plan recompute interval metadata N=1000",
        requires_plan=True,
        plan_source_name="flopslab-4096-slow-update-N1000-flash-seed0",
    ),
    DenseVariant(
        axis="minruntime",
        variant="noscatter-1024-4096",
        mode="noscatter_prototype",
        hypothesis="p3_minimal_runtime",
        description="dense backend best no-scatter path with profile-time diagnostics disabled",
        requires_plan=True,
        plan_source_name="flopslab-4096-noscatter-contiguous-1024-4096-flash-seed0",
        bucket_policy=bucket_policy("minruntime-noscatter-1024-4096"),
        minimal_runtime=True,
    ),
    DenseVariant(
        axis="headreorder",
        variant="lowerbound-1024-4096",
        mode="noscatter_prototype",
        hypothesis="h5_head_reorder_candidate",
        description="dense backend lower-bound assuming heads are already physically bucket-ordered",
        requires_plan=True,
        plan_source_name="flopslab-4096-noscatter-contiguous-1024-4096-flash-seed0",
        bucket_policy=bucket_policy("lowerbound-1024-4096"),
        minimal_runtime=True,
        assume_preordered_heads=True,
    ),
)


def clone_yaml(obj: dict) -> dict:
    return yaml.safe_load(yaml.safe_dump(obj))


def tune_train(cfg: dict, max_steps: int, checkpoint_steps: list[int], use_wandb: bool) -> None:
    train = cfg.setdefault("train", {})
    train["batch_size"] = 1
    train["precision"] = "bf16"
    train["max_steps"] = int(max_steps)
    train["checkpoint_steps"] = [int(s) for s in checkpoint_steps if int(s) <= int(max_steps)]
    train["eval_batches"] = 20
    train["eval_interval"] = 200
    train["use_wandb"] = bool(use_wandb)
    train["save_checkpoints"] = False


def make_dense_baseline(template: dict, max_steps: int, checkpoint_steps: list[int]) -> dict:
    cfg = clone_yaml(template)
    cfg["experiment"]["name"] = "flopslab-4096-baseline-pure-dense-seed0"
    cfg["experiment"]["variant"] = "baseline_pure_dense"
    cfg["experiment"]["out_dir"] = P7_RESULT_DIR
    model = cfg.setdefault("model", {})
    model["attention_backend"] = "dense_masked"
    model["flex_block_size"] = 128
    tune_train(cfg, max_steps, checkpoint_steps, use_wandb=False)
    cfg["profiling"] = {
        "ncu_gpu_flops_ratio_required": True,
        "profile_regions": ["total", "attention"],
        "notes": "Dense-framework denominator: standard dense MHA full attention.",
    }
    cfg["lab"] = {
        "enabled": False,
        "hypothesis": "p7_dense_framework",
        "axis": "baseline",
        "variant": "pure-dense",
        "mode": "dense_denominator",
        "description": "Standard dense MHA denominator for dense framework profiles.",
        "primary_metric": "dense_gpu_flops_total_ratio_ncu",
    }
    return cfg


def make_dense_variant(template: dict, variant: DenseVariant, max_steps: int, checkpoint_steps: list[int]) -> dict:
    cfg = clone_yaml(template)
    cfg["experiment"]["name"] = variant.name
    cfg["experiment"]["variant"] = f"{variant.axis}_{variant.variant}_dense".replace("-", "_")
    cfg["experiment"]["out_dir"] = P7_RESULT_DIR
    model = cfg.setdefault("model", {})
    model["aah_v3_attention_backend"] = "dense_masked"
    model["aah_v3_flex_block_size"] = 128
    if variant.same_codepath_full:
        model["aah_v3_grouping_enabled"] = False
        model["aah_v3_control_enabled"] = False
        model["aah_v3_build_hierarchy"] = False
        model["aah_v3_apply_window_control"] = False
        model["aah_v3_resolution_ema_alpha"] = 0.0
        model["aah_flopslab_enabled"] = False
    else:
        plan_path = f"{FLASH_PLAN_ROOT}/{variant.plan_source_name}.json" if variant.requires_plan else ""
        policy = dict(variant.bucket_policy or {"kind": "from_plan"})
        windows = list(policy.get("windows", []))
        model["aah_flopslab_enabled"] = True
        model["aah_flopslab_mode"] = variant.mode
        model["aah_flopslab_variant"] = variant.variant
        model["aah_flopslab_plan_path"] = plan_path
        model["aah_flopslab_bucket_policy_kind"] = str(policy.get("kind", ""))
        model["aah_flopslab_bucket_windows"] = windows
        model["aah_flopslab_bucket_threshold"] = int(policy.get("threshold", 0) or 0)
        model["aah_flopslab_minimal_runtime"] = bool(variant.minimal_runtime)
        if variant.assume_preordered_heads:
            model["aah_flopslab_assume_preordered_heads"] = True
    tune_train(cfg, max_steps, checkpoint_steps, use_wandb=False)
    cfg["profiling"] = {
        "ncu_gpu_flops_ratio_required": True,
        "baseline_config": f"{P7_CONFIG_DIR}/flopslab-4096-baseline-pure-dense-seed0.yaml",
        "profile_regions": ["total", "attention"],
        "notes": (
            "Dense-framework profile. Divide Nsight GPU FP ops by the matched "
            "standard dense MHA denominator, not by pure FlashAttention."
        ),
    }
    policy = dict(variant.bucket_policy or {"kind": "from_plan"})
    cfg["lab"] = {
        "enabled": not variant.same_codepath_full,
        "hypothesis": variant.hypothesis,
        "axis": variant.axis,
        "variant": variant.variant,
        "mode": variant.mode,
        "description": variant.description,
        "requires_plan": variant.requires_plan,
        "plan_path": f"{FLASH_PLAN_ROOT}/{variant.plan_source_name}.json" if variant.requires_plan else "",
        "bucket_policy": policy,
        "baseline_config": f"{P7_CONFIG_DIR}/flopslab-4096-baseline-pure-dense-seed0.yaml",
        "primary_metric": "dense_gpu_flops_total_ratio_ncu",
        "success_threshold": 1.0,
        "minimal_runtime": bool(variant.minimal_runtime),
        "assume_preordered_heads": bool(variant.assume_preordered_heads),
        "dense_framework": True,
    }
    return cfg


def parse_steps(raw: str) -> list[int]:
    steps = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not steps:
        raise ValueError("--checkpoint-steps must contain at least one integer")
    return steps


def write_yaml(path: Path, cfg: dict, dry_run: bool) -> None:
    print(path)
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-template", default=DENSE_BASELINE_TEMPLATE)
    parser.add_argument("--aah-template", default=AAH_TEMPLATE)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--checkpoint-steps", default="1000,2000,3000")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    checkpoint_steps = parse_steps(args.checkpoint_steps)
    with open(args.dense_template) as f:
        dense_template = yaml.safe_load(f)
    with open(args.aah_template) as f:
        aah_template = yaml.safe_load(f)

    baseline = make_dense_baseline(dense_template, args.max_steps, checkpoint_steps)
    write_yaml(Path(P7_CONFIG_DIR) / "flopslab-4096-baseline-pure-dense-seed0.yaml", baseline, args.dry_run)
    for variant in DENSE_VARIANTS:
        cfg = make_dense_variant(aah_template, variant, args.max_steps, checkpoint_steps)
        write_yaml(Path(P7_CONFIG_DIR) / variant.yaml_name, cfg, args.dry_run)

    if not args.dry_run:
        meta = {
            "dense_framework": True,
            "max_steps": int(args.max_steps),
            "checkpoint_steps": [s for s in checkpoint_steps if s <= int(args.max_steps)],
            "variants": [variant.name for variant in DENSE_VARIANTS],
        }
        Path(P7_CONFIG_DIR).mkdir(parents=True, exist_ok=True)
        with open(Path(P7_CONFIG_DIR) / "dense_framework_config_manifest.json", "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
