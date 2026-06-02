"""Names and variant contracts for the AAH FLOPs reduction lab."""

from __future__ import annotations

from dataclasses import dataclass


SEQ_LEN = 4096
SEED = 0
BACKEND = "flash"
PURE_FLASH_BASELINE = (
    "experiments/backend_realized_local_attention/"
    "FlashAttention/pure/configs/backend_4096_pure_flash_seed0.yaml"
)
AAH_FLASH_TEMPLATE = (
    "experiments/backend_realized_local_attention/"
    "FlashAttention/aah_modified/configs/backend_4096_full_adaptive_flash_seed0.yaml"
)
RUN_ROOT = "paper_results/aah_flops_reduction_lab"


@dataclass(frozen=True)
class LabVariant:
    hypothesis: str
    axis: str
    variant: str
    backend: str = BACKEND
    module: str = "aah"
    description: str = ""
    requires_plan: bool = False
    compares_to: str = PURE_FLASH_BASELINE

    @property
    def name(self) -> str:
        return f"flopslab-{SEQ_LEN}-{self.axis}-{self.variant}-{self.backend}-seed{SEED}"

    @property
    def yaml_name(self) -> str:
        return f"{self.name}.yaml"


VARIANTS = (
    LabVariant(
        hypothesis="h1_static_compiled_plan",
        axis="static-plan",
        variant="per-layer",
        description="one exported window vector per layer",
        requires_plan=True,
    ),
    LabVariant(
        hypothesis="h1_static_compiled_plan",
        axis="static-plan",
        variant="per-layer-head",
        description="one exported window per layer/head",
        requires_plan=True,
    ),
    LabVariant(
        hypothesis="h1_static_compiled_plan",
        axis="static-plan",
        variant="majority",
        description="majority window from calibration batches",
        requires_plan=True,
    ),
    LabVariant(
        hypothesis="h2_quantized_execution",
        axis="quantized",
        variant="single-1024",
        description="collapse all local execution to W=1024",
    ),
    LabVariant(
        hypothesis="h2_quantized_execution",
        axis="quantized",
        variant="single-2048",
        description="collapse all local execution to W=2048",
    ),
    LabVariant(
        hypothesis="h2_quantized_execution",
        axis="quantized",
        variant="two-bucket-1024-4096",
        description="short bucket W=1024 and full bucket W=4096",
    ),
    LabVariant(
        hypothesis="h2_quantized_execution",
        axis="quantized",
        variant="two-bucket-2048-4096",
        description="short bucket W=2048 and full bucket W=4096",
    ),
    LabVariant(
        hypothesis="h3_noscatter_prototype",
        axis="noscatter",
        variant="contiguous-1024-4096",
        description="contiguous head blocks for W=1024/full",
        requires_plan=True,
    ),
    LabVariant(
        hypothesis="h3_noscatter_prototype",
        axis="noscatter",
        variant="contiguous-layer-plan",
        description="contiguous head blocks from exported layer plan",
        requires_plan=True,
    ),
    LabVariant(
        hypothesis="h3_noscatter_prototype",
        axis="noscatter",
        variant="scatter-control-matched",
        description="matched scatter control for no-scatter variants",
        requires_plan=True,
    ),
    LabVariant(
        hypothesis="h4_fixed_plan_granularity",
        axis="fixed",
        variant="per-layer",
        description="one fixed window per layer",
        requires_plan=True,
    ),
    LabVariant(
        hypothesis="h4_fixed_plan_granularity",
        axis="fixed",
        variant="per-state",
        description="one fixed plan per cheap state bucket",
        requires_plan=True,
    ),
    LabVariant(
        hypothesis="h4_fixed_plan_granularity",
        axis="fixed",
        variant="per-head-group",
        description="one fixed window per layer/group",
        requires_plan=True,
    ),
    LabVariant(
        hypothesis="h4_fixed_plan_granularity",
        axis="fixed",
        variant="per-head",
        description="one fixed window per layer/head",
        requires_plan=True,
    ),
    LabVariant(
        hypothesis="h4_fixed_plan_granularity",
        axis="slow-update",
        variant="N200",
        description="recompute plan every 200 steps/batches",
        requires_plan=True,
    ),
    LabVariant(
        hypothesis="h4_fixed_plan_granularity",
        axis="slow-update",
        variant="N1000",
        description="recompute plan every 1000 steps/batches",
        requires_plan=True,
    ),
)


def hypothesis_dir(variant: LabVariant) -> str:
    return f"experiments/aah_flops_reduction_lab/{variant.hypothesis}"
