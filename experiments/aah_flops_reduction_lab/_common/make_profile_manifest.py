"""Create a JSONL manifest of lab profiling commands.

The manifest is intentionally explicit so remote runs can be inspected before
spending GPU time. It does not execute anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.aah_flops_reduction_lab._common.naming import (
    PURE_FLASH_BASELINE,
    RUN_ROOT,
    VARIANTS,
    hypothesis_dir,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncu", default="/usr/local/cuda/bin/ncu")
    parser.add_argument("--output", default=f"{RUN_ROOT}/profile_manifest.jsonl")
    parser.add_argument("--profile-timeout", type=int, default=7200)
    args = parser.parse_args()

    rows = []
    baseline_json = f"{RUN_ROOT}/gpu_flops_profiles/flashattention_pure_gpu_flops_profile.json"
    rows.append(
        {
            "name": "flopslab-4096-baseline-pure-flash-seed0",
            "module": "pure",
            "config": PURE_FLASH_BASELINE,
            "output": baseline_json,
            "baseline_json": "",
            "purpose": "matched pure FlashAttention denominator",
        }
    )
    for variant in VARIANTS:
        cfg = Path(hypothesis_dir(variant)) / "configs" / variant.yaml_name
        rows.append(
            {
                "name": variant.name,
                "module": variant.module,
                "config": str(cfg),
                "output": f"{RUN_ROOT}/gpu_flops_profiles/{variant.name}_gpu_flops_profile.json",
                "baseline_json": baseline_json,
                "purpose": variant.description,
                "requires_plan": variant.requires_plan,
                "status": "requires_static_plan_before_profile" if variant.requires_plan else "ready",
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for row in rows:
            cmd = [
                "python",
                "-m",
                "experiments.backend_realized_local_attention._common.profile_gpu_flops_ncu",
                "--module",
                row["module"],
                "--config",
                row["config"],
                "--ncu",
                args.ncu,
                "--warmup",
                "1",
                "--repeats",
                "1",
                "--timeout",
                str(args.profile_timeout),
                "--output",
                row["output"],
            ]
            if row["baseline_json"]:
                cmd.extend(["--baseline-json", row["baseline_json"]])
            row["command"] = cmd
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote_profile_manifest {out}")


if __name__ == "__main__":
    main()
