"""Create a dense-framework Nsight profile manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.aah_flops_reduction_lab.p7_dense_framework.scripts.make_dense_framework_configs import (
    DENSE_VARIANTS,
    P7_CONFIG_DIR,
    RUN_ROOT,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncu", default="/home/featurize/work/bin/ncu-sudo")
    parser.add_argument("--output", default=f"{RUN_ROOT}/profile_manifest_p7_dense_framework.jsonl")
    parser.add_argument("--profile-timeout", type=int, default=7200)
    parser.add_argument("--include-attention", action="store_true")
    args = parser.parse_args()

    profile_dir = f"{RUN_ROOT}/gpu_flops_profiles"
    baseline_json = f"{profile_dir}/flopslab-4096-baseline-pure-dense-seed0_gpu_flops_profile.json"
    baseline_cfg = f"{P7_CONFIG_DIR}/flopslab-4096-baseline-pure-dense-seed0.yaml"

    rows = [
        {
            "name": "flopslab-4096-baseline-pure-dense-seed0",
            "module": "pure",
            "config": baseline_cfg,
            "output": baseline_json,
            "baseline_json": "",
            "profile_scope": "total",
            "purpose": "standard dense MHA denominator",
        }
    ]

    for variant in DENSE_VARIANTS:
        rows.append(
            {
                "name": variant.name,
                "module": variant.module,
                "config": f"{P7_CONFIG_DIR}/{variant.yaml_name}",
                "output": f"{profile_dir}/{variant.name}_gpu_flops_profile.json",
                "baseline_json": baseline_json,
                "profile_scope": "total",
                "purpose": variant.description,
                "dense_framework": True,
            }
        )

    if args.include_attention:
        attention_baseline_json = (
            f"{profile_dir}/flopslab-4096-baseline-pure-dense-seed0_attention_gpu_flops_profile.json"
        )
        rows.append(
            {
                "name": "flopslab-4096-baseline-pure-dense-seed0_attention",
                "module": "pure",
                "config": baseline_cfg,
                "output": attention_baseline_json,
                "baseline_json": "",
                "profile_scope": "attention",
                "purpose": "standard dense MHA attention-scope denominator",
            }
        )
        for variant in DENSE_VARIANTS:
            rows.append(
                {
                    "name": f"{variant.name}_attention",
                    "module": variant.module,
                    "config": f"{P7_CONFIG_DIR}/{variant.yaml_name}",
                    "output": f"{profile_dir}/{variant.name}_attention_gpu_flops_profile.json",
                    "baseline_json": attention_baseline_json,
                    "profile_scope": "attention",
                    "purpose": f"attention-scope: {variant.description}",
                    "dense_framework": True,
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
                "--profile-scope",
                row["profile_scope"],
                "--output",
                row["output"],
            ]
            if row["baseline_json"]:
                cmd.extend(["--baseline-json", row["baseline_json"]])
            row["command"] = cmd
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote_dense_profile_manifest {out}")


if __name__ == "__main__":
    main()
