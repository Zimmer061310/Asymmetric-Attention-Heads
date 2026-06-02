"""Export static AAH execution plans from an AAH checkpoint.

The exporter intentionally observes the existing AAH implementation instead of
changing it. It runs calibration forwards, reads each block's `last_stats`, and
writes per-layer window/group statistics that later lab prototypes can compile
into cheaper execution paths.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from experiments.backend_realized_local_attention._common.profile_flops_ratio import (
    autocast_context,
    get_device,
    load_config,
    load_model,
    sync,
)


def _mode(values):
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def collect_plan(args) -> dict:
    cfg = load_config(args.config)
    train = cfg.get("train", {})
    device = get_device(args.device or train.get("device", "cuda"))
    precision = train.get("precision", "bf16")
    model, gpt_cfg, checkpoint_loaded, checkpoint_path = load_model(cfg, "aah", device, args.checkpoint)
    batch_size = int(train.get("batch_size", 1))
    seq_len = int(cfg["data"]["seq_len"])
    torch.manual_seed(int(args.seed))

    per_layer_windows = defaultdict(list)
    per_layer_groups = defaultdict(list)
    per_layer_branch_freq = defaultdict(list)
    per_layer_path_modes = defaultdict(list)

    with torch.no_grad():
        for batch_idx in range(int(args.calibration_batches)):
            if hasattr(model, "set_step"):
                model.set_step(int(args.start_step) + batch_idx * int(args.step_stride))
            idx = torch.randint(0, int(gpt_cfg.vocab_size), (batch_size, seq_len), device=device)
            with autocast_context(device, precision):
                model(idx)
            sync(device)
            for layer_idx, block in enumerate(getattr(model, "blocks", [])):
                stats = getattr(getattr(block, "attn", None), "last_stats", {}) or {}
                windows = [int(x) for x in stats.get("resolution_per_head", [])]
                groups = [int(x) for x in stats.get("head_groups", [])]
                if windows:
                    per_layer_windows[layer_idx].append(windows)
                if groups:
                    per_layer_groups[layer_idx].append(groups)
                per_layer_branch_freq[layer_idx].append(stats.get("branch_usage_freq", {}))
                per_layer_path_modes[layer_idx].append(str(stats.get("path_mode", "")))

    layers = []
    for layer_idx in range(int(cfg["model"]["n_layer"])):
        window_samples = per_layer_windows.get(layer_idx, [])
        group_samples = per_layer_groups.get(layer_idx, [])
        n_head = int(cfg["model"]["n_head"])
        majority_by_head = []
        for head_idx in range(n_head):
            majority_by_head.append(_mode([sample[head_idx] for sample in window_samples if len(sample) > head_idx]))
        layer_windows = [w for sample in window_samples for w in sample]
        layers.append(
            {
                "layer_idx": layer_idx,
                "majority_window_per_head": majority_by_head,
                "majority_window_layer": _mode(layer_windows),
                "majority_group_per_head": [
                    _mode([sample[head_idx] for sample in group_samples if len(sample) > head_idx])
                    for head_idx in range(n_head)
                ],
                "window_samples": window_samples if args.include_samples else [],
                "group_samples": group_samples if args.include_samples else [],
                "path_modes": sorted(set(per_layer_path_modes.get(layer_idx, []))),
                "branch_usage_freq_samples": per_layer_branch_freq.get(layer_idx, []) if args.include_samples else [],
            }
        )

    return {
        "schema": "aah_flops_reduction_lab.static_plan.v1",
        "config": args.config,
        "checkpoint_loaded": checkpoint_loaded,
        "checkpoint_path": checkpoint_path,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "precision": precision,
        "seed": int(args.seed),
        "calibration_batches": int(args.calibration_batches),
        "start_step": int(args.start_step),
        "step_stride": int(args.step_stride),
        "windows": [int(x) for x in cfg["model"].get("aah_v3_windows", [])],
        "layers": layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-batches", type=int, default=16)
    parser.add_argument("--start-step", type=int, default=3000)
    parser.add_argument("--step-stride", type=int, default=5)
    parser.add_argument("--include-samples", action="store_true")
    args = parser.parse_args()

    payload = collect_plan(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"wrote_static_plan {out}")


if __name__ == "__main__":
    main()
