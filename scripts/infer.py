#!/usr/bin/env python3
import argparse
import math
import os
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


def get_device(device_pref):
    if device_pref == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_pref


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
        aah_v3_W_min_gpu=model_cfg.get("aah_v3_W_min_gpu", 64),
        aah_v3_mask_cache_size=model_cfg.get("aah_v3_mask_cache_size", 16),
    )
    return GPT(gpt_cfg).to(device)


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
    t0 = time.time()
    with torch.no_grad(), autocast_ctx:
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
            total_tokens += x.numel()
    elapsed = time.time() - t0
    for block in model.blocks:
        if hasattr(block.attn, "set_eval_mode"):
            block.attn.set_eval_mode(False)
    avg_loss = sum(losses) / max(1, len(losses))
    ppl = math.exp(avg_loss) if losses else float("inf")
    tok_s = total_tokens / max(1e-9, elapsed)
    return avg_loss, ppl, tok_s, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--eval-batches", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp = cfg["experiment"]
    train = cfg["train"]
    data = cfg["data"]

    device = get_device(train.get("device", "auto"))
    precision = train.get("precision", "fp32").lower()
    use_bf16 = precision == "bf16" and device in ("cuda", "cpu")
    if use_bf16 and device == "cuda" and not torch.cuda.is_bf16_supported():
        use_bf16 = False

    _, val_loader, vocab_size = build_dataloaders(
        data["dataset"], data["tokenizer"], data["seq_len"], train["batch_size"], data["num_workers"]
    )
    model = build_model(cfg, vocab_size, device)

    ckpt = args.checkpoint or os.path.join(exp.get("out_dir", "experiments"), f"{exp['name']}.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=True)

    eval_batches = args.eval_batches if args.eval_batches is not None else train.get("eval_batches", 50)
    val_loss, val_ppl, tok_s, elapsed = evaluate(model, val_loader, device, eval_batches, use_bf16=use_bf16)
    print(f"config={args.config}")
    print(f"checkpoint={ckpt}")
    print(f"device={device} precision={precision}")
    print(f"eval_batches={eval_batches} val_loss={val_loss:.6f} val_ppl={val_ppl:.4f} tok_s={tok_s:.2f} elapsed_s={elapsed:.2f}")


if __name__ == "__main__":
    main()
