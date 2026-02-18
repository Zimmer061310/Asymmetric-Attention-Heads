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
        aah_v3_resolution_ema_alpha=model_cfg.get("aah_v3_resolution_ema_alpha", 0.0),
        aah_v3_resolution_collapse_min_frac=model_cfg.get("aah_v3_resolution_collapse_min_frac", 0.95),
        aah_v3_resolution_collapse_max_frac=model_cfg.get("aah_v3_resolution_collapse_max_frac", 0.95),
        aah_v3_post_warmup_ramp_steps=model_cfg.get("aah_v3_post_warmup_ramp_steps", 0),
    )
    return GPT(gpt_cfg).to(device)
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


def evaluate(model, loader, device, max_batches, use_bf16=False, use_wandb=False, wandb_run=None, log_interval=50):
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
    total_attn_flops = 0.0
    total_flops = 0.0
    total_full_flops = 0.0
    n = 0
    t0 = time.time()
    resolution_per_head_sum = None
    resolution_per_head_count = 0
    branch_usage_sum = {}
    branch_usage_count = 0
    with torch.no_grad(), autocast_ctx:
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
            total_tokens += x.numel()
            attn_elems = None
            if getattr(model.config, "aah_v2_enabled", False) or getattr(model.config, "aah_v3_enabled", False):
                te = 0.0
                be = 0.0
                for block in model.blocks:
                    attn = block.attn
                    if hasattr(attn, "last_stats"):
                        te += attn.last_stats.get("total_elements", 0.0)
                        be += attn.last_stats.get("baseline_elements", 0.0)
                if be > 0:
                    attn_elems = te
            fa, ft, fr, _ = estimate_flops(
                {
                    "n_layer": model.config.n_layer,
                    "n_head": model.config.n_head,
                    "n_embd": model.config.n_embd,
                    "n_ff": model.config.n_ff,
                },
                int(x.size(0)),
                int(x.size(1)),
                attn_elements_total=attn_elems,
            )
            _, ff_total, _, _ = estimate_flops(
                {
                    "n_layer": model.config.n_layer,
                    "n_head": model.config.n_head,
                    "n_embd": model.config.n_embd,
                    "n_ff": model.config.n_ff,
                },
                int(x.size(0)),
                int(x.size(1)),
                attn_elements_total=None,
            )
            total_attn_flops += fa
            total_flops += ft
            total_full_flops += ff_total
            n += 1
            if getattr(model.config, "aah_v2_enabled", False) or getattr(model.config, "aah_v3_enabled", False):
                for block in model.blocks:
                    attn = block.attn
                    if not hasattr(attn, "last_stats"):
                        continue
                    lk = attn.last_stats.get("lk", [])
                    if lk:
                        lk_t = torch.tensor(lk, dtype=torch.float32)
                        if resolution_per_head_sum is None:
                            resolution_per_head_sum = torch.zeros_like(lk_t)
                        if resolution_per_head_sum.shape == lk_t.shape:
                            resolution_per_head_sum += lk_t
                            resolution_per_head_count += 1
                    branch_usage = attn.last_stats.get("branch_usage_freq", {})
                    if branch_usage:
                        for k, v in branch_usage.items():
                            ks = str(k)
                            branch_usage_sum[ks] = branch_usage_sum.get(ks, 0.0) + float(v)
                        branch_usage_count += 1
    elapsed = time.time() - t0
    for block in model.blocks:
        if hasattr(block.attn, "set_eval_mode"):
            block.attn.set_eval_mode(False)
    avg_loss = sum(losses) / max(1, len(losses))
    ppl = math.exp(avg_loss) if losses else float("inf")
    tok_s = total_tokens / max(1e-9, elapsed)
    flops_attn_est = total_attn_flops / max(1, n)
    flops_total_est = total_flops / max(1, n)
    flops_ratio = (total_flops / total_full_flops) if total_full_flops > 0 else 1.0
    flops_reduction_pct = (1.0 - flops_ratio) * 100.0
    resolution_per_head_mean = None
    if resolution_per_head_sum is not None and resolution_per_head_count > 0:
        resolution_per_head_mean = (resolution_per_head_sum / float(resolution_per_head_count)).tolist()
    branch_usage_mean = {}
    if branch_usage_count > 0:
        for k, v in branch_usage_sum.items():
            branch_usage_mean[k] = v / float(branch_usage_count)
    return (
        avg_loss,
        ppl,
        tok_s,
        elapsed,
        flops_attn_est,
        flops_total_est,
        flops_ratio,
        flops_reduction_pct,
        resolution_per_head_mean,
        branch_usage_mean,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp = cfg["experiment"]
    train = cfg["train"]
    data = cfg["data"]
    use_wandb = train.get("use_wandb", False)

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

    run = None
    if use_wandb:
        try:
            import wandb
            run = wandb.init(
                project="ENA-AAH",
                name=f"{exp['name']}-infer",
                config=cfg,
                job_type="inference",
                reinit=True,
            )
        except Exception as exc:
            print(f"wandb init failed: {exc}")
            run = None

    eval_batches = args.eval_batches if args.eval_batches is not None else train.get("eval_batches", 50)
    (
        val_loss,
        val_ppl,
        tok_s,
        elapsed,
        flops_attn_est,
        flops_total_est,
        flops_ratio,
        flops_reduction_pct,
        resolution_per_head_mean,
        branch_usage_mean,
    ) = evaluate(
        model,
        val_loader,
        device,
        eval_batches,
        use_bf16=use_bf16,
        use_wandb=(run is not None),
        wandb_run=run,
        log_interval=args.log_interval,
    )
    print(f"config={args.config}")
    print(f"checkpoint={ckpt}")
    print(f"device={device} precision={precision}")
    print(f"eval_batches={eval_batches} val_loss={val_loss:.6f} val_ppl={val_ppl:.4f} tok_s={tok_s:.2f} elapsed_s={elapsed:.2f}")
    print(
        f"aah/flops_attn_est={flops_attn_est:.2f} aah/flops_total_est={flops_total_est:.2f} "
        f"aah/flops_ratio={flops_ratio:.6f} aah/flops_reduction_%={flops_reduction_pct:.4f}"
    )
    if resolution_per_head_mean is not None:
        print(f"aah/resolution_per_head_mean={resolution_per_head_mean}")
    if branch_usage_mean:
        print(f"aah/branch_usage_freq={branch_usage_mean}")
    if run is not None:
        try:
            run.log(
                {
                    "infer/val_loss": val_loss,
                    "infer/val_ppl": val_ppl,
                    "infer/tok_s": tok_s,
                    "infer/elapsed_s": elapsed,
                    "infer/eval_batches": eval_batches,
                    "infer/config": args.config,
                    "infer/checkpoint": ckpt,
                    "aah/flops_attn_est": flops_attn_est,
                    "aah/flops_total_est": flops_total_est,
                    "aah/flops_ratio": flops_ratio,
                    "aah/flops_reduction_%": flops_reduction_pct,
                    "aah/resolution_per_head_mean": resolution_per_head_mean,
                    "aah/branch_usage_freq": branch_usage_mean,
                }
            )
            run.finish()
        except Exception as exc:
            print(f"wandb logging failed: {exc}")


if __name__ == "__main__":
    main()
