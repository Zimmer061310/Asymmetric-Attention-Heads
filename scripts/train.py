import os
import time
import math
import csv
import sys
import argparse
import yaml
import platform
import resource
from contextlib import nullcontext
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
try:
    import psutil
except Exception:
    psutil = None

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


def linear_warmup_cosine(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def evaluate(model, loader, device, max_batches, log_progress=False, use_bf16=False):
    model.eval()
    for block in model.blocks:
        attn = block.attn
        if hasattr(attn, "set_eval_mode"):
            attn.set_eval_mode(True)
    losses = []
    t0 = time.time()
    autocast_ctx = nullcontext()
    if use_bf16:
        if device == "cuda":
            autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
        elif device == "cpu":
            autocast_ctx = torch.autocast("cpu", dtype=torch.bfloat16)
    with torch.no_grad(), autocast_ctx:
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            if log_progress and i % 5 == 0:
                print(f"eval batch {i}/{max_batches}")
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    for block in model.blocks:
        attn = block.attn
        if hasattr(attn, "set_eval_mode"):
            attn.set_eval_mode(False)
    if not losses:
        return float("inf"), float("inf"), 0.0
    avg_loss = sum(losses) / len(losses)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl, time.time() - t0


def get_memory_stats():
    gpu_alloc = None
    gpu_reserved = None
    gpu_alloc_max = None
    gpu_reserved_max = None
    if torch.cuda.is_available():
        gpu_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        gpu_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        gpu_alloc_max = torch.cuda.max_memory_allocated() / (1024 ** 2)
        gpu_reserved_max = torch.cuda.max_memory_reserved() / (1024 ** 2)
    cpu_rss = None
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            cpu_rss = rss / (1024 ** 2)
        else:
            cpu_rss = rss / 1024
    except Exception:
        pass
    return gpu_alloc, gpu_reserved, gpu_alloc_max, gpu_reserved_max, cpu_rss


def get_psutil_memory_mb():
    if psutil is None:
        return {}
    try:
        proc = psutil.Process(os.getpid())
        info = proc.memory_info()
        full = proc.memory_full_info()
        vm = psutil.virtual_memory()
        def mb(x):
            return x / (1024 ** 2) if x is not None else None
        return {
            "psutil_rss_mb": mb(getattr(info, "rss", None)),
            "psutil_vms_mb": mb(getattr(info, "vms", None)),
            "psutil_shared_mb": mb(getattr(info, "shared", None)),
            "psutil_text_mb": mb(getattr(info, "text", None)),
            "psutil_data_mb": mb(getattr(info, "data", None)),
            "psutil_uss_mb": mb(getattr(full, "uss", None)),
            "psutil_pss_mb": mb(getattr(full, "pss", None)),
            "psutil_swap_mb": mb(getattr(full, "swap", None)),
            "psutil_ram_used_mb": mb(getattr(vm, "used", None)),
            "psutil_ram_total_mb": mb(getattr(vm, "total", None)),
        }
    except Exception:
        return {}
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/small.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    exp = cfg["experiment"]
    data = cfg["data"]
    model_cfg = cfg["model"]
    train = cfg["train"]

    torch.manual_seed(exp["seed"])

    device = get_device(train["device"])
    precision = train.get("precision", "fp32").lower()
    use_bf16 = precision == "bf16"
    if use_bf16:
        if device == "cuda" and not torch.cuda.is_bf16_supported():
            print("Warning: BF16 requested but not supported on this CUDA device. Falling back to FP32.")
            use_bf16 = False
        if device == "mps":
            print("Warning: BF16 autocast not supported on MPS. Falling back to FP32.")
            use_bf16 = False

    train_loader, val_loader, vocab_size = build_dataloaders(
        data["dataset"],
        data["tokenizer"],
        data["seq_len"],
        train["batch_size"],
        data["num_workers"],
    )

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
    )
    model = GPT(gpt_cfg).to(device)

    opt = AdamW(model.parameters(), lr=train["lr"], weight_decay=train["weight_decay"])
    scheduler = LambdaLR(opt, lambda s: linear_warmup_cosine(s, train["warmup_steps"], train["max_steps"]))

    use_wandb = train.get("use_wandb", False)
    log_csv = train.get("log_csv", False)
    effective_log_interval = 50
    cfg_log_interval = int(train.get("log_interval", 50))
    if cfg_log_interval != effective_log_interval:
        print(f"Info: overriding log_interval {cfg_log_interval} -> {effective_log_interval} for unified 50-step logging.")
    out_dir = exp.get("out_dir", "experiments")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{exp['name']}_{exp.get('variant','run')}.csv")
    csv_file = None
    csv_writer = None
    if use_wandb:
        try:
            import wandb
            wandb.init(project="ENA-AAH", name=exp["name"], config=cfg)
        except Exception:
            use_wandb = False
    if log_csv:
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "step",
            "train_loss",
            "tok_s",
            "mem_mb",
            "gpu_alloc_mb",
            "gpu_reserved_mb",
            "gpu_alloc_max_mb",
            "gpu_reserved_max_mb",
            "cpu_rss_mb",
            "psutil_rss_mb",
            "psutil_vms_mb",
            "psutil_shared_mb",
            "psutil_text_mb",
            "psutil_data_mb",
            "psutil_uss_mb",
            "psutil_pss_mb",
            "psutil_swap_mb",
            "psutil_ram_used_mb",
            "psutil_ram_total_mb",
            "val_loss",
            "val_ppl",
            "attn_elems",
            "attn_ratio",
            "attn_reduction",
            "flops_attn_est",
            "flops_total_est",
            "flops_ratio",
            "flops_reduction_pct",
            "attn_lq",
            "attn_lk_per_layer",
            "head_entropy",
            "group_change_rate",
            "avg_window",
            "group_overlap",
            "head_reassign_rate",
            "group_lifespan_ema",
            "head_groups",
            "shadow_win_idx",
            "shadow_logit_mean",
            "group_heads",
            "group_ratios",
            "lk_mean",
            "lk_p90",
            "w_mean",
            "w_min",
            "w_max",
            "control_time_ms",
            "attn_time_ms",
            "mask_time_ms",
            "overhead_time_ms",
            "step_time_ms",
            "eval_time_s",
        ])

    step = 0
    prev_head_groups = None
    lifespan_ema = None
    model.train()
    t0 = time.time()
    while step < train["max_steps"]:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if step == 0 and device == "cuda":
                try:
                    print(f"autocast gpu dtype: {torch.get_autocast_gpu_dtype()}")
                except Exception as exc:
                    print(f"autocast gpu dtype: unavailable ({exc})")
            aah_v2_enabled = model_cfg.get("aah_v2_enabled", False)
            aah_v3_enabled = model_cfg.get("aah_v3_enabled", False)
            warmup_steps = model_cfg.get("aah_v2_warmup_steps", 0)
            activation_steps = model_cfg.get("aah_v2_activation_steps", 0)
            compute_lambda = model_cfg.get("aah_v2_compute_lambda", 0.0)
            min_head_norm = model_cfg.get("aah_v2_min_head_norm", 0.0)
            norm_lambda = model_cfg.get("aah_v2_norm_lambda", 0.0)
            min_head_entropy = model_cfg.get("aah_v2_min_head_entropy", 0.0)
            ent_lambda = model_cfg.get("aah_v2_entropy_lambda", 0.0)
            if aah_v2_enabled:
                if step < warmup_steps:
                    control_enabled = False
                    lambda_now = 0.0
                    guardrails_enabled = False
                elif step < (warmup_steps + activation_steps):
                    control_enabled = True
                    lambda_now = 0.0
                    guardrails_enabled = True
                else:
                    control_enabled = True
                    lambda_now = compute_lambda
                    guardrails_enabled = False
                if step == warmup_steps:
                    for block in model.blocks:
                        attn = block.attn
                        if hasattr(attn, "reset_cache"):
                            attn.reset_cache()
                for block in model.blocks:
                    attn = block.attn
                    if hasattr(attn, "set_control"):
                        attn.set_control(control_enabled)
                    if hasattr(attn, "set_step"):
                        attn.set_step(step)
            if aah_v3_enabled:
                for block in model.blocks:
                    attn = block.attn
                    if hasattr(attn, "set_control"):
                        attn.set_control(model_cfg.get("aah_v3_control_enabled", True))
                    if hasattr(attn, "set_step"):
                        attn.set_step(step)
            step_t0 = time.time()
            autocast_ctx = nullcontext()
            if use_bf16:
                if device == "cuda":
                    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
                elif device == "cpu":
                    autocast_ctx = torch.autocast("cpu", dtype=torch.bfloat16)
            with autocast_ctx:
                logits, loss = model(x, y)
            if model_cfg.get("aah_v2_enabled", False):
                if lambda_now > 0:
                    total_elements = 0.0
                    baseline_elements = 0.0
                    lq = None
                    lk_layers = []
                    for block in model.blocks:
                        attn = block.attn
                        if hasattr(attn, "last_stats"):
                            total_elements += attn.last_stats.get("total_elements", 0.0)
                            baseline_elements += attn.last_stats.get("baseline_elements", 0.0)
                            if lq is None:
                                lq = attn.last_stats.get("lq")
                            lk_layers.append(attn.last_stats.get("lk", []))
                    if baseline_elements > 0:
                        attn_ratio = total_elements / baseline_elements
                        loss = loss + (lambda_now * torch.tensor(attn_ratio, device=loss.device))
                if guardrails_enabled:
                    if min_head_norm > 0 and norm_lambda > 0:
                        norms = []
                        for block in model.blocks:
                            attn = block.attn
                            if hasattr(attn, "last_stats"):
                                norms.extend(attn.last_stats.get("head_norms", []))
                        if norms:
                            norms_t = torch.tensor(norms, device=loss.device)
                            shortfall = (min_head_norm - norms_t).clamp_min(0.0).mean()
                            loss = loss + (norm_lambda * shortfall)
                    if min_head_entropy > 0 and ent_lambda > 0:
                        ents = []
                        for block in model.blocks:
                            attn = block.attn
                            if hasattr(attn, "last_stats"):
                                ents.extend(attn.last_stats.get("head_entropy", []))
                        if ents:
                            ents_t = torch.tensor(ents, device=loss.device)
                            shortfall = (min_head_entropy - ents_t).clamp_min(0.0).mean()
                            loss = loss + (ent_lambda * shortfall)
                        loss = loss + (ent_lambda * shortfall)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            scheduler.step()
            step_time_ms = (time.time() - step_t0) * 1000.0

            step += 1

            if step % effective_log_interval == 0:
                elapsed = time.time() - t0
                tokens = train["batch_size"] * data["seq_len"] * effective_log_interval
                tok_per_sec = tokens / max(1e-9, elapsed)
                gpu_alloc, gpu_reserved, gpu_alloc_max, gpu_reserved_max, cpu_rss = get_memory_stats()
                ps_mem = get_psutil_memory_mb()
                if gpu_alloc is not None:
                    mem = gpu_alloc
                elif cpu_rss is not None:
                    mem = cpu_rss
                else:
                    mem = 0.0
                attn_elems = None
                attn_ratio = None
                attn_reduction = None
                flops_attn_est = None
                flops_total_est = None
                flops_ratio = None
                flops_reduction_pct = None
                lq = None
                lk_layers = []
                head_entropy = []
                group_change_rates = []
                avg_windows = []
                head_groups = []
                shadow_win_idx = []
                shadow_logit_mean = []
                group_heads = []
                group_ratios = []
                lk_means = []
                lk_p90s = []
                w_means = []
                w_mins = []
                w_maxs = []
                control_times = []
                attn_times = []
                mask_times = []
                overhead_times = []
                if model_cfg.get("aah_v2_enabled", False) or model_cfg.get("aah_v3_enabled", False):
                    total_elements = 0.0
                    baseline_elements = 0.0
                    for block in model.blocks:
                        attn = block.attn
                        if hasattr(attn, "last_stats"):
                            total_elements += attn.last_stats.get("total_elements", 0.0)
                            baseline_elements += attn.last_stats.get("baseline_elements", 0.0)
                            if lq is None:
                                lq = attn.last_stats.get("lq")
                            lk_layers.append(attn.last_stats.get("lk", []))
                            head_entropy.extend(attn.last_stats.get("head_entropy", []))
                            if "group_change_rate" in attn.last_stats:
                                group_change_rates.append(attn.last_stats.get("group_change_rate"))
                            if "avg_window" in attn.last_stats:
                                avg_windows.append(attn.last_stats.get("avg_window"))
                            if "head_groups" in attn.last_stats:
                                head_groups.append(attn.last_stats.get("head_groups"))
                            if "shadow_win_idx" in attn.last_stats:
                                shadow_win_idx.append(attn.last_stats.get("shadow_win_idx"))
                            if "shadow_logit_mean" in attn.last_stats:
                                shadow_logit_mean.append(attn.last_stats.get("shadow_logit_mean"))
                            if "group_heads" in attn.last_stats:
                                group_heads.append(attn.last_stats.get("group_heads"))
                            if "group_ratios" in attn.last_stats:
                                group_ratios.append(attn.last_stats.get("group_ratios"))
                            if "control_time_ms" in attn.last_stats:
                                control_times.append(attn.last_stats.get("control_time_ms"))
                            if "attn_time_ms" in attn.last_stats:
                                attn_times.append(attn.last_stats.get("attn_time_ms"))
                            if "mask_time_ms" in attn.last_stats:
                                mask_times.append(attn.last_stats.get("mask_time_ms"))
                            if "overhead_time_ms" in attn.last_stats:
                                overhead_times.append(attn.last_stats.get("overhead_time_ms"))
                            if "lk_mean" in attn.last_stats:
                                lk_means.append(attn.last_stats.get("lk_mean"))
                            if "lk_p90" in attn.last_stats:
                                lk_p90s.append(attn.last_stats.get("lk_p90"))
                            if "w_mean" in attn.last_stats:
                                w_means.append(attn.last_stats.get("w_mean"))
                            if "w_min" in attn.last_stats:
                                w_mins.append(attn.last_stats.get("w_min"))
                            if "w_max" in attn.last_stats:
                                w_maxs.append(attn.last_stats.get("w_max"))
                    if baseline_elements > 0:
                        attn_elems = total_elements
                        attn_ratio = total_elements / baseline_elements
                        attn_reduction = 1.0 - attn_ratio
                b_cur = int(x.size(0))
                t_cur = int(x.size(1))
                flops_attn_est, flops_total_est, flops_ratio, flops_reduction_pct = estimate_flops(
                    model_cfg,
                    b_cur,
                    t_cur,
                    attn_elements_total=attn_elems,
                )
                group_change_rates = [v for v in group_change_rates if v is not None]
                group_change_rate = sum(group_change_rates) / len(group_change_rates) if group_change_rates else None
                avg_window = sum(avg_windows) / len(avg_windows) if avg_windows else None
                lk_mean = sum(lk_means) / len(lk_means) if lk_means else None
                lk_p90 = sum(lk_p90s) / len(lk_p90s) if lk_p90s else None
                w_mean = sum(w_means) / len(w_means) if w_means else None
                w_min = min(w_mins) if w_mins else None
                w_max = max(w_maxs) if w_maxs else None
                control_time_ms = sum(control_times) / len(control_times) if control_times else None
                attn_time_ms = sum(attn_times) / len(attn_times) if attn_times else None
                mask_time_ms = sum(mask_times) / len(mask_times) if mask_times else None
                overhead_time_ms = sum(overhead_times) / len(overhead_times) if overhead_times else None
                overlap_rates = []
                reassign_rates = []
                if head_groups and prev_head_groups is not None:
                    for curr, prev in zip(head_groups, prev_head_groups):
                        if not curr or not prev:
                            continue
                        n = min(len(curr), len(prev))
                        curr = curr[:n]
                        prev = prev[:n]
                        reassign = sum(1 for i in range(n) if curr[i] != prev[i]) / max(1, n)
                        reassign_rates.append(reassign)
                        curr_pairs = {(i, j) for i in range(n) for j in range(i + 1, n) if curr[i] == curr[j]}
                        prev_pairs = {(i, j) for i in range(n) for j in range(i + 1, n) if prev[i] == prev[j]}
                        if curr_pairs or prev_pairs:
                            jaccard = len(curr_pairs & prev_pairs) / len(curr_pairs | prev_pairs)
                            overlap_rates.append(jaccard)
                avg_overlap = sum(overlap_rates) / len(overlap_rates) if overlap_rates else None
                avg_reassign = sum(reassign_rates) / len(reassign_rates) if reassign_rates else None
                if avg_reassign is not None:
                    if lifespan_ema is None:
                        lifespan_ema = 1.0 - avg_reassign
                    else:
                        lifespan_ema = 0.9 * lifespan_ema + 0.1 * (1.0 - avg_reassign)
                msg = f"step {step} | loss {loss.item():.4f} | tok/s {tok_per_sec:.1f}"
                if attn_ratio is not None:
                    msg += f" | attn_elems {attn_elems:.0f} | attn_ratio {attn_ratio:.3f}"
                print(msg)
                if use_wandb:
                    payload = {"train/loss": loss.item(), "perf/tok_s": tok_per_sec, "perf/mem_mb": mem, "step": step}
                    if gpu_alloc is not None:
                        payload["perf/gpu_alloc_mb"] = gpu_alloc
                    if gpu_reserved is not None:
                        payload["perf/gpu_reserved_mb"] = gpu_reserved
                    if gpu_alloc_max is not None:
                        payload["perf/gpu_alloc_max_mb"] = gpu_alloc_max
                    if gpu_reserved_max is not None:
                        payload["perf/gpu_reserved_max_mb"] = gpu_reserved_max
                    if cpu_rss is not None:
                        payload["perf/cpu_rss_mb"] = cpu_rss
                    if ps_mem:
                        payload.update(ps_mem)
                    if attn_ratio is not None:
                        payload["perf/attn_elems"] = attn_elems
                        payload["perf/attn_ratio"] = attn_ratio
                    if attn_reduction is not None:
                        payload["perf/attn_reduction"] = attn_reduction
                    if flops_attn_est is not None:
                        payload["aah/flops_attn_est"] = flops_attn_est
                    if flops_total_est is not None:
                        payload["aah/flops_total_est"] = flops_total_est
                    if flops_ratio is not None:
                        payload["aah/flops_ratio"] = flops_ratio
                    if flops_reduction_pct is not None:
                        payload["aah/flops_reduction_%"] = flops_reduction_pct
                    if group_change_rate is not None:
                        payload["aah/group_change_rate"] = group_change_rate
                    if avg_window is not None:
                        payload["aah/avg_window"] = avg_window
                    if lk_mean is not None:
                        payload["aah/Lk_mean"] = lk_mean
                    if lk_p90 is not None:
                        payload["aah/Lk_p90"] = lk_p90
                    if w_mean is not None:
                        payload["aah/W_mean"] = w_mean
                    if w_min is not None:
                        payload["aah/W_min"] = w_min
                    if w_max is not None:
                        payload["aah/W_max"] = w_max
                    if control_time_ms is not None:
                        payload["aah/time/control_ms"] = control_time_ms
                    if attn_time_ms is not None:
                        payload["aah/time/attention_ms"] = attn_time_ms
                    if mask_time_ms is not None:
                        payload["aah/time/mask_ms"] = mask_time_ms
                    if overhead_time_ms is not None:
                        payload["aah/time/overhead_ms"] = overhead_time_ms
                    if shadow_logit_mean:
                        payload["aah/shadow_logit_mean"] = shadow_logit_mean[0]
                    if group_ratios:
                        payload["aah/group_ratios"] = group_ratios[0]
                    if avg_overlap is not None:
                        payload["aah/group_overlap"] = avg_overlap
                    if avg_reassign is not None:
                        payload["aah/head_reassign_rate"] = avg_reassign
                    if lifespan_ema is not None:
                        payload["aah/group_lifespan_ema"] = lifespan_ema
                    wandb.log(payload)
                if csv_writer:
                    lk_serialized = ""
                    if lk_layers:
                        lk_serialized = "|".join([",".join(map(str, layer)) for layer in lk_layers])
                    ent_serialized = ""
                    if head_entropy:
                        ent_serialized = ",".join([f"{v:.4f}" for v in head_entropy])
                    def fmt(v):
                        return f"{v:.2f}" if v is not None else ""
                    csv_writer.writerow([
                        step,
                        f"{loss.item():.6f}",
                        f"{tok_per_sec:.2f}",
                        f"{mem:.2f}",
                        f"{gpu_alloc:.2f}" if gpu_alloc is not None else "",
                        f"{gpu_reserved:.2f}" if gpu_reserved is not None else "",
                        f"{gpu_alloc_max:.2f}" if gpu_alloc_max is not None else "",
                        f"{gpu_reserved_max:.2f}" if gpu_reserved_max is not None else "",
                        f"{cpu_rss:.2f}" if cpu_rss is not None else "",
                        fmt(ps_mem.get("psutil_rss_mb")) if ps_mem else "",
                        fmt(ps_mem.get("psutil_vms_mb")) if ps_mem else "",
                        fmt(ps_mem.get("psutil_shared_mb")) if ps_mem else "",
                        fmt(ps_mem.get("psutil_text_mb")) if ps_mem else "",
                        fmt(ps_mem.get("psutil_data_mb")) if ps_mem else "",
                        fmt(ps_mem.get("psutil_uss_mb")) if ps_mem else "",
                        fmt(ps_mem.get("psutil_pss_mb")) if ps_mem else "",
                        fmt(ps_mem.get("psutil_swap_mb")) if ps_mem else "",
                        fmt(ps_mem.get("psutil_ram_used_mb")) if ps_mem else "",
                        fmt(ps_mem.get("psutil_ram_total_mb")) if ps_mem else "",
                        "",
                        "",
                        f"{attn_elems:.2f}" if attn_elems is not None else "",
                        f"{attn_ratio:.6f}" if attn_ratio is not None else "",
                        f"{attn_reduction:.6f}" if attn_reduction is not None else "",
                        f"{flops_attn_est:.2f}" if flops_attn_est is not None else "",
                        f"{flops_total_est:.2f}" if flops_total_est is not None else "",
                        f"{flops_ratio:.6f}" if flops_ratio is not None else "",
                        f"{flops_reduction_pct:.4f}" if flops_reduction_pct is not None else "",
                        str(lq) if lq is not None else "",
                        lk_serialized,
                        ent_serialized,
                        f"{group_change_rate:.6f}" if group_change_rate is not None else "",
                        f"{avg_window:.2f}" if avg_window is not None else "",
                        f"{avg_overlap:.6f}" if avg_overlap is not None else "",
                        f"{avg_reassign:.6f}" if avg_reassign is not None else "",
                        f"{lifespan_ema:.6f}" if lifespan_ema is not None else "",
                        "|".join([",".join(map(str, g)) for g in head_groups]) if head_groups else "",
                        "|".join([",".join(map(str, g)) for g in shadow_win_idx]) if shadow_win_idx else "",
                        "|".join([",".join(f"{v:.6f}" for v in g) for g in shadow_logit_mean]) if shadow_logit_mean else "",
                        str(group_heads[0]) if group_heads else "",
                        str(group_ratios[0]) if group_ratios else "",
                        fmt(lk_mean),
                        fmt(lk_p90),
                        fmt(w_mean),
                        fmt(w_min),
                        fmt(w_max),
                        fmt(control_time_ms),
                        fmt(attn_time_ms),
                        fmt(mask_time_ms),
                        fmt(overhead_time_ms),
                        f"{step_time_ms:.2f}",
                        eval_time_s,
                    ])
                if head_groups:
                    prev_head_groups = [list(g) for g in head_groups]
                t0 = time.time()

            eval_time_s = ""
            if step % train["eval_interval"] == 0:
                val_loss, val_ppl, eval_time = evaluate(
                    model,
                    val_loader,
                    device,
                    train["eval_batches"],
                    log_progress=train.get("eval_log_progress", False),
                    use_bf16=use_bf16,
                )
                eval_time_s = f"{eval_time:.2f}"
                print(f"eval step {step} | loss {val_loss:.4f} | ppl {val_ppl:.2f}")
                if use_wandb:
                    wandb.log({"val/loss": val_loss, "val/ppl": val_ppl, "step": step, "perf/eval_time_s": eval_time})
                if csv_writer:
                    csv_writer.writerow([step, "", "", "", f"{val_loss:.6f}", f"{val_ppl:.4f}"])

            if step >= train["max_steps"]:
                break

    torch.save(model.state_dict(), os.path.join(out_dir, f"{exp['name']}.pt"))
    if csv_file:
        csv_file.close()


if __name__ == "__main__":
    main()
