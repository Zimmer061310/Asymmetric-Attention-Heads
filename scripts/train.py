import os
import time
import math
import csv
import sys
import argparse
import yaml
import platform
import resource
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


def evaluate(model, loader, device, max_batches, log_progress=False):
    model.eval()
    for block in model.blocks:
        attn = block.attn
        if hasattr(attn, "set_eval_mode"):
            attn.set_eval_mode(True)
    losses = []
    t0 = time.time()
    with torch.no_grad():
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
    if torch.cuda.is_available():
        gpu_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
        gpu_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    cpu_rss = None
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            cpu_rss = rss / (1024 ** 2)
        else:
            cpu_rss = rss / 1024
    except Exception:
        pass
    return gpu_alloc, gpu_reserved, cpu_rss


def get_psutil_memory_mb():
    if psutil is None:
        return {}
    try:
        proc = psutil.Process(os.getpid())
        info = proc.memory_info()
        full = proc.memory_full_info()
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
        }
    except Exception:
        return {}


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
            "cpu_rss_mb",
            "psutil_rss_mb",
            "psutil_vms_mb",
            "psutil_shared_mb",
            "psutil_text_mb",
            "psutil_data_mb",
            "psutil_uss_mb",
            "psutil_pss_mb",
            "psutil_swap_mb",
            "val_loss",
            "val_ppl",
            "attn_elems",
            "attn_ratio",
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

            if step % train["log_interval"] == 0:
                elapsed = time.time() - t0
                tokens = train["batch_size"] * data["seq_len"] * train["log_interval"]
                tok_per_sec = tokens / max(1e-9, elapsed)
                gpu_alloc, gpu_reserved, cpu_rss = get_memory_stats()
                ps_mem = get_psutil_memory_mb()
                if gpu_alloc is not None:
                    mem = gpu_alloc
                elif cpu_rss is not None:
                    mem = cpu_rss
                else:
                    mem = 0.0
                attn_elems = None
                attn_ratio = None
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
                    if baseline_elements > 0:
                        attn_elems = total_elements
                        attn_ratio = total_elements / baseline_elements
                group_change_rates = [v for v in group_change_rates if v is not None]
                group_change_rate = sum(group_change_rates) / len(group_change_rates) if group_change_rates else None
                avg_window = sum(avg_windows) / len(avg_windows) if avg_windows else None
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
                    if cpu_rss is not None:
                        payload["perf/cpu_rss_mb"] = cpu_rss
                    if ps_mem:
                        payload.update(ps_mem)
                    if attn_ratio is not None:
                        payload["perf/attn_elems"] = attn_elems
                        payload["perf/attn_ratio"] = attn_ratio
                    if group_change_rate is not None:
                        payload["aah/group_change_rate"] = group_change_rate
                    if avg_window is not None:
                        payload["aah/avg_window"] = avg_window
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
                        f"{cpu_rss:.2f}" if cpu_rss is not None else "",
                        fmt(ps_mem.get("psutil_rss_mb")) if ps_mem else "",
                        fmt(ps_mem.get("psutil_vms_mb")) if ps_mem else "",
                        fmt(ps_mem.get("psutil_shared_mb")) if ps_mem else "",
                        fmt(ps_mem.get("psutil_text_mb")) if ps_mem else "",
                        fmt(ps_mem.get("psutil_data_mb")) if ps_mem else "",
                        fmt(ps_mem.get("psutil_uss_mb")) if ps_mem else "",
                        fmt(ps_mem.get("psutil_pss_mb")) if ps_mem else "",
                        fmt(ps_mem.get("psutil_swap_mb")) if ps_mem else "",
                        "",
                        "",
                        f"{attn_elems:.2f}" if attn_elems is not None else "",
                        f"{attn_ratio:.6f}" if attn_ratio is not None else "",
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
