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
            "val_loss",
            "val_ppl",
            "attn_elems",
            "attn_ratio",
            "attn_lq",
            "attn_lk_per_layer",
            "head_entropy",
            "step_time_ms",
            "eval_time_s",
        ])

    step = 0
    model.train()
    t0 = time.time()
    while step < train["max_steps"]:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            warmup_steps = model_cfg.get("aah_v2_warmup_steps", 0)
            activation_steps = model_cfg.get("aah_v2_activation_steps", 0)
            compute_lambda = model_cfg.get("aah_v2_compute_lambda", 0.0)
            min_head_norm = model_cfg.get("aah_v2_min_head_norm", 0.0)
            norm_lambda = model_cfg.get("aah_v2_norm_lambda", 0.0)
            min_head_entropy = model_cfg.get("aah_v2_min_head_entropy", 0.0)
            ent_lambda = model_cfg.get("aah_v2_entropy_lambda", 0.0)
            if model_cfg.get("aah_v2_enabled", False):
                if step < warmup_steps:
                    control_enabled = False
                    lambda_now = 0.0
                elif step < (warmup_steps + activation_steps):
                    control_enabled = True
                    lambda_now = 0.0
                else:
                    control_enabled = True
                    lambda_now = compute_lambda
                if step == warmup_steps:
                    for block in model.blocks:
                        attn = block.attn
                        if hasattr(attn, "reset_cache"):
                            attn.reset_cache()
                for block in model.blocks:
                    attn = block.attn
                    if hasattr(attn, "set_control"):
                        attn.set_control(control_enabled)
            step_t0 = time.time()
            logits, loss = model(x, y)

            if model_cfg.get("aah_v2_enabled", False) and lambda_now > 0:
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
                mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
                gpu_alloc, gpu_reserved, cpu_rss = get_memory_stats()
                attn_elems = None
                attn_ratio = None
                lq = None
                lk_layers = []
                head_entropy = []
                if model_cfg.get("aah_v2_enabled", False):
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
                    if baseline_elements > 0:
                        attn_elems = total_elements
                        attn_ratio = total_elements / baseline_elements
                msg = f"step {step} | loss {loss.item():.4f} | tok/s {tok_per_sec:.1f} | mem {mem:.1f} MB"
                if gpu_alloc is not None:
                    msg += f" | gpu_alloc {gpu_alloc:.1f} MB"
                if gpu_reserved is not None:
                    msg += f" | gpu_resv {gpu_reserved:.1f} MB"
                if cpu_rss is not None:
                    msg += f" | cpu_rss {cpu_rss:.1f} MB"
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
                    if attn_ratio is not None:
                        payload["perf/attn_elems"] = attn_elems
                        payload["perf/attn_ratio"] = attn_ratio
                    wandb.log(payload)
                if csv_writer:
                    lk_serialized = ""
                    if lk_layers:
                        lk_serialized = "|".join([",".join(map(str, layer)) for layer in lk_layers])
                    ent_serialized = ""
                    if head_entropy:
                        ent_serialized = ",".join([f"{v:.4f}" for v in head_entropy])
                    csv_writer.writerow([
                        step,
                        f"{loss.item():.6f}",
                        f"{tok_per_sec:.2f}",
                        f"{mem:.2f}",
                        f"{gpu_alloc:.2f}" if gpu_alloc is not None else "",
                        f"{gpu_reserved:.2f}" if gpu_reserved is not None else "",
                        f"{cpu_rss:.2f}" if cpu_rss is not None else "",
                        "",
                        "",
                        f"{attn_elems:.2f}" if attn_elems is not None else "",
                        f"{attn_ratio:.6f}" if attn_ratio is not None else "",
                        str(lq) if lq is not None else "",
                        lk_serialized,
                        ent_serialized,
                        f"{step_time_ms:.2f}",
                        eval_time_s,
                    ])
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
