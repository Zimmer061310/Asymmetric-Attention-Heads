import os
import time
import math
import csv
import sys
import argparse
import yaml
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


def evaluate(model, loader, device, max_batches):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    if not losses:
        return float("inf"), float("inf")
    avg_loss = sum(losses) / len(losses)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


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
        aah_enabled=model_cfg.get("aah_enabled", False),
        aah_local_heads=model_cfg.get("aah_local_heads", 0),
        aah_window=model_cfg.get("aah_window", data["seq_len"]),
        aah_stride=model_cfg.get("aah_stride", 1),
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
        csv_writer.writerow(["step", "train_loss", "tok_s", "mem_mb", "val_loss", "val_ppl"])

    step = 0
    model.train()
    t0 = time.time()
    while step < train["max_steps"]:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            scheduler.step()

            step += 1

            if step % train["log_interval"] == 0:
                elapsed = time.time() - t0
                tokens = train["batch_size"] * data["seq_len"] * train["log_interval"]
                tok_per_sec = tokens / max(1e-9, elapsed)
                mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
                msg = f"step {step} | loss {loss.item():.4f} | tok/s {tok_per_sec:.1f} | mem {mem:.1f} MB"
                print(msg)
                if use_wandb:
                    wandb.log({"train/loss": loss.item(), "perf/tok_s": tok_per_sec, "perf/mem_mb": mem, "step": step})
                if csv_writer:
                    csv_writer.writerow([step, f"{loss.item():.6f}", f"{tok_per_sec:.2f}", f"{mem:.2f}", "", ""])
                t0 = time.time()

            if step % train["eval_interval"] == 0:
                val_loss, val_ppl = evaluate(model, val_loader, device, train["eval_batches"])
                print(f"eval step {step} | loss {val_loss:.4f} | ppl {val_ppl:.2f}")
                if use_wandb:
                    wandb.log({"val/loss": val_loss, "val/ppl": val_ppl, "step": step})
                if csv_writer:
                    csv_writer.writerow([step, "", "", "", f"{val_loss:.6f}", f"{val_ppl:.4f}"])

            if step >= train["max_steps"]:
                break

    torch.save(model.state_dict(), os.path.join(out_dir, f"{exp['name']}.pt"))
    if csv_file:
        csv_file.close()


if __name__ == "__main__":
    main()
