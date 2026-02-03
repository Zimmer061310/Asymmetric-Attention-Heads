import time
import os
import sys
import torch
from torch.profiler import profile, ProfilerActivity

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.transformer import GPT, GPTConfig


def build_model(aah=False, local_heads=0, window=128, stride=4):
    cfg = GPTConfig(
        vocab_size=50257,
        seq_len=512,
        n_layer=8,
        n_head=8,
        n_embd=512,
        n_ff=2048,
        dropout=0.0,
        aah_enabled=aah,
        aah_local_heads=local_heads,
        aah_window=window,
        aah_stride=stride,
    )
    return GPT(cfg)


def device_sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    if device == "mps":
        torch.mps.synchronize()


def attention_latency(model, device="cpu", steps=20):
    model.to(device)
    model.eval()
    idx = torch.randint(0, 50257, (2, 512), device=device)
    # warmup
    for _ in range(5):
        with torch.no_grad():
            model(idx)
    device_sync(device)
    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(idx)
        device_sync(device)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def profile_breakdown(model, device="cpu"):
    model.to(device)
    model.eval()
    idx = torch.randint(0, 50257, (2, 512), device=device)
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, record_shapes=False, with_stack=False) as prof:
        with torch.no_grad():
            model(idx)
        device_sync(device)
    key_avgs = prof.key_averages()
    key_map = {e.key: e for e in key_avgs}
    keys = [
        "attn_qkv",
        "attn_matmul_qk",
        "attn_mask",
        "attn_softmax",
        "attn_matmul_av",
        "attn_local_matmul_qk",
        "attn_local_mask",
        "attn_local_softmax",
        "attn_local_matmul_av",
        "attn_global_downsample",
        "attn_global_matmul_qk",
        "attn_global_mask",
        "attn_global_softmax",
        "attn_global_matmul_av",
    ]
    summary = {}
    for k in keys:
        evt = key_map.get(k)
        if evt is not None:
            summary[k] = evt.self_cpu_time_total
    return summary


def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    runs = [
        ("baseline_mha", build_model(aah=False), {}),
        ("aah_w128_s4", build_model(aah=True, local_heads=4, window=128, stride=4), {"local": 4, "window": 128, "stride": 4}),
        ("aah_w256_s2", build_model(aah=True, local_heads=2, window=256, stride=2), {"local": 2, "window": 256, "stride": 2}),
    ]
    for name, model, meta in runs:
        avg = attention_latency(model, device=device, steps=10)
        breakdown = profile_breakdown(model, device=device)
        print(f"=== {name} ===")
        print(f"device: {device}")
        print(f"avg_forward_s: {avg:.6f}")
        if meta:
            print(f"config: {meta}")
        print("breakdown_cpu_self_time_us:", {k: round(v, 2) for k, v in breakdown.items()})


if __name__ == "__main__":
    main()
