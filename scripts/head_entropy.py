import math
import os
import sys
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.transformer import GPT, GPTConfig


def entropy_from_attn(attn):
    # attn: [B, H, T, K]
    eps = 1e-9
    p = attn.clamp_min(eps)
    ent = -(p * p.log()).sum(dim=-1)  # [B, H, T]
    return ent.mean(dim=(0, 2))  # [H]


def run(model, device="cpu"):
    model.to(device)
    model.eval()
    idx = torch.randint(0, 50257, (2, 128), device=device)
    with torch.no_grad():
        logits, _, attn_stack = model(idx, return_attn=True)
    return attn_stack


def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    cfg_base = GPTConfig(
        vocab_size=50257,
        seq_len=128,
        n_layer=2,
        n_head=8,
        n_embd=512,
        n_ff=2048,
        dropout=0.0,
        aah_enabled=False,
    )
    cfg_aah = GPTConfig(
        vocab_size=50257,
        seq_len=128,
        n_layer=2,
        n_head=8,
        n_embd=512,
        n_ff=2048,
        dropout=0.0,
        aah_enabled=True,
        aah_local_heads=2,
        aah_window=64,
        aah_stride=2,
    )

    base = GPT(cfg_base)
    aah = GPT(cfg_aah)
    aah.load_state_dict(base.state_dict(), strict=False)

    attn_base = run(base, device=device)
    attn_aah = run(aah, device=device)

    # Base: list of [B,H,T,T] per layer
    print("=== baseline ===")
    for li, att in enumerate(attn_base):
        ent = entropy_from_attn(att)
        print(f"layer {li} entropy_per_head:", [round(x, 4) for x in ent.tolist()])

    # AAH: list of [local_attn, global_attn] per layer
    print("=== aah ===")
    for li, att in enumerate(attn_aah):
        local_attn, global_attn = att
        ent_local = entropy_from_attn(local_attn)
        ent_global = entropy_from_attn(global_attn)
        print(f"layer {li} local_entropy_per_head:", [round(x, 4) for x in ent_local.tolist()])
        print(f"layer {li} global_entropy_per_head:", [round(x, 4) for x in ent_global.tolist()])


if __name__ == "__main__":
    main()
