import math
import os
import sys
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.transformer import GPT, GPTConfig


def build_models():
    cfg_base = GPTConfig(
        vocab_size=101,
        seq_len=32,
        n_layer=2,
        n_head=4,
        n_embd=32,
        n_ff=64,
        dropout=0.0,
        aah_enabled=False,
    )
    cfg_aah = GPTConfig(
        vocab_size=101,
        seq_len=32,
        n_layer=2,
        n_head=4,
        n_embd=32,
        n_ff=64,
        dropout=0.0,
        aah_enabled=True,
        aah_local_heads=0,   # H_local = 0
        aah_window=64,       # W >= L
        aah_stride=1,        # s = 1
    )
    torch.manual_seed(1234)
    m_base = GPT(cfg_base)
    torch.manual_seed(1234)
    m_aah = GPT(cfg_aah)
    m_aah.load_state_dict(m_base.state_dict(), strict=False)
    return m_base, m_aah


def forward_parity():
    m_base, m_aah = build_models()
    m_base.eval()
    m_aah.eval()
    torch.manual_seed(42)
    idx = torch.randint(0, 101, (2, 32))
    with torch.no_grad():
        out_base, _ = m_base(idx)
        out_aah, _ = m_aah(idx)
    diff = (out_base - out_aah).abs().max().item()
    print(f"forward_max_abs_diff: {diff:.6e}")
    return diff


def loss_curve_parity(steps=100):
    m_base, m_aah = build_models()
    m_base.train()
    m_aah.train()
    opt_base = torch.optim.AdamW(m_base.parameters(), lr=1e-3)
    opt_aah = torch.optim.AdamW(m_aah.parameters(), lr=1e-3)

    torch.manual_seed(999)
    batches = [torch.randint(0, 101, (4, 32)) for _ in range(steps)]
    losses_base = []
    losses_aah = []
    for x in batches:
        logits_b, loss_b = m_base(x, x)
        loss_b.backward()
        opt_base.step()
        opt_base.zero_grad(set_to_none=True)
        losses_base.append(loss_b.item())

        logits_a, loss_a = m_aah(x, x)
        loss_a.backward()
        opt_aah.step()
        opt_aah.zero_grad(set_to_none=True)
        losses_aah.append(loss_a.item())

    max_diff = max(abs(a - b) for a, b in zip(losses_base, losses_aah))
    print(f"loss_curve_max_abs_diff: {max_diff:.6e}")
    print(f"first_loss_base: {losses_base[0]:.6f} | first_loss_aah: {losses_aah[0]:.6f}")
    print(f"last_loss_base: {losses_base[-1]:.6f} | last_loss_aah: {losses_aah[-1]:.6f}")
    return max_diff


if __name__ == "__main__":
    fwd = forward_parity()
    curve = loss_curve_parity(steps=100)
    tol = 1e-6
    ok = fwd < tol and curve < 1e-6
    print(f"parity_ok: {ok}")
