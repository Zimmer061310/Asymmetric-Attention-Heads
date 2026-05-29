# AAH-v3 8192 Real-Backend AutoDL Smoke

Date: 2026-05-30 CST

Branch: `AAH-v3`

Local committed code: `8b5c324` (`Disable dropout for real backend configs`)

Server repo during successful train smoke: `d27032c` with temporary smoke configs overriding
`model.dropout=0.0`, `aah_v3_warmup_steps=0`, `dataset=synthetic`, and `tokenizer=synthetic:1024`.

Hardware/runtime:

- Platform: AutoDL
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB
- PyTorch: `2.8.0+cu128`
- CUDA runtime: `12.8`
- FlexAttention: available from `torch.nn.attention.flex_attention`
- FlashAttention: not installed; direct GitHub wheel download for `flash_attn-2.8.3+cu12torch2.8-cp312` repeatedly timed out from this server.

## FlexAttention backend smoke

W&B run: <https://wandb.ai/zimmer061310-ena/ENA-AAH/runs/7yqu01tr>

Small model smoke used `T=8192`, windows `[1024, 2048, 4096, 8192]`, FlexAttention backend,
bf16 autocast, and checked finite logits plus one backward pass.

| Case | Status | Effective ACR | Backend realized ACR | Backend | Notes |
|---|---:|---:|---:|---|---|
| fixed 1024 forward | pass | 0.125 | 0.125 | flex_attention | first call includes compilation |
| fixed 2048 forward | pass | 0.250 | 0.250 | flex_attention | no fallback |
| fixed 4096 forward | pass | 0.500 | 0.500 | flex_attention | no fallback |
| fixed 8192 forward | pass | 1.000 | 1.000 | flex_attention | full causal |
| adaptive forward | pass | 0.625 | 0.625 | flex_attention | two backend buckets |
| fixed 1024 backward | pass | 0.125 | 0.125 | flex_attention | finite loss |

## 1B / 8192 synthetic train smoke

These are runtime/backend checks only. They use random synthetic tokens and do not support quality claims.

Fixed 1024 W&B run: <https://wandb.ai/zimmer061310-ena/ENA-AAH/runs/ww6p6gq2>

Adaptive W&B run: <https://wandb.ai/zimmer061310-ena/ENA-AAH/runs/f0mkjtd3>

| Run | Steps | Backend | ACR | Avg window | Window range | Post-compile tok/s | Status |
|---|---:|---|---:|---:|---|---:|---|
| fixed 1024 | 5 | flex_attention | 0.125 | 1024 | 1024-1024 | about 4.2k | pass |
| adaptive | 5 | flex_attention | 0.414 | 3392 | 1024-8192 | about 4.0k | pass |

Important observations:

- `dropout=0.1` made FlexAttention intentionally fall back to dense masked attention and caused OOM at 1B/8192.
- Real-backend 8192 configs were updated to `dropout=0.0` so backend experiments do not silently fall back to dense masked training.
- AAH warmup must be disabled for short smoke tests; otherwise early steps intentionally use full windows.
- Hugging Face dataset access was unavailable on this server, so smoke training used the committed synthetic data path.
