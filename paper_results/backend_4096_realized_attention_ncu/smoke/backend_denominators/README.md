# Backend denominator Nsight smoke

Smoke date: 2026-05-31

Server: Featurize RTX 4090, 48 GB modified VRAM.

Environment:
- PyTorch: 2.5.1+cu121
- FlashAttention: 2.7.4.post1 prebuilt wheel
- Nsight Compute: `/opt/nvidia/nsight-compute/2023.3.1/ncu` via `/home/featurize/work/bin/ncu-sudo`

Purpose:
- Verify Nsight Compute counter access on the 4090 server.
- Verify pure FlexAttention and pure FlashAttention 4096-token denominator profiles run.
- Verify denominator JSONs contain backend name, sequence length, batch size, precision, device, raw Nsight counter values, peak memory, and fallback reasons.

Results:

| Backend | JSON | Nsight OK | Backend name | GPU counter total | Peak memory MB | Fallbacks |
| --- | --- | --- | --- | ---: | ---: | --- |
| FlexAttention | `flex_pure_4096_ncu.json` | true | `flex_attention` | 80149547272.0 | 7327.80078125 | none |
| FlashAttention | `flash_pure_4096_ncu.json` | true | `flash_attn` | 80149556919.0 | 4536.384765625 | none |

These are denominator smokes only. They validate the profiler path and backend imports; they are not the full 10-row paper rerun.
