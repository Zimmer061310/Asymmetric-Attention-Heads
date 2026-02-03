# AAH-v1 — Answers for ChatGPT (Baseline Parity, Profiling, Diagnostics)

## 1) Baseline Parity & Sanity Checks
**Config:** H_local=0, s=1, W>=L, dropout=0  
**Result:** AAH degenerates exactly to MHA.

```
forward_max_abs_diff: 0.000000e+00
loss_curve_max_abs_diff: 0.000000e+00
first_loss_base: 0.443474 | first_loss_aah: 0.443474
last_loss_base: 0.010236 | last_loss_aah: 0.010236
parity_ok: True
```

## 2) Attention-Only Profiling (CPU self-time; device = MPS)
> Note: MPS does not expose GPU kernel time in this profiler view; numbers below are CPU self-time for profiler events.

### Avg forward time (full model)
* Baseline MHA: **0.0686 s**
* AAH (W=128, s=4): **0.0720 s**
* AAH (W=256, s=2): **0.0673 s**

### Breakdown (CPU self-time, µs)
**Baseline MHA**
```
attn_qkv: 209.45
attn_matmul_qk: 110.96
attn_mask: 133.92
attn_softmax: 76.62
attn_matmul_av: 66.50
```

**AAH W=128, s=4**
```
attn_qkv: 424.20
attn_local_matmul_qk: 205.00
attn_local_mask: 293.83
attn_local_softmax: 168.92
attn_local_matmul_av: 144.46
attn_global_downsample: 389.17
attn_global_matmul_qk: 280.12
attn_global_mask: 323.75
attn_global_softmax: 145.62
attn_global_matmul_av: 106.33
```

**AAH W=256, s=2**
```
attn_qkv: 571.16
attn_local_matmul_qk: 239.96
attn_local_mask: 282.16
attn_local_softmax: 157.58
attn_local_matmul_av: 200.42
attn_global_downsample: 272.71
attn_global_matmul_qk: 277.50
attn_global_mask: 373.58
attn_global_softmax: 193.21
attn_global_matmul_av: 151.96
```

## 3) Head-Level Diagnostics (Entropy)
Entropy = mean over batch + query positions.

**Baseline (8 heads)**
```
layer 0 entropy_per_head: [3.826, 3.8261, 3.8244, 3.8263, 3.8235, 3.826, 3.8242, 3.8269]
layer 1 entropy_per_head: [3.8265, 3.8258, 3.825, 3.8253, 3.8227, 3.8275, 3.8267, 3.8272]
```

**AAH (local_heads=2, W=64, s=2)**
```
layer 0 local_entropy_per_head: [3.6299, 3.6299]
layer 0 global_entropy_per_head: [3.1527, 3.1564, 3.1542, 3.1544, 3.1543, 3.1563]
layer 1 local_entropy_per_head: [3.6299, 3.6318]
layer 1 global_entropy_per_head: [3.1561, 3.1548, 3.1559, 3.1529, 3.1552, 3.1581]
```

## 4) Training-Length Sensitivity (10k steps)
Not run yet. Available if requested (baseline + chosen AAH variants).

## 5) Environment Confirmation
* PyTorch: **2.11.0.dev20260131**
* CUDA: **False**
* MPS: **True**

**Notes:**  
Masking/downsampling are on-device (Torch ops on same device).  
Profiler numbers above are CPU self-time, not GPU kernel time (MPS limitation).
