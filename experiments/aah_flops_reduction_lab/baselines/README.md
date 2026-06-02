# Baselines

Baselines define denominators for the FLOPs lab.

Primary denominator:

- pure FlashAttention full-attention forward pass
- `seq_len=4096`
- `batch_size=1`
- `bf16`
- same GPU and Nsight metric set as numerator rows

Dense baselines are allowed only as diagnostics. Any dense-baseline result must
be named with `dense-baseline` and must not be described as beating modern
FlashAttention.
