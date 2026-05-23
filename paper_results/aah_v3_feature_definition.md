# AAH-v3 Feature Definition Lock

This note resolves the review item about Q/K/V magnitude statistics.

## Implementation Statistic

The implemented AAH-v3 head feature uses mean absolute activation:

```text
mu(abs(q_h)), sigma(q_h), mu(abs(k_h)), sigma(k_h), mu(abs(v_h)), sigma(v_h)
```

not absolute value after scalar reduction:

```text
abs(mu(q_h)), sigma(q_h), abs(mu(k_h)), sigma(k_h), abs(mu(v_h)), sigma(v_h)
```

For the native Transformer path, `q`, `k`, and `v` have shape
`[batch, head, time, head_dim]`. The mean absolute entries reduce over
`batch`, `time`, and `head_dim`, preserving the head axis. Group-level Q/K/V
features reduce over `batch`, the heads inside the group, `time`, and
`head_dim`.

For the Qwen3 compatibility patch, the first six feature dimensions use the
same mean-absolute and standard-deviation convention over
`[batch, head, time, head_dim]`.

## Code Locations

- `src/models/transformer.py`: `_head_features` and `_group_features_from_qkv`.
- `scripts/qwen3_aah_patch.py`: `AAHRuntimeState._features`.

## Paper Formula Replacement

Use:

```text
x_h = [mu(|q_h|), sigma(q_h), mu(|k_h|), sigma(k_h), mu(|v_h|), sigma(v_h), e_h, n_h, rho_h]
```

Do not use:

```text
x_h = [|bar(q_h)|, sigma(q_h), |bar(k_h)|, sigma(k_h), |bar(v_h)|, sigma(v_h), e_h, n_h, rho_h]
```

This matters because `abs(mu(q_h))` and `mu(abs(q_h))` are not equivalent for
signed activations and can induce different grouping, hierarchy, and window
policies.
