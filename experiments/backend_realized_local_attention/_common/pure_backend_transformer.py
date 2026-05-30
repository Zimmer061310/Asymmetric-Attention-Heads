"""Pure backend Transformer for FlexAttention/FlashAttention baselines.

This module intentionally contains no AAH controller, grouping, hierarchy, or
window-selection path. It exists so pure backend baselines do not depend on
disabled AAH modules.
"""

import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function

_FLASH_ATTN_FUNC = None
_FLASH_ATTN_IMPORT_ERROR = None
_FLEX_ATTENTION = None
_FLEX_CREATE_BLOCK_MASK = None
_FLEX_IMPORT_ERROR = None


def _load_flash_attn_func():
    global _FLASH_ATTN_FUNC, _FLASH_ATTN_IMPORT_ERROR
    if _FLASH_ATTN_FUNC is not None or _FLASH_ATTN_IMPORT_ERROR is not None:
        return _FLASH_ATTN_FUNC, _FLASH_ATTN_IMPORT_ERROR
    try:
        from flash_attn import flash_attn_func

        _FLASH_ATTN_FUNC = flash_attn_func
    except Exception as exc:  # pragma: no cover - optional CUDA package
        _FLASH_ATTN_IMPORT_ERROR = exc
    return _FLASH_ATTN_FUNC, _FLASH_ATTN_IMPORT_ERROR


def _load_flex_attention():
    global _FLEX_ATTENTION, _FLEX_CREATE_BLOCK_MASK, _FLEX_IMPORT_ERROR
    if _FLEX_ATTENTION is not None or _FLEX_IMPORT_ERROR is not None:
        return _FLEX_ATTENTION, _FLEX_CREATE_BLOCK_MASK, _FLEX_IMPORT_ERROR
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        _FLEX_ATTENTION = flex_attention
        _FLEX_CREATE_BLOCK_MASK = create_block_mask
    except Exception as exc:  # pragma: no cover - PyTorch build dependent
        _FLEX_IMPORT_ERROR = exc
    return _FLEX_ATTENTION, _FLEX_CREATE_BLOCK_MASK, _FLEX_IMPORT_ERROR


class GPTConfig:
    def __init__(self, **kwargs):
        defaults = {
            "dropout": 0.1,
            "aah_v2_enabled": False,
            "aah_v3_enabled": False,
            "aah_v3_attention_backend": "flex_attention",
            "aah_v3_flex_block_size": 128,
            "aah_v3_mask_cache_size": 16,
        }
        defaults.update(kwargs)
        self.__dict__.update(defaults)


class BackendCausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = self.head_dim ** -0.5
        self.backend = str(getattr(config, "aah_v3_attention_backend", "flex_attention"))
        if self.backend not in {"flex_attention", "flash_attn", "dense_masked"}:
            raise ValueError(f"Unsupported pure backend: {self.backend}")
        self.flex_block_size = int(getattr(config, "aah_v3_flex_block_size", 128))
        self.mask_cache_size = int(getattr(config, "aah_v3_mask_cache_size", 16))
        self.flex_mask_cache = OrderedDict()
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.seq_len, config.seq_len)).view(1, 1, config.seq_len, config.seq_len),
        )
        self.last_stats = {}

    def _fallback(self, reason):
        return {"backend": "dense_masked", "requested_backend": self.backend, "fallback_reason": str(reason)}

    def _dense_attention(self, q, k, v):
        B, H, T, D = q.shape
        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        return att @ v, {"backend": "dense_masked", "requested_backend": self.backend, "fallback_reason": ""}

    def _flash_attention(self, q, k, v):
        if q.device.type != "cuda":
            return None, self._fallback("flash_attn_requires_cuda")
        if q.dtype not in (torch.float16, torch.bfloat16):
            return None, self._fallback(f"flash_attn_unsupported_dtype_{q.dtype}")
        flash_attn_func, import_error = _load_flash_attn_func()
        if flash_attn_func is None:
            return None, self._fallback(f"flash_attn_unavailable: {import_error}")
        try:
            y = flash_attn_func(
                q.transpose(1, 2).contiguous(),
                k.transpose(1, 2).contiguous(),
                v.transpose(1, 2).contiguous(),
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=True,
            )
        except Exception as exc:  # pragma: no cover - backend/runtime specific
            return None, self._fallback(f"flash_attn_failed: {exc}")
        return y.transpose(1, 2).contiguous(), {
            "backend": "flash_attn",
            "requested_backend": self.backend,
            "fallback_reason": "",
        }

    def _get_full_flex_mask(self, T, device):
        key = (int(T), device.type, device.index, int(self.flex_block_size))
        cached = self.flex_mask_cache.get(key)
        if cached is not None:
            self.flex_mask_cache.move_to_end(key)
            return cached
        _, create_block_mask, import_error = _load_flex_attention()
        if create_block_mask is None:
            raise RuntimeError(f"flex_attention_unavailable: {import_error}")

        def causal_mask(_b, _h, q_idx, kv_idx):
            return kv_idx <= q_idx

        try:
            mask = create_block_mask(
                causal_mask,
                B=None,
                H=None,
                Q_LEN=int(T),
                KV_LEN=int(T),
                device=device,
                BLOCK_SIZE=int(self.flex_block_size),
                _compile=True,
            )
        except TypeError:
            mask = create_block_mask(
                causal_mask,
                B=None,
                H=None,
                Q_LEN=int(T),
                KV_LEN=int(T),
                device=device,
                BLOCK_SIZE=int(self.flex_block_size),
            )
        if self.mask_cache_size > 0:
            self.flex_mask_cache[key] = mask
            self.flex_mask_cache.move_to_end(key)
            while len(self.flex_mask_cache) > self.mask_cache_size:
                self.flex_mask_cache.popitem(last=False)
        return mask

    def _flex_attention(self, q, k, v):
        if self.training and self.attn_drop.p > 0.0:
            return None, self._fallback("flex_attention_no_dropout_fallback")
        flex_attention, _, import_error = _load_flex_attention()
        if flex_attention is None:
            return None, self._fallback(f"flex_attention_unavailable: {import_error}")
        try:
            block_mask = self._get_full_flex_mask(q.size(-2), q.device)
            y = flex_attention(q.contiguous(), k.contiguous(), v.contiguous(), block_mask=block_mask, scale=self.scale)
        except Exception as exc:  # pragma: no cover - backend/runtime specific
            return None, self._fallback(f"flex_attention_failed: {exc}")
        return y, {"backend": "flex_attention", "requested_backend": self.backend, "fallback_reason": ""}

    def forward(self, x, return_attn: bool = False):
        B, T, C = x.size()
        t_start = time.perf_counter()
        with record_function("attn_qkv"):
            qkv = self.qkv(x)
            q, k, v = qkv.split(C, dim=2)
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        t_attn = time.perf_counter()
        y = None
        info = {"backend": "dense_masked", "requested_backend": self.backend, "fallback_reason": ""}
        if not return_attn and self.backend == "flash_attn":
            with record_function("attn_flash_backend"):
                y, info = self._flash_attention(q, k, v)
        elif not return_attn and self.backend == "flex_attention":
            with record_function("attn_flex_backend"):
                y, info = self._flex_attention(q, k, v)
        if y is None:
            with record_function("attn_dense_masked_backend"):
                y, dense_info = self._dense_attention(q, k, v)
            if info.get("fallback_reason"):
                dense_info["fallback_reason"] = info["fallback_reason"]
            info = dense_info
        attn_time_ms = (time.perf_counter() - t_attn) * 1000.0

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        full_elements = float(self.n_head * T * T)
        backend_used = str(info.get("backend", "dense_masked"))
        self.last_stats = {
            "lq": T,
            "lk": [T for _ in range(self.n_head)],
            "total_elements": full_elements,
            "baseline_elements": full_elements,
            "effective_attn_elements": full_elements,
            "effective_ACR": 1.0,
            "dense_kernel_actual_elements_est": full_elements,
            "backend_realized_elements_est": full_elements,
            "backend_realized_ACR_est": 1.0,
            "backend_name": backend_used,
            "requested_backend": self.backend,
            "backend_bucket_counts": {backend_used: 1},
            "backend_kernel_calls": 1,
            "backend_time_ms": attn_time_ms,
            "backend_fallback_reasons": [info["fallback_reason"]] if info.get("fallback_reason") else [],
            "attention_stats_available": False,
            "head_norms": [],
            "head_entropy": [],
            "head_usage": [],
            "avg_window": float(T),
            "w_mean": float(T),
            "w_min": float(T),
            "w_max": float(T),
            "lk_mean": float(T),
            "lk_p90": float(T),
            "attn_time_ms": attn_time_ms,
            "overhead_time_ms": (time.perf_counter() - t_start) * 1000.0,
            "path_mode": f"pure_{backend_used}_full_attention",
        }
        if return_attn:
            return y, None
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, config.n_ff)
        self.fc2 = nn.Linear(config.n_ff, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = BackendCausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, return_attn: bool = False):
        if return_attn:
            attn_out = self.attn(self.ln1(x), return_attn=True)
            x = x + attn_out[0]
            attn = attn_out[1]
        else:
            x = x + self.attn(self.ln1(x))
            attn = None
        x = x + self.mlp(self.ln2(x))
        if return_attn:
            return x, attn
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.seq_len, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, idx, targets=None, return_attn: bool = False):
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.wte(idx) + self.wpe(pos)
        x = self.drop(x)
        attn_stack = []
        for block in self.blocks:
            if return_attn:
                x, attn = block(x, return_attn=True)
                attn_stack.append(attn)
            else:
                x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if return_attn:
            return logits, loss, attn_stack
        return logits, loss
