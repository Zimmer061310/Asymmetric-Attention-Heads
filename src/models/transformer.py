import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function


@dataclass
class GPTConfig:
    vocab_size: int
    seq_len: int
    n_layer: int
    n_head: int
    n_embd: int
    n_ff: int
    dropout: float = 0.1
    aah_enabled: bool = False
    aah_local_heads: int = 0
    aah_window: int = 128
    aah_stride: int = 4


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(config.seq_len, config.seq_len)).view(1, 1, config.seq_len, config.seq_len))

    def forward(self, x, return_attn: bool = False):
        B, T, C = x.size()
        with record_function("attn_qkv"):
            qkv = self.qkv(x)
            q, k, v = qkv.split(C, dim=2)
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        with record_function("attn_matmul_qk"):
            att = (q @ k.transpose(-2, -1)) * self.scale
        with record_function("attn_mask"):
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        with record_function("attn_softmax"):
            att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        with record_function("attn_matmul_av"):
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        if return_attn:
            return y, att
        return y


class AAHAttention(nn.Module):
    """Asymmetric Attention Heads: fixed partition into local (sliding window) and global (downsampled) heads."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = self.head_dim ** -0.5
        self.n_local = config.aah_local_heads
        self.n_global = config.n_head - config.aah_local_heads
        self.window = max(1, min(config.aah_window, config.seq_len))
        self.stride = max(1, config.aah_stride)
        self.seq_len = config.seq_len
        
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        
        # Local heads: sliding window mask
        if self.window >= config.seq_len:
            local_mask = torch.tril(torch.ones(config.seq_len, config.seq_len))
        else:
            base = torch.tril(torch.ones(config.seq_len, config.seq_len))
            band = torch.tril(torch.ones(config.seq_len, config.seq_len), diagonal=-self.window)
            local_mask = base - band
        self.register_buffer("local_mask", local_mask.view(1, 1, config.seq_len, config.seq_len))

    def forward(self, x, return_attn: bool = False):
        B, T, C = x.size()
        with record_function("attn_qkv"):
            qkv = self.qkv(x)
            q, k, v = qkv.split(C, dim=2)
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        outputs = []
        attn_maps = []
        
        # Local heads
        if self.n_local > 0:
            q_local = q[:, :self.n_local]
            k_local = k[:, :self.n_local]
            v_local = v[:, :self.n_local]
            with record_function("attn_local_matmul_qk"):
                att_local = (q_local @ k_local.transpose(-2, -1)) * self.scale
            with record_function("attn_local_mask"):
                att_local = att_local.masked_fill(self.local_mask[:, :, :T, :T] == 0, float("-inf"))
            with record_function("attn_local_softmax"):
                att_local = F.softmax(att_local, dim=-1)
            att_local = self.attn_drop(att_local)
            with record_function("attn_local_matmul_av"):
                y_local = att_local @ v_local
            outputs.append(y_local)
            if return_attn:
                attn_maps.append(att_local)
        
        # Global heads (downsample K/V)
        if self.n_global > 0:
            q_global = q[:, self.n_local:]
            k_global = k[:, self.n_local:]
            v_global = v[:, self.n_local:]
            
            # Downsample K and V by stride
            with record_function("attn_global_downsample"):
                indices = torch.arange(0, T, self.stride, device=x.device)
                k_down = k_global[:, :, indices, :]
                v_down = v_global[:, :, indices, :]
            
            with record_function("attn_global_matmul_qk"):
                att_global = (q_global @ k_down.transpose(-2, -1)) * self.scale
            # Causal mask: each query position can attend to downsampled keys up to its position
            with record_function("attn_global_mask"):
                key_pos = indices
                q_pos = torch.arange(0, T, device=x.device).unsqueeze(1)
                causal = key_pos.unsqueeze(0) <= q_pos
                att_global = att_global.masked_fill(causal.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
            with record_function("attn_global_softmax"):
                att_global = F.softmax(att_global, dim=-1)
            att_global = self.attn_drop(att_global)
            with record_function("attn_global_matmul_av"):
                y_global = att_global @ v_down
            outputs.append(y_global)
            if return_attn:
                attn_maps.append(att_global)
        
        y = torch.cat(outputs, dim=1)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        if return_attn:
            return y, attn_maps
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
        if config.aah_enabled:
            self.attn = AAHAttention(config)
        else:
            self.attn = CausalSelfAttention(config)
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
