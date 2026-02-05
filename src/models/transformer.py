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
    aah_v2_enabled: bool = False
    aah_v2_windows: tuple = (64, 128, 256, 512)
    aah_v2_strides: tuple = (1, 2, 4)
    aah_v2_group_size: int = 1
    aah_v2_control_dim: int = 16
    aah_v2_temperature: float = 1.0
    aah_v2_dynamic_grouping: bool = False
    aah_v2_num_groups: int = 4
    aah_v2_min_head_norm: float = 0.0
    aah_v2_min_head_entropy: float = 0.0
    aah_v2_local_chunk: int = 128


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



class AAHV2Controller(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int, n_windows: int, n_strides: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_windows + n_strides),
        )
        self.n_windows = n_windows
        self.n_strides = n_strides

    def forward(self, feats):
        logits = self.mlp(feats)
        win_logits = logits[:, :self.n_windows]
        stride_logits = logits[:, self.n_windows:]
        return win_logits, stride_logits


class AAHV2Attention(nn.Module):
    """AAH-v2: dynamic, pre-matmul control over per-head compute."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = self.head_dim ** -0.5
        self.seq_len = config.seq_len
        self.windows = tuple(int(w) for w in config.aah_v2_windows)
        self.strides = tuple(int(s) for s in config.aah_v2_strides)
        self.group_size = max(1, int(config.aah_v2_group_size))
        self.dynamic_grouping = bool(config.aah_v2_dynamic_grouping)
        self.num_groups = max(1, int(config.aah_v2_num_groups))
        self.temperature = float(config.aah_v2_temperature)
        self.control_enabled = True
        self.min_head_norm = float(config.aah_v2_min_head_norm)
        self.min_head_entropy = float(config.aah_v2_min_head_entropy)
        self.local_chunk = max(1, int(config.aah_v2_local_chunk))
        self.eval_mode = False
        self.cached_head_to_group = None

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        self.register_buffer(
            "full_mask",
            torch.tril(torch.ones(config.seq_len, config.seq_len)).view(1, 1, config.seq_len, config.seq_len),
        )

        self.controller = AAHV2Controller(
            feat_dim=6,
            hidden_dim=max(4, int(config.aah_v2_control_dim)),
            n_windows=len(self.windows),
            n_strides=len(self.strides),
        )
        if self.dynamic_grouping:
            self.group_assign = nn.Sequential(
                nn.Linear(6, max(4, int(config.aah_v2_control_dim))),
                nn.GELU(),
                nn.Linear(max(4, int(config.aah_v2_control_dim)), self.num_groups),
            )

    def set_control(self, enabled: bool):
        self.control_enabled = bool(enabled)
        if not self.control_enabled:
            self.cached_head_to_group = None

    def reset_cache(self):
        self.cached_head_to_group = None

    def set_eval_mode(self, enabled: bool):
        self.eval_mode = bool(enabled)

    def _head_features(self, q, k, v):
        q_mean = q.abs().mean(dim=(0, 2, 3))
        q_std = q.std(dim=(0, 2, 3))
        k_mean = k.abs().mean(dim=(0, 2, 3))
        k_std = k.std(dim=(0, 2, 3))
        v_mean = v.abs().mean(dim=(0, 2, 3))
        v_std = v.std(dim=(0, 2, 3))
        feats = torch.stack([q_mean, q_std, k_mean, k_std, v_mean, v_std], dim=-1)
        return feats

    def _group_features(self, feats):
        if self.group_size == 1:
            return feats, torch.arange(self.n_head, device=feats.device)
        n_groups = math.ceil(self.n_head / self.group_size)
        pad = n_groups * self.group_size - self.n_head
        if pad > 0:
            feats = torch.cat([feats, feats[:pad]], dim=0)
        feats = feats.view(n_groups, self.group_size, -1).mean(dim=1)
        head_to_group = torch.arange(self.n_head + pad, device=feats.device) // self.group_size
        head_to_group = head_to_group[:self.n_head]
        return feats, head_to_group

    def _select_discrete(self, logits):
        probs = F.softmax(logits / max(1e-6, self.temperature), dim=-1)
        idx = probs.argmax(dim=-1)
        hard = F.one_hot(idx, num_classes=probs.size(-1)).float()
        probs = hard + (probs - probs.detach())
        return idx, probs

    def _dynamic_grouping(self, feats):
        assign_logits = self.group_assign(feats)
        assign_probs = F.softmax(assign_logits / max(1e-6, self.temperature), dim=-1)
        group_mass = assign_probs.sum(dim=0).clamp_min(1e-6)
        group_feats = (assign_probs.transpose(0, 1) @ feats) / group_mass.unsqueeze(1)
        group_idx = assign_probs.argmax(dim=-1)
        hard_assign = F.one_hot(group_idx, num_classes=self.num_groups).float()
        assign_probs = hard_assign + (assign_probs - assign_probs.detach())
        return group_feats, assign_probs, group_idx

    def _local_attention(self, q, k, v, window):
        B, T, D = q.shape
        W = max(1, min(int(window), T))
        if W == T:
            att = (q @ k.transpose(-2, -1)) * self.scale
            mask = self.full_mask[0, 0, :T, :T]
            att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v
            return y, att
        chunk = min(self.local_chunk, T)
        y_chunks = []
        att_chunks = []
        max_att_len = 0
        for t0 in range(0, T, chunk):
            t1 = min(T, t0 + chunk)
            q_chunk = q[:, t0:t1, :]
            k_start = max(0, t0 - W + 1)
            k_slice = k[:, k_start:t1, :]
            v_slice = v[:, k_start:t1, :]
            att = torch.einsum("btd,bkd->btk", q_chunk, k_slice) * self.scale
            q_pos = torch.arange(t0, t1, device=q.device).unsqueeze(1)
            k_pos = torch.arange(k_start, t1, device=q.device).unsqueeze(0)
            causal = k_pos <= q_pos
            window_ok = k_pos >= (q_pos - W + 1)
            att = att.masked_fill(~(causal & window_ok), float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = torch.einsum("btk,bkd->btd", att, v_slice)
            y_chunks.append(y)
            att_chunks.append(att)
            att_len = att.size(-1)
            if att_len > max_att_len:
                max_att_len = att_len
        y_full = torch.cat(y_chunks, dim=1)
        if max_att_len > 0:
            padded = []
            for att in att_chunks:
                pad = max_att_len - att.size(-1)
                if pad > 0:
                    att = F.pad(att, (0, pad), value=0.0)
                padded.append(att)
            att_full = torch.cat(padded, dim=1)
        else:
            att_full = torch.cat(att_chunks, dim=1)
        return y_full, att_full
        return y, att

    def _stride_attention(self, q, k, v, stride):
        B, T, D = q.shape
        s = max(1, int(stride))
        if s == 1:
            return self._local_attention(q, k, v, T)
        indices = torch.arange(0, T, s, device=q.device)
        k_down = k[:, indices, :]
        v_down = v[:, indices, :]
        att = (q @ k_down.transpose(-2, -1)) * self.scale
        key_pos = indices
        q_pos = torch.arange(0, T, device=q.device).unsqueeze(1)
        causal = key_pos.unsqueeze(0) <= q_pos
        att = att.masked_fill(~causal.unsqueeze(0), float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v_down
        return y, att

    def forward(self, x, return_attn: bool = False):
        B, T, C = x.size()
        with record_function("attn_qkv"):
            qkv = self.qkv(x)
            q, k, v = qkv.split(C, dim=2)
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.control_enabled:
            feats = self._head_features(q, k, v)
            if self.dynamic_grouping:
                if self.eval_mode and self.cached_head_to_group is not None:
                    head_to_group = self.cached_head_to_group
                    group_feats, _ = self._group_features(feats)
                    assign_probs = None
                else:
                    group_feats, assign_probs, head_to_group = self._dynamic_grouping(feats)
                    self.cached_head_to_group = head_to_group.detach().clone()
            else:
                group_feats, head_to_group = self._group_features(feats)
                assign_probs = None

            win_logits, stride_logits = self.controller(group_feats)
            win_idx, _ = self._select_discrete(win_logits)
            stride_idx, _ = self._select_discrete(stride_logits)
            win_idx = win_idx[head_to_group]
            stride_idx = stride_idx[head_to_group]
        else:
            win_idx = torch.full((self.n_head,), 0, device=x.device, dtype=torch.long)
            stride_idx = torch.full((self.n_head,), 0, device=x.device, dtype=torch.long)
            # force window to full length if possible
            if T in self.windows:
                win_full = self.windows.index(T)
                win_idx = torch.full((self.n_head,), win_full, device=x.device, dtype=torch.long)

        lq = T
        stride_vals = torch.tensor(self.strides, device=x.device, dtype=torch.float32)
        window_vals = torch.tensor(self.windows, device=x.device, dtype=torch.float32)
        s = stride_vals[stride_idx]
        w = window_vals[win_idx]
        lk_tensor = torch.where(
            s > 1,
            torch.ceil(torch.tensor(float(T), device=x.device) / s),
            torch.clamp(w, min=1.0, max=float(T)),
        )
        total_elements = float((lq * lk_tensor).sum().item())
        baseline_elements = float(self.n_head * lq * lq)

        outputs = torch.zeros(B, self.n_head, T, self.head_dim, device=x.device, dtype=q.dtype)
        head_norms = [0.0 for _ in range(self.n_head)]
        head_entropy = [0.0 for _ in range(self.n_head)]
        attn_maps = [None for _ in range(self.n_head)] if return_attn else None

        head_groups = {}
        for h in range(self.n_head):
            window = self.windows[int(win_idx[h].item())]
            stride = self.strides[int(stride_idx[h].item())]
            head_groups.setdefault((window, stride), []).append(h)

        for (window, stride), heads in head_groups.items():
            q_sel = q[:, heads]
            k_sel = k[:, heads]
            v_sel = v[:, heads]
            Bc, Hc, Tc, Dc = q_sel.shape
            qf = q_sel.reshape(Bc * Hc, Tc, Dc)
            kf = k_sel.reshape(Bc * Hc, Tc, Dc)
            vf = v_sel.reshape(Bc * Hc, Tc, Dc)
            if stride > 1:
                y_f, att_f = self._stride_attention(qf, kf, vf, stride)
            else:
                y_f, att_f = self._local_attention(qf, kf, vf, window)
            y_h = y_f.view(Bc, Hc, Tc, Dc)
            outputs[:, heads] = y_h
            norms = y_h.norm(dim=-1).mean(dim=(0, 2))
            for i, h in enumerate(heads):
                head_norms[h] = float(norms[i].item())
            att_h = att_f.view(Bc, Hc, att_f.size(1), att_f.size(2))
            ent = -(att_h * (att_h + 1e-9).log()).sum(dim=-1).mean(dim=(0, 2))
            ent = ent / max(1.0, math.log(att_h.size(-1)))
            for i, h in enumerate(heads):
                head_entropy[h] = float(ent[i].item())
            if return_attn:
                att_h = att_f.view(Bc, Hc, att_f.size(1), att_f.size(2))
                for i, h in enumerate(heads):
                    attn_maps[h] = att_h[:, i:i+1]

        y = outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        lk_list = lk_tensor.to(torch.int64).tolist()
        self.last_stats = {
            "lq": lq,
            "lk": lk_list,
            "total_elements": total_elements,
            "baseline_elements": baseline_elements,
            "head_norms": head_norms,
            "head_entropy": head_entropy,
        }
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
        if config.aah_v2_enabled:
            self.attn = AAHV2Attention(config)
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
