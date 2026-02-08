import math
import time
from collections import OrderedDict
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
    aah_v2_control_interval: int = 1
    aah_v2_stride_control_enabled: bool = True
    aah_v3_enabled: bool = False
    aah_v3_windows: tuple = (64, 128, 256, 512)
    aah_v3_control_dim: int = 16
    aah_v3_control_interval: int = 100
    aah_v3_sim_threshold: float = 0.7
    aah_v3_super_threshold: float = 0.7
    aah_v3_max_depth: int = 6
    aah_v3_ema_alpha: float = 0.9
    aah_v3_churn_penalty: float = 0.05
    aah_v3_min_group_size: int = 2
    aah_v3_warmup_steps: int = 0
    aah_v3_control_enabled: bool = True
    aah_v3_grouping_enabled: bool = True
    aah_v3_W_min_gpu: int = 64
    aah_v3_mask_cache_size: int = 16


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
        t_start = time.perf_counter()
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
        stride_enabled = bool(config.aah_v2_stride_control_enabled)
        self.strides = tuple(int(s) for s in config.aah_v2_strides) if stride_enabled else (1,)
        self.group_size = max(1, int(config.aah_v2_group_size))
        self.dynamic_grouping = bool(config.aah_v2_dynamic_grouping)
        self.num_groups = max(1, int(config.aah_v2_num_groups))
        self.temperature = float(config.aah_v2_temperature)
        self.control_enabled = True
        self.min_head_norm = float(config.aah_v2_min_head_norm)
        self.min_head_entropy = float(config.aah_v2_min_head_entropy)
        self.local_chunk = max(1, int(config.aah_v2_local_chunk))
        self.control_interval = max(1, int(config.aah_v2_control_interval))
        self.stride_control_enabled = stride_enabled
        self.eval_mode = False
        self.cached_head_to_group = None
        self.cached_win_idx = None
        self.cached_stride_idx = None
        self.last_control_step = None
        self.current_step = None

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        self.register_buffer(
            "full_mask",
            torch.tril(torch.ones(config.seq_len, config.seq_len)).view(1, 1, config.seq_len, config.seq_len),
        )
        self.mask_cache = OrderedDict()

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
            self.cached_win_idx = None
            self.cached_stride_idx = None
            self.last_control_step = None

    def reset_cache(self):
        self.cached_head_to_group = None
        self.cached_win_idx = None
        self.cached_stride_idx = None
        self.last_control_step = None

    def set_eval_mode(self, enabled: bool):
        self.eval_mode = bool(enabled)

    def set_step(self, step: int):
        self.current_step = int(step)

    def _should_update_control(self):
        if self.control_interval <= 1:
            return True
        if self.current_step is None:
            return True
        if self.last_control_step is None:
            return True
        return (self.current_step - self.last_control_step) >= self.control_interval

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
            return y, att, 0.0
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
        t_start = time.perf_counter()
        t_control0 = time.perf_counter()
        with record_function("attn_qkv"):
            qkv = self.qkv(x)
            q, k, v = qkv.split(C, dim=2)
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.control_enabled:
            use_cache = (not self._should_update_control()) and self.cached_win_idx is not None and self.cached_stride_idx is not None
            if use_cache:
                win_idx = self.cached_win_idx
                stride_idx = self.cached_stride_idx
            else:
                feats = self._head_features(q, k, v)
                if self.dynamic_grouping:
                    if self.eval_mode and self.cached_head_to_group is not None:
                        head_to_group = self.cached_head_to_group
                        group_feats, _ = self._group_features(feats)
                    else:
                        group_feats, _, head_to_group = self._dynamic_grouping(feats)
                        self.cached_head_to_group = head_to_group.detach().clone()
                else:
                    group_feats, head_to_group = self._group_features(feats)

                win_logits, stride_logits = self.controller(group_feats)
                win_idx, _ = self._select_discrete(win_logits)
                if self.stride_control_enabled:
                    stride_idx, _ = self._select_discrete(stride_logits)
                else:
                    stride_idx = torch.zeros_like(win_idx)
                win_idx = win_idx[head_to_group]
                stride_idx = stride_idx[head_to_group]
                self.cached_win_idx = win_idx.detach().clone()
                self.cached_stride_idx = stride_idx.detach().clone()
                self.last_control_step = self.current_step
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
        group_heads = {}
        group_ratios = {}
        if head_to_group is not None:
            for h in range(self.n_head):
                g = int(head_to_group[h].item())
                group_heads.setdefault(g, []).append(h)
            for g, heads in group_heads.items():
                win_vals = [float(w[int(win_idx[h].item())].item()) for h in heads]
                avg_w = sum(win_vals) / max(1, len(win_vals))
                group_ratios[g] = avg_w / float(lq)
        total_time_ms = (time.perf_counter() - t_start) * 1000.0
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


class AAHV3Controller(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int, n_windows: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_windows),
        )
        self.n_windows = n_windows

    def forward(self, feats):
        return self.mlp(feats)


class AAHV3Attention(nn.Module):
    """AAH-v3: adaptive, hierarchical resolution control with variable-size supergroups."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = self.head_dim ** -0.5
        self.seq_len = config.seq_len
        self.windows = tuple(int(w) for w in config.aah_v3_windows)
        self.control_interval = max(1, int(config.aah_v3_control_interval))
        self.sim_threshold = float(config.aah_v3_sim_threshold)
        self.super_threshold = float(config.aah_v3_super_threshold)
        self.max_depth = max(1, int(config.aah_v3_max_depth))
        self.ema_alpha = float(config.aah_v3_ema_alpha)
        self.churn_penalty = float(config.aah_v3_churn_penalty)
        self.min_group_size = max(1, int(config.aah_v3_min_group_size))
        self.warmup_steps = max(0, int(config.aah_v3_warmup_steps))
        self.control_enabled = bool(config.aah_v3_control_enabled)
        self.grouping_enabled = bool(config.aah_v3_grouping_enabled)
        self.w_min_gpu = max(1, int(config.aah_v3_W_min_gpu))
        self.mask_cache_size = max(0, int(config.aah_v3_mask_cache_size))
        self.eval_mode = False
        self.cached_win_idx = None
        self.cached_head_to_group = None
        self.cached_group_feats = None
        self.cached_group_feats_step = None
        self.last_control_step = None
        self.current_step = None
        self.last_stats = {}
        self.ema_feats = None

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        self.register_buffer(
            "full_mask",
            torch.tril(torch.ones(config.seq_len, config.seq_len)).view(1, 1, config.seq_len, config.seq_len),
        )
        self.mask_cache = OrderedDict()

        self.controller = AAHV3Controller(
            feat_dim=9,
            hidden_dim=max(4, int(config.aah_v3_control_dim)),
            n_windows=len(self.windows),
        )

    def set_control(self, enabled: bool):
        self.control_enabled = bool(enabled)
        if not self.control_enabled:
            self.cached_win_idx = None
            self.cached_head_to_group = None
            self.cached_group_feats = None
            self.cached_group_feats_step = None
            self.last_control_step = None

    def reset_cache(self):
        self.cached_win_idx = None
        self.cached_head_to_group = None
        self.cached_group_feats = None
        self.cached_group_feats_step = None
        self.last_control_step = None

    def set_eval_mode(self, enabled: bool):
        self.eval_mode = bool(enabled)

    def set_step(self, step: int):
        self.current_step = int(step)

    def _should_update_control(self):
        if self.control_interval <= 1:
            return True
        if self.current_step is None:
            return True
        if self.last_control_step is None:
            return True
        return (self.current_step - self.last_control_step) >= self.control_interval

    def _head_features(self, q, k, v):
        q_mean = q.abs().mean(dim=(0, 2, 3))
        q_std = q.std(dim=(0, 2, 3))
        k_mean = k.abs().mean(dim=(0, 2, 3))
        k_std = k.std(dim=(0, 2, 3))
        v_mean = v.abs().mean(dim=(0, 2, 3))
        v_std = v.std(dim=(0, 2, 3))
        if self.last_stats:
            last_entropy = torch.tensor(self.last_stats.get("head_entropy", [0.0] * self.n_head), device=q.device)
            last_norm = torch.tensor(self.last_stats.get("head_norms", [0.0] * self.n_head), device=q.device)
            last_usage = torch.tensor(self.last_stats.get("head_usage", [0.0] * self.n_head), device=q.device)
        else:
            last_entropy = torch.zeros(self.n_head, device=q.device)
            last_norm = torch.zeros(self.n_head, device=q.device)
            last_usage = torch.zeros(self.n_head, device=q.device)
        feats = torch.stack(
            [q_mean, q_std, k_mean, k_std, v_mean, v_std, last_entropy, last_norm, last_usage],
            dim=-1,
        )
        return feats
    def _groups_from_head_to_group(self, head_to_group):
        if head_to_group is None:
            return None
        num_groups = int(head_to_group.max().item()) + 1 if head_to_group.numel() > 0 else 0
        groups = [[] for _ in range(num_groups)]
        for h in range(self.n_head):
            g = int(head_to_group[h].item())
            if g >= 0:
                groups[g].append(h)
        return groups

    def _group_features_from_qkv(self, q, k, v, groups0):
        if not groups0:
            return torch.zeros((0, 9), device=q.device, dtype=q.dtype)
        if self.last_stats:
            last_entropy = torch.tensor(self.last_stats.get("head_entropy", [0.0] * self.n_head), device=q.device)
            last_norm = torch.tensor(self.last_stats.get("head_norms", [0.0] * self.n_head), device=q.device)
            last_usage = torch.tensor(self.last_stats.get("head_usage", [0.0] * self.n_head), device=q.device)
        else:
            last_entropy = torch.zeros(self.n_head, device=q.device)
            last_norm = torch.zeros(self.n_head, device=q.device)
            last_usage = torch.zeros(self.n_head, device=q.device)
        feats = []
        for heads in groups0:
            if not heads:
                continue
            qg = q[:, heads]
            kg = k[:, heads]
            vg = v[:, heads]
            q_mean = qg.abs().mean(dim=(0, 1, 2, 3))
            q_std = qg.std(dim=(0, 1, 2, 3))
            k_mean = kg.abs().mean(dim=(0, 1, 2, 3))
            k_std = kg.std(dim=(0, 1, 2, 3))
            v_mean = vg.abs().mean(dim=(0, 1, 2, 3))
            v_std = vg.std(dim=(0, 1, 2, 3))
            h_idx = torch.tensor(heads, device=q.device, dtype=torch.long)
            ent = last_entropy[h_idx].mean()
            norm = last_norm[h_idx].mean()
            usage = last_usage[h_idx].mean()
            feats.append(torch.stack([q_mean, q_std, k_mean, k_std, v_mean, v_std, ent, norm, usage], dim=0))
        if not feats:
            return torch.zeros((0, 9), device=q.device, dtype=q.dtype)
        return torch.stack(feats, dim=0)

    def _get_cached_group_feats(self, q, k, v, groups0):
        if self.current_step is not None and self.cached_group_feats is not None and self.cached_group_feats_step is not None:
            if (self.current_step - self.cached_group_feats_step) < self.control_interval:
                return self.cached_group_feats
        group_feats = self._group_features_from_qkv(q, k, v, groups0)
        self.cached_group_feats = group_feats.detach()
        self.cached_group_feats_step = self.current_step
        return group_feats

    def _build_group_hierarchy(self, groups0, group_feats):
        levels = []
        levels.append((groups0, group_feats))
        if len(groups0) == 1:
            return levels
        feats_prev = group_feats
        for _ in range(1, self.max_depth):
            groups = self._cluster(feats_prev, self.super_threshold)
            feats = self._aggregate(groups, feats_prev)
            levels.append((groups, feats))
            if len(groups) == 1:
                break
            feats_prev = feats
        return levels

    def _cluster(self, feats, threshold, prev_groups=None):
        n = feats.size(0)
        if n == 1:
            return [[0]]
        feats = F.normalize(feats, dim=-1, eps=1e-6)
        sim = feats @ feats.transpose(0, 1)
        if prev_groups is not None and self.churn_penalty > 0:
            same_group = prev_groups.unsqueeze(0) == prev_groups.unsqueeze(1)
            sim = torch.clamp(sim + (same_group.float() * self.churn_penalty), max=1.0)
        adj = sim >= threshold
        visited = [False] * n
        groups = []
        for i in range(n):
            if visited[i]:
                continue
            stack = [i]
            visited[i] = True
            comp = [i]
            while stack:
                u = stack.pop()
                neighbors = torch.where(adj[u])[0].tolist()
                for v in neighbors:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
                        comp.append(v)
            groups.append(comp)
        if self.min_group_size > 1 and len(groups) > 1:
            groups = self._merge_small_groups(groups, feats, sim)
        return groups

    def _merge_small_groups(self, groups, feats, sim):
        groups = [list(g) for g in groups]
        small = [i for i, g in enumerate(groups) if len(g) < self.min_group_size]
        if not small:
            return groups
        group_feats = []
        for g in groups:
            group_feats.append(feats[g].mean(dim=0))
        group_feats = torch.stack(group_feats, dim=0)
        for gi in small:
            if len(groups) <= 1:
                break
            if len(groups[gi]) >= self.min_group_size:
                continue
            head_idx = groups[gi][0]
            sims = group_feats @ group_feats[gi].unsqueeze(1)
            sims = sims.squeeze(1)
            sims[gi] = -1.0
            best = int(sims.argmax().item())
            groups[best].extend(groups[gi])
            groups[gi] = []
            group_feats[best] = feats[groups[best]].mean(dim=0)
        groups = [g for g in groups if g]
        return groups

    def _aggregate(self, groups, feats):
        agg = []
        for g in groups:
            g_feats = feats[g].mean(dim=0)
            agg.append(g_feats)
        return torch.stack(agg, dim=0)

    def _build_hierarchy(self, head_feats, prev_head_groups=None):
        levels = []
        groups0 = self._cluster(head_feats, self.sim_threshold, prev_groups=prev_head_groups)
        feats0 = self._aggregate(groups0, head_feats)
        levels.append((groups0, feats0))
        if len(groups0) == 1:
            return levels
        feats_prev = feats0
        for _ in range(1, self.max_depth):
            groups = self._cluster(feats_prev, self.super_threshold)
            feats = self._aggregate(groups, feats_prev)
            levels.append((groups, feats))
            if len(groups) == 1:
                break
            feats_prev = feats
        return levels

    def _parent_maps(self, levels):
        parent_maps = []
        for level in range(len(levels) - 1):
            groups_next = levels[level + 1][0]
            num_groups = len(levels[level][0])
            parent_map = torch.empty(num_groups, dtype=torch.long, device=levels[level][1].device)
            for parent_idx, children in enumerate(groups_next):
                for child_idx in children:
                    parent_map[child_idx] = parent_idx
            parent_maps.append(parent_map)
        return parent_maps

    def _select_windows(self, levels, parent_maps, device):
        n_levels = len(levels)
        win_indices = [None] * n_levels
        top_groups, top_feats = levels[-1]
        top_logits = self.controller(top_feats)
        top_idx = top_logits.argmax(dim=-1)
        win_indices[-1] = top_idx
        for level in range(n_levels - 2, -1, -1):
            groups, feats = levels[level]
            logits = self.controller(feats)
            idx = logits.argmax(dim=-1)
            parent_map = parent_maps[level].to(device)
            parent_idx = win_indices[level + 1][parent_map]
            idx = torch.minimum(idx, parent_idx)
            win_indices[level] = idx
        return win_indices[0]

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
            return y, att, 0.0
        mask, build_ms = self._get_window_mask(T, W, q.device)
        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        return y, att, build_ms

    def _get_window_mask(self, T, W, device):
        if not isinstance(self.mask_cache, OrderedDict):
            self.mask_cache = OrderedDict(self.mask_cache)
        key = (int(T), int(W), device.type, device.index)
        cached = self.mask_cache.get(key)
        if cached is not None:
            self.mask_cache.move_to_end(key)
            return cached, 0.0
        t0 = time.perf_counter()
        q_pos = torch.arange(0, T, device=device).unsqueeze(1)
        k_pos = torch.arange(0, T, device=device).unsqueeze(0)
        mask = (k_pos <= q_pos) & (k_pos >= (q_pos - W + 1))
        build_ms = (time.perf_counter() - t0) * 1000.0
        if self.mask_cache_size > 0:
            self.mask_cache[key] = mask
            self.mask_cache.move_to_end(key)
            while len(self.mask_cache) > self.mask_cache_size:
                self.mask_cache.popitem(last=False)
        return mask, build_ms

    def forward(self, x, return_attn: bool = False):
        B, T, C = x.size()
        t_start = time.perf_counter()
        t_control0 = time.perf_counter()
        with record_function("attn_qkv"):
            qkv = self.qkv(x)
            q, k, v = qkv.split(C, dim=2)
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        in_warmup = (self.current_step is not None) and (self.current_step < self.warmup_steps)
        shadow_win_idx = None
        shadow_logit_mean = None
        t_control0 = time.perf_counter()
        if not self.control_enabled:
            t_attn0 = time.perf_counter()
            att = (q @ k.transpose(-2, -1)) * self.scale
            mask = self.full_mask[:, :, :T, :T]
            att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            y = self.resid_drop(self.proj(y))
            attn_time_ms = (time.perf_counter() - t_attn0) * 1000.0
            total_time_ms = (time.perf_counter() - t_start) * 1000.0
            mask_time_ms = 0.0
            control_time_ms = 0.0
            lk_tensor = torch.full((self.n_head,), float(T), device=x.device)
            w_tensor = lk_tensor.clone()
            lk_mean = float(lk_tensor.mean().item())
            lk_p90 = float(torch.quantile(lk_tensor, 0.9).item())
            w_mean = float(w_tensor.mean().item())
            w_min = float(w_tensor.min().item())
            w_max = float(w_tensor.max().item())
            self.last_stats = {
                "lq": T,
                "lk": [T for _ in range(self.n_head)],
                "total_elements": float(self.n_head * T * T),
                "baseline_elements": float(self.n_head * T * T),
                "head_norms": [],
                "head_entropy": [],
                "head_usage": [],
                "avg_window": float(T),
                "group_change_rate": None,
                "head_groups": [],
                "shadow_win_idx": [],
                "shadow_logit_mean": [],
                "group_heads": {},
                "group_ratios": {},
                "control_time_ms": 0.0,
                "attn_time_ms": attn_time_ms,
                "overhead_time_ms": max(0.0, total_time_ms - attn_time_ms - mask_time_ms),
                "mask_time_ms": mask_time_ms,
                "lk_mean": lk_mean,
                "lk_p90": lk_p90,
                "w_mean": w_mean,
                "w_min": w_min,
                "w_max": w_max,
            }
            if return_attn:
                return y, att
            return y
        elif self.control_enabled and not in_warmup:
            use_cache = (not self._should_update_control()) and self.cached_win_idx is not None
            if use_cache:
                win_idx = self.cached_win_idx
                head_to_group = self.cached_head_to_group
                group_change_rate = 0.0
            else:
                if self.cached_head_to_group is not None and self.grouping_enabled:
                    head_to_group = self.cached_head_to_group
                    groups0 = self._groups_from_head_to_group(head_to_group)
                    group_feats = self._get_cached_group_feats(q, k, v, groups0)
                    levels = self._build_group_hierarchy(groups0, group_feats)
                    parent_maps = self._parent_maps(levels)
                    group_win_idx = self._select_windows(levels, parent_maps, device=x.device)
                    win_idx = group_win_idx[head_to_group]
                    group_change_rate = 0.0
                else:
                    feats = self._head_features(q, k, v)
                    if self.ema_feats is None or self.ema_feats.shape != feats.shape:
                        self.ema_feats = feats.detach()
                    else:
                        alpha = max(0.0, min(1.0, self.ema_alpha))
                        self.ema_feats = alpha * self.ema_feats + (1.0 - alpha) * feats.detach()
                    prev_groups = None
                    if self.cached_head_to_group is not None:
                        prev_groups = self.cached_head_to_group
                    levels = self._build_hierarchy(self.ema_feats, prev_head_groups=prev_groups)
                    parent_maps = self._parent_maps(levels)
                    group_win_idx = self._select_windows(levels, parent_maps, device=x.device)
                    groups0 = levels[0][0]
                    head_to_group = torch.empty(self.n_head, dtype=torch.long, device=x.device)
                    for gi, heads in enumerate(groups0):
                        for h in heads:
                            head_to_group[h] = gi
                    win_idx = group_win_idx[head_to_group]
                    if self.cached_head_to_group is None:
                        group_change_rate = 0.0
                    else:
                        group_change_rate = (head_to_group != self.cached_head_to_group).float().mean().item()
                    self.cached_head_to_group = head_to_group.detach().clone()
                    self.cached_group_feats = None
                    self.cached_group_feats_step = None
                self.cached_win_idx = win_idx.detach().clone()
                self.last_control_step = self.current_step
        else:
            head_to_group = None
            group_change_rate = None
            if self.grouping_enabled:
                if self.cached_head_to_group is not None:
                    head_to_group = self.cached_head_to_group
                    groups0 = self._groups_from_head_to_group(head_to_group)
                    group_feats = self._get_cached_group_feats(q, k, v, groups0)
                    levels = self._build_group_hierarchy(groups0, group_feats)
                    parent_maps = self._parent_maps(levels)
                    group_win_idx = self._select_windows(levels, parent_maps, device=x.device)
                    shadow_win_idx = group_win_idx[head_to_group]
                    logits = self.controller(levels[0][1])
                    shadow_logit_mean = logits.mean(dim=0).detach().cpu().tolist()
                    group_change_rate = 0.0
                else:
                    feats = self._head_features(q, k, v)
                    if self.ema_feats is None or self.ema_feats.shape != feats.shape:
                        self.ema_feats = feats.detach()
                    else:
                        alpha = max(0.0, min(1.0, self.ema_alpha))
                        self.ema_feats = alpha * self.ema_feats + (1.0 - alpha) * feats.detach()
                    prev_groups = None
                    if self.cached_head_to_group is not None:
                        prev_groups = self.cached_head_to_group
                    levels = self._build_hierarchy(self.ema_feats, prev_head_groups=prev_groups)
                    parent_maps = self._parent_maps(levels)
                    group_win_idx = self._select_windows(levels, parent_maps, device=x.device)
                    groups0 = levels[0][0]
                    head_to_group = torch.empty(self.n_head, dtype=torch.long, device=x.device)
                    for gi, heads in enumerate(groups0):
                        for h in heads:
                            head_to_group[h] = gi
                    shadow_win_idx = group_win_idx[head_to_group]
                    logits = self.controller(levels[0][1])
                    shadow_logit_mean = logits.mean(dim=0).detach().cpu().tolist()
                    if self.cached_head_to_group is None:
                        group_change_rate = 0.0
                    else:
                        group_change_rate = (head_to_group != self.cached_head_to_group).float().mean().item()
                    self.cached_head_to_group = head_to_group.detach().clone()
                    self.cached_group_feats = None
                    self.cached_group_feats_step = None
            win_idx = torch.full((self.n_head,), 0, device=x.device, dtype=torch.long)
            if T in self.windows:
                win_full = self.windows.index(T)
                win_idx = torch.full((self.n_head,), win_full, device=x.device, dtype=torch.long)
        control_time_ms = (time.perf_counter() - t_control0) * 1000.0 if self.control_enabled else 0.0

        lq = T
        window_vals = torch.tensor(self.windows, device=x.device, dtype=torch.float32)
        w = torch.clamp(window_vals[win_idx], min=float(self.w_min_gpu), max=float(T))
        lk_tensor = torch.clamp(w, min=1.0, max=float(T))
        total_elements = float((lq * lk_tensor).sum().item())
        baseline_elements = float(self.n_head * lq * lq)

        outputs = torch.zeros(B, self.n_head, T, self.head_dim, device=x.device, dtype=q.dtype)
        head_norms = [0.0 for _ in range(self.n_head)]
        head_entropy = [0.0 for _ in range(self.n_head)]
        head_usage = [0.0 for _ in range(self.n_head)]
        attn_maps = [None for _ in range(self.n_head)] if return_attn else None
        t_attn0 = time.perf_counter()
        mask_time_ms = 0.0

        head_groups = {}
        for h in range(self.n_head):
            window = self.windows[int(win_idx[h].item())]
            head_groups.setdefault(window, []).append(h)

        for window, heads in head_groups.items():
            q_sel = q[:, heads]
            k_sel = k[:, heads]
            v_sel = v[:, heads]
            Bc, Hc, Tc, Dc = q_sel.shape
            qf = q_sel.reshape(Bc * Hc, Tc, Dc)
            kf = k_sel.reshape(Bc * Hc, Tc, Dc)
            vf = v_sel.reshape(Bc * Hc, Tc, Dc)
            y_f, att_f, mask_ms = self._local_attention(qf, kf, vf, window)
            mask_time_ms += mask_ms
            y_h = y_f.view(Bc, Hc, Tc, Dc)
            outputs[:, heads] = y_h
            norms = y_h.norm(dim=-1).mean(dim=(0, 2))
            for i, h in enumerate(heads):
                head_norms[h] = float(norms[i].item())
            att_h = att_f.view(Bc, Hc, att_f.size(1), att_f.size(2))
            ent = -(att_h * (att_h + 1e-9).log()).sum(dim=-1).mean(dim=(0, 2))
            ent = ent / max(1.0, math.log(att_h.size(-1)))
            usage = att_h[..., -1].mean(dim=(0, 2))
            for i, h in enumerate(heads):
                head_entropy[h] = float(ent[i].item())
                head_usage[h] = float(usage[i].item())
            if return_attn:
                for i, h in enumerate(heads):
                    attn_maps[h] = att_h[:, i:i+1]
        attn_time_ms = (time.perf_counter() - t_attn0) * 1000.0

        y = outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        avg_window = float(w.mean().item())
        lk_list = lk_tensor.to(torch.int64).tolist()
        total_time_ms = (time.perf_counter() - t_start) * 1000.0
        lk_mean = float(lk_tensor.mean().item())
        lk_p90 = float(torch.quantile(lk_tensor, 0.9).item())
        w_mean = float(w.mean().item())
        w_min = float(w.min().item())
        w_max = float(w.max().item())
        group_heads = {}
        group_ratios = {}
        if head_to_group is not None:
            for h in range(self.n_head):
                g = int(head_to_group[h].item())
                group_heads.setdefault(g, []).append(h)
            for g, heads in group_heads.items():
                win_vals = [float(w[int(win_idx[h].item())].item()) for h in heads]
                avg_w = sum(win_vals) / max(1, len(win_vals))
                group_ratios[g] = avg_w / float(lq)
        self.last_stats = {
            "lq": lq,
            "lk": lk_list,
            "total_elements": total_elements,
            "baseline_elements": baseline_elements,
            "head_norms": head_norms,
            "head_entropy": head_entropy,
            "head_usage": head_usage,
            "avg_window": avg_window,
            "group_change_rate": group_change_rate,
            "head_groups": head_to_group.detach().cpu().tolist() if head_to_group is not None else [],
            "shadow_win_idx": shadow_win_idx.detach().cpu().tolist() if shadow_win_idx is not None else [],
            "shadow_logit_mean": shadow_logit_mean if shadow_logit_mean is not None else [],
            "group_heads": group_heads,
            "group_ratios": group_ratios,
            "control_time_ms": control_time_ms,
            "attn_time_ms": attn_time_ms,
            "overhead_time_ms": max(0.0, total_time_ms - attn_time_ms),
            "lk_mean": lk_mean,
            "lk_p90": lk_p90,
            "w_mean": w_mean,
            "w_min": w_min,
            "w_max": w_max,
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
        if config.aah_v3_enabled:
            self.attn = AAHV3Attention(config)
        elif config.aah_v2_enabled:
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
