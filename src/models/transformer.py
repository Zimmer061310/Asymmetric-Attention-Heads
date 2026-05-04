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
    aah_v3_build_hierarchy: bool = True
    aah_v3_apply_window_control: bool = True
    aah_v3_W_min_gpu: int = 64
    aah_v3_mask_cache_size: int = 16
    aah_v3_resolution_ema_alpha: float = 0.0
    aah_v3_resolution_collapse_min_frac: float = 0.95
    aah_v3_resolution_collapse_max_frac: float = 0.95
    aah_v3_post_warmup_ramp_steps: int = 0
    aah_v3_group_feature_mode: str = "mean"
    aah_v3_upper_cluster_metric: str = "cosine"
    aah_v3_upper_l2_threshold: float = 0.0
    aah_v3_cosine_normdiff_scale: float = 16.0


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
        q_f = q.float()
        k_f = k.float()
        v_f = v.float()
        q_mean = q_f.abs().mean(dim=(0, 2, 3))
        q_std = q_f.std(dim=(0, 2, 3))
        k_mean = k_f.abs().mean(dim=(0, 2, 3))
        k_std = k_f.std(dim=(0, 2, 3))
        v_mean = v_f.abs().mean(dim=(0, 2, 3))
        v_std = v_f.std(dim=(0, 2, 3))
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
        if not self._logged_attn_dtype:
            self._logged_attn_dtype = True
            print(f"AAHV3Attention q dtype: {q.dtype}")

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
            norms = y_h.float().norm(dim=-1).mean(dim=(0, 2))
            for i, h in enumerate(heads):
                head_norms[h] = float(norms[i].item())
            att_h = att_f.view(Bc, Hc, att_f.size(1), att_f.size(2))
            att_h_f = att_h.float()
            ent = -(att_h_f * (att_h_f + 1e-9).log()).sum(dim=-1).mean(dim=(0, 2))
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
        self.build_hierarchy = bool(getattr(config, "aah_v3_build_hierarchy", config.aah_v3_grouping_enabled))
        self.apply_window_control = bool(getattr(config, "aah_v3_apply_window_control", config.aah_v3_control_enabled))
        # Backward-compatible internal aliases for older code paths/configs.
        self.control_enabled = self.apply_window_control
        self.grouping_enabled = self.build_hierarchy
        self.w_min_gpu = max(1, int(config.aah_v3_W_min_gpu))
        self.mask_cache_size = max(0, int(config.aah_v3_mask_cache_size))
        self.resolution_ema_alpha = float(config.aah_v3_resolution_ema_alpha)
        self.resolution_collapse_min_frac = float(config.aah_v3_resolution_collapse_min_frac)
        self.resolution_collapse_max_frac = float(config.aah_v3_resolution_collapse_max_frac)
        self.post_warmup_ramp_steps = max(0, int(config.aah_v3_post_warmup_ramp_steps))
        self.group_feature_mode = str(getattr(config, "aah_v3_group_feature_mode", "mean"))
        self.upper_cluster_metric = str(getattr(config, "aah_v3_upper_cluster_metric", "cosine"))
        self.upper_l2_threshold = float(getattr(config, "aah_v3_upper_l2_threshold", 0.0))
        self.cosine_normdiff_scale = float(getattr(config, "aah_v3_cosine_normdiff_scale", 16.0))
        self.eval_mode = False
        self.cached_win_idx = None
        self.cached_head_to_group = None
        self.cached_group_feats = None
        self.cached_group_feats_step = None
        self.cached_group_counts_per_level = None
        self.last_control_step = None
        self.current_step = None
        self.last_stats = {}
        self.ema_feats = None
        self.ema_win_idx = None
        self._logged_attn_dtype = False

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
            self.cached_group_counts_per_level = None
            self.last_control_step = None
            self.ema_win_idx = None

    def reset_cache(self):
        self.cached_win_idx = None
        self.cached_head_to_group = None
        self.cached_group_feats = None
        self.cached_group_feats_step = None
        self.cached_group_counts_per_level = None
        self.last_control_step = None
        self.ema_win_idx = None

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
        q_f = q.float()
        k_f = k.float()
        v_f = v.float()
        q_mean = q_f.abs().mean(dim=(0, 2, 3))
        q_std = q_f.std(dim=(0, 2, 3))
        k_mean = k_f.abs().mean(dim=(0, 2, 3))
        k_std = k_f.std(dim=(0, 2, 3))
        v_mean = v_f.abs().mean(dim=(0, 2, 3))
        v_std = v_f.std(dim=(0, 2, 3))
        if self.last_stats:
            last_entropy = torch.tensor(self.last_stats.get("head_entropy", [0.0] * self.n_head), device=q.device, dtype=torch.float32)
            last_norm = torch.tensor(self.last_stats.get("head_norms", [0.0] * self.n_head), device=q.device, dtype=torch.float32)
            last_usage = torch.tensor(self.last_stats.get("head_usage", [0.0] * self.n_head), device=q.device, dtype=torch.float32)
        else:
            last_entropy = torch.zeros(self.n_head, device=q.device, dtype=torch.float32)
            last_norm = torch.zeros(self.n_head, device=q.device, dtype=torch.float32)
            last_usage = torch.zeros(self.n_head, device=q.device, dtype=torch.float32)
        feats = torch.stack(
            [q_mean, q_std, k_mean, k_std, v_mean, v_std, last_entropy, last_norm, last_usage],
            dim=-1,
        )
        return feats.float()
    def _feature_separability_stats(self, feats):
        if feats is None or feats.numel() == 0:
            return {
                "feature_dim_var_mean": 0.0,
                "feature_dim_var_std": 0.0,
                "feature_dim_var_min": 0.0,
                "feature_dim_var_max": 0.0,
                "feature_cos_sim_mean": 1.0,
                "feature_cos_sim_std": 0.0,
                "feature_cos_sim_min": 1.0,
                "feature_cos_sim_max": 1.0,
                "feature_l2_dist_mean": 0.0,
                "feature_l2_dist_std": 0.0,
                "feature_l2_dist_min": 0.0,
                "feature_l2_dist_max": 0.0,
                "feature_norm_mean": 0.0,
                "feature_norm_std": 0.0,
                "feature_top_singular_ratio": 1.0,
                "hierarchy_head_group_map_per_level": [],
                "hierarchy_group_members_per_level": [],
            }
        f = feats.float()
        n = f.size(0)
        dim_var = f.var(dim=0, unbiased=False)
        norms = f.norm(dim=-1)
        if n > 1:
            f_norm = F.normalize(f, dim=-1, eps=1e-6)
            sim = f_norm @ f_norm.transpose(0, 1)
            dist = torch.cdist(f, f, p=2)
            offdiag_mask = ~torch.eye(n, dtype=torch.bool, device=f.device)
            sim_offdiag = sim[offdiag_mask]
            dist_offdiag = dist[offdiag_mask]
            sim_mean = float(sim_offdiag.mean().item()) if sim_offdiag.numel() > 0 else 1.0
            sim_std = float(sim_offdiag.std(unbiased=False).item()) if sim_offdiag.numel() > 1 else 0.0
            sim_min = float(sim_offdiag.min().item()) if sim_offdiag.numel() > 0 else 1.0
            sim_max = float(sim_offdiag.max().item()) if sim_offdiag.numel() > 0 else 1.0
            l2_mean = float(dist_offdiag.mean().item()) if dist_offdiag.numel() > 0 else 0.0
            l2_std = float(dist_offdiag.std(unbiased=False).item()) if dist_offdiag.numel() > 1 else 0.0
            l2_min = float(dist_offdiag.min().item()) if dist_offdiag.numel() > 0 else 0.0
            l2_max = float(dist_offdiag.max().item()) if dist_offdiag.numel() > 0 else 0.0
        else:
            sim_mean, sim_std, sim_min, sim_max = 1.0, 0.0, 1.0, 1.0
            l2_mean, l2_std, l2_min, l2_max = 0.0, 0.0, 0.0, 0.0
        f_centered = f - f.mean(dim=0, keepdim=True)
        total_var = float((f_centered.pow(2).sum()).item())
        if n > 1 and total_var > 0.0:
            svals = torch.linalg.svdvals(f_centered)
            top_singular_ratio = float((svals[0].pow(2) / (svals.pow(2).sum() + 1e-9)).item())
        else:
            top_singular_ratio = 1.0
        return {
            "feature_dim_var_mean": float(dim_var.mean().item()) if dim_var.numel() > 0 else 0.0,
            "feature_dim_var_std": float(dim_var.std(unbiased=False).item()) if dim_var.numel() > 1 else 0.0,
            "feature_dim_var_min": float(dim_var.min().item()) if dim_var.numel() > 0 else 0.0,
            "feature_dim_var_max": float(dim_var.max().item()) if dim_var.numel() > 0 else 0.0,
            "feature_cos_sim_mean": sim_mean,
            "feature_cos_sim_std": sim_std,
            "feature_cos_sim_min": sim_min,
            "feature_cos_sim_max": sim_max,
            "feature_l2_dist_mean": l2_mean,
            "feature_l2_dist_std": l2_std,
            "feature_l2_dist_min": l2_min,
            "feature_l2_dist_max": l2_max,
            "feature_norm_mean": float(norms.mean().item()) if norms.numel() > 0 else 0.0,
            "feature_norm_std": float(norms.std(unbiased=False).item()) if norms.numel() > 1 else 0.0,
            "feature_top_singular_ratio": top_singular_ratio,
        }
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
    def _head_group_map_per_level(self, levels):
        if not levels:
            return []
        groups0 = levels[0][0]
        level0 = torch.full((self.n_head,), -1, dtype=torch.long)
        for gi, heads in enumerate(groups0):
            for h in heads:
                level0[h] = gi
        maps = [level0.tolist()]
        prev_map = level0
        for level in range(1, len(levels)):
            prev_group_count = len(levels[level - 1][0])
            groups_cur = levels[level][0]
            parent_of_child = torch.full((prev_group_count,), -1, dtype=torch.long)
            for parent_idx, children in enumerate(groups_cur):
                for child_idx in children:
                    parent_of_child[child_idx] = parent_idx
            prev_map = parent_of_child[prev_map]
            maps.append(prev_map.tolist())
        return maps
    def _group_members_from_head_maps(self, head_group_map_per_level):
        members = []
        for head_map in head_group_map_per_level:
            by_group = {}
            for h, g in enumerate(head_map):
                g = int(g)
                by_group.setdefault(g, []).append(int(h))
            members.append([by_group[k] for k in sorted(by_group.keys())])
        return members

    def _group_features_from_qkv(self, q, k, v, groups0):
        if not groups0:
            return torch.zeros((0, 9), device=q.device, dtype=torch.float32)
        if self.last_stats:
            last_entropy = torch.tensor(self.last_stats.get("head_entropy", [0.0] * self.n_head), device=q.device, dtype=torch.float32)
            last_norm = torch.tensor(self.last_stats.get("head_norms", [0.0] * self.n_head), device=q.device, dtype=torch.float32)
            last_usage = torch.tensor(self.last_stats.get("head_usage", [0.0] * self.n_head), device=q.device, dtype=torch.float32)
        else:
            last_entropy = torch.zeros(self.n_head, device=q.device, dtype=torch.float32)
            last_norm = torch.zeros(self.n_head, device=q.device, dtype=torch.float32)
            last_usage = torch.zeros(self.n_head, device=q.device, dtype=torch.float32)
        feats = []
        for heads in groups0:
            if not heads:
                continue
            qg = q[:, heads].float()
            kg = k[:, heads].float()
            vg = v[:, heads].float()
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
            return torch.zeros((0, 9), device=q.device, dtype=torch.float32)
        return torch.stack(feats, dim=0).float()

    def _get_cached_group_feats(self, q, k, v, groups0):
        if self.current_step is not None and self.cached_group_feats is not None and self.cached_group_feats_step is not None:
            if (self.current_step - self.cached_group_feats_step) < self.control_interval:
                return self.cached_group_feats
        group_feats = self._group_features_from_qkv(q, k, v, groups0)
        self.cached_group_feats = group_feats.detach()
        self.cached_group_feats_step = self.current_step
        return group_feats

    def _build_group_hierarchy(self, groups0, group_feats, return_debug: bool = False):
        levels = []
        cluster_debug = []
        levels.append((groups0, group_feats))
        if len(groups0) == 1:
            if return_debug:
                return levels, cluster_debug
            return levels
        feats_prev = group_feats
        for level_idx in range(1, self.max_depth):
            groups, dbg = self._cluster(feats_prev, self.super_threshold, return_debug=True, allow_forced_bipartition=False, metric=self.upper_cluster_metric)
            dbg["level_idx"] = level_idx
            dbg["threshold_kind"] = "super_threshold"
            if len(groups) == 1:
                dbg["hierarchy_level_added"] = False
                dbg["hierarchy_growth_stopped"] = True
                dbg["hierarchy_stop_reason"] = "upper_level_collapsed"
                cluster_debug.append(dbg)
                break
            dbg["hierarchy_level_added"] = True
            dbg["hierarchy_growth_stopped"] = False
            dbg["hierarchy_stop_reason"] = ""
            cluster_debug.append(dbg)
            feats = self._aggregate(groups, feats_prev)
            levels.append((groups, feats))
            feats_prev = feats
        if return_debug:
            return levels, cluster_debug
        return levels
    def _cluster(self, feats, threshold, prev_groups=None, return_debug: bool = False, allow_forced_bipartition: bool = True, metric: str = "cosine"):
        n = feats.size(0)
        feature_stats = self._feature_separability_stats(feats)
        metric = str(metric)
        if metric == "l2":
            threshold = float(self.upper_l2_threshold)
        if n == 1:
            groups = [[0]]
            if return_debug:
                return groups, {
                    "n_items": 1,
                    "threshold": float(threshold),
                    "cluster_metric": metric,
                    "sim_mean": 1.0,
                    "sim_std": 0.0,
                    "sim_min": 1.0,
                    "sim_max": 1.0,
                    "groups_before_merge": 1,
                    "groups_after_merge": 1,
                    "groups_merged": 0,
                    "small_groups_before_merge": 0,
                    "singleton_groups_before_merge": 0,
                    "min_group_size": int(self.min_group_size),
                    "groups_before_force": 1,
                    "forced_bipartition": False,
                    "force_split_anchor_similarity": None,
                    "cluster_origin": "single_item",
                    "forced_bipartition_allowed": bool(allow_forced_bipartition),
                    **feature_stats,
                }
            return groups

        raw_feats = feats.float()
        if metric == "centered_cosine":
            score_feats = F.normalize(raw_feats - raw_feats.mean(dim=0, keepdim=True), dim=-1, eps=1e-6)
            sim = score_feats @ score_feats.transpose(0, 1)
        elif metric == "l2":
            score_feats = raw_feats
            sim = -torch.cdist(raw_feats, raw_feats, p=2)
        elif metric == "cosine_normdiff":
            score_feats = F.normalize(raw_feats, dim=-1, eps=1e-6)
            cos = score_feats @ score_feats.transpose(0, 1)
            norms = raw_feats.norm(dim=-1)
            norm_den = torch.maximum(norms.unsqueeze(0), norms.unsqueeze(1)).clamp_min(1e-6)
            norm_diff = (norms.unsqueeze(0) - norms.unsqueeze(1)).abs() / norm_den
            sim = cos - (self.cosine_normdiff_scale * norm_diff)
        else:
            metric = "cosine"
            score_feats = F.normalize(raw_feats, dim=-1, eps=1e-6)
            sim = score_feats @ score_feats.transpose(0, 1)
        offdiag_mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
        sim_offdiag = sim[offdiag_mask]
        sim_mean = float(sim_offdiag.mean().item()) if sim_offdiag.numel() > 0 else 1.0
        sim_std = float(sim_offdiag.std(unbiased=False).item()) if sim_offdiag.numel() > 1 else 0.0
        sim_min = float(sim_offdiag.min().item()) if sim_offdiag.numel() > 0 else 1.0
        sim_max = float(sim_offdiag.max().item()) if sim_offdiag.numel() > 0 else 1.0
        if metric == "cosine" and prev_groups is not None and self.churn_penalty > 0:
            same_group = prev_groups.unsqueeze(0) == prev_groups.unsqueeze(1)
            sim = torch.clamp(sim + (same_group.float() * self.churn_penalty), max=1.0)
        # Use centroid-threshold assignment instead of transitive connected-components.
        # Connected-components on a thresholded graph behaves like single-linkage and can
        # collapse into one giant group via similarity chains even when structure exists.
        groups = []
        centroids = []
        for i in range(n):
            fi = score_feats[i]
            if not centroids:
                groups.append([i])
                centroids.append(fi.clone())
                continue
            c_stack = torch.stack(centroids, dim=0)
            if metric == "l2":
                sims = -torch.cdist(c_stack, fi.unsqueeze(0), p=2).squeeze(1)
            elif metric == "cosine_normdiff":
                cos = c_stack @ fi.unsqueeze(1)
                cos = cos.squeeze(1)
                c_norms = torch.stack([raw_feats[g].mean(dim=0).norm() for g in groups], dim=0)
                fi_norm = raw_feats[i].norm()
                norm_den = torch.maximum(c_norms, fi_norm.expand_as(c_norms)).clamp_min(1e-6)
                norm_diff = (c_norms - fi_norm).abs() / norm_den
                sims = cos - (self.cosine_normdiff_scale * norm_diff)
            else:
                sims = (c_stack @ fi.unsqueeze(1)).squeeze(1)
            best_sim, best_idx = torch.max(sims, dim=0)
            if float(best_sim.item()) >= float(threshold):
                gi = int(best_idx.item())
                groups[gi].append(i)
                # running centroid update
                centroids[gi] = score_feats[groups[gi]].mean(dim=0)
                if metric != "l2":
                    centroids[gi] = F.normalize(centroids[gi], dim=0, eps=1e-6)
            else:
                groups.append([i])
                centroids.append(fi.clone())
        groups_before_force = len(groups)
        forced_bipartition = False
        force_split_anchor_similarity = None
        cluster_origin = "natural"
        if len(groups) == 1 and n >= 2 and allow_forced_bipartition:
            groups, force_split_anchor_similarity = self._force_bipartition_groups(score_feats, sim)
            forced_bipartition = len(groups) > 1
            cluster_origin = "forced_bipartition" if forced_bipartition else "collapsed"
        elif len(groups) == 1 and n >= 2:
            cluster_origin = "collapsed"
        groups_before_merge = len(groups)
        small_groups_before_merge = sum(1 for g in groups if len(g) < self.min_group_size)
        singleton_groups_before_merge = sum(1 for g in groups if len(g) == 1)
        if self.min_group_size > 1 and len(groups) > 1 and not forced_bipartition:
            groups = self._merge_small_groups(groups, score_feats, sim, threshold)
        groups_after_merge = len(groups)
        if return_debug:
            return groups, {
                "n_items": int(n),
                "threshold": float(threshold),
                "cluster_metric": metric,
                "sim_mean": sim_mean,
                "sim_std": sim_std,
                "sim_min": sim_min,
                "sim_max": sim_max,
                "groups_before_merge": int(groups_before_merge),
                "groups_after_merge": int(groups_after_merge),
                "groups_merged": int(max(0, groups_before_merge - groups_after_merge)),
                "small_groups_before_merge": int(small_groups_before_merge),
                "singleton_groups_before_merge": int(singleton_groups_before_merge),
                "min_group_size": int(self.min_group_size),
                "groups_before_force": int(groups_before_force),
                "forced_bipartition": bool(forced_bipartition),
                "force_split_anchor_similarity": float(force_split_anchor_similarity) if force_split_anchor_similarity is not None else None,
                "cluster_origin": str(cluster_origin),
                "forced_bipartition_allowed": bool(allow_forced_bipartition),
                **feature_stats,
            }
        return groups

    def _force_bipartition_groups(self, feats, sim):
        n = feats.shape[0]
        if n < 2:
            return [[i for i in range(n)]], None

        sim_flat = sim.clone()
        sim_flat.fill_diagonal_(2.0)
        min_idx = int(torch.argmin(sim_flat).item())
        i0 = min_idx // n
        i1 = min_idx % n
        if i0 == i1:
            i1 = (i0 + 1) % n

        a0 = feats[i0]
        a1 = feats[i1]
        s0 = feats @ a0
        s1 = feats @ a1

        g0, g1 = [i0], [i1]
        for j in range(n):
            if j == i0 or j == i1:
                continue
            d = float(s0[j] - s1[j])
            if d > 1e-8:
                g0.append(j)
            elif d < -1e-8:
                g1.append(j)
            else:
                if len(g0) <= len(g1):
                    g0.append(j)
                else:
                    g1.append(j)

        if len(g0) == 0 or len(g1) == 0:
            order = torch.argsort((s0 - s1), descending=True).tolist()
            half = n // 2
            g0 = order[:half]
            g1 = order[half:]
            if len(g0) == 0:
                g0 = [g1.pop()]
            if len(g1) == 0:
                g1 = [g0.pop()]

        anchor_sim = float(sim[i0, i1].item())
        return [g0, g1], anchor_sim

    def _merge_small_groups(self, groups, feats, sim, threshold):
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
            best_sim, best_idx = torch.max(sims, dim=0)
            best = int(best_idx.item())
            if float(best_sim.item()) < float(threshold):
                continue
            if len([g for g in groups if len(g) > 0]) <= 2:
                continue
            groups[best].extend(groups[gi])
            groups[gi] = []
            group_feats[best] = feats[groups[best]].mean(dim=0)
        groups = [g for g in groups if g]
        return groups

    def _aggregate(self, groups, feats):
        agg = []
        global_mean = feats.mean(dim=0) if feats.numel() > 0 else None
        mode = self.group_feature_mode
        for g in groups:
            child_feats = feats[g]
            mean = child_feats.mean(dim=0)
            if mode == "mean":
                g_feats = mean
            elif mode == "mean_std":
                spread = child_feats.std(dim=0, unbiased=False) if child_feats.size(0) > 1 else torch.zeros_like(mean)
                g_feats = mean + spread
            elif mode == "mean_var":
                spread = child_feats.var(dim=0, unbiased=False) if child_feats.size(0) > 1 else torch.zeros_like(mean)
                g_feats = mean + spread
            elif mode == "mean_global_offset":
                g_feats = mean + (mean - global_mean)
            else:
                g_feats = mean
            agg.append(g_feats)
        return torch.stack(agg, dim=0)

    def _build_hierarchy(self, head_feats, prev_head_groups=None, return_debug: bool = False):
        levels = []
        cluster_debug = []
        groups0, dbg0 = self._cluster(head_feats, self.sim_threshold, prev_groups=prev_head_groups, return_debug=True, allow_forced_bipartition=True)
        dbg0["level_idx"] = 0
        dbg0["threshold_kind"] = "sim_threshold"
        dbg0["hierarchy_level_added"] = True
        dbg0["hierarchy_growth_stopped"] = len(groups0) == 1
        dbg0["hierarchy_stop_reason"] = "level0_collapsed" if len(groups0) == 1 else ""
        cluster_debug.append(dbg0)
        feats0 = self._aggregate(groups0, head_feats)
        levels.append((groups0, feats0))
        if len(groups0) == 1:
            if return_debug:
                return levels, cluster_debug
            return levels
        feats_prev = feats0
        for level_idx in range(1, self.max_depth):
            groups, dbg = self._cluster(feats_prev, self.super_threshold, return_debug=True, allow_forced_bipartition=False, metric=self.upper_cluster_metric)
            dbg["level_idx"] = level_idx
            dbg["threshold_kind"] = "super_threshold"
            if len(groups) == 1:
                dbg["hierarchy_level_added"] = False
                dbg["hierarchy_growth_stopped"] = True
                dbg["hierarchy_stop_reason"] = "upper_level_collapsed"
                cluster_debug.append(dbg)
                break
            dbg["hierarchy_level_added"] = True
            dbg["hierarchy_growth_stopped"] = False
            dbg["hierarchy_stop_reason"] = ""
            cluster_debug.append(dbg)
            feats = self._aggregate(groups, feats_prev)
            levels.append((groups, feats))
            feats_prev = feats
        if return_debug:
            return levels, cluster_debug
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

    def _select_windows(self, levels, parent_maps, device, return_debug: bool = False):
        n_levels = len(levels)
        win_indices = [None] * n_levels
        _, top_feats = levels[-1]
        top_logits = self.controller(top_feats.float()).float()
        top_idx = top_logits.argmax(dim=-1)
        win_indices[-1] = top_idx
        logits_std_per_level = [float(top_logits.std(unbiased=False).item())]
        pre_clamp_level0 = top_idx.detach().clone()
        post_clamp_level0 = top_idx.detach().clone()
        for level in range(n_levels - 2, -1, -1):
            _, feats = levels[level]
            logits = self.controller(feats.float()).float()
            idx = logits.argmax(dim=-1)
            idx_pre = idx.detach().clone()
            logits_std_per_level.insert(0, float(logits.std(unbiased=False).item()))
            parent_map = parent_maps[level].to(device)
            parent_idx = win_indices[level + 1][parent_map]
            idx = torch.minimum(idx, parent_idx)
            if level == 0:
                pre_clamp_level0 = idx_pre
                post_clamp_level0 = idx.detach().clone()
            win_indices[level] = idx
        if return_debug:
            return win_indices[0], {
                "pre_clamp_level0": pre_clamp_level0,
                "post_clamp_level0": post_clamp_level0,
                "logits_std_per_level": logits_std_per_level,
            }
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
        resolution_delta = None
        win_idx_pre_clamp = None
        win_idx_post_clamp = None
        hierarchy_levels_used = 0
        group_counts_per_level = []
        controller_logits_std_per_level = []
        path_mode = "unknown"
        cluster_debug = []
        feature_probe_stats = {}
        hierarchy_head_group_map_per_level = []
        hierarchy_group_members_per_level = []
        t_control0 = time.perf_counter()
        if not self.apply_window_control and not self.build_hierarchy:
            path_mode = "full_attention_fastpath"
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
                "resolution_per_head": [float(T) for _ in range(self.n_head)],
                "resolution_mean": float(T),
                "resolution_std": 0.0,
                "resolution_min_frac": 0.0,
                "resolution_max_frac": 1.0,
                "resolution_collapse_min": False,
                "resolution_collapse_max": True,
                "resolution_delta": 0.0,
                "branch_usage_freq": {int(T): 1.0},
                "hierarchy_levels_used": 0,
                "group_counts_per_level": [],
                "controller_logits_std_per_level": [],
                "path_mode": path_mode,
                "win_idx_pre_clamp": [],
                "win_idx_post_clamp": [],
                "cluster_threshold_kind_per_level": [],
                "cluster_threshold_per_level": [],
                "cluster_item_count_per_level": [],
                "cluster_groups_before_merge_per_level": [],
                "cluster_groups_after_merge_per_level": [],
                "cluster_groups_merged_per_level": [],
                "cluster_small_groups_before_merge_per_level": [],
                "cluster_singletons_before_merge_per_level": [],
                "cluster_sim_mean_per_level": [],
                "cluster_sim_std_per_level": [],
                "cluster_sim_min_per_level": [],
                "cluster_sim_max_per_level": [],
                "cluster_min_group_size": int(self.min_group_size),
                "cluster_sim_threshold": float(self.sim_threshold),
                "cluster_super_threshold": float(self.super_threshold),
                "group_feature_mode": self.group_feature_mode,
                "feature_dim_var_mean": 0.0,
                "feature_dim_var_std": 0.0,
                "feature_dim_var_min": 0.0,
                "feature_dim_var_max": 0.0,
                "feature_cos_sim_mean": 1.0,
                "feature_cos_sim_std": 0.0,
                "feature_cos_sim_min": 1.0,
                "feature_cos_sim_max": 1.0,
                "feature_l2_dist_mean": 0.0,
                "feature_l2_dist_std": 0.0,
                "feature_l2_dist_min": 0.0,
                "feature_l2_dist_max": 0.0,
                "feature_norm_mean": 0.0,
                "feature_norm_std": 0.0,
                "feature_top_singular_ratio": 1.0,
            }
            if return_attn:
                return y, att
            return y
        elif self.apply_window_control and not in_warmup:
            use_cache = (not self._should_update_control()) and self.cached_win_idx is not None
            if use_cache:
                path_mode = "grouped_control_cached" if (self.build_hierarchy and self.cached_head_to_group is not None) else "ungrouped_control_cached"
                win_idx = self.cached_win_idx
                head_to_group = self.cached_head_to_group
                group_change_rate = 0.0
                resolution_delta = 0.0
                if self.build_hierarchy:
                    group_counts_per_level = []
                    if self.cached_group_counts_per_level is not None:
                        group_counts_per_level = [int(v) for v in self.cached_group_counts_per_level]
                    elif head_to_group is not None:
                        group_counts_per_level = [int(torch.unique(head_to_group).numel())]
                    hierarchy_levels_used = len(group_counts_per_level)
                else:
                    hierarchy_levels_used = 1
                    group_counts_per_level = [int(self.n_head)]
                    self.cached_group_counts_per_level = list(group_counts_per_level)
            else:
                prev_win_idx = self.cached_win_idx.detach().clone() if self.cached_win_idx is not None else None
                if self.build_hierarchy:
                    path_mode = "grouped_control_update"
                    feats = self._head_features(q, k, v)
                    feature_probe_stats = self._feature_separability_stats(feats)
                    if self.ema_feats is None or self.ema_feats.shape != feats.shape:
                        self.ema_feats = feats.detach().float()
                    else:
                        alpha = max(0.0, min(1.0, self.ema_alpha))
                        self.ema_feats = alpha * self.ema_feats + (1.0 - alpha) * feats.detach().float()
                    prev_groups = None
                    if self.cached_head_to_group is not None:
                        prev_groups = self.cached_head_to_group
                    levels, cluster_debug = self._build_hierarchy(self.ema_feats, prev_head_groups=prev_groups, return_debug=True)
                    hierarchy_levels_used = len(levels)
                    group_counts_per_level = [len(groups) for groups, _ in levels]
                    self.cached_group_counts_per_level = list(group_counts_per_level)
                    hierarchy_head_group_map_per_level = self._head_group_map_per_level(levels)
                    hierarchy_group_members_per_level = self._group_members_from_head_maps(hierarchy_head_group_map_per_level)
                    parent_maps = self._parent_maps(levels)
                    group_win_idx, win_debug = self._select_windows(levels, parent_maps, device=x.device, return_debug=True)
                    controller_logits_std_per_level = win_debug["logits_std_per_level"]
                    groups0 = levels[0][0]
                    head_to_group = torch.empty(self.n_head, dtype=torch.long, device=x.device)
                    for gi, heads in enumerate(groups0):
                        for h in heads:
                            head_to_group[h] = gi
                    win_idx = group_win_idx[head_to_group]
                    win_idx_pre_clamp = win_debug["pre_clamp_level0"][head_to_group]
                    win_idx_post_clamp = win_debug["post_clamp_level0"][head_to_group]
                    if self.cached_head_to_group is None:
                        group_change_rate = 0.0
                    else:
                        group_change_rate = (head_to_group != self.cached_head_to_group).float().mean().item()
                    self.cached_head_to_group = head_to_group.detach().clone()
                    self.cached_group_feats = None
                    self.cached_group_feats_step = None
                else:
                    path_mode = "ungrouped_control_update"
                    head_to_group = None
                    group_change_rate = None
                    feats = self._head_features(q, k, v)
                    feature_probe_stats = self._feature_separability_stats(feats)
                    logits = self.controller(feats.float()).float()
                    controller_logits_std_per_level = [float(logits.std(unbiased=False).item())]
                    win_idx = logits.argmax(dim=-1)
                    win_idx_pre_clamp = win_idx.detach().clone()
                    win_idx_post_clamp = win_idx.detach().clone()
                    hierarchy_levels_used = 1
                    group_counts_per_level = [int(self.n_head)]
                    self.cached_group_counts_per_level = list(group_counts_per_level)
                if prev_win_idx is not None:
                    resolution_delta = float((win_idx.float() - prev_win_idx.float()).abs().mean().item())
                self.cached_win_idx = win_idx.detach().clone()
                self.last_control_step = self.current_step
        else:
            path_mode = "warmup_or_disabled_control" if self.apply_window_control else "grouping_only_full_attention"
            head_to_group = None
            group_change_rate = None
            if self.build_hierarchy:
                if self.cached_head_to_group is not None:
                    head_to_group = self.cached_head_to_group
                    groups0 = self._groups_from_head_to_group(head_to_group)
                    group_feats = self._get_cached_group_feats(q, k, v, groups0)
                    levels, cluster_debug = self._build_group_hierarchy(groups0, group_feats, return_debug=True)
                    hierarchy_levels_used = len(levels)
                    group_counts_per_level = [len(groups) for groups, _ in levels]
                    self.cached_group_counts_per_level = list(group_counts_per_level)
                    hierarchy_head_group_map_per_level = self._head_group_map_per_level(levels)
                    hierarchy_group_members_per_level = self._group_members_from_head_maps(hierarchy_head_group_map_per_level)
                    if self.apply_window_control:
                        parent_maps = self._parent_maps(levels)
                        group_win_idx, _ = self._select_windows(levels, parent_maps, device=x.device, return_debug=True)
                        shadow_win_idx = group_win_idx[head_to_group]
                        logits = self.controller(levels[0][1].float()).float()
                        controller_logits_std_per_level = [float(logits.std(unbiased=False).item())]
                        shadow_logit_mean = logits.mean(dim=0).detach().cpu().tolist()
                    group_change_rate = 0.0
                else:
                    feats = self._head_features(q, k, v)
                    feature_probe_stats = self._feature_separability_stats(feats)
                    if self.ema_feats is None or self.ema_feats.shape != feats.shape:
                        self.ema_feats = feats.detach().float()
                    else:
                        alpha = max(0.0, min(1.0, self.ema_alpha))
                        self.ema_feats = alpha * self.ema_feats + (1.0 - alpha) * feats.detach().float()
                    prev_groups = None
                    if self.cached_head_to_group is not None:
                        prev_groups = self.cached_head_to_group
                    levels, cluster_debug = self._build_hierarchy(self.ema_feats, prev_head_groups=prev_groups, return_debug=True)
                    hierarchy_levels_used = len(levels)
                    group_counts_per_level = [len(groups) for groups, _ in levels]
                    self.cached_group_counts_per_level = list(group_counts_per_level)
                    hierarchy_head_group_map_per_level = self._head_group_map_per_level(levels)
                    hierarchy_group_members_per_level = self._group_members_from_head_maps(hierarchy_head_group_map_per_level)
                    if self.apply_window_control:
                        parent_maps = self._parent_maps(levels)
                        group_win_idx, _ = self._select_windows(levels, parent_maps, device=x.device, return_debug=True)
                    else:
                        group_win_idx = None
                    groups0 = levels[0][0]
                    head_to_group = torch.empty(self.n_head, dtype=torch.long, device=x.device)
                    for gi, heads in enumerate(groups0):
                        for h in heads:
                            head_to_group[h] = gi
                    if self.apply_window_control and group_win_idx is not None:
                        shadow_win_idx = group_win_idx[head_to_group]
                        logits = self.controller(levels[0][1].float()).float()
                        controller_logits_std_per_level = [float(logits.std(unbiased=False).item())]
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
        if self.control_enabled and not in_warmup and self.resolution_ema_alpha > 0.0:
            alpha = max(0.0, min(1.0, self.resolution_ema_alpha))
            win_idx_f = win_idx.float()
            if self.ema_win_idx is None or self.ema_win_idx.shape != win_idx_f.shape:
                self.ema_win_idx = win_idx_f.detach().clone()
            else:
                self.ema_win_idx = alpha * self.ema_win_idx + (1.0 - alpha) * win_idx_f.detach()
            win_idx = self.ema_win_idx.round().clamp(0, len(self.windows) - 1).to(torch.long)
        if (
            self.control_enabled
            and not in_warmup
            and self.post_warmup_ramp_steps > 0
            and self.current_step is not None
        ):
            ramp_progress = (self.current_step - self.warmup_steps + 1) / max(1, self.post_warmup_ramp_steps)
            ramp_progress = max(0.0, min(1.0, float(ramp_progress)))
            if ramp_progress < 1.0:
                full_idx = self.windows.index(T) if T in self.windows else 0
                full_idx_f = torch.full_like(win_idx, full_idx, dtype=torch.float32)
                win_idx_f = win_idx.float()
                win_idx = (ramp_progress * win_idx_f + (1.0 - ramp_progress) * full_idx_f).round().to(torch.long)
        control_time_ms = (time.perf_counter() - t_control0) * 1000.0 if (self.apply_window_control or self.build_hierarchy) else 0.0

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
            norms = y_h.float().norm(dim=-1).mean(dim=(0, 2))
            for i, h in enumerate(heads):
                head_norms[h] = float(norms[i].item())
            att_h = att_f.view(Bc, Hc, att_f.size(1), att_f.size(2))
            att_h_f = att_h.float()
            ent = -(att_h_f * (att_h_f + 1e-9).log()).sum(dim=-1).mean(dim=(0, 2))
            ent = ent / max(1.0, math.log(att_h.size(-1)))
            usage = att_h_f[..., -1].mean(dim=(0, 2))
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
        resolution_std = float(w.std(unbiased=False).item()) if w.numel() > 1 else 0.0
        min_window = float(min(self.windows))
        max_window = float(max(self.windows))
        resolution_min_frac = float((w == min_window).float().mean().item())
        resolution_max_frac = float((w == max_window).float().mean().item())
        resolution_collapse_min = resolution_min_frac >= self.resolution_collapse_min_frac
        resolution_collapse_max = resolution_max_frac >= self.resolution_collapse_max_frac
        group_heads = {}
        group_ratios = {}
        branch_usage_freq = {}
        if head_to_group is not None:
            for h in range(self.n_head):
                g = int(head_to_group[h].item())
                group_heads.setdefault(g, []).append(h)
            for g, heads in group_heads.items():
                win_vals = [float(w[int(win_idx[h].item())].item()) for h in heads]
                avg_w = sum(win_vals) / max(1, len(win_vals))
                group_ratios[g] = avg_w / float(lq)
        for window, heads in head_groups.items():
            branch_usage_freq[int(window)] = len(heads) / float(self.n_head)
        cluster_metric_per_level = [str(d.get("cluster_metric", "cosine")) for d in cluster_debug]
        cluster_threshold_kind_per_level = [str(d.get("threshold_kind", "")) for d in cluster_debug]
        cluster_threshold_per_level = [float(d.get("threshold", 0.0)) for d in cluster_debug]
        cluster_item_count_per_level = [int(d.get("n_items", 0)) for d in cluster_debug]
        cluster_groups_before_merge_per_level = [int(d.get("groups_before_merge", 0)) for d in cluster_debug]
        cluster_groups_after_merge_per_level = [int(d.get("groups_after_merge", 0)) for d in cluster_debug]
        cluster_groups_merged_per_level = [int(d.get("groups_merged", 0)) for d in cluster_debug]
        cluster_small_groups_before_merge_per_level = [int(d.get("small_groups_before_merge", 0)) for d in cluster_debug]
        cluster_singletons_before_merge_per_level = [int(d.get("singleton_groups_before_merge", 0)) for d in cluster_debug]
        cluster_sim_mean_per_level = [float(d.get("sim_mean", 1.0)) for d in cluster_debug]
        cluster_sim_std_per_level = [float(d.get("sim_std", 0.0)) for d in cluster_debug]
        cluster_sim_min_per_level = [float(d.get("sim_min", 1.0)) for d in cluster_debug]
        cluster_sim_max_per_level = [float(d.get("sim_max", 1.0)) for d in cluster_debug]
        cluster_forced_bipartition_per_level = [bool(d.get("forced_bipartition", False)) for d in cluster_debug]
        cluster_force_split_anchor_similarity_per_level = [
            float(d.get("force_split_anchor_similarity")) if d.get("force_split_anchor_similarity") is not None else None
            for d in cluster_debug
        ]
        cluster_origin_per_level = [str(d.get("cluster_origin", "")) for d in cluster_debug]
        cluster_forced_bipartition_allowed_per_level = [bool(d.get("forced_bipartition_allowed", False)) for d in cluster_debug]
        cluster_groups_before_force_per_level = [int(d.get("groups_before_force", 0)) for d in cluster_debug]
        cluster_feature_norm_mean_per_level = [float(d.get("feature_norm_mean", 0.0)) for d in cluster_debug]
        cluster_feature_norm_std_per_level = [float(d.get("feature_norm_std", 0.0)) for d in cluster_debug]
        cluster_feature_dim_var_mean_per_level = [float(d.get("feature_dim_var_mean", 0.0)) for d in cluster_debug]
        cluster_feature_dim_var_std_per_level = [float(d.get("feature_dim_var_std", 0.0)) for d in cluster_debug]
        cluster_feature_l2_dist_mean_per_level = [float(d.get("feature_l2_dist_mean", 0.0)) for d in cluster_debug]
        cluster_feature_l2_dist_std_per_level = [float(d.get("feature_l2_dist_std", 0.0)) for d in cluster_debug]
        cluster_feature_top_singular_ratio_per_level = [float(d.get("feature_top_singular_ratio", 1.0)) for d in cluster_debug]
        hierarchy_level_added_per_level = [bool(d.get("hierarchy_level_added", True)) for d in cluster_debug]
        hierarchy_growth_stopped_per_level = [bool(d.get("hierarchy_growth_stopped", False)) for d in cluster_debug]
        hierarchy_stop_reason_per_level = [str(d.get("hierarchy_stop_reason", "")) for d in cluster_debug]
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
            "branch_usage_freq": branch_usage_freq,
            "control_time_ms": control_time_ms,
            "attn_time_ms": attn_time_ms,
            "overhead_time_ms": max(0.0, total_time_ms - attn_time_ms),
            "mask_time_ms": mask_time_ms,
            "lk_mean": lk_mean,
            "lk_p90": lk_p90,
            "w_mean": w_mean,
            "w_min": w_min,
            "w_max": w_max,
            "resolution_per_head": lk_list,
            "resolution_mean": w_mean,
            "resolution_std": resolution_std,
            "resolution_min_frac": resolution_min_frac,
            "resolution_max_frac": resolution_max_frac,
            "resolution_collapse_min": resolution_collapse_min,
            "resolution_collapse_max": resolution_collapse_max,
            "resolution_delta": resolution_delta,
            "hierarchy_levels_used": hierarchy_levels_used,
            "group_counts_per_level": group_counts_per_level,
            "controller_logits_std_per_level": controller_logits_std_per_level,
            "path_mode": path_mode,
            "win_idx_pre_clamp": win_idx_pre_clamp.detach().cpu().tolist() if win_idx_pre_clamp is not None else [],
            "win_idx_post_clamp": win_idx_post_clamp.detach().cpu().tolist() if win_idx_post_clamp is not None else [],
            "cluster_metric_per_level": cluster_metric_per_level,
            "cluster_threshold_kind_per_level": cluster_threshold_kind_per_level,
            "cluster_threshold_per_level": cluster_threshold_per_level,
            "cluster_item_count_per_level": cluster_item_count_per_level,
            "cluster_groups_before_merge_per_level": cluster_groups_before_merge_per_level,
            "cluster_groups_after_merge_per_level": cluster_groups_after_merge_per_level,
            "cluster_groups_merged_per_level": cluster_groups_merged_per_level,
            "cluster_small_groups_before_merge_per_level": cluster_small_groups_before_merge_per_level,
            "cluster_singletons_before_merge_per_level": cluster_singletons_before_merge_per_level,
            "cluster_sim_mean_per_level": cluster_sim_mean_per_level,
            "cluster_sim_std_per_level": cluster_sim_std_per_level,
            "cluster_sim_min_per_level": cluster_sim_min_per_level,
            "cluster_sim_max_per_level": cluster_sim_max_per_level,
            "cluster_forced_bipartition_per_level": cluster_forced_bipartition_per_level,
            "cluster_force_split_anchor_similarity_per_level": cluster_force_split_anchor_similarity_per_level,
            "cluster_origin_per_level": cluster_origin_per_level,
            "cluster_forced_bipartition_allowed_per_level": cluster_forced_bipartition_allowed_per_level,
            "cluster_groups_before_force_per_level": cluster_groups_before_force_per_level,
            "cluster_feature_norm_mean_per_level": cluster_feature_norm_mean_per_level,
            "cluster_feature_norm_std_per_level": cluster_feature_norm_std_per_level,
            "cluster_feature_dim_var_mean_per_level": cluster_feature_dim_var_mean_per_level,
            "cluster_feature_dim_var_std_per_level": cluster_feature_dim_var_std_per_level,
            "cluster_feature_l2_dist_mean_per_level": cluster_feature_l2_dist_mean_per_level,
            "cluster_feature_l2_dist_std_per_level": cluster_feature_l2_dist_std_per_level,
            "cluster_feature_top_singular_ratio_per_level": cluster_feature_top_singular_ratio_per_level,
            "hierarchy_level_added_per_level": hierarchy_level_added_per_level,
            "hierarchy_growth_stopped_per_level": hierarchy_growth_stopped_per_level,
            "hierarchy_stop_reason_per_level": hierarchy_stop_reason_per_level,
            "cluster_min_group_size": int(self.min_group_size),
            "cluster_sim_threshold": float(self.sim_threshold),
            "cluster_super_threshold": float(self.super_threshold),
            "feature_dim_var_mean": float(feature_probe_stats.get("feature_dim_var_mean", 0.0)),
            "feature_dim_var_std": float(feature_probe_stats.get("feature_dim_var_std", 0.0)),
            "feature_dim_var_min": float(feature_probe_stats.get("feature_dim_var_min", 0.0)),
            "feature_dim_var_max": float(feature_probe_stats.get("feature_dim_var_max", 0.0)),
            "feature_cos_sim_mean": float(feature_probe_stats.get("feature_cos_sim_mean", 1.0)),
            "feature_cos_sim_std": float(feature_probe_stats.get("feature_cos_sim_std", 0.0)),
            "feature_cos_sim_min": float(feature_probe_stats.get("feature_cos_sim_min", 1.0)),
            "feature_cos_sim_max": float(feature_probe_stats.get("feature_cos_sim_max", 1.0)),
            "feature_l2_dist_mean": float(feature_probe_stats.get("feature_l2_dist_mean", 0.0)),
            "feature_l2_dist_std": float(feature_probe_stats.get("feature_l2_dist_std", 0.0)),
            "feature_l2_dist_min": float(feature_probe_stats.get("feature_l2_dist_min", 0.0)),
            "feature_l2_dist_max": float(feature_probe_stats.get("feature_l2_dist_max", 0.0)),
            "feature_norm_mean": float(feature_probe_stats.get("feature_norm_mean", 0.0)),
            "feature_norm_std": float(feature_probe_stats.get("feature_norm_std", 0.0)),
            "feature_top_singular_ratio": float(feature_probe_stats.get("feature_top_singular_ratio", 1.0)),
            "hierarchy_head_group_map_per_level": hierarchy_head_group_map_per_level,
            "hierarchy_group_members_per_level": hierarchy_group_members_per_level,
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
