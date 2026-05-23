#!/usr/bin/env python3
"""AAH-style local attention patch for Hugging Face Qwen/Llama causal LMs.

This module is intentionally self-contained so it can be imported by smoke,
adaptation, and benchmark scripts without changing the custom GPT code path.
It targets attention modules exposing q_proj/k_proj/v_proj/o_proj and common
Qwen/Llama metadata such as num_heads, num_key_value_heads, and head_dim.
"""

from __future__ import annotations

import csv
import json
import math
import os
import types
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QwenAAHConfig:
    regime: str = "grouping_off"
    windows: Tuple[int, ...] = (512, 1024, 2048, 4096)
    control_interval: int = 5
    feature_ema_alpha: float = 0.9
    resolution_ema_alpha: float = 0.15
    controller_hidden_dim: int = 64
    parent_constraint: bool = True
    freeze_learned_topology: bool = False
    reuse_group_hierarchy: bool = False
    max_depth: int = 4
    min_group_size: int = 2
    diagnostic_detail: str = "light"


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    while cos.dim() < q.dim():
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, n_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, n_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(bsz, n_kv_heads * n_rep, slen, head_dim)


def build_local_mask(t: int, window: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(t, device=device)
    causal = pos[:, None] >= pos[None, :]
    if window >= t:
        return causal
    local = (pos[:, None] - pos[None, :]) < int(window)
    return causal & local


class AAHHeadController(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int, n_windows: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_windows),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)


class AAHRuntimeState(nn.Module):
    def __init__(self, layer_idx: int, n_heads: int, config: QwenAAHConfig):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.n_heads = int(n_heads)
        self.config = config
        self.controller = AAHHeadController(8, int(config.controller_hidden_dim), len(config.windows))
        self.register_buffer("ema_features", torch.zeros(n_heads, 8), persistent=False)
        self.register_buffer("ema_window_idx", torch.zeros(n_heads), persistent=False)
        self.feature_initialized = False
        self.step = 0
        self.cached_raw_idx: Optional[torch.Tensor] = None
        self.cached_final_idx: Optional[torch.Tensor] = None
        self.cached_head_to_group: Optional[torch.Tensor] = None
        self.last_stats: Dict[str, object] = {}

    def set_eval_mode(self, enabled: bool) -> None:
        self.training = not bool(enabled)

    def _features(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        qf = q.detach().float()
        kf = k.detach().float()
        vf = v.detach().float()
        feats = torch.stack(
            [
                # Mean absolute activation, not absolute value of the scalar mean.
                qf.abs().mean(dim=(0, 2, 3)),
                qf.std(dim=(0, 2, 3)),
                kf.abs().mean(dim=(0, 2, 3)),
                kf.std(dim=(0, 2, 3)),
                vf.abs().mean(dim=(0, 2, 3)),
                vf.std(dim=(0, 2, 3)),
                torch.linspace(0.0, 1.0, self.n_heads, device=q.device),
                torch.full((self.n_heads,), float(self.layer_idx), device=q.device) / 100.0,
            ],
            dim=-1,
        )
        if self.feature_initialized:
            alpha = float(self.config.feature_ema_alpha)
            self.ema_features.mul_(alpha).add_(feats.to(self.ema_features.dtype) * (1.0 - alpha))
        else:
            self.ema_features.copy_(feats.to(self.ema_features.dtype))
            self.feature_initialized = True
        return self.ema_features.to(device=q.device, dtype=q.dtype)

    def _grouping(self, feats: torch.Tensor) -> torch.Tensor:
        if self.config.regime == "grouping_off":
            return torch.arange(self.n_heads, device=feats.device)
        if self.cached_head_to_group is not None and (
            self.config.freeze_learned_topology or self.config.reuse_group_hierarchy
        ):
            return self.cached_head_to_group.to(feats.device)
        # Stable adjacent-pair topology. It is deliberately conservative for
        # pretrained retrofits and sufficient to exercise grouped execution.
        group_size = max(1, int(self.config.min_group_size))
        head_to_group = torch.arange(self.n_heads, device=feats.device) // group_size
        self.cached_head_to_group = head_to_group.detach().cpu()
        return head_to_group

    def controller_logits(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self._features(q, k, v)
        head_to_group = self._grouping(feats)
        n_groups = int(head_to_group.max().item()) + 1
        group_feats = torch.zeros(n_groups, feats.size(-1), device=feats.device, dtype=feats.dtype)
        group_counts = torch.zeros(n_groups, device=feats.device, dtype=feats.dtype)
        group_feats.index_add_(0, head_to_group, feats)
        group_counts.index_add_(0, head_to_group, torch.ones_like(head_to_group, dtype=feats.dtype))
        group_feats = group_feats / group_counts.clamp_min(1.0).unsqueeze(-1)

        logits = self.controller(group_feats)
        return head_to_group, logits

    def choose_windows(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        should_update = self.cached_final_idx is None or (self.step % max(1, int(self.config.control_interval)) == 0)
        self.step += 1
        if not should_update:
            return self.cached_raw_idx.to(q.device), self.cached_final_idx.to(q.device)

        head_to_group, logits = self.controller_logits(q, k, v)
        n_groups = logits.size(0)
        raw_group_idx = logits.argmax(dim=-1)
        final_group_idx = raw_group_idx
        if self.config.parent_constraint and self.config.regime in {"full_adaptive", "deep_practical_reuse", "shallow_freeze"}:
            # Conservative parent constraint: sibling groups may not jump more
            # than one bucket apart. This is a retrofit approximation of the
            # custom AAH-v3 hierarchy constraint.
            clamped = raw_group_idx.clone()
            for i in range(1, n_groups):
                lo = int(clamped[i - 1].item()) - 1
                hi = int(clamped[i - 1].item()) + 1
                clamped[i] = clamped[i].clamp(max(0, lo), min(len(self.config.windows) - 1, hi))
            final_group_idx = clamped

        raw_idx = raw_group_idx[head_to_group]
        final_idx = final_group_idx[head_to_group]
        if float(self.config.resolution_ema_alpha) > 0:
            alpha = float(self.config.resolution_ema_alpha)
            ema_update = final_idx.detach().float().to(self.ema_window_idx.device)
            self.ema_window_idx.mul_(1.0 - alpha).add_(ema_update * alpha)
            final_idx = self.ema_window_idx.round().long().clamp(0, len(self.config.windows) - 1).to(q.device)

        self.cached_raw_idx = raw_idx.detach().cpu()
        self.cached_final_idx = final_idx.detach().cpu()
        return raw_idx, final_idx


def grouped_local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    final_idx: torch.Tensor,
    windows: Tuple[int, ...],
    scale: float,
    attention_mask: Optional[torch.Tensor],
    output_attentions: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
    bsz, n_heads, t, head_dim = q.shape
    out = torch.empty_like(q)
    attn_weights_out = None
    if output_attentions:
        attn_weights_out = torch.zeros(bsz, n_heads, t, t, dtype=q.dtype, device=q.device)
    total_elements = 0.0
    for idx in torch.unique(final_idx).tolist():
        head_ids = torch.nonzero(final_idx == int(idx), as_tuple=False).flatten()
        window = min(int(windows[int(idx)]), t)
        qh = q[:, head_ids]
        kh = k[:, head_ids]
        vh = v[:, head_ids]
        scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
        mask = build_local_mask(t, window, q.device)
        scores = scores.masked_fill(~mask.view(1, 1, t, t), torch.finfo(scores.dtype).min)
        if attention_mask is not None and attention_mask.dim() == 4:
            scores = scores + attention_mask[:, :, :t, :t]
        elif attention_mask is not None and attention_mask.dim() == 2:
            pad_mask = attention_mask[:, None, None, :t].to(dtype=torch.bool)
            scores = scores.masked_fill(~pad_mask, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores.float(), dim=-1).to(q.dtype)
        out[:, head_ids] = torch.matmul(probs, vh)
        if attn_weights_out is not None:
            attn_weights_out[:, head_ids] = probs
        total_elements += float(bsz * len(head_ids) * t * window)
    baseline_elements = float(bsz * n_heads * t * t)
    stats = {
        "total_elements": total_elements,
        "baseline_elements": baseline_elements,
        "ACR": total_elements / baseline_elements if baseline_elements > 0 else 1.0,
    }
    return out, attn_weights_out, stats


def soft_window_local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    head_window_probs: torch.Tensor,
    windows: Tuple[int, ...],
    scale: float,
    attention_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    bsz, n_heads, t, _ = q.shape
    out = torch.zeros_like(q)
    total_elements = 0.0
    for idx, window_value in enumerate(windows):
        window = min(int(window_value), t)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask = build_local_mask(t, window, q.device)
        scores = scores.masked_fill(~mask.view(1, 1, t, t), torch.finfo(scores.dtype).min)
        if attention_mask is not None and attention_mask.dim() == 4:
            scores = scores + attention_mask[:, :, :t, :t]
        elif attention_mask is not None and attention_mask.dim() == 2:
            pad_mask = attention_mask[:, None, None, :t].to(dtype=torch.bool)
            scores = scores.masked_fill(~pad_mask, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores.float(), dim=-1).to(q.dtype)
        y = torch.matmul(probs, v)
        weight = head_window_probs[:, idx].to(dtype=q.dtype).view(1, n_heads, 1, 1)
        out = out + y * weight
        total_elements += float(bsz * t * window) * float(head_window_probs[:, idx].detach().float().sum().item())
    baseline_elements = float(bsz * n_heads * t * t)
    stats = {
        "total_elements": total_elements,
        "baseline_elements": baseline_elements,
        "ACR": total_elements / baseline_elements if baseline_elements > 0 else 1.0,
    }
    return out, stats


def make_aah_forward(attn: nn.Module, state: AAHRuntimeState):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        if past_key_value is not None or use_cache:
            # The paper diagnostics use full-prompt evaluation and greedy
            # generation with cache disabled. Fallback to original attention if
            # an external caller insists on KV cache semantics.
            return self._aah_original_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        bsz, q_len, _ = hidden_states.size()
        head_dim = int(getattr(self, "head_dim", 0) or 0)
        if head_dim <= 0:
            guessed_heads = int(getattr(self, "num_heads", getattr(self, "num_attention_heads", state.n_heads)))
            head_dim = int(self.q_proj.out_features // max(1, guessed_heads))
        # Qwen3 grouped-query attention exposes inconsistent metadata across
        # transformers versions, so derive both counts from projection widths.
        num_heads = int(self.q_proj.out_features // head_dim)
        num_kv_heads = int(self.k_proj.out_features // head_dim)
        q = self.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim)
        if hasattr(self, "q_norm"):
            q = self.q_norm(q)
        if hasattr(self, "k_norm"):
            k = self.k_norm(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        elif hasattr(self, "rotary_emb") and position_ids is not None:
            cos, sin = self.rotary_emb(v, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        n_rep = num_heads // max(1, num_kv_heads)
        k = repeat_kv(k, n_rep)
        v = repeat_kv(v, n_rep)

        scale = float(getattr(self, "scaling", head_dim ** -0.5))
        if state.training:
            head_to_group, logits = state.controller_logits(q, k, v)
            group_probs = F.softmax(logits.float(), dim=-1).to(q.dtype)
            head_window_probs = group_probs[head_to_group]
            raw_idx = head_window_probs.argmax(dim=-1)
            final_idx = raw_idx
            attn_output, stats = soft_window_local_attention(
                q,
                k,
                v,
                head_window_probs,
                state.config.windows,
                scale,
                attention_mask,
            )
            attn_weights = None
        else:
            raw_idx, final_idx = state.choose_windows(q, k, v)
            attn_output, attn_weights, stats = grouped_local_attention(
                q,
                k,
                v,
                final_idx,
                state.config.windows,
                scale,
                attention_mask,
                output_attentions=output_attentions,
            )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, num_heads * head_dim)
        attn_output = self.o_proj(attn_output)

        stats.update(
            {
                "layer_id": state.layer_idx,
                "selected_window_idx": [int(x) for x in final_idx.detach().cpu().tolist()],
                "pre_clamp_window_idx": [int(x) for x in raw_idx.detach().cpu().tolist()],
                "final_window_size": [int(state.config.windows[int(x)]) for x in final_idx.detach().cpu().tolist()],
            }
        )
        state.last_stats = stats
        return attn_output, attn_weights

    return types.MethodType(forward, attn)


def patch_model_attention(model: nn.Module, config: QwenAAHConfig) -> List[AAHRuntimeState]:
    states: List[AAHRuntimeState] = []
    layer_idx = 0
    for module in model.modules():
        if not all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj")):
            continue
        n_heads = int(getattr(module, "num_heads", getattr(module, "num_attention_heads", 0)))
        if n_heads <= 0:
            out_features = int(module.q_proj.out_features)
            head_dim = int(getattr(module, "head_dim", 0) or 128)
            n_heads = max(1, out_features // head_dim)
        state = AAHRuntimeState(layer_idx, n_heads, config)
        module.add_module("aah_state", state)
        module._aah_original_forward = module.forward
        module.forward = make_aah_forward(module, state)
        states.append(state)
        layer_idx += 1
    if not states:
        raise RuntimeError("No patchable attention modules found; expected q_proj/k_proj/v_proj/o_proj.")
    return states


def freeze_for_aah_adaptation(model: nn.Module, unfreeze_outputs: bool = False) -> None:
    for name, param in model.named_parameters():
        trainable = ".aah_state." in name
        if unfreeze_outputs and (".o_proj." in name or "norm" in name.lower()):
            trainable = True
        param.requires_grad_(trainable)


def collect_aah_summary(states: Iterable[AAHRuntimeState]) -> Dict[str, float]:
    states = list(states)
    acrs = []
    for state in states:
        acr = state.last_stats.get("ACR") if state.last_stats else None
        if acr is not None:
            acrs.append(float(acr))
    mean_acr = sum(acrs) / len(acrs) if acrs else 1.0
    return {"mean_ACR": mean_acr, "n_layers": len(states)}


def write_aah_diagnostics(states: Iterable[AAHRuntimeState], path: str, regime: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    for state in states:
        stats = state.last_stats or {}
        final_idx = stats.get("selected_window_idx", [])
        pre_idx = stats.get("pre_clamp_window_idx", [])
        final_sizes = stats.get("final_window_size", [])
        for head_id, idx in enumerate(final_idx):
            rows.append(
                {
                    "regime": regime,
                    "layer_id": state.layer_idx,
                    "head_id": head_id,
                    "selected_window_idx": idx,
                    "selected_window_size": final_sizes[head_id] if head_id < len(final_sizes) else "",
                    "pre_clamp_window_idx": pre_idx[head_id] if head_id < len(pre_idx) else idx,
                    "final_window_idx": idx,
                    "final_window_size": final_sizes[head_id] if head_id < len(final_sizes) else "",
                    "ACR": stats.get("ACR", ""),
                }
            )
    with open(path, "w", newline="") as f:
        fieldnames = [
            "regime",
            "layer_id",
            "head_id",
            "selected_window_idx",
            "selected_window_size",
            "pre_clamp_window_idx",
            "final_window_idx",
            "final_window_size",
            "ACR",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_aah_adapter(model: nn.Module, path: str, metadata: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items() if ".aah_state." in k},
        "metadata": metadata,
    }
    torch.save(payload, path)
    with open(f"{path}.json", "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def load_aah_adapter(model: nn.Module, path: str) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    # Missing pretrained weights are expected because adapter files only store AAH modules.
    hard_unexpected = [k for k in unexpected if ".aah_state." not in k]
    if hard_unexpected:
        raise RuntimeError(f"Unexpected adapter keys: {hard_unexpected[:8]}")
    return {"missing_count": len(missing), "unexpected_count": len(unexpected), **payload.get("metadata", {})}


def config_from_regime(regime: str) -> QwenAAHConfig:
    regime = str(regime)
    if regime == "grouping_off":
        return QwenAAHConfig(regime=regime, min_group_size=1, max_depth=0)
    if regime == "full_adaptive":
        return QwenAAHConfig(regime=regime, min_group_size=2, max_depth=4)
    if regime == "shallow_freeze":
        return QwenAAHConfig(regime=regime, min_group_size=2, max_depth=1, freeze_learned_topology=True)
    if regime == "deep_practical_reuse":
        return QwenAAHConfig(regime=regime, min_group_size=2, max_depth=4, reuse_group_hierarchy=True)
    raise ValueError(f"Unknown Qwen AAH regime: {regime}")


def config_to_json(config: QwenAAHConfig) -> str:
    payload = asdict(config)
    payload["windows"] = list(config.windows)
    return json.dumps(payload, sort_keys=True)
