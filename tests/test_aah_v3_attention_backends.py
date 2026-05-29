import os
import sys
import unittest

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.transformer import AAHV3Attention, GPTConfig


def tiny_config(**overrides):
    cfg = GPTConfig(
        vocab_size=64,
        seq_len=8,
        n_layer=1,
        n_head=4,
        n_embd=16,
        n_ff=32,
        dropout=0.0,
        aah_v3_enabled=True,
        aah_v3_windows=(2, 4, 8),
        aah_v3_control_dim=8,
        aah_v3_control_interval=1,
        aah_v3_warmup_steps=0,
        aah_v3_grouping_enabled=False,
        aah_v3_build_hierarchy=False,
        aah_v3_apply_window_control=True,
        aah_v3_control_enabled=True,
        aah_v3_W_min_gpu=1,
        aah_v3_controller_choice_mode="fixed_window_4",
        aah_v3_controller_pairwise_mode="none",
        aah_v3_attention_backend="dense_masked",
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


class AAHV3AttentionBackendTests(unittest.TestCase):
    def test_window_mask_is_trailing_causal(self):
        attn = AAHV3Attention(tiny_config())
        mask, _ = attn._get_window_mask(T=8, W=3, device=torch.device("cpu"))
        expected = torch.zeros(8, 8, dtype=torch.bool)
        for q_idx in range(8):
            lo = max(0, q_idx - 2)
            expected[q_idx, lo : q_idx + 1] = True
        self.assertTrue(torch.equal(mask.cpu(), expected))

    def test_fixed_window_choice_mode_selects_nearest_bucket(self):
        attn = AAHV3Attention(tiny_config(aah_v3_controller_choice_mode="fixed_window_4"))
        idx = attn._oracle_window_indices(6, torch.device("cpu"))
        self.assertEqual(idx.tolist(), [1, 1, 1, 1, 1, 1])

    def test_flash_backend_falls_back_to_dense_on_cpu(self):
        attn = AAHV3Attention(tiny_config(aah_v3_attention_backend="flash_attn"))
        q = torch.randn(1, 2, 8, 4)
        k = torch.randn(1, 2, 8, 4)
        v = torch.randn(1, 2, 8, 4)
        y, weights, _, info = attn._execute_attention_bucket(q, k, v, window=4)
        self.assertEqual(y.shape, q.shape)
        self.assertIsNotNone(weights)
        self.assertEqual(info["backend"], "dense_masked")
        self.assertIn("flash_attn_requires_cuda", info["fallback_reason"])

    def test_forward_logs_effective_and_backend_metrics(self):
        attn = AAHV3Attention(tiny_config())
        x = torch.randn(1, 8, 16)
        y = attn(x)
        self.assertEqual(y.shape, x.shape)
        self.assertAlmostEqual(attn.last_stats["effective_ACR"], 0.5)
        self.assertAlmostEqual(attn.last_stats["backend_realized_ACR_est"], 1.0)
        self.assertEqual(attn.last_stats["backend_name"], "dense_masked")


if __name__ == "__main__":
    unittest.main()
