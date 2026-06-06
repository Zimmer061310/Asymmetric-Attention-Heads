import os
import sys
import unittest
from collections import Counter

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.transformer import GPT, GPTConfig


def tiny_config(mode="adaptive", fixed_window=0):
    return GPTConfig(
        vocab_size=64,
        seq_len=16,
        n_layer=2,
        n_head=4,
        n_embd=32,
        n_ff=64,
        dropout=0.0,
        aah_v3_enabled=True,
        aah_v3_windows=(4, 8, 16),
        aah_v3_control_dim=8,
        aah_v3_control_interval=1,
        aah_v3_max_depth=1,
        aah_v3_min_group_size=1,
        aah_v3_warmup_steps=0,
        aah_v3_resolution_ema_alpha=0.0,
        aah_v3_group_feature_mode="mean",
        aah_v3_controller_input_mode="enriched",
        aah_v3_controller_pairwise_mode="joint_sibling",
        aah_v3_joint_hidden_dim=8,
        aah_v3_diagnostic_detail="minimal",
        aah_v3_hierarchy_ablation_mode="freeze_learned_topology",
        aah_v3_window_ablation_mode=mode,
        aah_v3_fixed_window=fixed_window,
        aah_v3_window_ablation_seed=123,
    )


class AAHWindowAblationTests(unittest.TestCase):
    def test_shuffle_preserves_selected_window_histogram(self):
        model = GPT(tiny_config("shuffle_post_select"))
        attn = model.blocks[0].attn
        attn.set_step(7)
        before = torch.tensor([0, 1, 1, 2, 2, 2], dtype=torch.long)
        after, debug = attn._apply_window_ablation(before)
        self.assertEqual(Counter(before.tolist()), Counter(after.tolist()))
        self.assertTrue(debug["preserves_histogram"])
        self.assertEqual(debug["mode"], "shuffle_post_select")

    def test_random_uniform_is_deterministic_for_step_and_layer(self):
        model = GPT(tiny_config("random_uniform"))
        attn = model.blocks[1].attn
        attn.set_step(11)
        before = torch.zeros(12, dtype=torch.long)
        after1, debug1 = attn._apply_window_ablation(before)
        after2, debug2 = attn._apply_window_ablation(before)
        self.assertTrue(torch.equal(after1, after2))
        self.assertFalse(debug1["preserves_histogram"])
        self.assertEqual(debug2["mode"], "random_uniform")

    def test_fixed_window_maps_every_head_to_configured_window(self):
        model = GPT(tiny_config("fixed_window", fixed_window=8))
        attn = model.blocks[0].attn
        attn.set_step(3)
        before = torch.tensor([0, 1, 2, 2], dtype=torch.long)
        after, debug = attn._apply_window_ablation(before)
        self.assertEqual(after.tolist(), [1, 1, 1, 1])
        self.assertEqual(debug["mode"], "fixed_window")

    def test_all_window_ablation_modes_run_finite_forward(self):
        modes = [
            ("adaptive", 0),
            ("shuffle_post_select", 0),
            ("random_uniform", 0),
            ("fixed_window", 8),
        ]
        idx = torch.randint(0, 64, (2, 16), dtype=torch.long)
        for mode, fixed_window in modes:
            torch.manual_seed(0)
            model = GPT(tiny_config(mode, fixed_window=fixed_window))
            model.eval()
            for block in model.blocks:
                block.attn.set_step(5)
            with torch.no_grad():
                logits, loss = model(idx, idx)
            self.assertTrue(torch.isfinite(logits).all(), mode)
            self.assertTrue(torch.isfinite(loss), mode)
            stats = model.blocks[0].attn.last_stats
            self.assertEqual(stats.get("window_ablation_mode"), mode)


if __name__ == "__main__":
    unittest.main()
