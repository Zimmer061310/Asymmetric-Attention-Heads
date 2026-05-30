import os
import sys
import tempfile
import unittest

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data import build_dataloaders


class DataLoaderTests(unittest.TestCase):
    def test_tokenized_file_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tokens.pt")
            torch.save(
                {
                    "train": torch.arange(33, dtype=torch.long),
                    "validation": torch.arange(17, dtype=torch.long),
                    "vocab_size": 64,
                },
                path,
            )
            train_loader, val_loader, vocab_size = build_dataloaders(
                f"tokenized:{path}",
                "ignored",
                seq_len=8,
                batch_size=2,
                num_workers=0,
            )
            self.assertEqual(vocab_size, 64)
            x, y = next(iter(train_loader))
            self.assertEqual(x.shape, (2, 8))
            self.assertEqual(y.shape, (2, 8))
            vx, vy = next(iter(val_loader))
            self.assertEqual(vx.shape, (2, 8))
            self.assertEqual(vy.shape, (2, 8))


if __name__ == "__main__":
    unittest.main()
