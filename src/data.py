import time
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


class TokenBlockDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def _load_wikitext(dataset_name, retries=5, backoff=5.0):
    last_err = None
    for attempt in range(retries):
        try:
            return load_dataset("wikitext", dataset_name)
        except Exception as e:
            last_err = e
            time.sleep(backoff * (2 ** attempt))
    raise RuntimeError(f"Failed to load dataset after {retries} attempts: {last_err}") from last_err


def build_dataloaders(dataset_name, tokenizer_name, seq_len, batch_size, num_workers):
    ds = _load_wikitext(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode_split(split):
        texts = ds[split]["text"]
        enc = tokenizer("\n\n".join(texts), return_tensors="pt")
        return enc["input_ids"].squeeze(0)

    train_tokens = encode_split("train")
    val_tokens = encode_split("validation")

    train_ds = TokenBlockDataset(train_tokens, seq_len)
    val_ds = TokenBlockDataset(val_tokens, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    return train_loader, val_loader, tokenizer.vocab_size
