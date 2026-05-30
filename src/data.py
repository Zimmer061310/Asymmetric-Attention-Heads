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
        x = chunk[:-1].detach().clone().to(dtype=torch.long)
        y = chunk[1:].detach().clone().to(dtype=torch.long)
        return x, y


def _parse_synthetic_vocab(tokenizer_name):
    if isinstance(tokenizer_name, str) and tokenizer_name.startswith("synthetic:"):
        return int(tokenizer_name.split(":", 1)[1])
    return 50304


def _build_synthetic_dataloaders(tokenizer_name, seq_len, batch_size, num_workers):
    vocab_size = _parse_synthetic_vocab(tokenizer_name)
    train_blocks = 256
    val_blocks = 32
    train_gen = torch.Generator().manual_seed(1234)
    val_gen = torch.Generator().manual_seed(5678)
    train_tokens = torch.randint(0, vocab_size, (seq_len * train_blocks + 1,), generator=train_gen)
    val_tokens = torch.randint(0, vocab_size, (seq_len * val_blocks + 1,), generator=val_gen)

    train_ds = TokenBlockDataset(train_tokens, seq_len)
    val_ds = TokenBlockDataset(val_tokens, seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    return train_loader, val_loader, vocab_size


def _load_tokenized_payload(path):
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        train_tokens = payload.get("train")
        val_tokens = payload.get("validation", payload.get("val"))
        vocab_size = payload.get("vocab_size")
    elif isinstance(payload, (list, tuple)) and len(payload) >= 2:
        train_tokens, val_tokens = payload[0], payload[1]
        vocab_size = payload[2] if len(payload) >= 3 else None
    else:
        raise ValueError(
            "Tokenized dataset must be a dict with train/validation tensors "
            "or a tuple/list of (train_tokens, val_tokens[, vocab_size])."
        )
    if train_tokens is None or val_tokens is None:
        raise ValueError(f"Tokenized dataset {path} is missing train or validation tokens.")
    train_tokens = torch.as_tensor(train_tokens, dtype=torch.long).cpu()
    val_tokens = torch.as_tensor(val_tokens, dtype=torch.long).cpu()
    if vocab_size is None:
        vocab_size = int(max(train_tokens.max().item(), val_tokens.max().item()) + 1)
    return train_tokens, val_tokens, int(vocab_size)


def _build_tokenized_file_dataloaders(path, seq_len, batch_size, num_workers):
    train_tokens, val_tokens, vocab_size = _load_tokenized_payload(path)
    train_ds = TokenBlockDataset(train_tokens, seq_len)
    val_ds = TokenBlockDataset(val_tokens, seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    return train_loader, val_loader, vocab_size


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
    dataset_name = str(dataset_name)
    if dataset_name.lower() == "synthetic":
        return _build_synthetic_dataloaders(tokenizer_name, seq_len, batch_size, num_workers)
    if dataset_name.lower() == "tokenized":
        return _build_tokenized_file_dataloaders(str(tokenizer_name), seq_len, batch_size, num_workers)
    if dataset_name.startswith("tokenized:"):
        return _build_tokenized_file_dataloaders(dataset_name.split(":", 1)[1], seq_len, batch_size, num_workers)

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
